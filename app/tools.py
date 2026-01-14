"""
Search Tools for Lexio Agent.

All tools use Railway PostgreSQL (pgvector) - no Qdrant.
Based on E-sbirka integration/mcp/tools.py pattern.
"""
import os
import asyncio
import hashlib
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

import asyncpg
import httpx

from app.config import settings


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class LawResult:
    """Law search result."""
    section_id: int
    citation: str
    law_citation: str
    law_title: str
    valid_from: Optional[str]
    text: str
    score: float
    source_url: str


@dataclass
class CaseResult:
    """Judgment search result."""
    judgment_id: int
    case_number: str
    court: str
    decision_date: Optional[str]
    text: str
    score: float


# =============================================================================
# DATABASE CONNECTION
# =============================================================================

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    """Get or create connection pool to Railway PostgreSQL."""
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
            min_size=1,
            max_size=5,
            command_timeout=30
        )
    return _pool


async def close_pool():
    """Close connection pool."""
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


# =============================================================================
# EMBEDDING CACHE
# =============================================================================

async def get_embedding(query: str) -> List[float]:
    """
    Get embedding for query. Uses PostgreSQL query_cache.
    """
    normalized = query.strip().lower()
    query_hash = hashlib.sha256(normalized.encode()).hexdigest()
    
    pool = await get_pool()
    
    # Check cache
    cached = await pool.fetchrow(
        "SELECT embedding FROM query_cache WHERE query_hash = $1",
        query_hash
    )
    
    if cached and cached['embedding']:
        await pool.execute(
            "UPDATE query_cache SET hit_count = hit_count + 1 WHERE query_hash = $1",
            query_hash
        )
        import ast
        return ast.literal_eval(cached['embedding'])
    
    # Call OpenAI
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://api.openai.com/v1/embeddings",
            headers={"Authorization": f"Bearer {settings.OPENAI_API_KEY}"},
            json={"model": settings.EMBEDDING_MODEL, "input": query}
        )
        response.raise_for_status()
        embedding = response.json()["data"][0]["embedding"]
    
    # Cache it
    try:
        await pool.execute(
            """INSERT INTO query_cache (query_text, query_hash, embedding)
               VALUES ($1, $2, $3)
               ON CONFLICT (query_hash) DO NOTHING""",
            normalized, query_hash, str(embedding)
        )
    except Exception:
        pass
    
    return embedding


# =============================================================================
# SEARCH TOOLS
# =============================================================================

async def search_laws(
    query: str,
    limit: int = 10
) -> List[LawResult]:
    """
    Hybrid search for laws (vector + BM25 with RRF).
    
    This is the PRIMARY search for legal questions.
    Laws take priority over judgments in legal reasoning.
    """
    embedding = await get_embedding(query)
    embedding_str = str(embedding)
    tsquery = ' | '.join(query.split())  # OR for better recall
    
    pool = await get_pool()
    
    sql = """
    WITH bm25_results AS (
        SELECT c.id, c.section_id,
               ROW_NUMBER() OVER (ORDER BY ts_rank(c.bm25_tokens, to_tsquery('simple', $1)) DESC) as rank
        FROM section_chunks c
        JOIN sections s ON c.section_id = s.id
        JOIN versions v ON s.version_id = v.id
        WHERE c.bm25_tokens @@ to_tsquery('simple', $1) AND v.is_current = true
        LIMIT 30
    ),
    vector_results AS (
        SELECT c.id, c.section_id,
               ROW_NUMBER() OVER (ORDER BY c.embedding <=> $2::vector) as rank
        FROM section_chunks c
        JOIN sections s ON c.section_id = s.id
        JOIN versions v ON s.version_id = v.id
        WHERE c.embedding IS NOT NULL AND v.is_current = true
        ORDER BY c.embedding <=> $2::vector
        LIMIT 30
    ),
    combined AS (
        SELECT COALESCE(b.section_id, v.section_id) as section_id,
               COALESCE(b.id, v.id) as chunk_id,
               (1.0 / (60 + COALESCE(b.rank, 1000))) + 
               (1.0 / (60 + COALESCE(v.rank, 1000))) as rrf_score
        FROM bm25_results b
        FULL OUTER JOIN vector_results v ON b.id = v.id
    )
    SELECT DISTINCT ON (s.id)
        s.id as section_id,
        COALESCE(s.parsed_section, s.citation) as citation,
        l.citation as law_citation,
        l.title as law_title,
        v.valid_from::text as valid_from,
        s.version_id,
        LEFT(c.chunk_text, 800) as text,
        comb.rrf_score as score
    FROM combined comb
    JOIN section_chunks c ON comb.chunk_id = c.id
    JOIN sections s ON c.section_id = s.id
    JOIN versions v ON s.version_id = v.id
    JOIN laws l ON v.law_id = l.id
    ORDER BY s.id, comb.rrf_score DESC, v.valid_from DESC
    """
    
    rows = await pool.fetch(f"""
        SELECT * FROM ({sql}) sub ORDER BY score DESC LIMIT $3
    """, tsquery, embedding_str, limit)
    
    return [
        LawResult(
            section_id=row['section_id'],
            citation=row['citation'],
            law_citation=row['law_citation'],
            law_title=row['law_title'],
            valid_from=row['valid_from'],
            text=row['text'],
            score=float(row['score']),
            source_url=f"https://www.e-sbirka.cz/{row['version_id']}"
        )
        for row in rows
    ]


async def search_judgments(
    query: str,
    court: Optional[str] = None,
    limit: int = 10
) -> List[CaseResult]:
    """
    Vector search for court judgments in PostgreSQL.
    
    SECONDARY to laws - use for legal interpretation and precedent.
    """
    embedding = await get_embedding(query)
    embedding_str = str(embedding)
    
    pool = await get_pool()
    
    sql = """
    SELECT 
        j.id as judgment_id,
        j.case_number,
        j.court,
        j.decision_date::text,
        LEFT(jc.chunk_text, 800) as text,
        1 - (jc.embedding <=> $1::vector) as score
    FROM judgment_chunks jc
    JOIN judgments j ON jc.judgment_id = j.id
    WHERE jc.embedding IS NOT NULL
    """
    
    params = [embedding_str]
    if court:
        sql += " AND j.court = $2"
        params.append(court)
    
    sql += f" ORDER BY jc.embedding <=> $1::vector LIMIT ${len(params) + 1}"
    params.append(limit)
    
    rows = await pool.fetch(sql, *params)
    
    return [
        CaseResult(
            judgment_id=row['judgment_id'],
            case_number=row['case_number'],
            court=row['court'],
            decision_date=row['decision_date'],
            text=row['text'],
            score=float(row['score']) if row['score'] else 0.0
        )
        for row in rows
    ]


async def get_full_section(law_citation: str, section_citation: str) -> Optional[Dict]:
    """Get full text of a specific law section."""
    pool = await get_pool()
    
    row = await pool.fetchrow("""
        SELECT s.id, s.citation, l.citation as law_citation, l.title as law_title,
               sc.full_text, s.version_id
        FROM sections s
        JOIN section_contents sc ON s.id = sc.section_id
        JOIN versions v ON s.version_id = v.id
        JOIN laws l ON v.law_id = l.id
        WHERE l.citation = $1 
          AND (s.citation LIKE $2 OR s.parsed_section LIKE $2)
          AND v.is_current = true
        LIMIT 1
    """, law_citation, f"%{section_citation}%")
    
    if row:
        return {
            "section_id": row['id'],
            "citation": row['citation'],
            "law_citation": row['law_citation'],
            "law_title": row['law_title'],
            "full_text": row['full_text'],
            "source_url": f"https://www.e-sbirka.cz/{row['version_id']}"
        }
    return None


async def get_full_judgment(case_number: str) -> Optional[Dict]:
    """Get full text of a specific judgment."""
    pool = await get_pool()
    
    row = await pool.fetchrow("""
        SELECT j.id, j.case_number, j.court, j.decision_date::text, jc.full_text
        FROM judgments j
        JOIN judgment_contents jc ON j.id = jc.judgment_id
        WHERE j.case_number ILIKE $1
        LIMIT 1
    """, f"%{case_number}%")
    
    if row:
        return {
            "judgment_id": row['id'],
            "case_number": row['case_number'],
            "court": row['court'],
            "decision_date": row['decision_date'],
            "full_text": row['full_text']
        }
    return None


async def web_search(query: str) -> Optional[Dict]:
    """
    Web search via Perplexity Sonar through OpenRouter.
    Use for current events, non-legal context, or when DB has no results.
    """
    async with httpx.AsyncClient(timeout=30) as client:
        response = await client.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "perplexity/sonar",
                "messages": [
                    {"role": "system", "content": "Odpověz stručně v češtině."},
                    {"role": "user", "content": query}
                ],
                "max_tokens": 500
            }
        )
        
        if response.status_code == 200:
            data = response.json()
            return {
                "answer": data["choices"][0]["message"]["content"],
                "citations": data.get("citations", [])
            }
    return None

