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
import logging

from app.config import settings

logger = logging.getLogger(__name__)


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
        logger.info(f"Creating DB pool to {settings.DATABASE_URL.split('@')[-1]}")
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
        logger.info("Closing DB pool")
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
# URL HELPER
# =============================================================================

def build_esbirka_url(version_id: str, citation: Optional[str] = None) -> str:
    """
    Build a URL to the official e-Sbírka website with fragment anchor.
    """
    import re
    
    base_url = "https://www.e-sbirka.cz"
    url = f"{base_url}/{version_id}"
    
    # Add fragment anchor if citation provided
    if citation:
        # § 51 → #par_51, Čl. 5 → #cl_5
        par_match = re.match(r'§\s*(\d+)', citation)
        cl_match = re.match(r'[Čč]l\.?\s*(\d+)', citation)
        
        if par_match:
            url += f"#par_{par_match.group(1)}"
        elif cl_match:
            url += f"#cl_{cl_match.group(1)}"
            
    return url


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
            source_url=build_esbirka_url(row['version_id'], row['citation'])
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
            "source_url": build_esbirka_url(row['version_id'], row['citation'])
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


# =============================================================================
# E-SBÍRKA API TOOLS (Direct API access)
# =============================================================================

async def search_esbirka_api(
    law_citation: str,
    section_citation: Optional[str] = None
) -> Optional[Dict]:
    """
    Search directly in the e-Sbírka API for a specific law and section.
    
    Use this as FALLBACK when search_laws doesn't find what you need.
    Returns the current official version from the source.
    
    Args:
        law_citation: Law citation (e.g., "262/2006 Sb.", "89/2012 Sb.")
        section_citation: Optional section (e.g., "§ 212", "§ 2235")
    
    Returns:
        Dict with section text and metadata, or None if not found
    """
    try:
        from app.esbirka_api import EsbirkaAPI
        
        api = EsbirkaAPI()
        
        if section_citation:
            # Find specific section
            fragment = await api.find_section(law_citation, section_citation)
            if fragment:
                return {
                    "citation": fragment.section_citation,
                    "full_citation": fragment.full_citation,
                    "text": fragment.text,
                    "is_current": fragment.is_current,
                    "source_url": f"https://www.e-sbirka.cz{fragment.stale_url}"
                }
        else:
            # Get all fragments for the law
            fragments = await api.get_law_fragments(law_citation)
            if fragments:
                return {
                    "law_citation": law_citation,
                    "fragments_count": len(fragments),
                    "sections": [
                        {
                            "citation": f.section_citation,
                            "text": f.text[:500] + "..." if len(f.text) > 500 else f.text
                        }
                        for f in fragments[:20]  # Limit to first 20
                    ]
                }
        
        return None
        
    except Exception as e:
        logger.error(f"E-Sbírka API error: {e}")
        return {"error": str(e)}


async def get_law_changes(since_days: int = 30) -> List[Dict]:
    """
    Get recent changes to laws from e-Sbírka API.
    
    Use this to check if a law has been recently amended or updated.
    Useful for ensuring you have the most current version.
    
    Args:
        since_days: How many days back to check (default 30)
    
    Returns:
        List of changed laws with metadata
    """
    try:
        from app.esbirka_api import EsbirkaAPI
        
        api = EsbirkaAPI()
        changes = await api.get_recent_changes(since_days)
        
        return [
            {
                "law_citation": change.get("citace", ""),
                "change_date": change.get("datumZmeny", ""),
                "change_type": change.get("typZmeny", ""),
                "effective_date": change.get("datumUcinnosti", "")
            }
            for change in changes[:20]  # Limit results
        ]
        
    except Exception as e:
        logger.error(f"E-Sbírka API changelog error: {e}")
        return []


async def get_historical_section(
    law_citation: str,
    section_citation: str,
    date: str
) -> Optional[Dict]:
    """
    Get a historical version of a law section as of a specific date.
    
    Use this when analyzing what the law said at a specific point in time,
    e.g., when an event occurred or for retroactive analysis.
    
    Args:
        law_citation: Law citation (e.g., "262/2006 Sb.")
        section_citation: Section citation (e.g., "§ 212")
        date: Date in YYYY-MM-DD format (e.g., "2020-01-15")
    
    Returns:
        Dict with historical section text, or None if not found
    """
    try:
        from app.esbirka_api import EsbirkaAPI
        
        api = EsbirkaAPI()
        fragments = await api.get_law_fragments(law_citation, date=date)
        
        # Find the matching section
        normalized_target = section_citation.strip().lower().replace(" ", "")
        
        for frag in fragments:
            normalized_frag = frag.section_citation.strip().lower().replace(" ", "")
            if normalized_target in normalized_frag or normalized_frag in normalized_target:
                return {
                    "citation": frag.section_citation,
                    "full_citation": frag.full_citation,
                    "text": frag.text,
                    "as_of_date": date,
                    "is_historical": True,
                    "source_url": f"https://www.e-sbirka.cz{frag.stale_url}"
                }
        
        return None
        
    except Exception as e:
        logger.error(f"E-Sbírka API historical error: {e}")
        return {"error": str(e)}


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
