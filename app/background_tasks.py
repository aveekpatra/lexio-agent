"""
Background task manager for async law processing.

When API fallback finds a law section, we:
1. Return result to user IMMEDIATELY (fast)
2. Queue background task to properly ingest the law (slow):
   - Fetch all fragments
   - Store in database
   - Generate embeddings
   - Create chunks
"""
import asyncio
from typing import Dict, Optional, Set
from dataclasses import dataclass
from datetime import datetime

import asyncpg
from app.config import settings
from app.esbirka_api import EsbirkaAPI


@dataclass
class IngestionTask:
    """Background task to ingest a law."""
    law_citation: str
    queued_at: datetime
    priority: int = 0  # Higher = more important


class BackgroundTaskQueue:
    """
    Async background task queue for law ingestion.
    
    Ensures we don't ingest the same law multiple times concurrently.
    """
    
    def __init__(self):
        self.pending_laws: Set[str] = set()
        self.queue: asyncio.Queue = asyncio.Queue()
        self.worker_task: Optional[asyncio.Task] = None
    
    def start_worker(self):
        """Start background worker."""
        if self.worker_task is None or self.worker_task.done():
            self.worker_task = asyncio.create_task(self._worker())
    
    async def enqueue_law_ingestion(self, law_citation: str, priority: int = 0):
        """
        Queue a law for full ingestion in the background.
        
        Args:
            law_citation: e.g., "262/2006 Sb."
            priority: Higher = processed sooner
        """
        # Prevent duplicate ingestion
        if law_citation in self.pending_laws:
            print(f"Law {law_citation} already queued for ingestion")
            return
        
        self.pending_laws.add(law_citation)
        
        task = IngestionTask(
            law_citation=law_citation,
            queued_at=datetime.now(),
            priority=priority
        )
        
        await self.queue.put(task)
        print(f"✓ Queued background ingestion: {law_citation}")
    
    async def _worker(self):
        """Background worker that processes ingestion tasks."""
        from app.tools import get_pool, get_embedding
        
        print("🚀 Background ingestion worker started")
        
        while True:
            try:
                # Get next task
                task: IngestionTask = await self.queue.get()
                
                print(f"\n📥 Processing background ingestion: {task.law_citation}")
                
                try:
                    # Step 1: Fetch ALL fragments from API
                    api = EsbirkaAPI()
                    fragments = await api.get_law_fragments(task.law_citation)
                    
                    print(f"   Fetched {len(fragments)} fragments")
                    
                    # Step 2: Store in database
                    pool = await get_pool()
                    
                    # Get law and version IDs
                    law_row = await pool.fetchrow(
                        "SELECT id FROM laws WHERE citation = $1",
                        task.law_citation
                    )
                    
                    if law_row:
                        law_id = law_row['id']
                    else:
                        law_id = await pool.fetchval("""
                            INSERT INTO laws (citation, title)
                            VALUES ($1, $2)
                            RETURNING id
                        """, task.law_citation, "")
                    
                    # Get current version
                    version_row = await pool.fetchrow("""
                        SELECT id FROM versions
                        WHERE law_id = $1 AND is_current = true
                        LIMIT 1
                    """, law_id)
                    
                    if version_row:
                        version_id = version_row['id']
                    else:
                        version_id = await pool.fetchval("""
                            INSERT INTO versions (law_id, is_current)
                            VALUES ($1, true)
                            RETURNING id
                        """, law_id)
                    
                    # Step 3: Store sections and generate embeddings
                    for i, frag in enumerate(fragments):
                        # Store section
                        section_id = await pool.fetchval("""
                            INSERT INTO sections (version_id, citation, parsed_section, position_in_document)
                            VALUES ($1, $2, $3, $4)
                            ON CONFLICT DO NOTHING
                            RETURNING id
                        """, version_id, frag.section_citation, frag.section_citation, i)
                        
                        if not section_id:
                            # Already exists
                            continue
                        
                        # Store content
                        await pool.execute("""
                            INSERT INTO section_contents (section_id, full_text)
                            VALUES ($1, $2)
                            ON CONFLICT (section_id) DO UPDATE
                            SET full_text = EXCLUDED.full_text
                        """, section_id, frag.text)
                        
                        # Step 4: Generate chunks and embeddings
                        # (Similar to ingest_correct_v2.py chunking logic)
                        chunks = self._chunk_text(frag.text, max_size=4000, overlap=400)
                        
                        for chunk_idx, chunk_text in enumerate(chunks):
                            if len(chunk_text) < 50:  # Skip tiny chunks
                                continue
                            
                            # Generate embedding
                            try:
                                embedding = await get_embedding(chunk_text)
                                
                                # Store chunk with embedding
                                await pool.execute("""
                                    INSERT INTO section_chunks (section_id, chunk_index, chunk_text, embedding)
                                    VALUES ($1, $2, $3, $4)
                                    ON CONFLICT DO NOTHING
                                """, section_id, chunk_idx, chunk_text, str(embedding))
                                
                            except Exception as e:
                                print(f"   ⚠️  Embedding failed for chunk {chunk_idx}: {e}")
                    
                    print(f"   ✅ Completed ingestion: {task.law_citation}")
                    
                except Exception as e:
                    print(f"   ❌ Ingestion failed: {e}")
                
                finally:
                    # Remove from pending set
                    self.pending_laws.discard(task.law_citation)
                    self.queue.task_done()
                
            except Exception as e:
                print(f"Worker error: {e}")
                await asyncio.sleep(1)
    
    def _chunk_text(self, text: str, max_size: int = 4000, overlap: int = 400) -> list[str]:
        """
        Split text into chunks with overlap.
        
        Args:
            text: Text to chunk
            max_size: Maximum chunk size in characters
            overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        if len(text) <= max_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + max_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence end within last 20% of chunk
                search_start = end - int(max_size * 0.2)
                chunk_text = text[start:end]
                
                # Find last sentence boundary
                for delimiter in ['. ', '.\n', '! ', '? ']:
                    last_idx = chunk_text.rfind(delimiter, search_start - start)
                    if last_idx != -1:
                        end = start + last_idx + len(delimiter)
                        break
            
            chunks.append(text[start:end])
            start = end - overlap if end < len(text) else end
        
        return chunks


# Global task queue
_task_queue: Optional[BackgroundTaskQueue] = None


def get_task_queue() -> BackgroundTaskQueue:
    """Get or create the global task queue."""
    global _task_queue
    if _task_queue is None:
        _task_queue = BackgroundTaskQueue()
        _task_queue.start_worker()
    return _task_queue
