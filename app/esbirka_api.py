"""
E-Sbírka REST API Client for Lexio Agent.

Async client with:
- 0-based pagination (starts at page 0, not 1)
- Conservative fallback (only used when data missing from DB)
- Rate limiting (0.2s between requests)
- Proper error handling (400, 404, 429)
"""
import os
import re
import asyncio
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import httpx
from pathlib import Path

from app.config import settings


@dataclass
class LawFragment:
    """Parsed fragment from E-Sbírka API."""
    fragment_id: int
    section_citation: str  # e.g., "§ 123 odst. 6"
    full_citation: str  # e.g., "§ 123 odst. 6 zákona č. 262/2006 Sb."
    text: str
    xhtml: str
    stale_url: str
    is_current: bool


class EsbirkaAPI:
    """E-Sbírka REST API client with conservative fallback logic."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize API client.
        
        Args:
            api_key: E-Sbírka API key (defaults to settings.ESBIRKA_API_KEY)
        """
        self.api_key = api_key or settings.ESBIRKA_API_KEY
        
        if not self.api_key:
            raise ValueError(
                "ESBIRKA_API_KEY not set. Add to .env file or set environment variable."
            )
        
        self.base_url = settings.ESBIRKA_API_BASE
        self.timeout = settings.ESBIRKA_API_TIMEOUT
    
    def _strip_html(self, html: str) -> str:
        """Remove HTML tags and normalize whitespace."""
        if not html:
            return ""
        # Simple regex-based stripping
        text = re.sub(r'<[^>]+>', '', html)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def _convert_to_stale_url(self, law_citation: str) -> str:
        """
        Convert law citation to stale URL format.
        
        Args:
            law_citation: e.g., "262/2006 Sb."
        
        Returns:
            Stale URL: "/eli/cz/sb/2006/262"
        """
        # Parse citation: "262/2006 Sb." -> sb/2006/262
        match = re.match(r'(\d+)/(\d+)\s+Sb\.?', law_citation)
        if not match:
            raise ValueError(f"Invalid law citation format: {law_citation}")
        
        number, year = match.groups()
        return f"/eli/cz/sb/{year}/{number}"
    
    async def get_law_fragments(
        self,
        law_citation: str,
        date: Optional[str] = None
    ) -> List[LawFragment]:
        """
        Get all fragments for a law (current or historical version).
        
        Args:
            law_citation: e.g., "262/2006 Sb."
            date: Optional ISO date (YYYY-MM-DD) for historical version
        
        Returns:
            List of LawFragment objects
        
        Raises:
            httpx.HTTPError: On API errors
        """
        stale_url = self._convert_to_stale_url(law_citation)
        
        # Add date to stale URL if provided (for historical versions)
        if date:
            stale_url = f"{stale_url}/{date}"
        
        # URL-encode the stale URL
        encoded = stale_url.replace("/", "%2F")
        
        all_fragments: List[LawFragment] = []
        page = 0  # 0-BASED PAGINATION!
        
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"esel-api-access-key": self.api_key},
            timeout=self.timeout,
            verify=False  # E-Sbírka API requires this
        ) as client:
            
            while page < 100:  # Safety limit
                try:
                    response = await client.get(
                        f"/dokumenty-sbirky/{encoded}/fragmenty",
                        params={"cisloStranky": page}
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        fragments = data.get("seznam", [])
                        
                        if not fragments:
                            break
                        
                        # Parse fragments
                        for frag in fragments:
                            all_fragments.append(LawFragment(
                                fragment_id=frag.get("id", 0),
                                section_citation=self._extract_section_citation(
                                    frag.get("zkracenaCitace", "")
                                ),
                                full_citation=frag.get("uplnaCitace", ""),
                                text=self._strip_html(frag.get("xhtml", "")),
                                xhtml=frag.get("xhtml", ""),
                                stale_url=frag.get("staleUrl", ""),
                                is_current=frag.get("jeUcinny", True)
                            ))
                        
                        page += 1
                        await asyncio.sleep(0.2)  # Rate limiting
                        
                    elif response.status_code == 400:
                        # Reached end of available pages
                        break
                    else:
                        response.raise_for_status()
                        
                except httpx.HTTPError as e:
                    # Log but don't crash
                    print(f"API error on page {page}: {e}")
                    break
        
        return all_fragments
    
    def _extract_section_citation(self, full_citation: str) -> str:
        """
        Extract just the section part from full citation.
        
        "§ 123 odst. 6 zákona č. 262/2006 Sb." -> "§ 123 odst. 6"
        """
        match = re.match(r'(§\s*\d+[a-z]?(?:\s*odst\.\s*\d+)?(?:\s*písm\.\s*[a-z]\))?)', full_citation)
        if match:
            return match.group(1)
        return full_citation
    
    async def find_section(
        self,
        law_citation: str,
        section_citation: str
    ) -> Optional[LawFragment]:
        """
        Find a specific section within a law.
        
        Args:
            law_citation: e.g., "262/2006 Sb."
            section_citation: e.g., "§ 212" or "§ 123 odst. 6"
        
        Returns:
            Matching LawFragment or None
        """
        fragments = await self.get_law_fragments(law_citation)
        
        # Normalize section citation for matching
        normalized_target = section_citation.strip().lower().replace(" ", "")
        
        for frag in fragments:
            normalized_frag = frag.section_citation.strip().lower().replace(" ", "")
            if normalized_target in normalized_frag or normalized_frag in normalized_target:
                return frag
        
        return None
    
    async def get_law_versions(self, law_citation: str) -> List[Dict]:
        """
        Get all historical versions of a law.
        
        NOTE: This functionality requires a different API endpoint
        that may not be fully documented. Returns empty list if not available.
        
        Args:
            law_citation: e.g., "262/2006 Sb."
        
        Returns:
            List of version metadata (dates, status, etc.)
        """
        # Placeholder - would need specific endpoint
        # The API documentation suggests this exists but doesn't specify the exact endpoint
        return []
    
    async def get_recent_changes(self, since_days: int = 7) -> List[Dict]:
        """
        Get laws that have been modified in the last N days.
        
        Args:
            since_days: Number of days to look back
        
        Returns:
            List of changed law metadata
        """
        from datetime import datetime, timedelta, timezone
        
        since = datetime.now(timezone.utc) - timedelta(days=since_days)
        since_str = since.strftime("%Y-%m-%dT%H:%M:%S+00:00")
        
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"esel-api-access-key": self.api_key},
            timeout=self.timeout,
            verify=False
        ) as client:
            
            try:
                response = await client.get(
                    "/dokumenty-sbirky/zmeny-zneni",
                    params={"datumCasOdberuOd": since_str}
                )
                response.raise_for_status()
                data = response.json()
                
                return data.get("seznamZneniPravnichPredpisu", [])
                
            except httpx.HTTPError as e:
                print(f"Error fetching recent changes: {e}")
                return []
