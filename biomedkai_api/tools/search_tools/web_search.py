import aiohttp
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
import json
from bs4 import BeautifulSoup

from ...tools.base_tool import BaseTool
from ...config.settings import settings


class WebSearchTool(BaseTool):
    """
    General web search tool using Google Custom Search API
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="web_search",
            description="Search the web for medical information",
            config=config
        )
        
        self.api_key = settings.google_search_api_key
        self.search_engine_id = settings.google_search_engine_id
        self.base_url = "https://www.googleapis.com/customsearch/v1"
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute web search"""
        query = kwargs.get("query", "")
        num_results = kwargs.get("num_results", 5)
        time_range = kwargs.get("time_range", None)
        
        # Add medical context to query if not present
        if "medical" not in query.lower() and "health" not in query.lower():
            query = f"{query} medical health"
            
        # Perform search
        search_results = await self._search_google(query, num_results, time_range)
        
        # Extract and clean results
        cleaned_results = await self._process_results(search_results)
        
        # Rank results by medical relevance
        ranked_results = self._rank_by_medical_relevance(cleaned_results)
        
        return {
            "query": query,
            "results": ranked_results[:num_results],
            "total_results": len(search_results),
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def validate_params(self, **kwargs) -> bool:
        """Validate search parameters"""
        return bool(kwargs.get("query"))
        
    async def _search_google(self, query: str, num_results: int, time_range: Optional[str]) -> List[Dict[str, Any]]:
        """Perform Google Custom Search"""
        
        params = {
            "key": self.api_key,
            "cx": self.search_engine_id,
            "q": query,
            "num": min(num_results, 10)  # Google limits to 10 per request
        }
        
        # Add time range if specified
        if time_range:
            date_restrict = {
                "day": "d1",
                "week": "w1",
                "month": "m1",
                "year": "y1"
            }.get(time_range)
            if date_restrict:
                params["dateRestrict"] = date_restrict
                
        results = []
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.base_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        items = data.get("items", [])
                        
                        for item in items:
                            results.append({
                                "title": item.get("title", ""),
                                "url": item.get("link", ""),
                                "snippet": item.get("snippet", ""),
                                "source": item.get("displayLink", ""),
                                "metadata": item.get("pagemap", {})
                            })
                    else:
                        self.logger.error(f"Google search failed: {response.status}")
                        
            except Exception as e:
                self.logger.error(f"Search error: {str(e)}")
                
        return results
        
    async def _process_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Process and clean search results"""
        processed = []
        
        for result in results:
            # Clean snippet
            snippet = result.get("snippet", "")
            snippet = snippet.replace("\n", " ").strip()
            
            # Extract additional metadata
            metadata = result.get("metadata", {})
            
            # Determine content type
            content_type = self._determine_content_type(result["url"], metadata)
            
            # Check if medical source
            is_medical = self._is_medical_source(result["source"])
            
            processed.append({
                "title": result["title"],
                "url": result["url"],
                "snippet": snippet,
                "source": result["source"],
                "content_type": content_type,
                "is_medical_source": is_medical,
                "credibility_score": self._calculate_credibility(result["source"], is_medical)
            })
            
        return processed
        
    def _determine_content_type(self, url: str, metadata: Dict[str, Any]) -> str:
        """Determine the type of content"""
        url_lower = url.lower()
        
        if "pubmed" in url_lower or "ncbi" in url_lower:
            return "research_article"
        elif "pdf" in url_lower:
            return "pdf_document"
        elif any(domain in url_lower for domain in [".gov", ".edu", ".org"]):
            return "authoritative"
        elif any(term in url_lower for term in ["blog", "news", "article"]):
            return "article"
        else:
            return "general"
            
    def _is_medical_source(self, source: str) -> bool:
        """Check if source is a recognized medical authority"""
        medical_domains = [
            "nih.gov", "cdc.gov", "who.int", "mayo", "webmd",
            "medlineplus", "cleveland", "hopkins", "nejm.org",
            "jamanetwork", "bmj.com", "thelancet", "nature.com/medicine"
        ]
        
        source_lower = source.lower()
        return any(domain in source_lower for domain in medical_domains)
        
    def _calculate_credibility(self, source: str, is_medical: bool) -> float:
        """Calculate source credibility score"""
        score = 0.5  # Base score
        
        if is_medical:
            score += 0.3
            
        # Government sites
        if ".gov" in source:
            score += 0.2
            
        # Educational institutions
        if ".edu" in source:
            score += 0.15
            
        # Professional organizations
        if ".org" in source:
            score += 0.1
            
        # Specific trusted sources
        trusted_sources = {
            "nih.gov": 0.2,
            "cdc.gov": 0.2,
            "who.int": 0.2,
            "mayoclinic": 0.15,
            "hopkinsmedicine": 0.15
        }
        
        for trusted, bonus in trusted_sources.items():
            if trusted in source.lower():
                score += bonus
                break
                
        return min(1.0, score)
        
    def _rank_by_medical_relevance(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Rank results by medical relevance and credibility"""
        
        # Sort by combined score
        for result in results:
            # Calculate relevance score
            relevance = 0.0
            
            # Medical source bonus
            if result["is_medical_source"]:
                relevance += 0.4
                
            # Content type bonus
            content_bonuses = {
                "research_article": 0.3,
                "authoritative": 0.2,
                "pdf_document": 0.1
            }
            relevance += content_bonuses.get(result["content_type"], 0)
            
            # Combine with credibility
            result["relevance_score"] = (relevance + result["credibility_score"]) / 2
            
        # Sort by relevance score
        return sorted(results, key=lambda x: x["relevance_score"], reverse=True)