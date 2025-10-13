import aiohttp
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional
from datetime import datetime
import asyncio

from ...tools.base_tool import BaseTool
from ...config.settings import settings


class PubMedSearchTool(BaseTool):
    """
    Search PubMed for medical literature and research
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="pubmed_search",
            description="Search PubMed for medical research and literature",
            config=config
        )
        
        self.api_key = settings.pubmed_api_key
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        self.search_url = f"{self.base_url}/esearch.fcgi"
        self.fetch_url = f"{self.base_url}/efetch.fcgi"
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute PubMed search"""
        query = kwargs.get("query", "")
        max_results = kwargs.get("max_results", 5)
        filters = kwargs.get("filters", {})
        
        # Search for article IDs
        article_ids = await self._search_articles(query, max_results, filters)
        
        if not article_ids:
            return {
                "query": query,
                "articles": [],
                "count": 0,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        # Fetch article details
        articles = await self._fetch_article_details(article_ids)
        
        # Generate summary
        summary = self._generate_summary(articles)
        
        return {
            "query": query,
            "articles": articles,
            "count": len(articles),
            "summary": summary,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    def validate_params(self, **kwargs) -> bool:
        """Validate search parameters"""
        return bool(kwargs.get("query"))
        
    async def _search_articles(self, 
                              query: str, 
                              max_results: int,
                              filters: Dict[str, Any]) -> List[str]:
        """Search PubMed for article IDs"""
        
        params = {
            "db": "pubmed",
            "term": query,
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "api_key": self.api_key
        }
        
        # Apply filters
        if filters.get("date_range"):
            params["datetype"] = "pdat"
            params["mindate"] = filters["date_range"].get("start", "")
            params["maxdate"] = filters["date_range"].get("end", "")
            
        if filters.get("article_type"):
            types = filters["article_type"]
            if isinstance(types, list):
                type_query = " OR ".join([f'"{t}"[Publication Type]' for t in types])
                params["term"] += f" AND ({type_query})"
                
        article_ids = []
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.search_url, params=params) as response:
                    if response.status == 200:
                        data = await response.json()
                        result = data.get("esearchresult", {})
                        article_ids = result.get("idlist", [])
                    else:
                        self.logger.error(f"PubMed search failed: {response.status}")
                        
            except Exception as e:
                self.logger.error(f"PubMed search error: {str(e)}")
                
        return article_ids
        
    async def _fetch_article_details(self, article_ids: List[str]) -> List[Dict[str, Any]]:
        """Fetch detailed information for articles"""
        
        if not article_ids:
            return []
            
        params = {
            "db": "pubmed",
            "id": ",".join(article_ids),
            "retmode": "xml",
            "api_key": self.api_key
        }
        
        articles = []
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(self.fetch_url, params=params) as response:
                    if response.status == 200:
                        xml_data = await response.text()
                        articles = self._parse_articles_xml(xml_data)
                    else:
                        self.logger.error(f"PubMed fetch failed: {response.status}")
                        
            except Exception as e:
                self.logger.error(f"PubMed fetch error: {str(e)}")
                
        return articles
        
    def _parse_articles_xml(self, xml_data: str) -> List[Dict[str, Any]]:
        """Parse PubMed XML response"""
        articles = []
        
        try:
            root = ET.fromstring(xml_data)
            
            for article in root.findall(".//PubmedArticle"):
                article_data = {}
                
                # Extract PMID
                pmid_elem = article.find(".//PMID")
                if pmid_elem is not None:
                    article_data["pmid"] = pmid_elem.text
                    
                # Extract title
                title_elem = article.find(".//ArticleTitle")
                if title_elem is not None:
                    article_data["title"] = title_elem.text
                    
                # Extract abstract
                abstract_elem = article.find(".//AbstractText")
                if abstract_elem is not None:
                    article_data["abstract"] = abstract_elem.text
                    
                # Extract authors
                authors = []
                for author in article.findall(".//Author"):
                    last_name = author.find("LastName")
                    first_name = author.find("ForeName")
                    if last_name is not None and first_name is not None:
                        authors.append(f"{last_name.text} {first_name.text}")
                article_data["authors"] = ", ".join(authors[:3])  # First 3 authors
                if len(authors) > 3:
                    article_data["authors"] += " et al."
                    
                # Extract publication date
                pub_date = article.find(".//PubDate")
                if pub_date is not None:
                    year = pub_date.find("Year")
                    if year is not None:
                        article_data["year"] = year.text
                        
                # Extract journal
                journal_elem = article.find(".//Journal/Title")
                if journal_elem is not None:
                    article_data["journal"] = journal_elem.text
                    
                # Extract keywords
                keywords = []
                for keyword in article.findall(".//Keyword"):
                    if keyword.text:
                        keywords.append(keyword.text)
                article_data["keywords"] = keywords
                
                # Extract DOI
                doi_elem = article.find(".//ArticleId[@IdType='doi']")
                if doi_elem is not None:
                    article_data["doi"] = doi_elem.text
                    
                articles.append(article_data)
                
        except Exception as e:
            self.logger.error(f"XML parsing error: {str(e)}")
            
        return articles
        
    def _generate_summary(self, articles: List[Dict[str, Any]]) -> str:
        """Generate a summary of the search results"""
        
        if not articles:
            return "No relevant articles found."
            
        summary_parts = []
        
        # Key findings
        summary_parts.append(f"Found {len(articles)} relevant articles.")
        
        # Recent studies
        recent_articles = [a for a in articles if a.get("year") and int(a.get("year", 0)) >= 2020]
        if recent_articles:
            summary_parts.append(f"{len(recent_articles)} articles published since 2020.")
            
        # Top article summary
        if articles[0].get("abstract"):
            summary_parts.append(f"\nMost relevant finding: {articles[0]['title']}")
            # First 200 characters of abstract
            abstract_preview = articles[0]["abstract"][:200] + "..."
            summary_parts.append(f"Summary: {abstract_preview}")
            
        return "\n".join(summary_parts)