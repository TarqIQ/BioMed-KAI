from typing import Dict, List, Any, Optional
import json
import asyncio
from datetime import datetime

from src.agents.base_agent import BaseMedicalAgent
from src.core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from src.core.state_manager import MedicalAssistantState


class WebSearchAgent(BaseMedicalAgent):
    """
    Agent for searching web for latest medical information
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="web_search",
            role=AgentRole.WEB_SEARCH.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.search_synthesis_prompt = """
You are a medical research specialist and evidence-based medicine expert. Your role is to find, interpret, and synthesize current medical research, clinical guidelines, and scientific evidence to answer complex medical questions.

**Your Research Strategy:**
- Prioritize peer-reviewed, high-quality sources (systematic reviews, RCTs, clinical guidelines)
- Search for the most current evidence and recent developments
- Consider evidence hierarchy and study quality
- Synthesize findings from multiple sources
- Identify gaps or controversies in current knowledge

**Response Structure:**
1. **Current State of Evidence:**
   - Brief overview of what the research shows
   - Level of evidence quality (strong, moderate, limited)

2. **Key Findings:**
   - Most important research conclusions
   - Emerging trends or breakthroughs
   - Clinical significance of findings

3. **Research Quality Assessment:**
   - Types of studies available
   - Limitations of current evidence
   - Areas needing more research

4. **Clinical Applications:**
   - How research translates to practice
   - Guidelines or recommendations based on evidence
   - Future implications

5. **Reliable Sources:**
   - Direct links to key studies or guidelines
   - Recommendations for further reading

**Search Priorities:**
- Recent publications (last 2-3 years when possible)
- Authoritative medical organizations (WHO, CDC, medical societies)
- High-impact medical journals
- Systematic reviews and meta-analyses
- Clinical practice guidelines

**Research Question:** {input_query}

**Methodology Note:**
I will search for the most current and reliable medical evidence to provide you with an accurate, evidence-based response. I'll indicate the strength of evidence and note any limitations or ongoing controversies.

Provide comprehensive, evidence-based research synthesis.
"""
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Web search agent can process any query"""
        return bool(state.get("messages"))
        
    async def process(self, 
                     state: MedicalAssistantState,
                     message: Optional[AgentMessage] = None) -> Dict[str, Any]:
        """Process web search request"""
        
        # Extract query
        query = self._extract_search_query(state, message)
        
        # Determine search sources
        search_sources = self._determine_search_sources(query, state)
        
        # Perform searches in parallel
        search_results = await self._perform_searches(query, search_sources)
        
        # Synthesize results
        synthesis = await self._synthesize_results(query, search_results)
        
        # Extract medical insights
        medical_insights = self._extract_medical_insights(synthesis, search_results)
        
        # Calculate confidence
        confidence = self._calculate_search_confidence(search_results)
        
        # Prepare updates
        updates = {
            "research_findings": state.get("research_findings", []) + [{
                "timestamp": datetime.utcnow().isoformat(),
                "query": query,
                "sources": search_sources,
                "synthesis": synthesis,
                "insights": medical_insights,
                "raw_results": search_results,
                "agent": self.name
            }],
            "confidence_scores": {
                **state.get("confidence_scores", {}),
                "web_search": confidence
            },
            "current_agent": "validation"  # Usually go to validation after search
        }
        
        # Add findings to messages
        updates["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": self._format_search_message(synthesis, medical_insights, search_results),
            "metadata": {
                "agent": self.name,
                "confidence": confidence,
                "sources": len(search_results)
            }
        }]
        
        # Log search completion
        await self.log_decision(
            decision="web_search_completed",
            reasoning=f"Searched {len(search_sources)} sources for: {query}",
            confidence=confidence,
            state=state
        )
        
        return updates
        
    def _extract_search_query(self, 
                             state: MedicalAssistantState,
                             message: Optional[AgentMessage]) -> str:
        """Extract search query from state or message"""
        
        # Check if explicit query in message
        if message and message.content.get("query"):
            return message.content["query"]
            
        # Build query from context
        query_parts = []
        
        # Add symptoms
        symptoms = state.get("symptoms", [])
        if symptoms:
            query_parts.append(" ".join(symptoms[:3]))  # Top 3 symptoms
            
        # Add primary diagnosis if available
        if state.get("diagnosis_history"):
            latest_diagnosis = state["diagnosis_history"][-1]
            primary = latest_diagnosis.get("diagnosis", {}).get("primary_diagnosis", {})
            if primary.get("condition"):
                query_parts.append(primary["condition"])
                
        # Add specific question from latest message
        if state.get("messages"):
            latest_msg = state["messages"][-1].get("content", "")
            if len(latest_msg) < 200:  # Short enough to be a query
                query_parts.append(latest_msg)
                
        return " ".join(query_parts) or "general medical information"
        
    def _determine_search_sources(self, 
                                 query: str,
                                 state: MedicalAssistantState) -> List[str]:
        """Determine which search sources to use"""
        sources = []
        
        # Always include general web search
        sources.append("web_search")
        
        # Add specialized sources based on query
        query_lower = query.lower()
        
        if any(term in query_lower for term in ["study", "research", "trial", "evidence"]):
            sources.append("pubmed_search")
            
        if any(term in query_lower for term in ["drug", "medication", "interaction", "side effect"]):
            sources.append("drug_database")
            
        if any(term in query_lower for term in ["clinical trial", "experimental", "recruiting"]):
            sources.append("clinical_trials")
            
        if any(term in query_lower for term in ["latest", "news", "update", "recent"]):
            sources.append("medical_news")
            
        return sources
        
    async def _perform_searches(self, 
                               query: str,
                               sources: List[str]) -> Dict[str, Any]:
        """Perform searches across multiple sources"""
        search_tasks = []
        
        for source in sources:
            if source in self.tools:
                task = self._search_single_source(source, query)
                search_tasks.append(task)
                
        # Run searches in parallel
        results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Organize results by source
        search_results = {}
        for i, source in enumerate(sources):
            if i < len(results) and not isinstance(results[i], Exception):
                search_results[source] = results[i]
            else:
                self.logger.warning(f"Search failed for {source}", 
                                  error=str(results[i]) if i < len(results) else "No result")
                                  
        return search_results
        
    async def _search_single_source(self, source: str, query: str) -> Dict[str, Any]:
        """Search a single source"""
        try:
            if source == "web_search":
                return await self.call_tool("web_search", {
                    "query": query + " medical health",
                    "num_results": 5
                })
            elif source == "pubmed_search":
                return await self.call_tool("pubmed_search", {
                    "query": query,
                    "max_results": 5
                })
            elif source == "clinical_trials":
                return await self.call_tool("clinical_trials_search", {
                    "condition": query,
                    "status": "recruiting",
                    "max_results": 3
                })
            elif source == "drug_database":
                return await self.call_tool("drug_database", {
                    "query": query,
                    "search_type": "general"
                })
            elif source == "medical_news":
                return await self.call_tool("web_search", {
                    "query": query + " medical news " + datetime.utcnow().strftime("%Y"),
                    "num_results": 3,
                    "time_range": "month"
                })
            else:
                return {}
        except Exception as e:
            self.logger.error(f"Search failed for {source}", error=str(e))
            return {}
            
    async def _synthesize_results(self, 
                                 query: str,
                                 search_results: Dict[str, Any]) -> str:
        """Synthesize search results into coherent answer"""
        
        # Format search results for prompt
        formatted_results = []
        
        for source, results in search_results.items():
            formatted_results.append(f"\n=== {source.upper()} ===")
            
            if source == "web_search" and results.get("results"):
                for i, result in enumerate(results["results"][:3]):
                    formatted_results.append(f"\n{i+1}. {result.get('title', 'No title')}")
                    formatted_results.append(f"   Source: {result.get('url', 'Unknown')}")
                    formatted_results.append(f"   Summary: {result.get('snippet', 'No summary')}")
                    
            elif source == "pubmed_search" and results.get("articles"):
                for article in results["articles"][:3]:
                    formatted_results.append(f"\n- {article.get('title', 'No title')}")
                    formatted_results.append(f"  Authors: {article.get('authors', 'Unknown')}")
                    formatted_results.append(f"  Abstract: {article.get('abstract', 'No abstract')[:200]}...")
                    
            # Add other source formats as needed
            
        prompt = self.search_synthesis_prompt.format(
            query=query,
            search_results="\n".join(formatted_results)
        )
        
        response = ""
        async for chunk in self.generate_llm_response(prompt):
            response += chunk
            
        return response
        
    def _extract_medical_insights(self, 
                                 synthesis: str,
                                 search_results: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key medical insights from search results"""
        insights = {
            "key_findings": [],
            "recent_updates": [],
            "clinical_relevance": [],
            "patient_resources": []
        }
        
        # Extract from synthesis
        synthesis_lower = synthesis.lower()
        
        # Look for key phrases
        if "study" in synthesis_lower or "research" in synthesis_lower:
            insights["clinical_relevance"].append("Recent research findings available")
            
        if "fda" in synthesis_lower or "approved" in synthesis_lower:
            insights["recent_updates"].append("Regulatory updates found")
            
        # Extract from PubMed results
        if "pubmed_search" in search_results:
            articles = search_results["pubmed_search"].get("articles", [])
            for article in articles[:2]:
                if article.get("title"):
                    insights["key_findings"].append(f"Study: {article['title']}")
                    
        # Extract patient resources
        if "web_search" in search_results:
            for result in search_results["web_search"].get("results", []):
                url = result.get("url", "")
                if any(domain in url for domain in [".gov", ".org", "mayo", "cleveland"]):
                    insights["patient_resources"].append({
                        "title": result.get("title"),
                        "url": url
                    })
                    
        return insights
        
    def _calculate_search_confidence(self, search_results: Dict[str, Any]) -> float:
        """Calculate confidence based on search results quality"""
        factors = {}
        
        # Number of sources
        factors["source_count"] = min(1.0, len(search_results) / 3)
        
        # Quality of sources
        quality_score = 0
        if "pubmed_search" in search_results:
            quality_score += 0.3
        if "drug_database" in search_results:
            quality_score += 0.2
        if "web_search" in search_results:
            quality_score += 0.2
        factors["source_quality"] = quality_score
        
        # Result count
        total_results = sum(
            len(results.get("results", results.get("articles", [])))
            for results in search_results.values()
        )
        factors["result_count"] = min(1.0, total_results / 10)
        
        return self.calculate_confidence(factors)
        
    def _format_search_message(self, 
                              synthesis: str,
                              insights: Dict[str, Any],
                              search_results: Dict[str, Any]) -> str:
        """Format search results for display"""
        message = "## Web Search Results\n\n"
        message += synthesis + "\n\n"
        
        if insights.get("key_findings"):
            message += "### Key Findings\n"
            for finding in insights["key_findings"]:
                message += f"- {finding}\n"
            message += "\n"
            
        if insights.get("recent_updates"):
            message += "### Recent Updates\n"
            for update in insights["recent_updates"]:
                message += f"- {update}\n"
            message += "\n"
            
        if insights.get("patient_resources"):
            message += "### Helpful Resources\n"
            for resource in insights["patient_resources"][:3]:
                message += f"- [{resource['title']}]({resource['url']})\n"
                
        message += f"\n*Searched {len(search_results)} sources*"
        
        return message

    def _get_system_prompt(self):
        return self.search_synthesis_prompt