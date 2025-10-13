from typing import Dict, List, Any, Optional
import json
from datetime import datetime
from src.agents.base_agent import BaseMedicalAgent
from src.core.message_protocol import AgentMessage, MessageType, Priority, AgentRole
from src.core.state_manager import MedicalAssistantState

class ResearchAgent(BaseMedicalAgent):
    """
    Specialist agent for medical research and evidence-based information gathering
    """
    
    def __init__(self, model: Any, tools: Dict[str, Any], config: Dict[str, Any]):
        super().__init__(
            name="research",
            role=AgentRole.RESEARCH.value,
            model=model,
            tools=tools,
            config=config
        )
        
        self.research_prompt = """
You are an expert medical researcher. Conduct comprehensive research on the given medical query.

Research Query: {query}
Condition/Disease: {condition}
Patient Context: {patient_context}

Research Areas:
1. LATEST EVIDENCE
   - Recent publications (last 2 years)
   - Systematic reviews and meta-analyses
   - Clinical practice guidelines
   - Emerging treatments

2. CLINICAL TRIALS
   - Ongoing trials
   - Recent completed trials
   - Trial outcomes and implications
   - Eligibility criteria

3. TREATMENT EFFICACY
   - Evidence levels for current treatments
   - Comparative effectiveness
   - Adverse effects and safety data
   - Patient outcomes

4. EMERGING RESEARCH
   - Novel therapeutic approaches
   - Biomarkers and diagnostics
   - Precision medicine applications
   - Future research directions

Provide a comprehensive research summary with:
- Key findings with evidence levels
- Clinical implications
- Recommendations based on current evidence
- Areas of uncertainty or conflicting evidence

Format your response as structured JSON for parsing.
"""

        self.literature_search_prompt = """
Search and analyze medical literature for: {topic}

Focus on:
- High-quality evidence (RCTs, systematic reviews)
- Recent publications (prioritize last 5 years)
- Clinical relevance
- Practice-changing findings

Provide summary with key findings and clinical implications.
"""
        
    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate research agent has necessary input"""
        # Need research query, condition, or diagnostic context
        return bool(
            state.get("research_query") or 
            state.get("diagnosis_history") or
            state.get("treatment_plans") or
            (state.get("messages") and len(state["messages"]) > 0)
        )
        
    async def process(self, 
                     state: MedicalAssistantState,
                     message: Optional[AgentMessage] = None) -> Dict[str, Any]:
        """Process research request and gather evidence-based information"""
        
        # Determine research focus
        research_query = await self._determine_research_query(state, message)
        
        # Conduct literature search
        literature_results = await self._conduct_literature_search(research_query)
        
        # Search clinical trials
        clinical_trials = await self._search_clinical_trials(research_query)
        
        # Search knowledge graph
        knowledge_context = await self._search_knowledge_graph(research_query, state)
        
        # Generate comprehensive research summary
        research_summary = await self._generate_research_summary(
            query=research_query,
            literature=literature_results,
            trials=clinical_trials,
            knowledge=knowledge_context,
            patient_context=state.get("patient_context", {})
        )
        
        # Calculate research confidence
        confidence = self._calculate_research_confidence(research_summary, literature_results)
        
        # Determine if findings are significant
        significant_findings = self._has_significant_findings(research_summary)
        
        # Prepare updates
        updates = {
            "research_history": state.get("research_history", []) + [{
                "timestamp": datetime.utcnow().isoformat(),
                "query": research_query,
                "summary": research_summary,
                "literature": literature_results,
                "clinical_trials": clinical_trials,
                "confidence": confidence,
                "agent": self.name
            }],
            "confidence_scores": {
                **state.get("confidence_scores", {}),
                "research": confidence
            },
            "current_agent": "validation"
        }
        
        # Add research findings to messages
        updates["messages"] = state.get("messages", []) + [{
            "role": "assistant",
            "content": self._format_research_message(research_summary, research_query),
            "metadata": {
                "agent": self.name,
                "confidence": confidence,
                "research_data": research_summary,
                "significant_findings": significant_findings
            }
        }]
        
        # Flag for human review if conflicting evidence found
        if research_summary.get("conflicting_evidence") or confidence < 0.6:
            updates["requires_human_review"] = True
            updates["review_reason"] = "Conflicting or insufficient research evidence"
        
        # Log research activity
        await self.log_decision(
            decision="research_completed",
            reasoning=f"Researched: {research_query}",
            confidence=confidence,
            state=state
        )
        
        return updates
        
    async def _determine_research_query(self, 
                                       state: MedicalAssistantState,
                                       message: Optional[AgentMessage]) -> str:
        """Determine what to research based on state and message"""
        
        # Check for explicit research query
        if state.get("research_query"):
            return state["research_query"]
            
        # Check message for research request
        if message and message.content.get("research_topic"):
            return message.content["research_topic"]
            
        # Extract from diagnosis
        if state.get("diagnosis_history"):
            latest_diagnosis = state["diagnosis_history"][-1]
            primary_condition = latest_diagnosis.get("diagnosis", {}).get("primary_diagnosis", {}).get("condition")
            if primary_condition:
                return f"latest evidence treatment {primary_condition}"
                
        # Extract from treatment plans
        if state.get("treatment_plans"):
            latest_plan = state["treatment_plans"][-1]
            if latest_plan.get("medications"):
                medication = latest_plan["medications"][0].get("name", "")
                condition = latest_plan.get("condition", "")
                return f"efficacy safety {medication} {condition}"
                
        # Extract from latest user message
        if state.get("messages"):
            for msg in reversed(state["messages"]):
                if msg.get("role") == "user":
                    return f"research {msg.get('content', '')}"
                    
        return "general medical research"
        
    async def _conduct_literature_search(self, query: str) -> Dict[str, Any]:
        """Search medical literature using PubMed"""
        if "pubmed_search" not in self.tools:
            return {"articles": [], "summary": "PubMed search not available"}
            
        try:
            # Search for recent high-quality evidence
            search_params = {
                "query": query,
                "max_results": self.config.get("max_articles", 10),
                "filters": {
                    "publication_types": ["systematic review", "meta-analysis", "randomized controlled trial"],
                    "years": self.config.get("recent_years", 5)
                }
            }
            
            results = await self.call_tool("pubmed_search", search_params)
            
            # Analyze literature quality
            quality_score = self._assess_literature_quality(results.get("articles", []))
            
            return {
                "articles": results.get("articles", []),
                "summary": results.get("summary", ""),
                "quality_score": quality_score,
                "search_query": query
            }
            
        except Exception as e:
            self.logger.error(f"Literature search failed: {e}")
            return {"articles": [], "summary": f"Literature search failed: {str(e)}"}
            
    async def _search_clinical_trials(self, query: str) -> Dict[str, Any]:
        """Search for relevant clinical trials"""
        if "clinical_trials_search" not in self.tools:
            return {"trials": [], "summary": "Clinical trials search not available"}
            
        try:
            search_params = {
                "query": query,
                "max_results": self.config.get("max_trials", 5),
                "status": ["recruiting", "active", "completed"]
            }
            
            results = await self.call_tool("clinical_trials_search", search_params)
            
            return {
                "trials": results.get("trials", []),
                "summary": results.get("summary", ""),
                "search_query": query
            }
            
        except Exception as e:
            self.logger.error(f"Clinical trials search failed: {e}")
            return {"trials": [], "summary": f"Clinical trials search failed: {str(e)}"}
            
    async def _search_knowledge_graph(self, 
                                     query: str,
                                     state: MedicalAssistantState) -> str:
        """Search knowledge graph for related information"""
        if "knowledge_graph_search" not in self.tools:
            return "Knowledge graph search not available"
            
        try:
            search_params = {
                "query": query,
                "entity_types": ["Disease", "Treatment", "Drug", "Study"],
                "limit": 5
            }
            
            results = await self.call_tool("knowledge_graph_search", search_params)
            return results.get("context", "No knowledge graph context found")
            
        except Exception as e:
            self.logger.error(f"Knowledge graph search failed: {e}")
            return f"Knowledge graph search failed: {str(e)}"
            
    async def _generate_research_summary(self, **kwargs) -> Dict[str, Any]:
        """Generate comprehensive research summary using LLM"""
        
        prompt = self.research_prompt.format(
            query=kwargs.get("query", ""),
            condition=self._extract_condition(kwargs.get("patient_context", {})),
            patient_context=json.dumps(kwargs.get("patient_context", {}))
        )
        
        # Add literature context
        if kwargs.get("literature", {}).get("summary"):
            prompt += f"\n\nLiterature Findings:\n{kwargs['literature']['summary']}"
            
        # Add clinical trials context
        if kwargs.get("trials", {}).get("summary"):
            prompt += f"\n\nClinical Trials:\n{kwargs['trials']['summary']}"
            
        # Add knowledge graph context
        if kwargs.get("knowledge"):
            prompt += f"\n\nKnowledge Base:\n{kwargs['knowledge']}"
            
        response = ""
        async for chunk in self.generate_llm_response(prompt):
            response += chunk
            
        try:
            research_data = json.loads(response)
            
            # Ensure required fields
            required_fields = ["latest_evidence", "clinical_trials", "treatment_efficacy", 
                             "emerging_research", "key_findings", "clinical_implications"]
            for field in required_fields:
                if field not in research_data:
                    research_data[field] = []
                    
            # Add metadata
            research_data["search_quality"] = kwargs.get("literature", {}).get("quality_score", 0.5)
            research_data["evidence_level"] = self._determine_evidence_level(research_data)
            
            return research_data
            
        except json.JSONDecodeError:
            # Fallback structure
            return {
                "latest_evidence": [],
                "clinical_trials": [],
                "treatment_efficacy": [],
                "emerging_research": [],
                "key_findings": ["Research summary generation failed"],
                "clinical_implications": [response],
                "conflicting_evidence": False,
                "evidence_level": "insufficient"
            }
            
    def _extract_condition(self, patient_context: Dict[str, Any]) -> str:
        """Extract primary condition from patient context"""
        # Try medical history
        history = patient_context.get("medical_history", {})
        if history.get("primary_condition"):
            return history["primary_condition"]
            
        # Try current conditions
        conditions = patient_context.get("conditions", [])
        if conditions:
            return conditions[0] if isinstance(conditions[0], str) else conditions[0].get("name", "")
            
        return "unspecified condition"
        
    def _assess_literature_quality(self, articles: List[Dict[str, Any]]) -> float:
        """Assess overall quality of literature found"""
        if not articles:
            return 0.0
            
        quality_factors = {}
        
        # Publication types
        high_quality_types = ["systematic review", "meta-analysis", "randomized controlled trial"]
        quality_count = sum(1 for article in articles 
                          if any(qt in article.get("publication_type", "").lower() 
                                for qt in high_quality_types))
        quality_factors["publication_type"] = quality_count / len(articles)
        
        # Recency
        current_year = datetime.now().year
        recent_count = sum(1 for article in articles 
                         if current_year - article.get("year", 0) <= 5)
        quality_factors["recency"] = recent_count / len(articles)
        
        # Journal impact (if available)
        impact_scores = [article.get("impact_factor", 0) for article in articles]
        quality_factors["impact"] = min(1.0, sum(impact_scores) / len(articles) / 10)
        
        return self.calculate_confidence(quality_factors)
        
    def _determine_evidence_level(self, research_data: Dict[str, Any]) -> str:
        """Determine overall evidence level"""
        evidence_count = len(research_data.get("latest_evidence", []))
        trial_count = len(research_data.get("clinical_trials", []))
        
        if evidence_count >= 3 and trial_count >= 1:
            return "high"
        elif evidence_count >= 2:
            return "moderate"
        elif evidence_count >= 1:
            return "low"
        else:
            return "insufficient"
            
    def _calculate_research_confidence(self, 
                                     research_summary: Dict[str, Any],
                                     literature_results: Dict[str, Any]) -> float:
        """Calculate overall research confidence"""
        factors = {}
        
        # Literature quality
        factors["literature"] = literature_results.get("quality_score", 0.5)
        
        # Evidence level
        evidence_level = research_summary.get("evidence_level", "insufficient")
        evidence_scores = {"high": 1.0, "moderate": 0.7, "low": 0.4, "insufficient": 0.2}
        factors["evidence"] = evidence_scores.get(evidence_level, 0.2)
        
        # Number of key findings
        findings_count = len(research_summary.get("key_findings", []))
        factors["findings"] = min(1.0, findings_count * 0.2)
        
        # Conflicting evidence penalty
        if research_summary.get("conflicting_evidence"):
            factors["consistency"] = 0.3
        else:
            factors["consistency"] = 1.0
            
        return self.calculate_confidence(factors)
        
    def _has_significant_findings(self, research_summary: Dict[str, Any]) -> bool:
        """Check if research has significant clinical findings"""
        key_findings = research_summary.get("key_findings", [])
        clinical_implications = research_summary.get("clinical_implications", [])
        
        return (len(key_findings) > 0 and len(clinical_implications) > 0 and
                research_summary.get("evidence_level") in ["moderate", "high"])
        
    def _format_research_message(self, 
                                research_summary: Dict[str, Any],
                                query: str) -> str:
        """Format research findings for display"""
        
        message = f"## Research Findings: {query}\n\n"
        
        # Evidence level
        evidence_level = research_summary.get("evidence_level", "insufficient")
        message += f"**Evidence Level:** {evidence_level.title()}\n\n"
        
        # Key findings
        if research_summary.get("key_findings"):
            message += "### Key Findings:\n"
            for finding in research_summary["key_findings"]:
                message += f"• {finding}\n"
            message += "\n"
            
        # Latest evidence
        if research_summary.get("latest_evidence"):
            message += "### Latest Evidence:\n"
            for evidence in research_summary["latest_evidence"][:3]:  # Top 3
                message += f"• {evidence}\n"
            message += "\n"
            
        # Clinical implications
        if research_summary.get("clinical_implications"):
            message += "### Clinical Implications:\n"
            for implication in research_summary["clinical_implications"]:
                message += f"• {implication}\n"
            message += "\n"
            
        # Clinical trials
        if research_summary.get("clinical_trials"):
            message += "### Relevant Clinical Trials:\n"
            for trial in research_summary["clinical_trials"][:2]:  # Top 2
                message += f"• {trial}\n"
            message += "\n"
            
        # Conflicting evidence warning
        if research_summary.get("conflicting_evidence"):
            message += "### ⚠️ Conflicting Evidence Found\n"
            message += "Multiple studies show conflicting results. Professional review recommended.\n\n"
            
        # Research limitations
        if evidence_level in ["low", "insufficient"]:
            message += "### Research Limitations\n"
            message += "Limited high-quality evidence available. Findings should be interpreted with caution.\n"
            
        return message

    async def validate_input(self, state: MedicalAssistantState) -> bool:
        """Validate research input"""
        messages = state.get("messages", [])
        return len(messages) > 0
    
    async def process(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Process research request"""
        return await self.process_with_streaming(state)
    
    def _get_system_prompt(self) -> str:
        return """You are a Medical Research AI Assistant specializing in evidence-based medicine. Your expertise includes:

1. **Literature Review** - Systematic analysis of medical literature
2. **Evidence Synthesis** - Integrate findings from multiple sources
3. **Clinical Trial Analysis** - Evaluate clinical trial data and outcomes
4. **Research Methodology** - Assess study quality and methodology
5. **Emerging Therapies** - Identify cutting-edge treatments and research

**Research Approach:**
- Conduct systematic literature searches
- Evaluate evidence quality and relevance
- Synthesize findings from multiple studies
- Assess clinical applicability
- Identify research gaps and limitations
- Provide evidence-based recommendations

**Evidence Standards:**
- Prioritize high-quality studies (RCTs, meta-analyses)
- Consider study limitations and biases
- Evaluate clinical relevance and applicability
- Assess statistical significance and clinical significance
- Consider patient population and setting
- Provide balanced perspective on controversial topics"""
    
    def _get_relevant_entity_types(self) -> List[str]:
        return ["Gene", "Protein", "Pathway", "Clinical_Trial", "Research", "Study"]
    
    async def _use_tool(self, tool_name: str, tool: Any, query: str, state: MedicalAssistantState) -> Optional[Dict[str, Any]]:
        """Use research-specific tools"""
        try:
            if tool_name == "pubmed_search":
                return await tool.execute(query=query, limit=10)
            elif tool_name == "clinical_trials_search":
                return await tool.execute(query=query)
        except Exception as e:
            self.logger.warning(f"Tool {tool_name} failed in research agent", error=str(e))
        return None
    
    def _determine_next_agent(self, state: MedicalAssistantState, response: str) -> str:
        """After research, go to validation"""
        return "validation"
    
    async def _get_agent_specific_updates(self, response: str, context: Dict[str, Any], state: MedicalAssistantState) -> Dict[str, Any]:
        """Update research findings"""
        updates = {}
        
        research_findings = {
            "timestamp": datetime.utcnow().isoformat(),
            "agent": self.name,
            "literature_review": response,
            "key_studies": self._extract_key_studies(response),
            "evidence_level": self._assess_evidence_level(response),
            "clinical_implications": self._extract_clinical_implications(response),
            "context_used": context
        }
        
        updates["research_findings"] = state.get("research_findings", []) + [research_findings]
        
        return updates
    
    def _extract_key_studies(self, response: str) -> List[Dict[str, Any]]:
        """Extract key studies from response"""
        studies = []
        lines = response.split('\n')
        
        current_study = None
        for line in lines:
            if 'study' in line.lower() or 'trial' in line.lower():
                if current_study:
                    studies.append(current_study)
                current_study = {
                    "description": line.strip(),
                    "type": self._identify_study_type(line),
                    "quality": "Unknown"
                }
            elif current_study and any(word in line.lower() for word in ['rct', 'randomized', 'meta-analysis']):
                current_study["type"] = self._identify_study_type(line)
                current_study["quality"] = "High"
        
        if current_study:
            studies.append(current_study)
        
        return studies
    
    def _assess_evidence_level(self, response: str) -> str:
        """Assess overall evidence level"""
        response_lower = response.lower()
        if "systematic review" in response_lower or "meta-analysis" in response_lower:
            return "high"
        elif "randomized controlled trial" in response_lower or "rct" in response_lower:
            return "moderate"
        elif "observational study" in response_lower or "cohort study" in response_lower:
            return "low"
        else:
            return "insufficient"
        
    def _extract_clinical_implications(self, response: str) -> List[str]:
        """Extract clinical implications from response"""
        implications = []
        lines = response.split('\n')
        
        for line in lines:
            if 'implication' in line.lower() or 'recommendation' in line.lower():
                implications.append(line.strip())
        
        return implications if implications else ["No specific clinical implications found."]
    
    