import asyncio
from typing import Dict, List, Any, Optional, AsyncGenerator
import uuid
from datetime import datetime

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph
import structlog

from ..core.state_manager import MedicalAssistantState, StateManager
from ..core.message_protocol import AgentMessage, MessageType, Priority
from ..agents.base_agent import BaseMedicalAgent
from ..agents.supervisor_agent import SupervisorAgent
from ..agents.preventive_care_and_risk_assessment import PreventiveCareAndRiskAssessmentAgent
from ..agents.diagnostic_agent import DiagnosticAgent
from ..agents.treatment_agent import TreatmentAgent
from ..agents.drug_interaction_agent import DrugInteractionAgent
from ..agents.research_agent import ResearchAgent
from ..agents.validation_agent import ValidationAgent
from ..agents.web_search_agent import WebSearchAgent
from ..agents.general_medical_agent import GeneralMedicalAgent
from rank_bm25 import BM25Okapi
import string
import re


class MedicalAgentOrchestrator:
    """
    Main orchestrator that manages the medical agent workflow with proper streaming
    """
    
    def __init__(self, 
                 model: Any,
                 tools: Dict[str, Any],
                 memory_system: Any,
                 config: Dict[str, Any]):
        self.model = model
        self.tools = tools
        self.memory_system = memory_system
        self.config = config
        self.logger = structlog.get_logger(name="orchestrator")
        self.state_manager = StateManager()
        
        # Initialize agents
        self.agents = self._initialize_agents()
        
        # Build workflow graph - simplified for direct streaming
        self.workflow = self._build_workflow()
        
        # Message queue for inter-agent communication
        self.message_queue = asyncio.Queue()
        
    def _initialize_agents(self) -> Dict[str, BaseMedicalAgent]:
        """Initialize all agents with their tools and configs"""
        
        agent_configs = self.config.get("agents", {})
        
        agents = {
            # "supervisor": SupervisorAgent(
            #     model=self.model,
            #     tools={
            #         "entity_extractor": self.tools.get("entity_extractor"),
            #         "query_classifier": self.tools.get("query_classifier"),
            #         "knowledge_graph_search": self.tools.get("knowledge_graph_search")
            #     },
            #     config=agent_configs.get("supervisor", {})
            # ),
            "diagnostic": DiagnosticAgent(
                model=self.model,
                tools={
                    "symptom_extractor": self.tools.get("symptom_extractor"),
                    "knowledge_graph_search": self.tools.get("knowledge_graph_search"),
                    "pubmed_search": self.tools.get("pubmed_search"),
                    "lab_interpreter": self.tools.get("lab_interpreter")
                },
                config=agent_configs.get("diagnostic", {})
            ),
            "treatment": TreatmentAgent(
                model=self.model,
                tools={
                    "guideline_checker": self.tools.get("guideline_checker"),
                    "drug_database": self.tools.get("drug_database"),
                    "pubmed_search": self.tools.get("pubmed_search"),
                    "knowledge_graph_search": self.tools.get("knowledge_graph_search")
                },
                config=agent_configs.get("treatment", {})
            ),
            "drug_interaction": DrugInteractionAgent(
                model=self.model,
                tools={
                    "drug_interaction_checker": self.tools.get("drug_interaction_checker"),
                    "drug_database": self.tools.get("drug_database"),
                    "allergy_checker": self.tools.get("allergy_checker"),
                    "knowledge_graph_search": self.tools.get("knowledge_graph_search")
                },
                config=agent_configs.get("drug_interaction", {})
            ),
            # "research": ResearchAgent(
            #     model=self.model,
            #     tools={
            #         "pubmed_search": self.tools.get("pubmed_search"),
            #         "clinical_trials_search": self.tools.get("clinical_trials_search"),
            #         "knowledge_graph_search": self.tools.get("knowledge_graph_search")
            #     },
            #     config=agent_configs.get("research", {})
            # ),
            # "validation": ValidationAgent(
            #     model=self.model,
            #     tools={
            #         "guideline_checker": self.tools.get("guideline_checker"),
            #         "safety_validator": self.tools.get("safety_validator")
            #     },
            #     config=agent_configs.get("validation", {})
            # ),
            "preventive_care_and_risk_assessment": PreventiveCareAndRiskAssessmentAgent(
                model=self.model,
                tools={
                    "screening_tool": self.tools.get("screening_tool"),
                    "risk_assessment_tool": self.tools.get("risk_assessment_tool"),
                    "knowledge_graph_search": self.tools.get("knowledge_graph_search")
                },
                config=agent_configs.get("preventive_care", {})
            ),
            "research_and_web_search": WebSearchAgent(
                model=self.model,
                tools={
                    "web_search": self.tools.get("web_search"),
                    "pubmed_search": self.tools.get("pubmed_search"),
                    "clinical_trials_search": self.tools.get("clinical_trials_search"),
                    "drug_database": self.tools.get("drug_database")
                },
                config=agent_configs.get("web_search", {})
            ),
            "general_medical_query": GeneralMedicalAgent(
                model=self.model,
                tools={
                    "symptom_analyzer": self.tools.get("symptom_analyzer"),
                    "vital_signs_analyzer": self.tools.get("vital_signs_analyzer")
                },
                config=agent_configs.get("general", {})
            )
        }
        
        return agents
        
    def _build_workflow(self) -> StateGraph:
        """Build a simplified workflow for direct streaming based on available agents"""
        
        # Create workflow
        workflow = StateGraph(MedicalAssistantState)
        
        # Add nodes for each available agent
        for agent_name, agent in self.agents.items():
            workflow.add_node(agent_name, self._create_agent_node(agent))
            
        # Add special nodes
        workflow.add_node("human_review", self._human_review_node)
        workflow.add_node("final_output", self._final_output_node)
        
        # Set entry point to general_medical_query as fallback since supervisor is commented out
        workflow.set_entry_point("general_medical_query")
        
        # Add conditional edges from general_medical_query
        workflow.add_conditional_edges(
            "general_medical_query",
            self._supervisor_router,
            {
            "diagnostic": "diagnostic",
            "treatment": "treatment",
            "drug_interaction": "drug_interaction",
            "preventive_care_and_risk_assessment": "preventive_care_and_risk_assessment",
            "research_and_web_search": "research_and_web_search",
            "human_review": "human_review",
            "final_output": "final_output",
            "end": END
            }
        )
        
        # Add edges between available agents in logical flow
        workflow.add_edge("diagnostic", "treatment")
        workflow.add_edge("treatment", "drug_interaction")
        workflow.add_edge("drug_interaction", "final_output")
        workflow.add_edge("preventive_care_and_risk_assessment", "final_output")
        workflow.add_edge("research_and_web_search", "final_output")
        
        # Add edges from special nodes
        workflow.add_edge("human_review", "final_output")
        workflow.add_edge("final_output", END)
        
        # Compile with memory
        memory = MemorySaver()
        compiled = workflow.compile(checkpointer=memory)
        
        return compiled
        
    def _create_agent_node(self, agent: BaseMedicalAgent):
        """Create a node function for an agent that supports streaming"""
        
        async def agent_node(state: MedicalAssistantState) -> Dict[str, Any]:
            """Execute agent processing with streaming support"""
            
            self.logger.info(f"Executing agent: {agent.name}",
                           session_id=state.get("session_id"))
            
            try:
                # Update agent state
                state["agent_states"][agent.name] = "processing"
                
                # Load patient context from memory
                if state.get("patient_id"):
                    patient_context = await self.memory_system.get_patient_context(
                        state["patient_id"]
                    )
                    state["patient_context"].update(patient_context)
                
                # Validate input
                if not await agent.validate_input(state):
                    self.logger.warning(f"Agent {agent.name} validation failed")
                    return {"current_agent": "supervisor"}
                
                # Process with streaming
                updates = await agent.process_with_streaming(state)
                
                # Store conversation in memory
                if updates.get("messages"):
                    await self._store_conversation(state, updates["messages"][-1])
                
                # Update agent count
                updates["total_agents_involved"] = state.get("total_agents_involved", 0) + 1
                
                # Mark agent as completed
                state["agent_states"][agent.name] = "completed"
                
                return updates
                
            except Exception as e:
                self.logger.error(f"Agent {agent.name} failed", 
                                error=str(e),
                                session_id=state.get("session_id"))
                
                # Add error to state
                error_entry = {
                    "agent": agent.name,
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                return {
                    "error_log": state.get("error_log", []) + [error_entry],
                    "current_agent": "supervisor",
                    "agent_states": {
                        **state.get("agent_states", {}),
                        agent.name: "error"
                    }
                }
                
        return agent_node
        
    def _supervisor_router(self, state: MedicalAssistantState) -> str:
        """Route from supervisor to next agent"""
        return state.get("current_agent", "end")
        
    def _validation_router(self, state: MedicalAssistantState) -> str:
        """Route from validation agent"""
        
        # Check if human review required
        if state.get("requires_human_review"):
            return "human_review"
            
        # Check confidence threshold
        avg_confidence = self._calculate_average_confidence(state)
        if avg_confidence < self.config.get("confidence_threshold", 0.7):
            return "human_review"
            
        # Check if more processing needed
        if state.get("current_agent") == "supervisor":
            return "supervisor"
            
        # Otherwise, go to final output
        return "final_output"
        
    async def _human_review_node(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Handle human review requirement"""
        
        self.logger.info("Human review requested", 
                        session_id=state.get("session_id"),
                        reason=state.get("review_reason", "Low confidence"))
        
        review_message = {
            "role": "assistant",
            "content": "This case requires human medical professional review due to:\n" +
                      f"- Average confidence: {self._calculate_average_confidence(state):.1%}\n" +
                      f"- Emergency flag: {state.get('emergency_flag', False)}\n" +
                      f"- Complex medical conditions detected",
            "metadata": {
                "agent": "human_review",
                "timestamp": datetime.utcnow().isoformat()
            }
        }
        
        return {
            "messages": state.get("messages", []) + [review_message],
            "human_review_completed": True,
            "current_agent": "final_output"
        }
        
    async def _final_output_node(self, state: MedicalAssistantState) -> Dict[str, Any]:
        """Generate final output summary"""
        
        # Generate summary
        summary = self._generate_summary(state)
        
        final_message = {
            "role": "assistant",
            "content": summary,
            "metadata": {
                "agent": "final_output",
                "timestamp": datetime.utcnow().isoformat(),
                "session_summary": {
                    "agents_involved": state.get("total_agents_involved", 0),
                    "confidence": self._calculate_average_confidence(state),
                    "has_diagnosis": bool(state.get("diagnosis_history")),
                    "has_treatment": bool(state.get("treatment_plans")),
                    "required_human_review": state.get("requires_human_review", False)
                }
            }
        }
        
        return {
            "messages": state.get("messages", []) + [final_message],
            "workflow_completed": True
        }
        
    def _calculate_average_confidence(self, state: MedicalAssistantState) -> float:
        """Calculate average confidence across all agents"""
        confidence_scores = state.get("confidence_scores", {})
        if not confidence_scores:
            return 0.5
            
        return sum(confidence_scores.values()) / len(confidence_scores)
        
    def _generate_summary(self, state: MedicalAssistantState) -> str:
        """Generate comprehensive summary of the session"""
        
        summary = "## Medical Consultation Summary\n\n"
        
        # Symptoms
        if state.get("symptoms"):
            summary += "### Presenting Symptoms\n"
            for symptom in state["symptoms"]:
                summary += f"- {symptom}\n"
            summary += "\n"
            
        # Diagnosis
        if state.get("diagnosis_history"):
            latest_diagnosis = state["diagnosis_history"][-1]
            diagnosis = latest_diagnosis.get("diagnosis", {})
            primary = diagnosis.get("primary_diagnosis", {})
            
            summary += "### Diagnosis\n"
            summary += f"**{primary.get('condition', 'Unknown')}**\n"
            summary += f"- Confidence: {primary.get('confidence', 0):.1%}\n\n"
            
        # Treatment Plan
        if state.get("treatment_plans"):
            latest_plan = state["treatment_plans"][-1]
            summary += "### Treatment Recommendations\n"
            summary += "See detailed treatment plan above\n\n"
            
        # Important Notes
        if state.get("emergency_flag"):
            summary += "### ⚠️ URGENT ATTENTION REQUIRED\n"
            summary += "This case was flagged as requiring immediate medical attention.\n\n"
            
        if state.get("requires_human_review"):
            summary += "### 👨‍⚕️ Professional Review Recommended\n"
            summary += "Due to complexity or uncertainty, professional medical review is advised.\n\n"
            
        # Confidence
        avg_confidence = self._calculate_average_confidence(state)
        summary += f"### Overall Confidence: {avg_confidence:.1%}\n"
        
        return summary
        
    async def _store_conversation(self, state: MedicalAssistantState, message: Dict[str, Any]):
        """Store conversation in memory system"""
        if self.memory_system and state.get("session_id"):
            await self.memory_system.add_conversation_entry(
                session_id=state["session_id"],
                patient_id=state.get("patient_id"),
                message=message
            )
            
    async def process_query_direct(self, 
                                 query: str,
                                 patient_id: Optional[str] = None,
                                 context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """
        Direct processing without complex workflow - for immediate streaming
        """
        
        session_id = str(uuid.uuid4())
        
        try:
            # Step 1: Get context from knowledge graph
            yield "<||action||>🔍 Searching medical knowledge base...\n"
            
            kg_tool = self.tools.get("knowledge_graph_search")
            if kg_tool:
                kg_results = await kg_tool.execute(
                    query=query,
                    entity_types=["Disease", "Drug", "Symptom", "Gene", "Protein", "Metabolite", "Pathway"],
                    limit=5,
                    include_relationships=True
                )
                
                context_info = kg_results.get("context", "")
                entities_found = len(kg_results.get("entities", []))
                relationships_found = len(kg_results.get("relationships", []))
                
                yield f"<||action||> ✅ Found {entities_found} entities and {relationships_found} relationships\n\n"
            else:
                context_info = ""
                yield "<||action||> ⚠️ Knowledge graph not available\n\n"
            
            # Step 2: Determine the right agent based on query
            yield "<||action||> 🤖 Analyzing query and selecting appropriate agent...\n"
            
            agent_name = await self._determine_agent(query)
            agent = self.agents.get(agent_name)
            
            if not agent:
                agent = self.agents.get("supervisor")
                agent_name = "supervisor"
            
            yield f"<||action||>👨‍⚕️ **{agent_name.title()} Agent** responding:\n\n"
            
            # Step 3: Prepare enhanced prompt with context
            enhanced_prompt = self._create_enhanced_prompt(query, context_info, kg_results)
            enhanced_prompt = agent._get_system_prompt()
            # substitue query in the {input_query} in the enhanced_prompt
            enhanced_prompt = enhanced_prompt.replace("{input_query}", query) 
            # Step 4: Stream response from the selected agent
            chat_history = []
            
            async for chunk in self.model.generate_with_context(
                enhanced_prompt, 
                chat_history, 
                use_rag=True  # We already have context
            ):
                yield chunk
                
            # Step 5: Store in memory if needed
            if self.memory_system:
                await self.memory_system.add_conversation_entry(
                    session_id=session_id,
                    patient_id=patient_id,
                    message={
                        "role": "user",
                        "content": query,
                        "timestamp": datetime.utcnow().isoformat()
                    }
                )
            
        except Exception as e:
            self.logger.error("Error in direct query processing", error=str(e))
            # yield f"\n❌ **Error**: {str(e)}\n"
            yield "<||action||> ❌ An error occurred while processing your query. Please try again later.\n"
    
    async def _determine_agent(self, query: str) -> str:
        """Determine which agent should handle the query using BM25 algorithm"""
        
        # Preprocess function
        def preprocess_text(text: str) -> List[str]:
            """Clean and tokenize text for BM25"""
            # Convert to lowercase and remove punctuation
            text = text.lower().translate(str.maketrans('', '', string.punctuation))
            # Remove extra whitespace and split
            tokens = re.split(r'\s+', text.strip())
            return [token for token in tokens if len(token) > 2]  # Filter short tokens
        
        # Define agent descriptions with comprehensive keywords and contexts
        agent_descriptions = {
            "general_medical_query": [
                "What’s the normal body temperature for a healthy adult?",
                "How many hours of sleep should an average adult get each night?",
                "Why do I get dizzy when I stand up too fast?",
                "Is it normal to feel tired after eating a big meal?",
                "What causes muscle cramps during the night?",
                "How much water should I drink per day to stay hydrated?",
                "Why do some people get motion sickness more than others?",
                "Can weather changes really affect joint pain?",
                "What’s the difference between a virus and bacteria?",
                "Is cracking your knuckles bad for your joints?",
                "How does caffeine affect the body over time?",
                "What are the health benefits of walking daily?",
                "What happens to your body when you fast for a day?",
                "How can I tell if my immune system is weak?",
                "Why do I feel more tired during the winter months?",
                "What causes hiccups and how can I stop them?",
                "Is it safe to sleep with a fan on all night?",
                "How long should a typical cold last?",
                "Are multivitamins actually effective or necessary?",
                "Why do I feel sore the day after exercising?"
            ],
            "diagnostic": [
                "I am a 52 year old woman and I have a terrible stomach ache. I had something like this 2 years ago! I also feel sick, sweaty and have a fever at 38.2 degrees. The pain is more in the left side of the stomach, more down - kind of a pressure pain. I have been suffering from hypertension and widespread athrosis for a long time. The only medications I take are Ramipril and HCT. What are most likely diagnoses? Name up to five.",
                "I am a 67-year-old woman, a farmer by profession. I have had muscle pain all over for 4 months. I am exhausted and can hardly sleep. At first it was more the neck. Then it moved to the hands. But most of all in the upper arm. But it also goes from my back to the back of my knees. I could hardly move my legs. For example, I have to lift my leg out when I get out of the car. When I get up from a sitting position, I first bend down until I can walk. That's how much it hurt my back. Also in the knees. It was especially bad in my arms at night. I couldn't stretch because there was always a pulling sensation at the bottom. I thought that it came from the stomach.  What are most likely diagnoses? Name up to five.",
                "My one-year-old daughter had a persistent high fever up to 40 degrees. I have already been to the doctor's office and there it was classified as \"uncharacteristic fever\". Other than a reddened throat, I didn't notice anything else. We were prescribed an antibiotic two days ago (clarithromycin) because the fever was still persistent. She responded poorly to paracetamol. Now the fever is finally gone, but a rash has appeared on the abdomen. What are most likely diagnoses? Name up to five.",
                "I am a 63 year old male and have had a cough for over 3 months. At first I thought it was a cold, but it can't last that long! I have been suffering from high blood pressure and coronary heart disease for a long time. I have also had 2 stents in this context. I take Ramipril, Atorvastatin and ASS as medication. Otherwise I am symptom-free. What are most likely diagnoses? Name up to five.",
                "My husband is 78 years old and since last night he can no longer lift his arm properly and he can no longer walk. My son and I take care of him (care level 2). 3 months ago he was in the hospital because of bleeding with prostate cancer. A certain therapy was discontinued at that time. What are most likely diagnoses? Name up to five.",
                "I am a 68 year old man and I always have such a twinge in my chest. It is a feeling of pressure in the chest that has been there for 6 weeks. Especially when I exert myself, it stings in there under the breastbone. I am slightly overweight, but otherwise healthy. What are most likely diagnoses? Name up to five.",
                "Our neighbor is an avid motorcyclist and was stung by a wasp under his leather suit on his way home today. He has quaddle-shaped skin rashes all over his body. What are most likely diagnoses? Name up to five."

                # "What could be causing my persistent fatigue and joint pain?",
                # "I have a rash that won't go away—what might it be?",
                # "What’s the difference between a cold and the flu?",
                # "signs symptoms presentation clinical manifestations",
                # "How is Lyme disease diagnosed?",
                # "I have night sweats and swollen lymph nodes—what could this mean?",
                # "biopsy pathology genetic testing molecular diagnostics",
                # "screening early detection preventive medicine risk assessment",
                # "rare diseases orphan conditions genetic disorders",
                # "infectious diseases bacterial viral fungal parasitic infections"
            ],
            "treatment": [
                "I am a 52 year old woman and I have a terrible stomach ache. I had something like this 2 years ago! I also feel sick, sweaty and have a fever at 38.2 degrees. The pain is more in the left side of the stomach, more down - kind of a pressure pain. I have been suffering from hypertension and widespread athrosis for a long time. The only medications I take are Ramipril and HCT. My doctor has diagnosed me with Diverticulitis. What are the most appropriate therapies in my case? Name up to five.",
                "I am a 67-year-old woman, a farmer by profession. I have had muscle pain all over for 4 months. I am exhausted and can hardly sleep. At first it was more the neck. Then it moved to the hands. But most of all in the upper arm. But it also goes from my back to the back of my knees. I could hardly move my legs. For example, I have to lift my leg out when I get out of the car. When I get up from a sitting position, I first bend down until I can walk. That's how much it hurt my back. Also in the knees. It was especially bad in my arms at night. I couldn't stretch because there was always a pulling sensation at the bottom. I thought that it came from the stomach.  My doctor has diagnosed me with Polymyalgia rheumatica. What are the most appropriate therapies in my case? Name up to five.",
                "My one-year-old daughter had a persistent high fever up to 40 degrees. I have already been to the doctor's office and there it was classified as \"uncharacteristic fever\". Other than a reddened throat, I didn't notice anything else. We were prescribed an antibiotic two days ago (clarithromycin) because the fever was still persistent. She responded poorly to paracetamol. Now the fever is finally gone, but a rash has appeared on the abdomen. My doctor has diagnosed me with Roseola/Sixth disease. What are the most appropriate therapies in my case? Name up to five.",
                "I am a 63 year old male and have had a cough for over 3 months. At first I thought it was a cold, but it can't last that long! I have been suffering from high blood pressure and coronary heart disease for a long time. I have also had 2 stents in this context. I take Ramipril, Atorvastatin and ASS as medication. Otherwise I am symptom-free. My doctor has diagnosed me with bronchial asthma . What are the most appropriate therapies in my case? Name up to five.",
                "My husband is 78 years old and since last night he can no longer lift his arm properly and he can no longer walk. My son and I take care of him (care level 2). 3 months ago he was in the hospital because of bleeding with prostate cancer. A certain therapy was discontinued at that time. My doctor has diagnosed me with Stroke. What are the most appropriate therapies in my case? Name up to five.",
                "Our neighbor is an avid motorcyclist and was stung by a wasp under his leather suit on his way home today. He has quaddle-shaped skin rashes all over his body. My doctor has diagnosed me with Anaphylaxis/Anaphylactic shock. What are the most appropriate therapies in my case? Name up to five.",
                "I am a 68 year old man and I always have such a twinge in my chest. It is a feeling of pressure in the chest that has been there for 6 weeks. Especially when I exert myself, it stings in there under the breastbone. I am slightly overweight, but otherwise healthy. My doctor has diagnosed me with Coronary artery disease. What are the most appropriate therapies in my case? Name up to five.",
                # "treatment therapy management intervention protocol",
                # "medication drug prescription pharmaceutical therapy",
                # "surgery surgical procedure operation intervention",
                # "rehabilitation physical therapy occupational therapy",
                # "lifestyle modifications diet exercise behavioral changes",
                # "alternative medicine complementary therapy holistic approach",
                # "prognosis outcome recovery healing cure remission",
                # "chronic disease management long term care maintenance",
                # "palliative care end of life comfort care hospice",
                # "immunotherapy targeted therapy personalized medicine"
            ],
            "drug_interaction": [
                "Can I take ibuprofen if I’m already on blood thinners?",
                "Are there known interactions between St. John’s Wort and antidepressants?",
                "What should I do if I accidentally took two doses of my anxiety medication?",
                "I’m experiencing dizziness after starting a new blood pressure pill—is this normal?",
                "Is it safe to drink alcohol while on antibiotics like amoxicillin?",
                "How do I know if two of my medications might interact dangerously?",
                "Can antihistamines interfere with heart medication?",
                "What side effects should I expect from starting a statin drug?",
                "Are there medications I should avoid while pregnant or breastfeeding?",
                "How are drug doses adjusted for elderly patients with kidney issues?"
            ],
            "research_and_web_search": [
                "What’s the latest research on treatments for Alzheimer’s disease?",
                "Can you explain what a randomized controlled trial is and why it’s important?",
                "Where can I find trustworthy information about the long-term effects of COVID-19?",
                "What’s the difference between a systematic review and a meta-analysis?",
                "Are there any recent breakthroughs in cancer immunotherapy?",
                "How do AI tools like ChatGPT assist in modern healthcare research?",
                "Where can I access a database of clinical trials for rare diseases?",
                "What are the key findings of recent studies on intermittent fasting?",
                "How does personalized medicine work based on my genetic profile?",
                "Can you help me interpret the results of this medical research paper?"
            ],
            "preventive_care_and_risk_assessment": [
                "At what age should I start getting screened for colon cancer?",
                "How often should women get mammograms based on current guidelines?",
                "I read an article claiming vaccines cause autism. How can I fact-check that?",
                "What health screenings should I get during my annual checkup?",
                "Is it ethical to test experimental drugs on terminal patients?",
                "How do I know if a health claim I saw on social media is true?",
                "Does my insurance cover preventive genetic testing for cancer?",
                "I want a second opinion—how do I request one without offending my doctor?",
                "What are the standards for ensuring quality in hospitals or clinics?",
                "Can I refuse a medical treatment based on my personal beliefs, and what are the implications?"
            ]
        }
        
        # Preprocess query
        query_tokens = preprocess_text(query)
        
        if not query_tokens:
            return "supervisor"
        
        # Prepare corpus (all agent descriptions combined)
        corpus = []
        agent_mapping = []
        
        for agent_name, descriptions in agent_descriptions.items():
            for desc in descriptions:
                corpus.append(preprocess_text(desc))
                agent_mapping.append(agent_name)
        
        # Initialize BM25
        bm25 = BM25Okapi(corpus)
        
        # Get BM25 scores for the query
        scores = bm25.get_scores(query_tokens)
        
        # Find the best matching agent
        print("BM25 scores:", scores, agent_mapping)

        if len(scores) > 0:
            best_match_idx = scores.argmax()
            best_agent = agent_mapping[best_match_idx]
            best_score = scores[best_match_idx]
            
            # Set a minimum threshold to avoid low-quality matches
            threshold = 0.5
            print("Best agent:", best_agent, "with score:", best_score)
            if best_score > threshold:
                self.logger.info(
                    f"BM25 agent selection: {best_agent}",
                    score=float(best_score),
                    query=query[:100],
                    session_id=getattr(self, 'current_session_id', None)
                )
                return best_agent
        
        # Fallback to keyword-based matching if BM25 score is too low
        self.logger.info(
            "BM25 score below threshold, using keyword fallback",
            query=query[:100]
        )

        print("BM25 scores:", scores)
        
        # Enhanced keyword fallback with weighted scoring
        agent_scores = {
            "general": 0,
            "diagnostic": 0,
            "treatment": 0,
            "drug_interaction": 0,
            "research": 0,
            "web_search": 0,
            "validation": 0
        }
        
        # High priority emergency keywords (weight: 3)
        emergency_critical = ["emergency", "urgent", "critical", "severe", "acute", "chest pain", 
                             "difficulty breathing", "unconscious", "bleeding", "poisoning"]
        for keyword in emergency_critical:
            if keyword in query.lower():
                agent_scores["general"] += 3
        
        # Diagnostic keywords (weight: 2)
        diagnostic_terms = ["symptoms", "diagnosis", "what is", "condition", "disease", 
                           "signs", "causes", "disorder"]
        for keyword in diagnostic_terms:
            if keyword in query.lower():
                agent_scores["diagnostic"] += 2
        
        # Treatment keywords (weight: 2)
        treatment_terms = ["treatment", "therapy", "cure", "medication", "surgery", 
                          "management", "intervention"]
        for keyword in treatment_terms:
            if keyword in query.lower():
                agent_scores["treatment"] += 2
        
        # Drug interaction keywords (weight: 2)
        drug_terms = ["interaction", "side effects", "contraindication", "allergy", 
                     "adverse", "dosage", "drug"]
        for keyword in drug_terms:
            if keyword in query.lower():
                agent_scores["drug_interaction"] += 2
        
        # Research keywords (weight: 1)
        research_terms = ["research", "study", "trial", "latest", "new", "evidence", 
                         "systematic review"]
        for keyword in research_terms:
            if keyword in query.lower():
                agent_scores["research"] += 1
        
        # Validation keywords (weight: 1)
        validation_terms = ["verify", "fact check", "guidelines", "safety", "validate"]
        for keyword in validation_terms:
            if keyword in query.lower():
                agent_scores["validation"] += 1
        
        # General health info keywords (weight: 1)
        general_terms = ["information", "education", "general", "basic", "explain"]
        for keyword in general_terms:
            if keyword in query.lower():
                agent_scores["web_search"] += 1
        
        # Return the agent with the highest score
        if max(agent_scores.values()) > 0:
            best_agent = max(agent_scores, key=agent_scores.get)
            self.logger.info(
                f"Keyword-based agent selection: {best_agent}",
                scores=agent_scores,
                query=query[:100]
            )
            return best_agent
        
        # Final fallback to supervisor
        return "supervisor"
    
    def _create_enhanced_prompt(self, query: str, context: str, kg_results: Dict[str, Any]) -> str:
        """Create an enhanced prompt with context"""
        
        entities = kg_results.get("entities", [])
        relationships = kg_results.get("relationships", [])
        recommendations = kg_results.get("recommendations", [])
        
        enhanced_prompt = f"""You are a medical AI assistant. Use the following context to provide accurate, helpful medical information.

**IMPORTANT**: 
- Always provide evidence-based information
- Acknowledge limitations and uncertainties
- Recommend consulting healthcare professionals when appropriate
- Do not provide definitive diagnoses

**Context from Medical Knowledge Base:**
{context}

**Detected Entities:** {len(entities)} medical entities found
**Relationships:** {len(relationships)} connections identified

**User Query:** {query}

**Instructions:**
1. Analyze the query in the context of the provided medical information
2. Provide a comprehensive, evidence-based response
3. Include relevant medical entities and relationships when applicable
4. Suggest follow-up questions or recommendations if helpful
5. Always emphasize the importance of professional medical consultation

Please provide your response:"""

        return enhanced_prompt
    
    async def process_query(self, 
                           query: str,
                           patient_id: Optional[str] = None,
                           context: Optional[Dict[str, Any]] = None) -> AsyncGenerator[str, None]:
        """
        Main entry point - use direct processing for better streaming
        """
        async for chunk in self.process_query_direct(query, patient_id, context):
            yield chunk