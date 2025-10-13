from typing import Dict, List, Any, Optional, Tuple
from neo4j import GraphDatabase
import json
import re
import spacy
from sentence_transformers import SentenceTransformer
from spacy.tokens import Doc

from ...tools.base_tool import BaseTool
from ...config.settings import settings


class KnowledgeGraphSearchTool(BaseTool):
    """
    Advanced search medical knowledge graph in Neo4j with entity detection,
    hybrid search, and recommendation generation
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(
            name="knowledge_graph_search",
            description="Search medical knowledge graph for entities and relationships with AI-powered entity detection",
            config=config
        )
        
        self.driver = GraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password)
        )
        
        # Initialize NLP components
        self._init_nlp_components()
        
        # Initialize sentence transformer for embeddings
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5', trust_remote_code=True)
        
        # Medical entity labels
        self.medical_labels = {
            'DISEASE', 'CHEMICAL', 'GENE', 'PROTEIN', 'DRUG', 
            'MEDCOND', 'DIAGNOSIS', 'MEDPROC', 'ANATOMY', 'SYMPTOM', 'COMPOUND',
            'METABOLITE', 'PATHWAY', 'BIOLOGICAL_PROCESS', 'PEPTIDE', 'TRANSCRIPT', 'TISSUE'
        }
        
        # Medical stopwords
        self.medical_stopwords = {
            "tell", "me", "about", "what", "is", "are", "can", "you", "explain",
            "describe", "know", "understand", "mean", "definition", "define"
        }
        
        # Medical compound terms
        self.medical_compounds = {
            "crohn's disease", "alzheimer's disease", "parkinson's disease",
            "multiple sclerosis", "breast cancer", "lung cancer",
            "type 2 diabetes", "high blood pressure"
        }
        
    def _init_nlp_components(self):
        """Initialize spaCy NLP model with medical entity recognition"""
        try:
            self.nlp = spacy.load("en_core_sci_md")
        except OSError:
            # Fallback to base English model if medical model not available
            self.nlp = spacy.load("en_core_web_sm")
            
        # Add entity ruler for medical terms
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", config={"validate": True})
            
            # Add patterns for common medical terms and question prefixes
            patterns = [
                {"label": "QUESTION_PREFIX", "pattern": [{"LOWER": {"IN": ["tell", "explain", "describe", "what", "how"]}}]},
                {"label": "QUESTION_PREFIX", "pattern": [{"LOWER": "what"}, {"LOWER": "is"}]},
                {"label": "QUESTION_PREFIX", "pattern": [{"LOWER": "tell"}, {"LOWER": "me"}, {"LOWER": "about"}]},
                {"label": "DISEASE", "pattern": "crohn's disease"},
                {"label": "DISEASE", "pattern": "crohns disease"},
                {"label": "DISEASE", "pattern": [{"LOWER": "crohn"}, {"LOWER": "'s"}, {"LOWER": "disease"}]},
                {"label": "DISEASE", "pattern": [{"LOWER": "crohn"}, {"LOWER": "disease"}]},
                {"label": "DISEASE", "pattern": "diabetes"},
                {"label": "CHEMICAL", "pattern": [{"LOWER": "tnf"}, {"LOWER": "-"}, {"LOWER": "alpha"}]},
            ]
            ruler.add_patterns(patterns)
        
    async def execute(self, **kwargs) -> Dict[str, Any]:
        """Execute enhanced knowledge graph search with entity detection"""
        query = kwargs.get("query", "")
        entity_types = kwargs.get("entity_types", ["Disease", "Drug", "Symptom", "Gene", "Protein"])
        limit = kwargs.get("limit", 5)
        include_relationships = kwargs.get("include_relationships", True)
        use_hybrid_search = kwargs.get("use_hybrid_search", True)
        generate_recommendations = kwargs.get("generate_recommendations", True)
        
        if not query:
            return {
                "query": query,
                "entities": [],
                "relationships": [],
                "context": "",
                "recommendations": [],
                "entity_count": 0,
                "relationship_count": 0,
                "error": "No query provided"
            }
        
        try:
            # Preprocess query and detect entities
            processed_entities = self._preprocess_query(query)
            detected_entities = await self._detect_entities_from_index(query)
            
            # Determine search strategy
            if use_hybrid_search and detected_entities:
                # Use hybrid search with best detected entity
                best_match = detected_entities[0]
                search_results = await self._hybrid_search(best_match['text'], limit)
                entities = self._format_hybrid_results(search_results)
            else:
                # Fallback to traditional entity search
                entities = await self._search_entities(query, entity_types, limit)
            
            # Get relationships if requested
            relationships = []
            if include_relationships and entities:
                relationships = await self._get_relationships(entities[:3])
            
            # Generate recommendations
            recommendations = []
            if generate_recommendations and entities:
                recommendations = self._generate_recommendations(entities[0] if entities else {}, query)
            
            # Generate context
            context = self._generate_enhanced_context(entities, relationships, detected_entities)
            
            return {
                "query": query,
                "processed_entities": processed_entities,
                "detected_entities": detected_entities,
                "entities": entities,
                "relationships": relationships,
                "context": context,
                "recommendations": recommendations,
                "entity_count": len(entities),
                "relationship_count": len(relationships),
                "search_method": "hybrid" if use_hybrid_search and detected_entities else "traditional"
            }
            
        except Exception as e:
            return {
                "query": query,
                "entities": [],
                "relationships": [],
                "context": "",
                "recommendations": [],
                "entity_count": 0,
                "relationship_count": 0,
                "error": str(e)
            }
        
    def validate_params(self, **kwargs) -> bool:
        """Validate search parameters"""
        return bool(kwargs.get("query"))
    
    def _preprocess_query(self, text: str) -> List[str]:
        """Preprocess query to extract relevant medical entities"""
        entities = self._extract_query_focus(text)
        
        if not entities:
            return [re.sub(r'\s+', ' ', text).strip()]
        
        return [entity["text"] for entity in entities]
    
    def _extract_query_focus(self, text: str) -> List[Dict[str, Any]]:
        """Extract main medical entities from query"""
        doc = self.nlp(text)
        
        # Merge adjacent entities
        merged_entities = self._merge_adjacent_entities(doc)
        
        # Filter relevant entities
        relevant_entities = []
        for ent_text, ent_label in merged_entities:
            if (ent_text.lower() not in self.medical_stopwords and 
                self._is_medical_entity(ent_text, ent_label)):
                relevant_entities.append({
                    "text": ent_text,
                    "type": ent_label,
                    "original_text": text
                })
        
        # Fallback to all entities if no medical ones found
        if not relevant_entities:
            for ent in doc.ents:
                relevant_entities.append({
                    "text": ent.text,
                    "type": ent.label_,
                    "original_text": text
                })
                
        return relevant_entities
    
    def _merge_adjacent_entities(self, doc: Doc) -> List[Tuple[str, str]]:
        """Merge adjacent entities that might be part of the same medical term"""
        merged_entities = []
        i = 0
        
        while i < len(doc.ents):
            current_ent = doc.ents[i]
            
            # Look for adjacent entities
            if i + 1 < len(doc.ents):
                next_ent = doc.ents[i + 1]
                
                # Check if entities are adjacent and could form a medical term
                if (next_ent.start == current_ent.end or 
                    (next_ent.start == current_ent.end + 1 and 
                     doc[current_ent.end].text.lower() in {"'s", "of", "and"})):
                    
                    combined_text = doc[current_ent.start:next_ent.end].text
                    if self._is_medical_entity(combined_text, current_ent.label_):
                        merged_entities.append((combined_text, "MERGED_MEDICAL"))
                        i += 2
                        continue
            
            if current_ent.label_ == "ENTITY" or self._is_medical_entity(current_ent.text, current_ent.label_):
                merged_entities.append((current_ent.text, current_ent.label_))
            i += 1
            
        return merged_entities
    
    def _is_medical_entity(self, text: str, ent_label: str) -> bool:
        """Determine if entity is medical based on text and label"""
        if text.lower() in self.medical_compounds:
            return True
            
        if ent_label in self.medical_labels:
            return True
            
        # Check for medical patterns
        medical_patterns = [
            "disease", "syndrome", "disorder", "cancer", "itis",
            "osis", "emia", "gene", "protein", "receptor"
        ]
        if any(pattern in text.lower() for pattern in medical_patterns):
            return True
            
        return False
    
    async def _detect_entities_from_index(self, text: str) -> List[Dict[str, Any]]:
        """Detect medical entities by querying Neo4j fulltext index"""
        embedding = self.model.encode(text).tolist()
        
        def run_query(tx):
            cypher_query = """
            CALL db.index.vector.queryNodes($index, $k, $embedding) 
            YIELD node, score 
            WHERE score > 0.8
            RETURN DISTINCT {
                text: node.name,
                label: head(labels(node)),
                o_label: node.o_label,
                score: score
            } as result
            ORDER BY result.score DESC
            """
            result = tx.run(
                cypher_query,
                index="AllEntities",
                embedding=embedding,
                k=5
            )
            return [record["result"] for record in result]
        
        with self.driver.session() as session:
            results = session.read_transaction(run_query)
            
            entities = []
            seen = set()
            for result in results:
                if result['text'] and result['label']:
                    key = f"{result['o_label']}"
                    if key not in seen:
                        entities.append({
                            "text": result['text'],
                            "label": result['label'],
                            "score": result['score'],
                            "o_label": result['o_label'],
                            "f_key": key
                        })
                        seen.add(key)
            
            return entities
    
    async def _hybrid_search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Perform hybrid vector and fulltext search"""
        embedding = self.model.encode(query).tolist()
        
        def run_query(tx):
            cypher_query = """
            CALL {
                CALL db.index.vector.queryNodes($index, $k, $embedding)
                YIELD node, score
                RETURN node, score
                UNION
                CALL db.index.fulltext.queryNodes($keyword_index, $text_query, {limit: $k})
                YIELD node, score
                RETURN node, score
            }
            WITH node, score
            ORDER BY score DESC

            MATCH (n:Protein|Drug|Phenotype|Disease|Gene|Metabolite|Pathway|Biological_process|Peptide|Transcript|Compound|Tissue|Symptom) 
            WHERE id(n) = node.f_key
            WITH collect({node: n, score: score}) AS top_nodes

            UNWIND top_nodes AS top_node_data
            WITH top_node_data.node AS root_node, top_node_data.score AS score

            // Find related entities
            OPTIONAL MATCH (root_node)-[r1]-(disease:Disease)
            WHERE root_node <> disease
            WITH root_node, score, collect(DISTINCT {entity: disease{.*, embedding:null}, rel_type: type(r1)})[..3] AS related_diseases

            OPTIONAL MATCH (root_node)-[r2]-(protein:Protein)
            WHERE root_node <> protein
            WITH root_node, score, related_diseases, collect(DISTINCT {entity: protein{.*, embedding:null}, rel_type: type(r2)})[..3] AS related_proteins

            OPTIONAL MATCH (root_node)-[r3]-(drug:Drug|Compound)
            WHERE root_node <> drug
            WITH root_node, score, related_diseases, related_proteins, collect(DISTINCT {entity: drug{.*, embedding:null}, rel_type: type(r3)})[..3] AS related_drugs

            OPTIONAL MATCH (root_node)-[r4]-(metabolite:Metabolite)
            WHERE root_node <> metabolite
            WITH root_node, score, related_diseases, related_proteins, related_drugs, collect(DISTINCT {entity: metabolite{.*, embedding:null}, rel_type: type(r4)})[..3] AS related_metabolites

            OPTIONAL MATCH (root_node)-[r5]-(gene:Gene)
            WHERE root_node <> gene
            WITH root_node, score, related_diseases, related_proteins, related_drugs, related_metabolites, collect(DISTINCT {entity: gene{.*, embedding:null}, rel_type: type(r5)})[..3] AS related_genes

            RETURN {
                score: score,
                metadata: labels(root_node)[0],
                root: root_node{.*, embedding:null},
                related_nodes: {
                    diseases: [d IN related_diseases WHERE d.entity IS NOT NULL | {
                        properties: d.entity,
                        relationship: d.rel_type
                    }],
                    proteins: [p IN related_proteins WHERE p.entity IS NOT NULL | {
                        properties: p.entity,
                        relationship: p.rel_type
                    }],
                    drugs: [d IN related_drugs WHERE d.entity IS NOT NULL | {
                        properties: d.entity,
                        relationship: d.rel_type
                    }],
                    metabolites: [m IN related_metabolites WHERE m.entity IS NOT NULL | {
                        properties: m.entity,
                        relationship: m.rel_type
                    }],
                    genes: [g IN related_genes WHERE g.entity IS NOT NULL | {
                        properties: g.entity,
                        relationship: g.rel_type
                    }]
                }
            } AS result
            """
            result = tx.run(
                cypher_query,
                index="AllEntities",
                keyword_index="all_entities_index",
                k=k,
                embedding=embedding,
                text_query=query
            )
            return [record["result"] for record in result]
        
        with self.driver.session() as session:
            return session.read_transaction(run_query)
    
    def _format_hybrid_results(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format hybrid search results to standard entity format"""
        entities = []
        for result in results:
            root = result.get('root', {})
            entities.append({
                "id": root.get('f_key', 'unknown'),
                "name": root.get('name', 'Unknown'),
                "labels": [result.get('metadata', 'Unknown')],
                "score": result.get('score', 0),
                "properties": root,
                "related_nodes": result.get('related_nodes', {})
            })
        return entities
    
    async def _search_entities(self, 
                              query: str,
                              entity_types: List[str],
                              limit: int) -> List[Dict[str, Any]]:
        """Traditional entity search using fulltext index"""
        entities = []
        
        with self.driver.session() as session:
            cypher_query = """
            CALL db.index.fulltext.queryNodes('all_entities_index', $query) 
            YIELD node, score
            WHERE any(label IN labels(node) WHERE label IN $entity_types)
            RETURN node, score, labels(node) as labels
            ORDER BY score DESC
            LIMIT $limit
            """
            
            result = session.run(
                cypher_query,
                query=query,
                entity_types=entity_types,
                limit=limit
            )
            
            for record in result:
                node = record["node"]
                entities.append({
                    "id": node.id,
                    "name": node.get("name", "Unknown"),
                    "labels": record["labels"],
                    "score": record["score"],
                    "properties": dict(node)
                })
                
        return entities
    
    async def _get_relationships(self, entities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Get relationships between entities"""
        relationships = []
        entity_ids = [e["id"] for e in entities]
        
        with self.driver.session() as session:
            cypher_query = """
            MATCH (n)-[r]-(m)
            WHERE id(n) IN $entity_ids AND id(m) IN $entity_ids
            RETURN n, r, m, type(r) as rel_type
            """
            
            result = session.run(cypher_query, entity_ids=entity_ids)
            
            for record in result:
                relationships.append({
                    "source": {
                        "id": record["n"].id,
                        "name": record["n"].get("name", "Unknown")
                    },
                    "target": {
                        "id": record["m"].id,
                        "name": record["m"].get("name", "Unknown")
                    },
                    "type": record["rel_type"],
                    "properties": dict(record["r"])
                })
                
        return relationships
    
    def _generate_recommendations(self, context: Dict[str, Any], original_query: str) -> List[str]:
        """Generate follow-up question recommendations"""
        recommendations = []
        
        if not context:
            return recommendations
            
        root_name = context.get('name', context.get('properties', {}).get('name', ''))
        entity_type = context.get('labels', [''])[0] if context.get('labels') else ''
        related_nodes = context.get('related_nodes', {})
        
        # Type-specific recommendations
        if related_nodes.get('proteins'):
            recommendations.append(f"What are the proteins associated with {root_name}?")
        
        if related_nodes.get('drugs'):
            if entity_type == 'Disease':
                recommendations.append(f"What are the drugs used to treat {root_name}?")
            else:
                recommendations.append(f"What drugs interact with {root_name}?")
        
        if related_nodes.get('genes'):
            recommendations.append(f"What genes are involved in {root_name}?")
        
        if related_nodes.get('metabolites'):
            recommendations.append(f"What metabolites are associated with {root_name}?")
        
        if entity_type == 'Disease':
            recommendations.extend([
                f"What are the symptoms of {root_name}?",
                f"What are the risk factors for {root_name}?",
                f"What are the common biomarkers for {root_name}?"
            ])
        
        return recommendations[:4]
    
    def _generate_enhanced_context(self, 
                                 entities: List[Dict[str, Any]],
                                 relationships: List[Dict[str, Any]],
                                 detected_entities: List[Dict[str, Any]]) -> str:
        """Generate enhanced human-readable context"""
        context_parts = []
        
        # Detected entities section
        if detected_entities:
            context_parts.append("Detected Entities:")
            for entity in detected_entities[:3]:
                context_parts.append(
                    f"- {entity['text']} ({entity.get('o_label', entity['label'])}) - Score: {entity['score']:.2f}"
                )
            context_parts.append("")
        
        # Main entities section
        context_parts.append("Search Results:")
        for i, entity in enumerate(entities[:3], 1):
            entity_type = entity["labels"][0] if entity["labels"] else "Entity"
            name = entity["name"]
            
            context_parts.append(f"Result {i}:")
            context_parts.append(f"  Type: {entity_type}")
            context_parts.append(f"  Name: {name}")
            context_parts.append(f"  Score: {entity.get('score', 'N/A')}")
            
            # Add key properties
            props = entity.get("properties", {})
            if "description" in props:
                context_parts.append(f"  Description: {props['description']}")
            if "synonyms" in props:
                context_parts.append(f"  Also known as: {props['synonyms']}")
            
            # Add related nodes information
            related_nodes = entity.get("related_nodes", {})
            if related_nodes:
                context_parts.append("  Related Entities:")
                for category, nodes in related_nodes.items():
                    if nodes:
                        context_parts.append(f"    {category.capitalize()}:")
                        for node in nodes[:2]:  # Limit to 2 per category
                            node_name = node.get('properties', {}).get('name', 'Unknown')
                            relationship = node.get('relationship', 'related_to')
                            context_parts.append(f"      - {node_name} ({relationship})")
            
            context_parts.append("")
        
        # Relationships section
        if relationships:
            context_parts.append("Direct Relationships:")
            for rel in relationships[:5]:
                context_parts.append(
                    f"- {rel['source']['name']} {rel['type']} {rel['target']['name']}"
                )
        
        return "\n".join(context_parts)
    
    def __del__(self):
        """Close Neo4j driver connection"""
        if hasattr(self, 'driver'):
            self.driver.close()