import asyncio
import structlog
from typing import Optional, Dict, Any, List, AsyncGenerator
from pathlib import Path
import numpy as np
import torch

logger = structlog.get_logger()

try:
    import llama_cpp
except:
    llama_cpp = None

try:
    import llama_cpp_cuda
except:
    llama_cpp_cuda = None

try:
    import llama_cpp_cuda_tensorcores
except:
    llama_cpp_cuda_tensorcores = None


def llama_cpp_lib():
    """Select the best available llama-cpp library"""
    if llama_cpp_cuda_tensorcores is not None:
        return llama_cpp_cuda_tensorcores
    elif llama_cpp_cuda is not None:
        return llama_cpp_cuda
    else:
        return llama_cpp


class LlamaModelWrapper:
    """Wrapper for LLaMA model with async support using llama-cpp-python"""
    
    # Default models dictionary - can be overridden
    models = {
        'llama-3': 'models/Meta-Llama-3-8B-Instruct.Q8_0.gguf',
        'llama-3.1': 'models/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf',
        'llama-3.2': 'models/Llama-3.2-3B-Instruct-uncensored-Q8_0.gguf',
        'llama-3.3': 'models/Llama-3.3-70B-Instruct-IQ1_M.gguf',
        'deepseek': 'models/DeepSeek-R1-Distill-Qwen-14B-Q8_0.gguf',
    }
    
    def __init__(self, model_path: str = None, model_name: str = 'llama-3.1', device: str = "auto"):
        if model_path:
            self.model_path = Path(model_path)
        else:
            self.model_path = Path(self.models.get(model_name, self.models['llama-3.1']))
        
        self.model_name = model_name
        self.device = device
        self.model = None
        self._initialized = False
        self.grammar_string = ''
        self.grammar = None
    
    def set_context_retreiver(self, context_retriever):
        """Set the context retriever for RAG"""
        self.context_retriever = context_retriever
        
    async def initialize(self) -> 'LlamaModelWrapper':
        """Initialize the model asynchronously"""
        if self._initialized:
            return self
            
        logger.info("Initializing LLaMA model", model_path=str(self.model_path))
        
        try:
            Llama = llama_cpp_lib().Llama
            LlamaCache = llama_cpp_lib().LlamaCache
            # Model parameters similar to your working code
            params = {
                'model_path': str(self.model_path),
                'n_ctx': 4096 * 4,  # Large context window
                'n_threads': 8,
                'n_threads_batch': None,
                'n_batch': 512,
                'use_mmap': True,
                'use_mlock': False,
                'mul_mat_q': True,
                'numa': False,
                'flash_attn': True,
                'n_gpu_layers': 256,  # Offload layers to GPU
                'rope_freq_base': 500000,
                'rope_freq_scale': 1.0,
                'offload_kqv': True,
                'split_mode': 2
            }
            
            logger.info("Loading model with parameters", params=params)
            self.model = Llama(**params)
            
            # Optional cache setup
            cache_capacity = 0  # Set to > 0 if you want caching
            if cache_capacity > 0:
                self.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))
            
            self._initialized = True
            logger.info("LLaMA model initialized successfully")
            return self
            
        except Exception as e:
            logger.error("Failed to initialize LLaMA model", error=str(e))
            raise

    async def initialize_with_params(self, **params) -> 'LlamaModelWrapper':
        """Initialize the model asynchronously"""
        if self._initialized:
            return self
            
        logger.info("Initializing LLaMA model", model_path=str(self.model_path))
        
        try:
            Llama = llama_cpp_lib().Llama
            LlamaCache = llama_cpp_lib().LlamaCache
            # Model parameters similar to your working code
            params = {
                'model_path': str(self.model_path),
                'n_ctx': params.get('context_size', 4096 * 4),  # Large context window
                'n_threads': params.get('threads', 8),
                'n_threads_batch': params.get('batch_size', 512),
                'n_batch': 512,
                'use_mmap': True,
                'use_mlock': False,
                'mul_mat_q': True,
                'numa': False,
                'flash_attn': True,
                'n_gpu_layers': params.get('gpu_layers', 256),  # Offload layers to GPU
                'rope_freq_base': 500000,
                'rope_freq_scale': 1.0,
                'offload_kqv': True,
                'split_mode': 2
            }

            logger.info("Loading model with parameters", params=params)
            self.model = Llama(**params)
            
            # Optional cache setup
            cache_capacity = 0  # Set to > 0 if you want caching
            if cache_capacity > 0:
                self.model.set_cache(LlamaCache(capacity_bytes=cache_capacity))
            
            self._initialized = True
            logger.info("LLaMA model initialized successfully")
            return self
            
        except Exception as e:
            logger.error("Failed to initialize LLaMA model", error=str(e))
            raise
    
    def encode(self, string: str) -> List[int]:
        """Encode string to token IDs"""
        if type(string) is str:
            string = string.encode()
        return self.model.tokenize(string)

    def decode(self, ids: List[int]) -> str:
        """Decode token IDs to string"""
        return self.model.detokenize(ids).decode('utf-8')

    def sanitize_prompt(self, prompt: str) -> str:
        """Sanitize prompt for any special characters"""
        if "/" in prompt and "//" not in prompt:
            prompt = prompt.replace("/", "//")
        return prompt

    def load_grammar(self, string: str):
        """Load grammar for structured generation"""
        if string != self.grammar_string:
            self.grammar_string = string
            if string.strip() != '':
                self.grammar = llama_cpp_lib().LlamaGrammar.from_string(string)
            else:
                self.grammar = None

    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = 512,
        temperature: Optional[float] = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """Generate response from the model"""
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        try:
            # Prepare the prompt with chat template
            if system_prompt:
                formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            else:
                formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            
            # Sanitize and truncate prompt
            formatted_prompt = self.sanitize_prompt(formatted_prompt)
            prompt_tokens = self.encode(formatted_prompt)
            prompt_tokens = prompt_tokens[-10000:]  # Truncate if too long
            formatted_prompt = self.decode(prompt_tokens)
            
            # Generation parameters
            generation_params = {
                'max_tokens': max_tokens or 512,
                'temperature': temperature or 0.7,
                'top_p': kwargs.get('top_p', 0.9),
                'min_p': kwargs.get('min_p', 0.05),
                'typical_p': kwargs.get('typical_p', 1.0),
                'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
                'presence_penalty': kwargs.get('presence_penalty', 0.0),
                'repeat_penalty': kwargs.get('repeat_penalty', 1.1),
                'top_k': kwargs.get('top_k', 40),
                'stream': False,
                'seed': kwargs.get('seed', -1),
                'tfs_z': kwargs.get('tfs_z', 1.0),
                'mirostat_mode': kwargs.get('mirostat_mode', 0),
                'mirostat_tau': kwargs.get('mirostat_tau', 1.0),
                'mirostat_eta': kwargs.get('mirostat_eta', 0.1),
            }
            
            # Generate response
            completion = self.model.create_completion(formatted_prompt, **generation_params)
            
            response = completion['choices'][0]['text']
            return response.strip()
            
        except Exception as e:
            logger.error("Error generating response", error=str(e))
            raise

    async def generate_streaming(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = 512,
        temperature: Optional[float] = 0.7,
        system_prompt: Optional[str] = None,
        **kwargs
    ):
        """Generate streaming response from the model"""
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        try:
            # Prepare the prompt with chat template
            if system_prompt:
                formatted_prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|>
<|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            else:
                formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
{prompt}<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
"""
            
            # Sanitize and truncate prompt
            formatted_prompt = self.sanitize_prompt(formatted_prompt)
            prompt_tokens = self.encode(formatted_prompt)
            prompt_tokens = prompt_tokens[-10000:]  # Truncate if too long
            formatted_prompt = self.decode(prompt_tokens)
            
            # Generation parameters
            generation_params = {
                'max_tokens': max_tokens or 512,
                'temperature': temperature or 0.7,
                'top_p': kwargs.get('top_p', 0.9),
                'min_p': kwargs.get('min_p', 0.05),
                'typical_p': kwargs.get('typical_p', 1.0),
                'frequency_penalty': kwargs.get('frequency_penalty', 0.0),
                'presence_penalty': kwargs.get('presence_penalty', 0.0),
                'repeat_penalty': kwargs.get('repeat_penalty', 1.1),
                'top_k': kwargs.get('top_k', 40),
                'stream': True,
                'seed': kwargs.get('seed', -1),
                'tfs_z': kwargs.get('tfs_z', 1.0),
                'mirostat_mode': kwargs.get('mirostat_mode', 0),
                'mirostat_tau': kwargs.get('mirostat_tau', 1.0),
                'mirostat_eta': kwargs.get('mirostat_eta', 0.1),
            }
            
            # Generate streaming response
            completion_chunks = self.model.create_completion(formatted_prompt, **generation_params)
            
            for completion_chunk in completion_chunks:
                delta = completion_chunk["choices"][0]["text"]
                yield delta
                
        except Exception as e:
            logger.error("Error generating streaming response", error=str(e))
            raise

    async def batch_generate(
        self, 
        prompts: List[str], 
        batch_size: int = 4,
        **kwargs
    ) -> List[str]:
        """Generate responses for multiple prompts"""
        if not self._initialized:
            raise RuntimeError("Model not initialized. Call initialize() first.")
            
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_results = []
            
            for prompt in batch_prompts:
                response = await self.generate_response(prompt, **kwargs)
                batch_results.append(response)
                
            results.extend(batch_results)
            
        return results

    async def generate(self, 
                      prompt: str, 
                      chat_history: List[Dict[str, Any]] = None,
                      stream: bool = True) -> AsyncGenerator[str, None]:
        """
        Generate response using your existing LlamaCppModel
        Adapts the existing generate2 method to async generator
        """
        if not self._initialized:
            await self.initialize()
            
        chat_history = chat_history or []
        
        # Use your existing generate2 method which already supports streaming
        # async for chunk in self.model.generate1(prompt, chat_history):
        #     # Skip recommendation markers
        #     if "$$-+Recommendations+-$$" not in chunk:
        #         yield chunk
        print("################### Generating response with LlamaCppModel")
        
        # Format the prompt properly (similar to other methods)
        formatted_prompt = f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
    {prompt}<|eot_id|>
    <|start_header_id|>assistant<|end_header_id|>
    """
        
        completion_chunks = self.model.create_completion(
            formatted_prompt,
            max_tokens=8192,
            temperature=0.9,
            top_p=0.1,
            min_p=0.5,
            typical_p=1,
            frequency_penalty=0.6,
            presence_penalty=0,
            repeat_penalty=1.18,
            top_k=40,
            stream=True,
            seed=-1,
            tfs_z=1,
            mirostat_mode=0,
            mirostat_tau=1,
            mirostat_eta=0.1,
        )

        output = ""
        tool_call = ""
        can_yield = False
        print("################### Completion chunks: ", completion_chunks)
        for completion_chunk in completion_chunks:
            # print("---!!!!-----============== :::::: completion_chunk: ", completion_chunk)
            delta = completion_chunk["choices"][0]["text"]
            # print("output text: ", delta)
            # text = ""
            # if "content" in delta:
            #     text = delta["content"]
            
            output += delta
            if len(output) > 20:
                if "<function" not in output and "Please wait" not in output and can_yield == False:
                    yield f"{output}"
                    can_yield = True
                elif "</function>" in output:
                    tool_call = output
                    output = ""
                if can_yield == True:
                    yield f"{delta}"
                
    async def generate_with_context(self,
                                   prompt: str,
                                   chat_history: List[Dict[str, Any]] = None,
                                   use_rag: bool = True) -> AsyncGenerator[str, None]:
        """
        Generate with RAG context using your existing fetch_context
        """
        if not self._initialized:
            await self.initialize()
            
        print("################### Generating with context", prompt, use_rag)
        # Fetch context if RAG is enabled
        if use_rag:
            # context, recommendations = fetch_context({"text": prompt})
            """
            query = kwargs.get("query", "")
            entity_types = kwargs.get("entity_types", ["Disease", "Drug", "Symptom", "Gene", "Protein"])
            limit = kwargs.get("limit", 5)
            include_relationships = kwargs.get("include_relationships", True)
            use_hybrid_search = kwargs.get("use_hybrid_search", True)
            generate_recommendations = kwargs.get("generate_recommendations", True)

            response:
            {
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
            """
            response = await self.context_retriever.execute(query=prompt,
                                            limit=5,
                                            include_relationships=True,
                                            use_hybrid_search=True,
                                            generate_recommendations=True)
            context = response.get("context", "")
            recommendations = response.get("recommendations", [])
            
            if context:
                # Enhance prompt with context (like your existing code)
                enhanced_prompt = f"""
                Context:
                {context}
                
                User Question: {prompt}
                """
                prompt = enhanced_prompt
            print("################### Generating response", context, recommendations)
                
        # Generate response
        
        async for chunk in self.generate(prompt, chat_history):
            yield chunk

        if use_rag and recommendations and len(recommendations) > 0:
            # Yield recommendations if available
            yield "\n\n$$-+Recommendations+-$$\n" + str(recommendations)
            # for rec in recommendations:
            #     yield f"- {rec}\n"
            # yield "$$-+EndRecommendations+-$$\n"
            
    async def generate_single(self, 
                             prompt: str,
                             chat_history: List[Dict[str, Any]] = None) -> str:
        """
        Generate complete response (non-streaming)
        Uses your existing generate1 method
        """
        if not self._initialized:
            await self.initialize()
            
        chat_history = chat_history or []
        
        # Use your existing generate1 method
        response = await self.model.generate1(prompt, chat_history)
        return response
        
    def detect_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect medical entities using your existing function
        """
        return self.detect_entities_from_index(text)

    def detect_entities_from_index(self, text: str) -> List[Dict[str, Any]]:
        """
        Detect entities from a text using the existing index
        This is a placeholder for your actual entity detection logic
        """
        entities = []

        # This is a placeholder implementation for entity detection
        # You'll need to implement this based on your specific medical entity index
        # Here's a generic structure you can adapt:

        try:
            # Split text into tokens/words for analysis
            tokens = text.lower().split()
            
            # Example medical entity patterns (replace with your actual index)
            medical_patterns = {
                'symptoms': ['headache', 'fever', 'cough', 'pain', 'nausea'],
                'conditions': ['diabetes', 'hypertension', 'asthma', 'covid'],
                'medications': ['aspirin', 'ibuprofen', 'metformin', 'insulin'],
                'anatomy': ['heart', 'lung', 'kidney', 'brain', 'liver']
            }
            
            for token in tokens:
                for category, terms in medical_patterns.items():
                    if token in terms:
                        # Find the position in original text
                        start_pos = text.lower().find(token)
                        end_pos = start_pos + len(token)
                        
                        entities.append({
                            'text': token,
                            'label': category,
                            'start': start_pos,
                            'end': end_pos,
                            'confidence': 0.95  # Placeholder confidence score
                        })
            
            # Remove duplicates
            seen = set()
            unique_entities = []
            for entity in entities:
                key = (entity['text'], entity['start'], entity['end'])
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            
            entities = unique_entities
            
        except Exception as e:
            logger.error("Error in entity detection", error=str(e))
            entities = []

        return entities
        
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self._initialized:
            return {"status": "not_initialized"}
            
        return {
            "status": "initialized",
            "model_path": str(self.model_path),
            "model_name": self.model_name,
            "n_ctx": self.model.n_ctx() if self.model else None,
            "n_vocab": self.model.n_vocab() if self.model else None,
            "library": str(type(llama_cpp_lib()).__module__)
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up LLaMA model resources")
        
        if self.model:
            del self.model
            self.model = None
            
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self._initialized = False
        logger.info("LLaMA model cleanup complete")
    
    def __repr__(self) -> str:
        return f"LlamaModelWrapper(model_path={self.model_path}, model_name={self.model_name}, initialized={self._initialized})"