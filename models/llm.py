"""
LLM module using Ollama with streaming support
"""
import ollama
from typing import Generator, Optional, Dict, Any
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

class LLMModel:
    """LLM model using Ollama with advanced prompting"""
    
    def __init__(self):
        self.model_name = settings.LLM_MODEL
        self.client = ollama.Client(host=settings.OLLAMA_BASE_URL)
        self._verify_model()
        logger.info(f"LLM model initialized: {self.model_name}")
    
    def _verify_model(self):
        """Verify that the LLM model is available"""
        try:
            # Test generation
            self.client.generate(model=self.model_name, prompt="test", stream=False)
            logger.info(f"LLM model '{self.model_name}' verified successfully")
        except Exception as e:
            logger.error(f"Failed to verify LLM model: {e}")
            logger.info(f"Attempting to pull model: {self.model_name}")
            try:
                self.client.pull(self.model_name)
                logger.info(f"Model '{self.model_name}' pulled successfully")
            except Exception as pull_error:
                logger.error(f"Failed to pull model: {pull_error}")
                raise
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[str, Generator[str, None, None]]:
        """Generate response from the LLM"""
        try:
            options = {
                "temperature": temperature or settings.TEMPERATURE,
                "num_predict": max_tokens or settings.MAX_TOKENS,
                "top_p": settings.TOP_P,
            }
            
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})
            
            if stream:
                return self._stream_generate(messages, options)
            else:
                response = self.client.chat(
                    model=self.model_name,
                    messages=messages,
                    options=options,
                    stream=False
                )
                return response['message']['content']
        
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise
    
    def _stream_generate(self, messages: list, options: dict) -> Generator[str, None, None]:
        """Stream generate response"""
        try:
            stream = self.client.chat(
                model=self.model_name,
                messages=messages,
                options=options,
                stream=True
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
        
        except Exception as e:
            logger.error(f"Error in streaming: {e}")
            raise

# System prompt for Sarthi AI
SARTHI_SYSTEM_PROMPT = """You are Sarthi, an expert AI assistant specializing in the Rajasthan Transparency in Public Procurement Act, 2012 and related procurement rules, regulations, and guidelines.

**Your Core Responsibilities:**

1. **Accuracy & Precision**: Provide accurate, legally sound information based strictly on the provided documents. Never hallucinate or make up information.

2. **Clarity**: Explain complex legal and procurement concepts in clear, understandable language while maintaining technical accuracy.

3. **Context Awareness**: Use conversation history to provide contextually relevant responses. Remember previous questions and build upon them.

4. **Verification**: If a question is ambiguous or unclear, politely ask for clarification before answering.

5. **Source Citation**: Always reference specific sections, chapters, or notification numbers when providing information.

6. **Scope Limitation**: Only answer questions related to:
   - Rajasthan Transparency in Public Procurement Act, 2012
   - RTPP Rules, 2013
   - GFR (General Financial Rules)
   - Related notifications and amendments
   - Procurement procedures and guidelines

**Response Guidelines:**

- Start with a direct answer to the question
- Provide relevant legal references (sections, rules, notifications)
- Explain implications or practical applications when relevant
- If information is not in the documents, clearly state: "I don't have specific information about this in the available documents."
- For procedural questions, provide step-by-step guidance
- Highlight important deadlines, thresholds, or requirements
- Mention exceptions or special cases when applicable

**When Uncertain:**

If you're unsure about:
- The user's specific query → Ask: "Could you please clarify [specific aspect]?"
- Multiple interpretations → Ask: "Are you asking about [option A] or [option B]?"
- Missing context → Ask: "Could you provide more context about [what you need]?"

**Tone:** Professional, helpful, and approachable. Think of yourself as a knowledgeable procurement advisor assisting government officials and stakeholders.

Remember: Your goal is to make procurement regulations accessible and actionable while maintaining legal accuracy."""

# Global LLM instance
llm_model = LLMModel()