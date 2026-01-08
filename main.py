"""
Main processing pipeline for Sarthi AI
"""
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from config import settings
from utils.logger import get_logger
from processing.document_loader import document_loader, Document
from processing.chunking import chunker, Chunk
from storage.vector_store import vector_store
from models.graph_rag import graph_rag
from storage.conversation_memory import conversation_memory
from models.llm import llm_model, SARTHI_SYSTEM_PROMPT

logger = get_logger(__name__)

class SarthiPipeline:
    """Main processing pipeline for Sarthi AI"""
    
    def __init__(self):
        self.vector_store = vector_store
        self.graph_rag = graph_rag
        self.memory = conversation_memory
        self.llm = llm_model
        logger.info("Sarthi AI Pipeline initialized")
    
    def initialize_system(self) -> Dict[str, Any]:
        """Initialize the system and process documents if needed"""
        logger.info("=" * 80)
        logger.info("Initializing Sarthi AI System")
        logger.info("=" * 80)
        
        start_time = time.time()
        result = {
            "status": "success",
            "new_documents_processed": 0,
            "total_documents": 0,
            "total_chunks": 0,
            "processing_time": 0
        }
        
        try:
            # Get all documents
            documents = document_loader.load_all_documents()
            result["total_documents"] = len(documents)
            
            if not documents:
                logger.warning("No documents found in documents directory")
                result["status"] = "warning"
                result["message"] = "No documents found"
                return result
            
            # Check which documents need processing
            new_documents = []
            for doc in documents:
                if not self.vector_store.is_document_processed(doc.doc_id):
                    new_documents.append(doc)
                    logger.info(f"New document detected: {doc.doc_id}")
            
            if new_documents:
                logger.info(f"Processing {len(new_documents)} new documents...")
                
                for doc in new_documents:
                    success = self._process_document(doc)
                    if success:
                        result["new_documents_processed"] += 1
            else:
                logger.info("All documents already processed. Loading from cache...")
            
            # Get collection info
            collection_info = self.vector_store.get_collection_info()
            result["total_chunks"] = collection_info.get("points_count", 0)
            
            result["processing_time"] = time.time() - start_time
            
            logger.info("=" * 80)
            logger.info(f"System Initialization Complete!")
            logger.info(f"Total Documents: {result['total_documents']}")
            logger.info(f"New Documents Processed: {result['new_documents_processed']}")
            logger.info(f"Total Chunks: {result['total_chunks']}")
            logger.info(f"Time Taken: {result['processing_time']:.2f}s")
            logger.info("=" * 80)
            
            return result
            
        except Exception as e:
            logger.error(f"Error during system initialization: {e}")
            result["status"] = "error"
            result["error"] = str(e)
            return result
    
    def _process_document(self, doc: Document) -> bool:
        """Process a single document"""
        try:
            logger.info(f"Processing document: {doc.doc_id}")
            start_time = time.time()
            
            # Chunk the document
            chunks = chunker.chunk_text(
                text=doc.content,
                metadata=doc.metadata,
                doc_id=doc.doc_id
            )
            
            if not chunks:
                logger.warning(f"No chunks created for document {doc.doc_id}")
                return False
            
            # Add to vector store
            vector_success = self.vector_store.add_chunks(chunks, doc.doc_id)
            
            # Add to graph
            if vector_success:
                self.graph_rag.add_document(doc.doc_id, chunks, doc.metadata)
            
            elapsed = time.time() - start_time
            logger.info(f"Document {doc.doc_id} processed successfully in {elapsed:.2f}s")
            
            return vector_success
            
        except Exception as e:
            logger.error(f"Error processing document {doc.doc_id}: {e}")
            return False
    
    def query(
        self,
        user_query: str,
        use_graph_expansion: bool = True,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Process a user query and generate response"""
        logger.info(f"Processing query: {user_query[:100]}...")
        start_time = time.time()
        
        try:
            # Search vector store
            search_results = self.vector_store.search(
                query=user_query,
                top_k=settings.RETRIEVAL_TOP_K
            )
            
            if not search_results:
                return self._generate_no_context_response(user_query)
            
            # Expand context using graph if enabled
            if use_graph_expansion:
                chunk_ids = [r["chunk_id"] for r in search_results[:3]]
                related_chunks = self.graph_rag.get_related_chunks(chunk_ids, max_chunks=3)
                
                if related_chunks:
                    logger.info(f"Expanded context with {len(related_chunks)} related chunks")
                    # Add related chunks to results (simplified - in production, fetch from store)
            
            # Get conversation context
            conversation_context = self.memory.get_recent_context(n=3)
            
            # Build context for LLM
            context_parts = []
            for i, result in enumerate(search_results[:settings.RERANK_TOP_K], 1):
                context_parts.append(
                    f"[Source {i}] (from {result['metadata'].get('filename', 'unknown')}, "
                    f"page {result['metadata'].get('page_number', 'N/A')})\n"
                    f"{result['text']}\n"
                )
            
            context = "\n".join(context_parts)
            
            # Build prompt
            prompt = self._build_prompt(user_query, context, conversation_context)
            
            # Generate response
            if stream:
                return self._generate_streaming_response(
                    prompt=prompt,
                    user_query=user_query,
                    search_results=search_results,
                    start_time=start_time
                )
            else:
                response = self.llm.generate(
                    prompt=prompt,
                    system_prompt=SARTHI_SYSTEM_PROMPT,
                    stream=False
                )
                
                # Add to memory
                self.memory.add_interaction(
                    user_message=user_query,
                    assistant_response=response,
                    context_chunks=search_results
                )
                
                elapsed = time.time() - start_time
                
                return {
                    "response": response,
                    "sources": [
                        {
                            "filename": r["metadata"].get("filename", "unknown"),
                            "page": r["metadata"].get("page_number"),
                            "chunk_id": r["chunk_id"],
                            "score": round(r["score"], 3)
                        }
                        for r in search_results[:5]
                    ],
                    "processing_time": round(elapsed, 2),
                    "context_used": len(search_results)
                }
        
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "response": "I apologize, but I encountered an error while processing your question. Please try again or rephrase your question.",
                "error": str(e),
                "sources": []
            }
    
    def _build_prompt(
        self,
        query: str,
        context: str,
        conversation_context: str
    ) -> str:
        """Build the prompt for LLM"""
        prompt_parts = []
        
        if conversation_context:
            prompt_parts.append("=== Previous Conversation ===")
            prompt_parts.append(conversation_context)
            prompt_parts.append("")
        
        prompt_parts.append("=== Relevant Information from Documents ===")
        prompt_parts.append(context)
        prompt_parts.append("")
        prompt_parts.append("=== User Question ===")
        prompt_parts.append(query)
        prompt_parts.append("")
        prompt_parts.append("=== Your Response ===")
        prompt_parts.append("Based on the provided information and conversation history, please answer the question accurately and concisely:")
        
        return "\n".join(prompt_parts)
    
    def _generate_no_context_response(self, query: str) -> Dict[str, Any]:
        """Generate response when no relevant context is found"""
        response = (
            "I apologize, but I couldn't find specific information about your question in the "
            "Rajasthan Procurement documents. This could mean:\n\n"
            "1. The information isn't covered in the available documents\n"
            "2. Your question might need to be rephrased for better results\n"
            "3. The topic might be covered under different terminology\n\n"
            "Could you please:\n"
            "- Rephrase your question with different keywords?\n"
            "- Provide more context about what you're looking for?\n"
            "- Specify which aspect of procurement you're asking about?"
        )
        
        return {
            "response": response,
            "sources": [],
            "processing_time": 0,
            "context_used": 0
        }
    
    def _generate_streaming_response(
        self,
        prompt: str,
        user_query: str,
        search_results: List[Dict],
        start_time: float
    ):
        """Generate streaming response (for WebSocket support)"""
        # This is a simplified version - full implementation would use async generators
        response_parts = []
        
        for chunk in self.llm.generate(
            prompt=prompt,
            system_prompt=SARTHI_SYSTEM_PROMPT,
            stream=True
        ):
            response_parts.append(chunk)
            yield {
                "type": "chunk",
                "content": chunk
            }
        
        full_response = "".join(response_parts)
        
        # Add to memory
        self.memory.add_interaction(
            user_message=user_query,
            assistant_response=full_response,
            context_chunks=search_results
        )
        
        # Send completion
        yield {
            "type": "complete",
            "sources": [
                {
                    "filename": r["metadata"].get("filename", "unknown"),
                    "page": r["metadata"].get("page_number"),
                    "score": round(r["score"], 3)
                }
                for r in search_results[:5]
            ],
            "processing_time": round(time.time() - start_time, 2)
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "vector_store": {
                **self.vector_store.get_collection_info(),
                "processed_documents": len(self.vector_store.get_processed_documents())
            },
            "graph": self.graph_rag.get_graph_stats(),
            "memory": self.memory.get_conversation_summary(),
            "documents_available": len(document_loader.get_document_info())
        }
    
    def clear_all_data(self):
        """Clear all processed data (use with caution!)"""
        logger.warning("Clearing all system data...")
        self.vector_store.clear_all()
        self.graph_rag.clear_all()
        self.memory.clear_history()
        logger.info("All data cleared")

# Global pipeline instance
pipeline = SarthiPipeline()