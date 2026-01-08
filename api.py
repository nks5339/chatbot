"""
FastAPI application for Sarthi AI
"""
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
from pathlib import Path
from config import settings
from utils.logger import get_logger
from main import pipeline

logger = get_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Sarthi AI - Rajasthan Procurement Assistant",
    description="AI-powered chatbot for Rajasthan Transparency in Public Procurement Act, 2012",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class QueryRequest(BaseModel):
    query: str
    use_graph_expansion: bool = True
    stream: bool = False

class QueryResponse(BaseModel):
    response: str
    sources: List[Dict[str, Any]]
    processing_time: float
    context_used: int

class DocumentInfo(BaseModel):
    doc_id: str
    filename: str
    pages: Optional[int]
    chunks: Optional[int]
    processed: bool

class SystemStatus(BaseModel):
    status: str
    vector_store: Dict[str, Any]
    graph: Dict[str, Any]
    memory: Dict[str, Any]
    documents_available: int

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    logger.info("Starting Sarthi AI API server...")
    
    # Run initialization in background
    def init_system():
        try:
            result = pipeline.initialize_system()
            logger.info(f"System initialization result: {result}")
        except Exception as e:
            logger.error(f"Error during startup initialization: {e}")
    
    # Run in thread pool to avoid blocking
    import threading
    init_thread = threading.Thread(target=init_system)
    init_thread.start()
    
    logger.info("API server started successfully")

@app.get("/")
async def read_root():
    """Serve the frontend"""
    static_dir = Path(__file__).parent / "static"
    index_file = static_dir / "index.html"
    
    if index_file.exists():
        return FileResponse(index_file)
    else:
        return JSONResponse({
            "message": "Sarthi AI API",
            "version": "1.0.0",
            "docs": "/docs"
        })

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Sarthi AI",
        "version": "1.0.0"
    }

@app.get("/api/status", response_model=SystemStatus)
async def get_status():
    """Get comprehensive system status"""
    try:
        status_data = pipeline.get_system_status()
        return SystemStatus(
            status="operational",
            **status_data
        )
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents", response_model=List[DocumentInfo])
async def get_documents():
    """Get information about all documents"""
    try:
        from processing.document_loader import document_loader
        
        docs_info = document_loader.get_document_info()
        processed_docs = pipeline.vector_store.get_processed_documents()
        
        result = []
        for doc_info in docs_info:
            doc_id = doc_info["doc_id"]
            result.append(DocumentInfo(
                doc_id=doc_id,
                filename=doc_info["filename"],
                pages=doc_info.get("pages"),
                chunks=doc_info.get("chunks"),
                processed=doc_id in processed_docs
            ))
        
        return result
        
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(request: QueryRequest):
    """Query endpoint for non-streaming responses"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Received query: {request.query[:100]}...")
        
        result = pipeline.query(
            user_query=request.query,
            use_graph_expansion=request.use_graph_expansion,
            stream=False
        )
        
        return QueryResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")

@app.post("/api/query/stream")
async def query_stream_endpoint(request: QueryRequest):
    """Query endpoint with streaming responses"""
    try:
        if not request.query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Received streaming query: {request.query[:100]}...")
        
        async def event_generator():
            """Generate SSE events"""
            try:
                # Get the streaming generator from pipeline
                for chunk in pipeline.query(
                    user_query=request.query,
                    use_graph_expansion=request.use_graph_expansion,
                    stream=True
                ):
                    import json
                    # Format as SSE
                    yield f"data: {json.dumps(chunk)}\n\n"
                    await asyncio.sleep(0.01)  # Small delay for smooth streaming
                
            except Exception as e:
                logger.error(f"Error in streaming: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in streaming endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/conversations")
async def get_conversations():
    """Get conversation history"""
    try:
        conversations = pipeline.memory.get_all_conversations()
        return {
            "total": len(conversations),
            "conversations": conversations[-50:]  # Last 50 conversations
        }
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/conversations")
async def clear_conversations():
    """Clear conversation history"""
    try:
        pipeline.memory.clear_history()
        return {
            "status": "success",
            "message": "Conversation history cleared"
        }
    except Exception as e:
        logger.error(f"Error clearing conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/initialize")
async def reinitialize_system(background_tasks: BackgroundTasks):
    """Reinitialize the system and process new documents"""
    try:
        # Run initialization in background
        def init_task():
            try:
                result = pipeline.initialize_system()
                logger.info(f"Reinitialization complete: {result}")
            except Exception as e:
                logger.error(f"Error during reinitialization: {e}")
        
        background_tasks.add_task(init_task)
        
        return {
            "status": "initiated",
            "message": "System reinitialization started in background"
        }
        
    except Exception as e:
        logger.error(f"Error initiating reinitialization: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/graph/stats")
async def get_graph_stats():
    """Get graph statistics"""
    try:
        stats = pipeline.graph_rag.get_graph_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting graph stats: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/document/{doc_id}/structure")
async def get_document_structure(doc_id: str):
    """Get hierarchical structure of a document"""
    try:
        structure = pipeline.graph_rag.get_document_structure(doc_id)
        if not structure:
            raise HTTPException(status_code=404, detail="Document not found")
        return structure
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document structure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/system/clear")
async def clear_all_data():
    """Clear all system data (use with caution!)"""
    try:
        pipeline.clear_all_data()
        return {
            "status": "success",
            "message": "All system data cleared. Please reinitialize."
        }
    except Exception as e:
        logger.error(f"Error clearing system data: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/search/conversations")
async def search_conversations(q: str, limit: int = 5):
    """Search through conversation history"""
    try:
        if not q.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        results = pipeline.memory.search_conversations(q, limit=limit)
        return {
            "query": q,
            "results": results,
            "count": len(results)
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching conversations: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (for CSS, JS, images if needed)
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,
        log_level="info"
    )