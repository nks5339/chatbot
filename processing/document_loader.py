"""
Document loader for PDF files
"""
import pypdf
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Document:
    """Document dataclass"""
    content: str
    metadata: Dict[str, Any]
    doc_id: str

class DocumentLoader:
    """Load and process PDF documents"""
    
    def __init__(self):
        self.documents_dir = settings.DOCUMENTS_DIR
        logger.info(f"Document loader initialized. Documents directory: {self.documents_dir}")
    
    def load_pdf(self, file_path: Path) -> Optional[Document]:
        """Load a single PDF file"""
        try:
            logger.info(f"Loading PDF: {file_path.name}")
            
            with open(file_path, 'rb') as file:
                pdf_reader = pypdf.PdfReader(file)
                
                # Extract metadata
                metadata = {
                    "filename": file_path.name,
                    "filepath": str(file_path),
                    "num_pages": len(pdf_reader.pages),
                    "title": pdf_reader.metadata.get('/Title', file_path.stem) if pdf_reader.metadata else file_path.stem,
                }
                
                # Extract text from all pages
                full_text = []
                for page_num, page in enumerate(pdf_reader.pages, 1):
                    try:
                        page_text = page.extract_text()
                        if page_text.strip():
                            full_text.append(page_text)
                    except Exception as e:
                        logger.warning(f"Error extracting text from page {page_num} of {file_path.name}: {e}")
                
                content = "\n\n".join(full_text)
                
                if not content.strip():
                    logger.warning(f"No text content extracted from {file_path.name}")
                    return None
                
                doc_id = file_path.stem
                
                logger.info(f"Successfully loaded {file_path.name}: {len(content)} characters, {metadata['num_pages']} pages")
                
                return Document(
                    content=content,
                    metadata=metadata,
                    doc_id=doc_id
                )
        
        except Exception as e:
            logger.error(f"Error loading PDF {file_path.name}: {e}")
            return None
    
    def load_all_documents(self) -> List[Document]:
        """Load all PDF documents from the documents directory"""
        documents = []
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        
        if not pdf_files:
            logger.warning(f"No PDF files found in {self.documents_dir}")
            return documents
        
        logger.info(f"Found {len(pdf_files)} PDF files to process")
        
        for pdf_file in pdf_files:
            doc = self.load_pdf(pdf_file)
            if doc:
                documents.append(doc)
        
        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents
    
    def get_document_info(self) -> List[Dict[str, Any]]:
        """Get information about all documents"""
        pdf_files = list(self.documents_dir.glob("*.pdf"))
        info = []
        
        for pdf_file in pdf_files:
            try:
                with open(pdf_file, 'rb') as file:
                    pdf_reader = pypdf.PdfReader(file)
                    info.append({
                        "filename": pdf_file.name,
                        "num_pages": len(pdf_reader.pages),
                        "size_bytes": pdf_file.stat().st_size,
                    })
            except Exception as e:
                logger.error(f"Error getting info for {pdf_file.name}: {e}")
        
        return info

# Global document loader instance
document_loader = DocumentLoader()