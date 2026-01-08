"""
Advanced text chunking with semantic awareness
"""
import re
from typing import List, Dict, Any
from dataclasses import dataclass
from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class Chunk:
    """Text chunk dataclass"""
    text: str
    metadata: Dict[str, Any]
    chunk_id: str

class SemanticChunker:
    """Advanced semantic-aware text chunking"""
    
    def __init__(
        self,
        chunk_size: int = settings.CHUNK_SIZE,
        chunk_overlap: int = settings.CHUNK_OVERLAP
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        logger.info(f"Semantic chunker initialized: size={chunk_size}, overlap={chunk_overlap}")
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences using regex"""
        # Pattern to split on sentence boundaries
        sentence_pattern = r'(?<=[.!?।])\s+(?=[A-Z]|[०-९]|\d)'
        sentences = re.split(sentence_pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify document sections (chapters, articles, etc.)"""
        sections = []
        
        # Patterns for section headers
        patterns = [
            (r'^(CHAPTER|अध्याय)\s+([IVX\d]+)', 'chapter'),
            (r'^(\d+)\.\s+([A-Z][^.]+)', 'section'),
            (r'^\(([a-z])\)', 'subsection'),
        ]
        
        lines = text.split('\n')
        current_section = None
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            for pattern, section_type in patterns:
                match = re.match(pattern, line, re.IGNORECASE)
                if match:
                    sections.append({
                        'line_num': line_num,
                        'type': section_type,
                        'title': line,
                        'start': line_num
                    })
                    break
        
        return sections
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        doc_id: str
    ) -> List[Chunk]:
        """Chunk text with semantic awareness"""
        chunks = []
        
        # Identify sections for better chunking
        sections = self._identify_sections(text)
        
        # Split into sentences
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            logger.warning(f"No sentences found in document {doc_id}")
            return chunks
        
        current_chunk = []
        current_length = 0
        chunk_num = 0
        
        for i, sentence in enumerate(sentences):
            sentence_length = len(sentence)
            
            # Check if adding this sentence would exceed chunk size
            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Create chunk
                chunk_text = ' '.join(current_chunk)
                chunk_id = f"{doc_id}_chunk_{chunk_num}"
                
                chunk_metadata = {
                    **metadata,
                    "chunk_id": chunk_id,
                    "chunk_num": chunk_num,
                    "start_sentence": i - len(current_chunk),
                    "end_sentence": i,
                    "chunk_length": current_length
                }
                
                chunks.append(Chunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                    chunk_id=chunk_id
                ))
                
                # Calculate overlap
                overlap_sentences = []
                overlap_length = 0
                
                for sent in reversed(current_chunk):
                    if overlap_length + len(sent) <= self.chunk_overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_length += len(sent)
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_length = overlap_length
                chunk_num += 1
            
            current_chunk.append(sentence)
            current_length += sentence_length
        
        # Add final chunk
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk_id = f"{doc_id}_chunk_{chunk_num}"
            
            chunk_metadata = {
                **metadata,
                "chunk_id": chunk_id,
                "chunk_num": chunk_num,
                "start_sentence": len(sentences) - len(current_chunk),
                "end_sentence": len(sentences),
                "chunk_length": current_length
            }
            
            chunks.append(Chunk(
                text=chunk_text,
                metadata=chunk_metadata,
                chunk_id=chunk_id
            ))
        
        logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        return chunks

# Global chunker instance
chunker = SemanticChunker()