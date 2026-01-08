"""
Graph-based RAG for enhanced context understanding
"""
import json
import networkx as nx
from pathlib import Path
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from config import settings
from utils.logger import get_logger
from processing.chunking import Chunk

logger = get_logger(__name__)

class GraphRAG:
    """Graph-based RAG for document relationships and enhanced retrieval"""
    
    def __init__(self):
        self.graph = nx.DiGraph()
        self.graph_file = settings.GRAPH_PATH / "document_graph.json"
        self._load_graph()
        logger.info("Graph RAG initialized")
    
    def _load_graph(self):
        """Load existing graph from disk"""
        if self.graph_file.exists():
            try:
                with open(self.graph_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct graph
                self.graph = nx.node_link_graph(data)
                logger.info(f"Loaded graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
            except Exception as e:
                logger.error(f"Error loading graph: {e}")
                self.graph = nx.DiGraph()
        else:
            logger.info("No existing graph found, starting fresh")
    
    def _save_graph(self):
        """Save graph to disk"""
        try:
            data = nx.node_link_data(self.graph)
            with open(self.graph_file, 'w') as f:
                json.dump(data, f, indent=2)
            logger.info("Graph saved successfully")
        except Exception as e:
            logger.error(f"Error saving graph: {e}")
    
    def add_document(self, doc_id: str, chunks: List[Chunk], metadata: Dict[str, Any]):
        """Add document and its chunks to the graph"""
        try:
            # Add document node
            self.graph.add_node(
                doc_id,
                node_type="document",
                **metadata
            )
            
            # Add chunk nodes and edges
            for i, chunk in enumerate(chunks):
                chunk_id = chunk.chunk_id
                
                # Add chunk node
                self.graph.add_node(
                    chunk_id,
                    node_type="chunk",
                    text=chunk.text,
                    chunk_num=i,
                    **chunk.metadata
                )
                
                # Add edge from document to chunk
                self.graph.add_edge(doc_id, chunk_id, relation="contains")
                
                # Add sequential edges between chunks
                if i > 0:
                    prev_chunk_id = chunks[i-1].chunk_id
                    self.graph.add_edge(prev_chunk_id, chunk_id, relation="precedes")
                
                # Extract and link entities (sections, chapters, etc.)
                self._extract_and_link_entities(chunk)
            
            self._save_graph()
            logger.info(f"Added document {doc_id} to graph with {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error adding document to graph: {e}")
    
    def _extract_and_link_entities(self, chunk: Chunk):
        """Extract and link entities from chunk text"""
        import re
        
        # Extract section references
        section_pattern = r'section\s+(\d+)'
        sections = re.findall(section_pattern, chunk.text, re.IGNORECASE)
        
        for section in sections:
            entity_id = f"section_{section}"
            if not self.graph.has_node(entity_id):
                self.graph.add_node(entity_id, node_type="section", section_num=section)
            self.graph.add_edge(chunk.chunk_id, entity_id, relation="references")
        
        # Extract chapter references
        chapter_pattern = r'chapter\s+([IVX\d]+)'
        chapters = re.findall(chapter_pattern, chunk.text, re.IGNORECASE)
        
        for chapter in chapters:
            entity_id = f"chapter_{chapter}"
            if not self.graph.has_node(entity_id):
                self.graph.add_node(entity_id, node_type="chapter", chapter_num=chapter)
            self.graph.add_edge(chunk.chunk_id, entity_id, relation="references")
    
    def get_related_chunks(
        self,
        chunk_ids: List[str],
        max_depth: int = 2,
        max_chunks: int = 10
    ) -> List[str]:
        """Get related chunks using graph traversal"""
        try:
            related = set()
            
            for chunk_id in chunk_ids:
                if not self.graph.has_node(chunk_id):
                    continue
                
                # Get neighbors within max_depth
                for node in nx.single_source_shortest_path_length(
                    self.graph, chunk_id, cutoff=max_depth
                ):
                    if self.graph.nodes[node].get('node_type') == 'chunk':
                        related.add(node)
                
                if len(related) >= max_chunks:
                    break
            
            return list(related)[:max_chunks]
            
        except Exception as e:
            logger.error(f"Error getting related chunks: {e}")
            return []
    
    def get_document_structure(self, doc_id: str) -> Dict[str, Any]:
        """Get hierarchical structure of a document"""
        try:
            if not self.graph.has_node(doc_id):
                return {}
            
            structure = {
                "doc_id": doc_id,
                "metadata": dict(self.graph.nodes[doc_id]),
                "chunks": [],
                "entities": defaultdict(list)
            }
            
            # Get all chunks
            for successor in self.graph.successors(doc_id):
                if self.graph.nodes[successor].get('node_type') == 'chunk':
                    chunk_info = {
                        "chunk_id": successor,
                        "chunk_num": self.graph.nodes[successor].get('chunk_num'),
                        "references": []
                    }
                    
                    # Get entity references
                    for entity in self.graph.successors(successor):
                        entity_type = self.graph.nodes[entity].get('node_type')
                        if entity_type in ['section', 'chapter']:
                            chunk_info["references"].append({
                                "type": entity_type,
                                "id": entity
                            })
                            structure["entities"][entity_type].append(entity)
                    
                    structure["chunks"].append(chunk_info)
            
            return structure
            
        except Exception as e:
            logger.error(f"Error getting document structure: {e}")
            return {}
    
    def find_cross_references(self, chunk_id: str) -> List[str]:
        """Find chunks that reference similar entities"""
        try:
            if not self.graph.has_node(chunk_id):
                return []
            
            # Get entities referenced by this chunk
            entities = [
                n for n in self.graph.successors(chunk_id)
                if self.graph.nodes[n].get('node_type') in ['section', 'chapter']
            ]
            
            # Find other chunks referencing the same entities
            cross_refs = set()
            for entity in entities:
                for predecessor in self.graph.predecessors(entity):
                    if (self.graph.nodes[predecessor].get('node_type') == 'chunk' 
                        and predecessor != chunk_id):
                        cross_refs.add(predecessor)
            
            return list(cross_refs)
            
        except Exception as e:
            logger.error(f"Error finding cross-references: {e}")
            return []
    
    def get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the graph"""
        try:
            node_types = defaultdict(int)
            for node in self.graph.nodes():
                node_type = self.graph.nodes[node].get('node_type', 'unknown')
                node_types[node_type] += 1
            
            return {
                "total_nodes": self.graph.number_of_nodes(),
                "total_edges": self.graph.number_of_edges(),
                "node_types": dict(node_types),
                "is_connected": nx.is_weakly_connected(self.graph)
            }
        except Exception as e:
            logger.error(f"Error getting graph stats: {e}")
            return {}
    
    def delete_document(self, doc_id: str):
        """Remove document and its chunks from the graph"""
        try:
            if self.graph.has_node(doc_id):
                # Get all chunk nodes
                chunks_to_remove = [
                    n for n in self.graph.successors(doc_id)
                    if self.graph.nodes[n].get('node_type') == 'chunk'
                ]
                
                # Remove chunks and their edges
                for chunk in chunks_to_remove:
                    self.graph.remove_node(chunk)
                
                # Remove document node
                self.graph.remove_node(doc_id)
                
                self._save_graph()
                logger.info(f"Removed document {doc_id} from graph")
        except Exception as e:
            logger.error(f"Error deleting document from graph: {e}")
    
    def clear_all(self):
        """Clear entire graph"""
        self.graph = nx.DiGraph()
        self._save_graph()
        logger.info("Graph cleared")

# Global graph RAG instance
graph_rag = GraphRAG()