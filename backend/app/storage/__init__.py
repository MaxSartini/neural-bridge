"""
Neural Bridge Storage Layer

Local graph storage replacing Zep Cloud:
- Neo4j CE for graph persistence
- Ollama for embeddings (nomic-embed-text)
- LLM-based NER/RE extraction
- Hybrid search (vector + keyword)
"""

from importlib import import_module

__all__ = [
    "GraphStorage",
    "Neo4jStorage",
    "EmbeddingService",
    "EmbeddingError",
    "NERExtractor",
    "SearchService",
]

_MODULES = {
    "GraphStorage": "graph_storage",
    "Neo4jStorage": "neo4j_storage",
    "EmbeddingService": "embedding_service",
    "EmbeddingError": "embedding_service",
    "NERExtractor": "ner_extractor",
    "SearchService": "search_service",
}


def __getattr__(name):
    if name not in _MODULES:
        raise AttributeError(name)
    value = getattr(import_module(f"{__name__}.{_MODULES[name]}"), name)
    globals()[name] = value
    return value
