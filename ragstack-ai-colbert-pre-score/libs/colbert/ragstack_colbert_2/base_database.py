"""
This module defines abstract base classes for implementing storage mechanisms for text chunk
embeddings, specifically designed to work with ColBERT or similar embedding models.
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from .objects import Chunk, Vector


class BaseDatabase(ABC):
    """
    Abstract base class (ABC) for a storage system designed to hold vector representations of text chunks,
    typically generated by a ColBERT model or similar embedding model.

    This class defines the interface for storing and managing the embedded text chunks, supporting
    operations like adding new chunks to the store and deleting existing documents by their identifiers.
    """

    @abstractmethod
    def add_chunks(self, chunks: List[Chunk]) -> List[Tuple[str, int]]:
        """
        Stores a list of embedded text chunks in the vector store

        Parameters:
            chunks (List[Chunk]): A list of `Chunk` instances to be stored.

        Returns:
            a list of tuples: (doc_id, chunk_id)
        """

    @abstractmethod
    def delete_chunks(self, doc_ids: List[str]) -> bool:
        """
        Deletes chunks from the vector store based on their document id.

        Parameters:
            doc_ids (List[str]): A list of document identifiers specifying the chunks to be deleted.

        Returns:
            True if the delete was successful.
        """

    @abstractmethod
    async def search_relevant_chunks(self, vector: Vector, n: int) -> List[Chunk]:
        """
        Retrieves 'n' ANN results for an embedded token vector.

        Returns:
            A list of Chunks with only `doc_id`, `chunk_id`, and `embedding` set.
            Here `embedding` is only a single vector, the one which matched on the ANN search.
            Fewer than 'n' results may be returned.
        """

    @abstractmethod
    async def get_chunk_embedding(self, doc_id: str, chunk_id: int) -> Chunk:
        """
        Retrieve the embedding data for a chunk.

        Returns:
            A chunk with `doc_id`, `chunk_id`, and `embedding` set.
        """

    @abstractmethod
    async def get_chunk_data(
        self, doc_id: str, chunk_id: int, include_embedding: Optional[bool]
    ) -> Chunk:
        """
        Retrieve the text and metadata for a chunk.

        Returns:
            A chunk with `doc_id`, `chunk_id`, `text`, `metadata`, and optionally `embedding` set.
        """

    @abstractmethod
    def close(self) -> None:
        """
        Cleans up any open resources.
        """
