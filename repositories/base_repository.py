"""Base repository with common Firestore operations"""
from typing import TypeVar, Generic, Dict, Any, List, Optional
from abc import ABC, abstractmethod
from firebase_admin import firestore


T = TypeVar('T')


class BaseRepository(ABC, Generic[T]):
    """Abstract base repository for Firestore CRUD operations"""
    
    def __init__(self, collection_name: str):
        """
        Initialize repository with collection name
        
        Args:
            collection_name: Name of Firestore collection
        """
        self.db = firestore.client(database_id='vista-db')
        self.collection_name = collection_name
        self.collection = self.db.collection(collection_name)
    
    @abstractmethod
    def to_model(self, data: Dict[str, Any]) -> T:
        """Convert Firestore document to model"""
        pass
    
    @abstractmethod
    def to_dict(self, model: T) -> Dict[str, Any]:
        """Convert model to Firestore document"""
        pass
    
    def create(self, doc_id: str, model: T) -> T:
        """
        Create a new document
        
        Args:
            doc_id: Document ID
            model: Model instance to create
        
        Returns:
            Created model
        """
        doc_data = self.to_dict(model)
        self.collection.document(doc_id).set(doc_data)
        return model
    
    def get(self, doc_id: str) -> Optional[T]:
        """
        Get a document by ID
        
        Args:
            doc_id: Document ID
        
        Returns:
            Model instance or None if not found
        """
        doc = self.collection.document(doc_id).get()
        if doc.exists:
            return self.to_model(doc.to_dict())
        return None
    
    def update(self, doc_id: str, model: T) -> T:
        """
        Update an existing document
        
        Args:
            doc_id: Document ID
            model: Updated model instance
        
        Returns:
            Updated model
        """
        doc_data = self.to_dict(model)
        self.collection.document(doc_id).set(doc_data, merge=True)
        return model
    
    def delete(self, doc_id: str) -> bool:
        """
        Delete a document by ID
        
        Args:
            doc_id: Document ID
        
        Returns:
            True if deleted, False if not found
        """
        self.collection.document(doc_id).delete()
        return True
    
    def list_all(self) -> List[tuple[str, T]]:
        """
        Get all documents in collection
        
        Returns:
            List of tuples (doc_id, model)
        """
        docs = self.collection.stream()
        return [(doc.id, self.to_model(doc.to_dict())) for doc in docs]
    
    def query(self, field: str, operator: str, value: Any) -> List[tuple[str, T]]:
        """
        Query documents by field
        
        Args:
            field: Field name
            operator: Comparison operator (==, <, >, <=, >=, !=, array-contains)
            value: Value to compare
        
        Returns:
            List of tuples (doc_id, model)
        """
        query = self.collection.where(field, operator, value)
        docs = query.stream()
        return [(doc.id, self.to_model(doc.to_dict())) for doc in docs]
    
    def batch_write(self, operations: List[tuple[str, T, str]]) -> None:
        """
        Perform batch write operations
        
        Args:
            operations: List of tuples (doc_id, model, operation_type)
                       operation_type can be 'set', 'update', 'delete'
        """
        batch = self.db.batch()
        
        for doc_id, model, op_type in operations:
            doc_ref = self.collection.document(doc_id)
            
            if op_type == 'set':
                batch.set(doc_ref, self.to_dict(model))
            elif op_type == 'update':
                batch.set(doc_ref, self.to_dict(model), merge=True)
            elif op_type == 'delete':
                batch.delete(doc_ref)
        
        batch.commit()
