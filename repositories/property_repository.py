"""Repository for Property model"""
from typing import Dict, Any, List, Optional
from models.property import Property
from repositories.base_repository import BaseRepository
from repositories.virtual_staging_repository import VirtualStagingRepository


# Passes the base repository with Property type
class PropertyRepository(BaseRepository[Property]):
    """Repository for managing properties"""
    
    virtual_staging_repo: VirtualStagingRepository = VirtualStagingRepository()

    def __init__(self):
        super().__init__('properties')
    
    def to_model(self, data: Dict[str, Any]) -> Property:
        """Convert Firestore document to Property model"""
        return Property.from_dict(data)
    
    def to_dict(self, model: Property) -> Dict[str, Any]:
        """Convert Property model to Firestore document"""
        return model.to_dict()
    
    def create_property(self, property_model: Property) -> Property:
        """
        Create a new property
        
        Args:
            property_model: Property model instance
        
        Returns:
            Created Property model
        """
        doc_id = str(property_model.id)
        return self.create(doc_id, property_model)
    
    def get_property(self, property_id: int) -> Optional[Property]:
        """
        Get property by ID
        
        Args:
            property_id: Property ID
        
        Returns:
            Property model or None
        """
        doc_id = str(property_id)
        return self.get(doc_id)
    
    def get_properties_by_host(self, host_id: int) -> List[tuple[str, Property]]:
        """
        Get all properties for a host
        
        Args:
            host_id: Host ID
        
        Returns:
            List of (doc_id, Property) tuples
        """
        return self.query('host_id', '==', host_id)
    
    def get_all_properties(self) -> List[tuple[str, Property]]:
        """
        Get all properties
        
        Returns:
            List of (doc_id, Property) tuples
        """
        return self.list_all()
    
    def update_property(self, property_model: Property) -> Property:
        """
        Update an existing property
        
        Args:
            property_model: Updated Property model
        
        Returns:
            Updated Property model
        """
        doc_id = str(property_model.id)
        return self.update(doc_id, property_model)
    
    def add_virtual_staging_session(self, property_id: int, session_id: str) -> bool:
        """
        Add a virtual staging session to property's session list
        
        Args:
            property_id: Property ID
            session_id: Virtual staging session ID
        
        Returns:
            True if successful
        """
        property_model = self.get_property(property_id)
        if not property_model:
            return False
        
        if session_id not in property_model.virtual_staging_sessions:
            property_model.virtual_staging_sessions.append(session_id)
            self.update_property(property_model)
        
        return True
    
    def remove_virtual_staging_session(self, property_id: int, session_id: str) -> bool:
        """
        Remove a virtual staging session from property's session list
        
        Args:
            property_id: Property ID
            session_id: Virtual staging session ID
        
        Returns:
            True if successful
        """
        property_model = self.get_property(property_id)
        if not property_model:
            return False
        
        virtual_staging = self.virtual_staging_repo.get(session_id)

        if not virtual_staging:
            return False

        self.virtual_staging_repo.delete(session_id)

        self.update_property(property_model)

        return True

    
    def get_virtual_staging_sessions(self, property_id: int) -> List[str]:
        """
        Get all virtual staging session IDs for a property
        
        Args:
            property_id: Property ID
        
        Returns:
            List of session IDs
        """
        property_model = self.get_property(property_id)
        if not property_model:
            return []
        
        return property_model.virtual_staging_sessions
    
    def update_rooms(self, property_id: int, rooms: List[Dict[str, Any]]) -> bool:
        """
        Update the rooms/spaces in a property
        
        Args:
            property_id: Property ID
            rooms: List of room dictionaries
        
        Returns:
            True if successful
        """
        property_model = self.get_property(property_id)
        if not property_model:
            return False
        
        property_model.rooms = rooms
        self.update_property(property_model)
        return True
    
    def delete_property(self, property_id: int) -> bool:
        """
        Delete a property
        
        Args:
            property_id: Property ID
        
        Returns:
            True if deleted
        """
        doc_id = str(property_id)
        return self.delete(doc_id)
