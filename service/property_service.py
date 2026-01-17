"""Service layer for Property entity"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from models.property import Property
from repositories.property_repository import PropertyRepository


class PropertyService:
    """Business logic for property management"""
    
    def __init__(self):
        self.repository = PropertyRepository()
    
    def create_property(self,
                       property_id: int,
                       host_id: int,
                       address: str,
                       base_price: float,
                       description: str,
                       panoramic_image_url: str) -> Optional[Property]:
        """
        Create a new property
        
        Args:
            property_id: Unique property identifier
            host_id: Property owner/host
            address: Full property address
            base_price: Base property price
            description: Property description
            panoramic_image_url: Main property image URL
        
        Returns:
            Created Property or None if validation fails
        """
        if not self._validate_property(property_id, host_id, address, base_price, description, panoramic_image_url):
            return None
        
        property_model = Property(
            id=property_id,
            host_id=host_id,
            address=address,
            base_price=base_price,
            description=description,
            panoramic_image_url=panoramic_image_url
        )
        return self.repository.create_property(property_model)
    
    def get_property(self, property_id: int) -> Optional[Property]:
        """Get property by ID"""
        return self.repository.get_property(property_id)
    
    def get_properties_by_host(self, host_id: int) -> List[tuple[str, Property]]:
        """Get all properties for a host"""
        return self.repository.get_properties_by_host(host_id)
    
    def get_all_properties(self) -> List[tuple[str, Property]]:
        """Get all properties"""
        return self.repository.get_all_properties()
    
    def update_property(self, property_model: Property) -> Optional[Property]:
        """Update existing property"""
        if not self._validate_property_model(property_model):
            return None
        return self.repository.update_property(property_model)
    
    def add_staging_session(self, property_id: int, session_id: str) -> bool:
        """Link virtual staging session to property"""
        return self.repository.add_virtual_staging_session(property_id, session_id)
    
    def remove_staging_session(self, property_id: int, session_id: str) -> bool:
        """Unlink virtual staging session from property"""
        return self.repository.remove_virtual_staging_session(property_id, session_id)
    
    def get_staging_sessions(self, property_id: int) -> List[str]:
        """Get all virtual staging session IDs for property"""
        return self.repository.get_virtual_staging_sessions(property_id)
    
    def get_staging_details(self, property_id: int) -> Optional[Dict[str, Any]]:
        """Get detailed staging information for property"""
        property_model = self.get_property(property_id)
        if not property_model:
            return None
        
        return {
            'property_id': property_model.id,
            'address': property_model.address,
            'host_id': property_model.host_id,
            'total_staging_sessions': len(property_model.virtual_staging_sessions),
            'staging_session_ids': property_model.virtual_staging_sessions
        }
    
    def update_rooms(self, property_id: int, rooms: List[Dict[str, Any]]) -> bool:
        """Update room metadata for property"""
        return self.repository.update_rooms(property_id, rooms)
    
    def delete_property(self, property_id: int) -> bool:
        """Delete property"""
        return self.repository.delete_property(property_id)
    
    def _validate_property(self, property_id: int, host_id: int, address: str, 
                          base_price: float, description: str, panoramic_image_url: str) -> bool:
        """Validate property fields"""
        return (
            property_id > 0 and
            host_id > 0 and
            address and address.strip() and
            base_price > 0 and
            description and description.strip() and
            panoramic_image_url and panoramic_image_url.strip()
        )
    
    def _validate_property_model(self, property_model: Property) -> bool:
        """Validate property model"""
        return self._validate_property(
            property_model.id,
            property_model.host_id,
            property_model.address,
            property_model.base_price,
            property_model.description,
            property_model.panoramic_image_url
        )
