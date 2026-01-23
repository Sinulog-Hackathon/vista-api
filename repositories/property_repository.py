"""Repository for Property model"""
from typing import Dict, Any, List, Optional
from models.property import Property
from repositories.base_repository import BaseRepository


class PropertyRepository(BaseRepository[Property]):
    """Repository for managing properties"""

    def __init__(self):
        super().__init__('properties')

    def to_model(self, data: Dict[str, Any]) -> Property:
        """Convert Firestore document to Property model"""
        return Property.from_dict(data)

    def to_dict(self, model: Property) -> Dict[str, Any]:
        """Convert Property model to Firestore document"""
        return model.to_dict()

    def create_property(self, property_model: Property, property_id: str) -> Property:
        """
        Create a new property

        Args:
            property_model: Property model instance
            property_id: Unique property ID string

        Returns:
            Created Property model
        """
        return self.create(property_id, property_model)

    def get_property(self, property_id: str) -> Optional[Property]:
        """
        Get property by ID

        Args:
            property_id: Property ID string

        Returns:
            Property model or None
        """
        return self.get(property_id)

    def get_properties_by_user(self, user_id: str) -> List[tuple[str, Property]]:
        """
        Get all properties for a user

        Args:
            user_id: User ID

        Returns:
            List of (doc_id, Property) tuples
        """
        return self.query('createdBy', '==', user_id)

    def get_all_properties(self) -> List[tuple[str, Property]]:
        """
        Get all properties

        Returns:
            List of (doc_id, Property) tuples
        """
        return self.list_all()

    def update_property(self, property_id: str, property_model: Property) -> Property:
        """
        Update an existing property

        Args:
            property_id: Property ID string
            property_model: Updated Property model

        Returns:
            Updated Property model
        """
        return self.update(property_id, property_model)

    def delete_property(self, property_id: str) -> bool:
        """
        Delete a property

        Args:
            property_id: Property ID string

        Returns:
            True if deleted
        """
        return self.delete(property_id)

    def get_properties_by_status(self, status: str) -> List[tuple[str, Property]]:
        """
        Get properties by status

        Args:
            status: Property status

        Returns:
            List of (doc_id, Property) tuples
        """
        return self.query('status', '==', status)
