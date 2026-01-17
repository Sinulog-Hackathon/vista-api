"""Repository for VirtualStaging model"""
from typing import Dict, Any, List, Optional
from models.virtual_staging import VirtualStaging
from repositories.base_repository import BaseRepository


class VirtualStagingRepository(BaseRepository[VirtualStaging]):    
    def __init__(self):
        super().__init__('virtual_staging_sessions')
    
    def to_model(self, data: Dict[str, Any]) -> VirtualStaging:
        """Convert Firestore document to VirtualStaging model"""
        return VirtualStaging.from_dict(data)
    
    def to_dict(self, model: VirtualStaging) -> Dict[str, Any]:
        """Convert VirtualStaging model to Firestore document"""
        return model.to_dict()
    
    def create_session(self, staging: VirtualStaging) -> VirtualStaging:
        """
        Create a new virtual staging session
        
        Args:
            staging: VirtualStaging model instance
        
        Returns:
            Created VirtualStaging model
        """
        return self.create(staging.session_id, staging)
    
    def get_session(self, session_id: str) -> Optional[VirtualStaging]:
        """
        Get virtual staging session by ID
        
        Args:
            session_id: Session ID
        
        Returns:
            VirtualStaging model or None
        """
        return self.get(session_id)
    
    def get_sessions_by_property(self, property_id: int) -> List[tuple[str, VirtualStaging]]:
        """
        Get all virtual staging sessions for a property
        
        Args:
            property_id: Property ID
        
        Returns:
            List of (session_id, VirtualStaging) tuples
        """
        return self.query('property_id', '==', property_id)
    
    def get_sessions_by_user(self, user_id: int) -> List[tuple[str, VirtualStaging]]:
        """
        Get all virtual staging sessions for a user
        
        Args:
            user_id: User ID
        
        Returns:
            List of (session_id, VirtualStaging) tuples
        """
        return self.query('user_id', '==', user_id)

    
    def update_session(self, session: VirtualStaging) -> VirtualStaging:
        """
        Update an existing virtual staging session
        
        Args:
            session: Updated VirtualStaging model
        
        Returns:
            Updated VirtualStaging model
        """
        return self.update(session.session_id, session)
    
    
    def update_generated_image(self, session_id: str, image_url: str) -> bool:
        """
        Update the generated image URL and increment version
        
        Args:
            session_id: Session ID
            image_url: New image URL
        
        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.generated_image_url = image_url
        session.version += 1
        session.status = StagingStatus.COMPLETED
        
        self.update_session(session)
        return True
    
    def add_prompt(self, session_id: str, prompt: str) -> bool:
        """
        Add a prompt to the session's prompt history
        
        Args:
            session_id: Session ID
            prompt: Prompt text to add
        
        Returns:
            True if successful
        """
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.prompts.append(prompt)
        self.update_session(session)
        return True
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a virtual staging session
        
        Args:
            session_id: Session ID
        
        Returns:
            True if deleted
        """
        return self.delete(session_id)
