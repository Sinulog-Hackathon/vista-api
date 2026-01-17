"""Service layer for VirtualStaging entity"""
from typing import Optional, List
from datetime import datetime
from models.virtual_staging import VirtualStaging, StagingParameters
from repositories.virtual_staging_repository import VirtualStagingRepository


class VirtualStagingService:
    """Business logic for virtual staging sessions"""
    
    def __init__(self):
        self.repository = VirtualStagingRepository()
    
    def create_staging_session(self,
                              session_id: str,
                              property_id: int,
                              user_id: int,
                              room_name: str,
                              original_image_key: str,
                              staging_parameters: StagingParameters) -> Optional[VirtualStaging]:
        """Create a new virtual staging session"""
        if not self._validate_staging(session_id, property_id, user_id, room_name, original_image_key):
            return None
        
        staging = VirtualStaging(
            session_id=session_id,
            property_id=property_id,
            user_id=user_id,
            room_name=room_name,
            orignal_image_key=original_image_key,
            staging_parameters=staging_parameters
        )
        return self.repository.create_session(staging)
    
    def get_session(self, session_id: str) -> Optional[VirtualStaging]:
        """Get staging session by ID"""
        return self.repository.get_session(session_id)
    
    def get_sessions_by_property(self, property_id: int) -> List[tuple[str, VirtualStaging]]:
        """Get all sessions for a property"""
        return self.repository.get_sessions_by_property(property_id)
    
    def get_sessions_by_user(self, user_id: int) -> List[tuple[str, VirtualStaging]]:
        """Get all sessions for a user"""
        return self.repository.get_sessions_by_user(user_id)
    
    
    def complete_staging(self, session_id: str, generated_image_key: str) -> bool:
        """Complete staging with generated image"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.generated_image_key = generated_image_key
        session.completed_at = datetime.now()
        return self.repository.update_generated_image(session_id, generated_image_key)
    
    
    def add_refinement_prompt(self, session_id: str, prompt: str) -> bool:
        """Add refinement prompt to session"""
        return self.repository.add_prompt(session_id, prompt)
    
    def refine_staging(self, session_id: str, new_image_key: str, prompt: str) -> bool:
        """Refine existing staging with new image and prompt"""
        session = self.get_session(session_id)
        if not session:
            return False
        
        session.prompts.append(prompt)
        session.generated_image_key = new_image_key
        session.version += 1
        session.updated_at = datetime.utcnow()
        
        return self.repository.update_session(session) is not None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete staging session"""
        return self.repository.delete_session(session_id)
    
    def _validate_staging(self, session_id: str, property_id: int, user_id: int,
                         room_name: str, original_image_key: str) -> bool:
        """Validate staging fields"""
        return (
            session_id and session_id.strip() and
            property_id > 0 and
            user_id > 0 and
            room_name and room_name.strip() and
            original_image_key and original_image_key.strip()
        )
