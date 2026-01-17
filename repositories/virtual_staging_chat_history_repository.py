"""Repository for VirtualStagingChatHistory model"""
from typing import Dict, Any, List, Optional
from models.virtual_staging_chat_history import (
    VirtualStagingChatHistory, 
    VirtualStagingChatMessage,
    MessageRole
)
from repositories.base_repository import BaseRepository
from datetime import datetime


class VirtualStagingChatHistoryRepository(BaseRepository[VirtualStagingChatHistory]):
    """Repository for managing virtual staging chat history"""
    
    def __init__(self):
        super().__init__('virtual_staging_chat_history')
    
    def to_model(self, data: Dict[str, Any]) -> VirtualStagingChatHistory:
        """Convert Firestore document to VirtualStagingChatHistory model"""
        return VirtualStagingChatHistory.from_dict(data)
    
    def to_dict(self, model: VirtualStagingChatHistory) -> Dict[str, Any]:
        """Convert VirtualStagingChatHistory model to Firestore document"""
        return model.to_dict()
    
    def create_history(self, history: VirtualStagingChatHistory) -> VirtualStagingChatHistory:
        """
        Create a new chat history record
        
        Args:
            history: VirtualStagingChatHistory model instance
        
        Returns:
            Created VirtualStagingChatHistory model
        """
        return self.create(history.history_id, history)
    
    def get_history(self, history_id: str) -> Optional[VirtualStagingChatHistory]:
        """
        Get chat history by ID
        
        Args:
            history_id: History ID
        
        Returns:
            VirtualStagingChatHistory model or None
        """
        return self.get(history_id)
    
    def get_history_by_session(self, session_id: str) -> Optional[VirtualStagingChatHistory]:
        """
        Get chat history for a specific virtual staging session
        
        Args:
            session_id: Virtual staging session ID
        
        Returns:
            VirtualStagingChatHistory model or None
        """
        results = self.query('session_id', '==', session_id)
        if results:
            return results[0][1]
        return None
    
    def get_histories_by_property(self, property_id: int) -> List[tuple[str, VirtualStagingChatHistory]]:
        """
        Get all chat histories for a property
        
        Args:
            property_id: Property ID
        
        Returns:
            List of (history_id, VirtualStagingChatHistory) tuples
        """
        return self.query('property_id', '==', property_id)
    
    def get_histories_by_user(self, user_id: int) -> List[tuple[str, VirtualStagingChatHistory]]:
        """
        Get all chat histories for a user
        
        Args:
            user_id: User ID
        
        Returns:
            List of (history_id, VirtualStagingChatHistory) tuples
        """
        return self.query('user_id', '==', user_id)
    
    def add_message(self, history_id: str, message: VirtualStagingChatMessage) -> bool:
        """
        Add a message to chat history
        
        Args:
            history_id: History ID
            message: VirtualStagingChatMessage to add
        
        Returns:
            True if successful
        """
        history = self.get_history(history_id)
        if not history:
            return False
        
        history.add_message(message)
        self.update_history(history)
        return True
    
    def add_user_message(self, history_id: str, message_id: str, content: str, 
                        refinement_iteration: int = 1) -> bool:
        """
        Add a user message to chat history
        
        Args:
            history_id: History ID
            message_id: Unique message ID
            content: Message content
            refinement_iteration: Which iteration/version this corresponds to
        
        Returns:
            True if successful
        """
        message = VirtualStagingChatMessage(
            message_id=message_id,
            session_id=self.get_history(history_id).session_id,
            role=MessageRole.USER,
            content=content,
            refinement_iteration=refinement_iteration
        )
        return self.add_message(history_id, message)
    
    def add_assistant_message(self, history_id: str, message_id: str, content: str,
                             refinement_iteration: int = 1,
                             staging_parameters: Optional[Dict[str, Any]] = None) -> bool:
        """
        Add an assistant message to chat history
        
        Args:
            history_id: History ID
            message_id: Unique message ID
            content: Message content
            refinement_iteration: Which iteration/version this corresponds to
            staging_parameters: Parameters used for this staging
        
        Returns:
            True if successful
        """
        history = self.get_history(history_id)
        if not history:
            return False
        
        message = VirtualStagingChatMessage(
            message_id=message_id,
            session_id=history.session_id,
            role=MessageRole.ASSISTANT,
            content=content,
            refinement_iteration=refinement_iteration,
            staging_parameters_used=staging_parameters
        )
        return self.add_message(history_id, message)
    
    def update_context_summary(self, history_id: str, summary: str) -> bool:
        """
        Update the context summary
        
        Args:
            history_id: History ID
            summary: New context summary
        
        Returns:
            True if successful
        """
        history = self.get_history(history_id)
        if not history:
            return False
        
        history.context_summary = summary
        history.updated_at = datetime.utcnow()
        self.update_history(history)
        return True
    
    def update_accumulated_refinements(self, history_id: str, refinements: Dict[str, Any]) -> bool:
        """
        Update accumulated refinements
        
        Args:
            history_id: History ID
            refinements: Dictionary of accumulated refinements
        
        Returns:
            True if successful
        """
        history = self.get_history(history_id)
        if not history:
            return False
        
        history.accumulated_refinements = refinements
        history.updated_at = datetime.utcnow()
        self.update_history(history)
        return True
    
    def increment_iteration(self, history_id: str) -> bool:
        """
        Increment the total iterations counter
        
        Args:
            history_id: History ID
        
        Returns:
            True if successful
        """
        history = self.get_history(history_id)
        if not history:
            return False
        
        history.total_iterations += 1
        history.updated_at = datetime.utcnow()
        self.update_history(history)
        return True
    
    def update_history(self, history: VirtualStagingChatHistory) -> VirtualStagingChatHistory:
        """
        Update an existing chat history record
        
        Args:
            history: Updated VirtualStagingChatHistory model
        
        Returns:
            Updated VirtualStagingChatHistory model
        """
        history.updated_at = datetime.utcnow()
        return self.update(history.history_id, history)
    
    def get_context_for_llm(self, history_id: str, include_full_history: bool = False, 
                           last_n_messages: int = 10) -> Optional[str]:
        """
        Get formatted context string for LLM
        
        Args:
            history_id: History ID
            include_full_history: Whether to include full conversation
            last_n_messages: Number of recent messages to include
        
        Returns:
            Formatted context string or None if history not found
        """
        history = self.get_history(history_id)
        if not history:
            return None
        
        return history.get_context_for_llm(include_full_history, last_n_messages)
    
    def delete_history(self, history_id: str) -> bool:
        """
        Delete a chat history record
        
        Args:
            history_id: History ID
        
        Returns:
            True if deleted
        """
        return self.delete(history_id)
