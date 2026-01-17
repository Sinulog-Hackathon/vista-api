"""Service layer for VirtualStagingChatHistory entity"""
from typing import Optional, List, Dict, Any
from datetime import datetime
from models.virtual_staging_chat_history import (
    VirtualStagingChatHistory,
    VirtualStagingChatMessage,
    MessageRole
)
from repositories.virtual_staging_chat_history_repository import VirtualStagingChatHistoryRepository


class VirtualStagingChatHistoryService:
    """Business logic for chat history and context management"""
    
    def __init__(self):
        self.repository = VirtualStagingChatHistoryRepository()
    
    def create_chat_history(self,
                           history_id: str,
                           session_id: str,
                           property_id: int,
                           user_id: int) -> Optional[VirtualStagingChatHistory]:
        """Create new chat history for staging session"""
        if not self._validate_history(history_id, session_id, property_id, user_id):
            return None
        
        history = VirtualStagingChatHistory(
            history_id=history_id,
            session_id=session_id,
            property_id=property_id,
            user_id=user_id
        )
        return self.repository.create_history(history)
    
    def get_history(self, history_id: str) -> Optional[VirtualStagingChatHistory]:
        """Get chat history by ID"""
        return self.repository.get_history(history_id)
    
    def get_history_by_session(self, session_id: str) -> Optional[VirtualStagingChatHistory]:
        """Get chat history for a staging session"""
        return self.repository.get_history_by_session(session_id)
    
    def get_histories_by_property(self, property_id: int) -> List[tuple[str, VirtualStagingChatHistory]]:
        """Get all chat histories for a property"""
        return self.repository.get_histories_by_property(property_id)
    
    def get_histories_by_user(self, user_id: int) -> List[tuple[str, VirtualStagingChatHistory]]:
        """Get all chat histories for a user"""
        return self.repository.get_histories_by_user(user_id)
    
    def add_user_message(self, history_id: str, message_id: str, content: str,
                        refinement_iteration: int = 1) -> bool:
        """Add user message to chat history"""
        return self.repository.add_user_message(
            history_id,
            message_id,
            content,
            refinement_iteration
        )
    
    def add_assistant_message(self, history_id: str, message_id: str, content: str,
                             refinement_iteration: int = 1,
                             staging_parameters: Optional[Dict[str, Any]] = None) -> bool:
        """Add assistant response to chat history"""
        return self.repository.add_assistant_message(
            history_id,
            message_id,
            content,
            refinement_iteration,
            staging_parameters
        )
    
    def get_llm_context(self, history_id: str,
                       include_full_history: bool = False,
                       last_n_messages: int = 10) -> Optional[str]:
        """Generate formatted context for LLM API calls"""
        return self.repository.get_context_for_llm(
            history_id,
            include_full_history,
            last_n_messages
        )
    
    def update_context_summary(self, history_id: str, summary: str) -> bool:
        """Update conversation context summary"""
        return self.repository.update_context_summary(history_id, summary)
    
    def update_refinements(self, history_id: str, refinements: Dict[str, Any]) -> bool:
        """Update accumulated refinements from conversation"""
        return self.repository.update_accumulated_refinements(history_id, refinements)
    
    def process_refinement_request(self, history_id: str, user_message: str,
                                  assistant_response: str,
                                  refinement_iteration: int,
                                  staging_params: Optional[Dict[str, Any]] = None) -> bool:
        """Process a complete refinement request-response cycle"""
        history = self.get_history(history_id)
        if not history:
            return False
        
        self.add_user_message(
            history_id,
            f"msg_{len(history.messages) + 1}",
            user_message,
            refinement_iteration
        )
        
        self.add_assistant_message(
            history_id,
            f"msg_{len(history.messages) + 2}",
            assistant_response,
            refinement_iteration,
            staging_params
        )
        
        self.repository.increment_iteration(history_id)
        return True
    
    def get_conversation_summary(self, history_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of conversation state"""
        history = self.get_history(history_id)
        if not history:
            return None
        
        return {
            'total_messages': history.total_messages,
            'total_iterations': history.total_iterations,
            'context_summary': history.context_summary,
            'last_message_at': history.last_message_at,
            'accumulated_refinements': history.accumulated_refinements,
            'created_at': history.created_at,
            'updated_at': history.updated_at
        }
    
    def delete_history(self, history_id: str) -> bool:
        """Delete chat history"""
        return self.repository.delete_history(history_id)
    
    def _validate_history(self, history_id: str, session_id: str,
                         property_id: int, user_id: int) -> bool:
        """Validate history fields"""
        return (
            history_id and history_id.strip() and
            session_id and session_id.strip() and
            property_id > 0 and
            user_id > 0
        )
