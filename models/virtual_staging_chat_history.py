from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum


class MessageRole(str, Enum):
    """Role of the message sender in chat"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class VirtualStagingChatMessage(BaseModel):
    """Individual chat message for virtual staging context"""
    
    message_id: str = Field(..., description="Unique identifier for the message")
    session_id: str = Field(..., description="Reference to the virtual staging session")
    role: MessageRole = Field(..., description="Role of the message sender")
    content: str = Field(..., description="Actual message content")
    
    # Context and refinement tracking
    refinement_iteration: int = Field(default=1, description="Which iteration/version this message corresponds to")
    staging_parameters_used: Optional[Dict[str, Any]] = Field(default=None, description="Staging parameters used when this message was processed")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the message was created")
    

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firestore"""
        data = self.dict()
        data['created_at'] = self.created_at.isoformat()
        data['role'] = self.role.value
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VirtualStagingChatMessage':
        """Create from Firestore dictionary"""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        return cls(**data)


class VirtualStagingChatHistory(BaseModel):
    """Chat history for virtual staging session - maintains context for stateless LLM"""
    
    history_id: str = Field(..., description="Unique identifier for the chat history record")
    session_id: str = Field(..., description="Reference to the virtual staging session")
    property_id: int = Field(..., description="Reference to the property")
    user_id: int = Field(..., description="Reference to the user")
    
    # Chat messages
    messages: List[VirtualStagingChatMessage] = Field(default_factory=list, description="Ordered list of chat messages")
    
    # Context accumulation
    context_summary: Optional[str] = Field(default=None, description="Summary of key decisions and preferences from conversation")
    accumulated_refinements: Dict[str, Any] = Field(default_factory=dict, description="Accumulated refinement requests and outcomes")
    
    # Session tracking
    total_messages: int = Field(default=0, description="Total count of messages in history")
    total_iterations: int = Field(default=1, description="Total number of staging iterations")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the chat history was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the chat history was last updated")
    last_message_at: Optional[datetime] = Field(default=None, description="Timestamp of the last message")
    


    def add_message(self, message: VirtualStagingChatMessage) -> None:
        """Add a message to the chat history"""
        self.messages.append(message)
        self.total_messages = len(self.messages)
        self.updated_at = datetime.utcnow()
        self.last_message_at = message.created_at

    def get_context_for_llm(self, include_full_history: bool = False, last_n_messages: int = 10) -> str:
        """
        Generate context string for stateless LLM calls
        
        Args:
            include_full_history: Whether to include full conversation or just recent messages
            last_n_messages: Number of recent messages to include if not using full history
        
        Returns:
            Formatted context string for LLM
        """
        context_lines = []
        
        # Add session context
        context_lines.append(f"Session ID: {self.session_id}")
        context_lines.append(f"Property ID: {self.property_id}")
        if self.context_summary:
            context_lines.append(f"\nContext Summary:\n{self.context_summary}")
        
        # Add accumulated refinements
        if self.accumulated_refinements:
            context_lines.append("\nAccumulated Refinements:")
            for key, value in self.accumulated_refinements.items():
                context_lines.append(f"  - {key}: {value}")
        
        # Add message history
        context_lines.append("\nConversation History:")
        messages_to_include = self.messages if include_full_history else self.messages[-last_n_messages:]
        
        for msg in messages_to_include:
            role_label = msg.role.value
            context_lines.append(f"[{role_label}] {msg.content}")
        
        return "\n".join(context_lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firestore"""
        data = self.dict()
        data['messages'] = [msg.to_dict() for msg in self.messages]
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.last_message_at:
            data['last_message_at'] = self.last_message_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VirtualStagingChatHistory':
        """Create from Firestore dictionary"""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if isinstance(data.get('last_message_at'), str):
            data['last_message_at'] = datetime.fromisoformat(data['last_message_at'])
        
        if 'messages' in data:
            data['messages'] = [
                VirtualStagingChatMessage.from_dict(msg) if isinstance(msg, dict) else msg
                for msg in data['messages']
            ]
        
        return cls(**data)
