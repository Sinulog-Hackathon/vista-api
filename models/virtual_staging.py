from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum




class StagingParameters(BaseModel):
    """Parameters for customizing virtual staging"""
    role: str = Field(default="professional interior designer", description="Role description for the AI model")
    style: str = Field(default="modern", description="Staging style (e.g., modern, contemporary, minimalist)")
    color_scheme: Optional[str] = Field(default=None, description="Preferred color scheme")
    furniture_style: Optional[str] = Field(default=None, description="Furniture style preference")
    specific_requests: Optional[str] = Field(default=None, description="Additional customization requests")


class VirtualStaging(BaseModel):
    """Virtual staging session model with chat context support"""
    
    session_id: str = Field(..., description="Unique identifier for the virtual staging session")
    property_id: int = Field(..., description="Identifier for the associated property")
    user_id: int = Field(..., description="Identifier for the user requesting the virtual staging")
    room_name: str = Field(..., description="Name/identifier of the room being staged")
    
    # Image and output
    orignal_image_key: str = Field(..., description="URL of the original property image")
    generated_image_key: Optional[str] = Field(default=None, description="URL of the generated virtual staging image")
    
    # Staging parameters and history
    staging_parameters: StagingParameters = Field(..., description="Parameters for customizing the staging")
    prompts: List[str] = Field(default_factory=list, description="List of all prompts used for this staging session")
    
    # Versioning
    version: int = Field(default=1, description="Version/iteration number of the staging")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when session was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when session was last updated")
    completed_at: Optional[datetime] = Field(default=None, description="Timestamp when staging was completed")
    
    # Additional metadata
    error_message: Optional[str] = Field(default=None, description="Error message if staging failed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for Firestore"""
        data = self.model_dump();
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        if self.completed_at:
            data['completed_at'] = self.completed_at.isoformat()
        data['staging_parameters'] = self.staging_parameters.model_dump()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VirtualStaging':
        """Create model from Firestore dictionary"""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        if isinstance(data.get('completed_at'), str):
            data['completed_at'] = datetime.fromisoformat(data['completed_at'])
        if isinstance(data.get('staging_parameters'), dict):
            data['staging_parameters'] = StagingParameters(**data['staging_parameters'])
        return cls(**data)