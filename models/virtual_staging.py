from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum


class StyleEnum(str, Enum):
    """Available staging styles"""
    MODERN = "modern"
    CONTEMPORARY = "contemporary"
    MINIMALIST = "minimalist"
    TRADITIONAL = "traditional"
    INDUSTRIAL = "industrial"
    SCANDINAVIAN = "scandinavian"
    BOHEMIAN = "bohemian"
    RUSTIC = "rustic"
    LUXURY = "luxury"
    ECLECTIC = "eclectic"


class FurnitureThemeEnum(str, Enum):
    """Available furniture themes"""
    MINIMALIST = "minimalist"
    MAXIMALIST = "maximalist"
    MID_CENTURY = "mid_century"
    VINTAGE = "vintage"
    CONTEMPORARY = "contemporary"
    CLASSIC = "classic"
    SCANDINAVIAN = "scandinavian"
    INDUSTRIAL = "industrial"
    BOHEMIAN = "bohemian"
    TRANSITIONAL = "transitional"


class StagingParameters(BaseModel):
    """Parameters for customizing virtual staging"""
    role: str = Field(default="professional interior designer", description="Role description for the AI model")
    style: str = Field(default="modern", description="Staging style (e.g., modern, contemporary, minimalist)")
    color_scheme: Optional[str] = Field(default=None, description="Preferred color scheme")
    furniture_style: Optional[str] = Field(default=None, description="Furniture style preference")
    specific_requests: Optional[str] = Field(default=None, description="Additional customization requests")


class ImageVersion(BaseModel):
    """Represents a version/revision of a generated image"""
    version_number: int = Field(..., description="Version number in sequence")
    image_key: str = Field(..., description="S3 key/path to the image")
    image_url: str = Field(..., description="Full URL to the image in S3")
    prompt_used: str = Field(..., description="The prompt used to generate this version")
    parameters: StagingParameters = Field(..., description="Staging parameters used for this version")
    is_saved: bool = Field(default=False, description="Whether this version has been saved (committed)")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When this version was generated")
    saved_at: Optional[datetime] = Field(default=None, description="When this version was saved")


class VirtualStaging(BaseModel):
    """Virtual staging session model with version history and safety features"""
    
    session_id: str = Field(..., description="Unique identifier for the virtual staging session")
    property_id: int = Field(..., description="Identifier for the associated property")
    user_id: int = Field(..., description="Identifier for the user requesting the virtual staging")
    room_name: str = Field(..., description="Name/identifier of the room being staged")
    
    # Chat history reference
    chat_history_id: Optional[str] = Field(default=None, description="Reference to the persistent virtual staging chat history")
    
    # Image storage
    orignal_image_key: Optional[str] = Field(default=None, description="S3 key of the original property image")
    original_image_url: Optional[str] = Field(default=None, description="Full URL of the original image in S3")
    original_image_path: Optional[str] = Field(default=None, description="Local file path of the original image")
    
    # Current state (working version - not yet saved)
    current_image_key: Optional[str] = Field(default=None, description="S3 key of current working image")
    current_image_url: Optional[str] = Field(default=None, description="Full URL of current working image")
    current_image_path: Optional[str] = Field(default=None, description="Local file path of current working image")
    current_parameters: Optional[StagingParameters] = Field(default=None, description="Parameters for current working image")
    current_prompt: Optional[str] = Field(default=None, description="Prompt used for current working image")
    
    # Saved versions history
    saved_versions: List[ImageVersion] = Field(default_factory=list, description="List of all saved image versions")
    
    # Current version tracking
    version: int = Field(default=0, description="Current version number (increments when saved)")
    last_saved_version: int = Field(default=0, description="The last version that was saved")
    
    # Generation history (all attempts, saved or not)
    generation_history: List[Dict[str, Any]] = Field(default_factory=list, description="All generation attempts in this session")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when session was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Timestamp when session was last updated")
    completed_at: Optional[datetime] = Field(default=None, description="Timestamp when staging was completed")
    
    # Additional metadata
    error_message: Optional[str] = Field(default=None, description="Error message if staging failed")

    def to_dict(self) -> Dict[str, Any]:
        """Convert model to dictionary for Firestore"""
        # Explicitly include all fields to ensure nothing is lost
        data = {
            'session_id': self.session_id,
            'property_id': self.property_id,
            'user_id': self.user_id,
            'room_name': self.room_name,
            'chat_history_id': self.chat_history_id,
            'orignal_image_key': self.orignal_image_key,
            'original_image_url': self.original_image_url,
            'original_image_path': self.original_image_path,
            'current_image_key': self.current_image_key,
            'current_image_url': self.current_image_url,
            'current_image_path': self.current_image_path,
            'current_prompt': self.current_prompt,
            'version': self.version,
            'last_saved_version': self.last_saved_version,
            'generation_history': self.generation_history,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'error_message': self.error_message,
            'current_parameters': self.current_parameters.model_dump() if self.current_parameters else None,
            'saved_versions': [
                {
                    **v.model_dump(),
                    'created_at': v.created_at.isoformat(),
                    'saved_at': v.saved_at.isoformat() if v.saved_at else None,
                    'parameters': v.parameters.model_dump()
                }
                for v in self.saved_versions
            ] if self.saved_versions else []
        }
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
        
        # Handle current parameters
        if isinstance(data.get('current_parameters'), dict):
            data['current_parameters'] = StagingParameters(**data['current_parameters'])
        
        # Handle saved versions
        if isinstance(data.get('saved_versions'), list):
            data['saved_versions'] = [
                ImageVersion(
                    **{
                        **v,
                        'created_at': datetime.fromisoformat(v['created_at']) if isinstance(v.get('created_at'), str) else v.get('created_at'),
                        'saved_at': datetime.fromisoformat(v['saved_at']) if isinstance(v.get('saved_at'), str) else v.get('saved_at'),
                        'parameters': StagingParameters(**v['parameters']) if isinstance(v.get('parameters'), dict) else v.get('parameters')
                    }
                )
                for v in data['saved_versions']
            ]
        
        return cls(**data)