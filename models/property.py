from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime


class Property(BaseModel):
    """Property model with virtual staging references"""
    
    id: int = Field(..., description="Unique identifier for the property")
    host_id: int = Field(..., description="Identifier for the host/owner")
    address: str = Field(..., description="Full address of the property")
    base_price: float = Field(..., description="Base price of the property")
    description: str = Field(..., description="Description of the property")
    panoramic_image_url: str = Field(..., description="URL of the panoramic/main image")
    
    # Virtual staging references
    virtual_staging_sessions: List[str] = Field(default_factory=list, description="List of virtual staging session IDs for this property")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow, description="When the property was created")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="When the property was last updated")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firestore"""
        data = self.model_dump();
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Property':
        """Create from Firestore dictionary"""
        if isinstance(data.get('created_at'), str):
            data['created_at'] = datetime.fromisoformat(data['created_at'])
        if isinstance(data.get('updated_at'), str):
            data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        return cls(**data)
