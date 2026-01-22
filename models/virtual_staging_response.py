"""
Response models for Virtual Staging API
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime


class StagingMetadata(BaseModel):
    """Metadata about a staging result"""
    session_id: str = Field(..., description="Unique staging session identifier")
    property_id: int = Field(..., description="Associated property ID")
    user_id: int = Field(..., description="Associated user ID")
    room_name: str = Field(..., description="Room name/identifier")
    version: int = Field(..., description="Staging version/iteration number")
    
    style: str = Field(..., description="Applied staging style")
    furniture_theme: str = Field(..., description="Applied furniture theme")
    color_scheme: Optional[str] = Field(None, description="Applied color scheme (hex)")
    specific_request: Optional[str] = Field(None, description="Applied specific request")
    
    created_at: datetime = Field(..., description="When session was created")
    updated_at: datetime = Field(..., description="When this version was created")
    completed_at: Optional[datetime] = Field(None, description="When staging was completed")


class VirtualStagingResponse(BaseModel):
    """Response for virtual staging request with image and metadata - UNSAVED working version"""
    image_url: str = Field(..., description="URL to the generated virtual staged image (working version, not yet saved)")
    metadata: StagingMetadata = Field(..., description="Metadata about the staging")
    prompt_used: Optional[str] = Field(None, description="The prompt used to generate this staging")
    is_saved: bool = Field(default=False, description="Whether this version has been saved")
    can_revert: bool = Field(default=False, description="Whether previous versions exist to revert to")


class SaveChangeResponse(BaseModel):
    """Response for saving a change - commits current working version to history"""
    success: bool = Field(..., description="Whether save was successful")
    version: int = Field(..., description="Version number this was saved as")
    message: str = Field(..., description="Success or error message")
    image_url: str = Field(..., description="URL of the saved image in S3")
    saved_at: datetime = Field(..., description="When this version was saved")


class RevertChangeResponse(BaseModel):
    """Response for reverting to a previous version"""
    success: bool = Field(..., description="Whether revert was successful")
    version: int = Field(..., description="Version number reverted to")
    message: str = Field(..., description="Success or error message")
    image_url: str = Field(..., description="URL of the reverted image in S3")
    reverted_at: datetime = Field(..., description="When the revert was performed")


class VersionHistoryItem(BaseModel):
    """Single item in version history"""
    version_number: int = Field(..., description="Version number")
    image_url: str = Field(..., description="URL to the image")
    parameters: Dict[str, Any] = Field(..., description="Parameters used for this version")
    prompt_used: str = Field(..., description="Prompt used to generate this version")
    created_at: datetime = Field(..., description="When generated")
    saved_at: datetime = Field(..., description="When saved")
    is_current: bool = Field(default=False, description="Whether this is the current working version")


class VersionHistoryResponse(BaseModel):
    """Response with complete version history"""
    session_id: str = Field(..., description="Session ID")
    total_versions: int = Field(..., description="Total number of saved versions")
    current_version: int = Field(..., description="Current version number")
    has_unsaved_changes: bool = Field(..., description="Whether there are unsaved changes")
    versions: List[VersionHistoryItem] = Field(..., description="List of all saved versions")


class StagingSessionResponse(BaseModel):
    """Response for getting a staging session"""
    session_id: str = Field(..., description="Unique staging session identifier")
    property_id: int = Field(..., description="Associated property ID")
    user_id: int = Field(..., description="Associated user ID")
    room_name: str = Field(..., description="Room name/identifier")
    
    original_image_url: Optional[str] = Field(None, description="URL to the original image in S3")
    current_image_url: Optional[str] = Field(None, description="URL to the current working image (unsaved)")
    last_saved_image_url: Optional[str] = Field(None, description="URL to the last saved image version")
    
    staging_parameters: Dict[str, Any] = Field(..., description="Current working staging parameters")
    current_version: int = Field(..., description="Current version number (last saved)")
    total_versions: int = Field(..., description="Total saved versions")
    has_unsaved_changes: bool = Field(..., description="Whether there are unsaved changes")
    
    created_at: datetime = Field(..., description="When session was created")
    updated_at: datetime = Field(..., description="When session was last updated")
    completed_at: Optional[datetime] = Field(None, description="When staging was completed")
    error_message: Optional[str] = Field(None, description="Error message if staging failed")


class RefinementResponse(BaseModel):
    """Response for staging refinement - creates unsaved working version"""
    image_url: str = Field(..., description="URL to the refined virtual staged image (working version, not yet saved)")
    version: int = Field(..., description="Current version number (before saving)")
    updated_at: datetime = Field(..., description="When the refinement was applied")
    prompt_used: str = Field(..., description="The prompt used for this refinement")
    is_saved: bool = Field(default=False, description="Whether this version has been saved")

