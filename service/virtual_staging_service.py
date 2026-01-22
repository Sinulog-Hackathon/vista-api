"""Service layer for VirtualStaging entity with versioning and save/revert functionality"""
from typing import Optional, List, Tuple, Dict, Any
from datetime import datetime
from models.virtual_staging import VirtualStaging, StagingParameters, ImageVersion
from models.virtual_staging_response import (
    VirtualStagingResponse, 
    StagingMetadata, 
    RefinementResponse, 
    StagingSessionResponse,
    SaveChangeResponse,
    RevertChangeResponse,
    VersionHistoryResponse,
    VersionHistoryItem
)
from repositories.virtual_staging_repository import VirtualStagingRepository
from service.gemini_service import GeminiService
from service.aws_service import AWSService
from service.virtual_staging_chat_history_service import VirtualStagingChatHistoryService
from config.prompt_config import build_staging_prompt, build_refinement_context
from pathlib import Path
import os
import uuid


class VirtualStagingService:
    """Business logic for virtual staging sessions with versioning support"""
    
    def __init__(self, gemini_model: str = "gemini-2.5-flash-image"):
        self.repository = VirtualStagingRepository()
        self.gemini_service = GeminiService()
        self.aws_service = AWSService()
        self.chat_history_service = VirtualStagingChatHistoryService()
        self.gemini_model = gemini_model
        self.base_url = "https://vista-resources.s3.ap-southeast-2.amazonaws.com/"
    
    def create_staging_session(self,
                              session_id: str,
                              property_id: int,
                              user_id: int,
                              room_name: str,
                              original_image_path: str,
                              staging_parameters: StagingParameters) -> Optional[VirtualStaging]:
        """
        Create a new virtual staging session with local image file
        
        Args:
            session_id: Unique session identifier
            property_id: Property ID
            user_id: User ID
            room_name: Room name being staged
            original_image_path: Local file path to the original image
            staging_parameters: Staging parameters for customization
            
        Returns:
            Created VirtualStaging session or None if failed
        """
        if not self._validate_staging(session_id, property_id, user_id, room_name):
            return None
        
        try:
            # Validate file exists
            if not os.path.exists(original_image_path):
                print(f"Error: Original image file not found: {original_image_path}")
                return None
            
            print(f"[SESSION] Using original image from: {original_image_path}")
            
            # Create chat history for this session
            chat_history_id = f"chat_{session_id}"
            chat_history = self.chat_history_service.create_chat_history(
                history_id=chat_history_id,
                session_id=session_id,
                property_id=property_id,
                user_id=user_id
            )
            
            if not chat_history:
                print(f"[SESSION] Failed to create chat history for session {session_id}")
                return None
            
            print(f"[SESSION] Chat history created: {chat_history_id}")
            
            # Create session in database with local path and chat history reference
            staging = VirtualStaging(
                session_id=session_id,
                property_id=property_id,
                user_id=user_id,
                room_name=room_name,
                chat_history_id=chat_history_id,
                original_image_path=original_image_path,
                orignal_image_key=None,
                original_image_url=None,
                current_parameters=staging_parameters,
                version=0,
                last_saved_version=0
            )
            
            print(f"[SESSION] VirtualStaging created with path: {staging.original_image_path}")
            
            created_session = self.repository.create_session(staging)
            
            if created_session:
                print(f"[SESSION] Session {session_id} created with chat history")
                print(f"[SESSION] Stored path in database: {created_session.original_image_path}")
            
            return created_session
        
        except Exception as e:
            print(f"Error creating staging session: {str(e)}")
            return None
    
    def generate_staging(self,
                        session_id: str,
                        staging_parameters: Optional[StagingParameters] = None,
                        custom_prompt: Optional[str] = None,
                        mask_image_url: Optional[str] = None,
                        user_message: Optional[str] = None) -> Optional[VirtualStagingResponse]:
        """
        Generate virtual staging with Gemini and save as unsaved working version.
        This does NOT save to history until save_change() is called.
        Also adds the initial message to chat history.
        
        Args:
            session_id: Session ID
            staging_parameters: Staging parameters (optional, uses session defaults if not provided)
            custom_prompt: Custom prompt for Gemini (optional)
            mask_image_url: Optional mask image URL for specific area editing
            user_message: Optional user message for chat history
        
        Returns:
            VirtualStagingResponse with unsaved working image, or None if failed
        """
        try:
            session = self.get_session(session_id)
            if not session:
                print(f"Session {session_id} not found")
                return None
            
            # Ensure we have an image path (with fallback reconstruction)
            image_path = session.original_image_path
            if not image_path or not os.path.exists(image_path):
                # Try to reconstruct path from session_id
                reconstructed_path = str(Path('uploads') / f"original_{session_id}.png")
                if os.path.exists(reconstructed_path):
                    image_path = reconstructed_path
                    print(f"[STAGING] Using reconstructed path: {reconstructed_path}")
                else:
                    print(f"Error: Original image not found for session {session_id}")
                    print(f"  Expected path: {session.original_image_path}")
                    print(f"  Reconstructed: {reconstructed_path}")
                    return None
            
            # Use provided parameters or fall back to session parameters
            params = staging_parameters or session.current_parameters
            
            # Build prompt
            if custom_prompt:
                prompt = custom_prompt
            elif params:
                prompt = build_staging_prompt(
                    role=params.role,
                    style=params.style,
                    furniture_theme=params.furniture_style or "modern",
                    color_scheme=params.color_scheme,
                    specific_request=params.specific_requests
                )
            else:
                return None
            
            # Generate image using Gemini with local file path
            generated_image_bytes = self.gemini_service.generate_image_from_image(
                model=self.gemini_model,
                image_path=image_path,
                prompt=prompt,
                mask_image_path=mask_image_url
            )
            
            if not generated_image_bytes:
                print(f"Failed to generate image for session {session_id}")
                return None
            
            # Save generated image locally (working version - not yet saved to AWS)
            upload_folder = Path('uploads')
            upload_folder.mkdir(parents=True, exist_ok=True)
            
            local_filename = f"generated_{session_id}_v{session.version + 1}.png"
            local_path = upload_folder / local_filename
            
            with open(local_path, 'wb') as f:
                f.write(generated_image_bytes)
            
            print(f"[STAGING] Saved generated image locally: {local_path}")
            
            # Store in session as unsaved working version (local only)
            session.current_image_path = str(local_path)
            session.current_image_key = None
            session.current_image_url = None
            session.current_parameters = params
            session.current_prompt = prompt
            session.updated_at = datetime.utcnow()
            
            # Add to generation history
            session.generation_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "image_key": session.current_image_key,
                "prompt": prompt,
                "parameters": params.model_dump() if params else None,
                "saved": False
            })
            
            self.repository.update_session(session)
            
            # Persist initial message to chat history
            if session.chat_history_id:
                # Add user message if provided
                if user_message:
                    self.chat_history_service.add_user_message(
                        history_id=session.chat_history_id,
                        message_id=f"msg_{uuid.uuid4().hex[:12]}",
                        content=user_message,
                        refinement_iteration=1
                    )
                
                # Add assistant message with the staging prompt and parameters
                self.chat_history_service.add_assistant_message(
                    history_id=session.chat_history_id,
                    message_id=f"msg_{uuid.uuid4().hex[:12]}",
                    content=f"Generated initial virtual staging with the specified parameters",
                    refinement_iteration=1,
                    staging_parameters=params.model_dump() if params else None
                )
                
                print(f"[STAGING] Chat history updated for session {session_id}")
            
            # Convert local image to base64 for response
            import base64
            with open(session.current_image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            image_data_url = f"data:image/png;base64,{image_data}"
            
            # Build response
            metadata = StagingMetadata(
                session_id=session_id,
                property_id=session.property_id,
                user_id=session.user_id,
                room_name=session.room_name,
                version=session.version,
                style=params.style if params else "custom",
                furniture_theme=params.furniture_style or "modern" if params else "modern",
                color_scheme=params.color_scheme if params else None,
                specific_request=params.specific_requests if params else None,
                created_at=session.created_at,
                updated_at=datetime.utcnow(),
                completed_at=session.completed_at
            )
            
            return VirtualStagingResponse(
                image_url=image_data_url,
                metadata=metadata,
                prompt_used=prompt,
                is_saved=False,
                can_revert=len(session.saved_versions) > 0
            )
        
        except Exception as e:
            print(f"Error generating staging: {str(e)}")
            return None
    
    def save_change(self, session_id: str) -> Optional[SaveChangeResponse]:
        """
        Save the current working version to history and upload to AWS S3.
        This is where the working image becomes part of the permanent history in AWS.
        
        Args:
            session_id: Session ID
        
        Returns:
            SaveChangeResponse with success status and new version, or None if failed
        """
        try:
            session = self.get_session(session_id)
            if not session:
                print(f"Session {session_id} not found")
                return None
            
            if not session.current_image_path or not os.path.exists(session.current_image_path):
                print(f"No working image to save for session {session_id}")
                return None
            
            # Upload current working image to AWS S3
            with open(session.current_image_path, 'rb') as f:
                image_bytes = f.read()
            
            new_version = session.version + 1
            
            upload_result = self.aws_service.upload_bytes(
                image_bytes=image_bytes,
                filename=f"v{new_version}_{session_id}.png",
                folder=f"staging/{session_id}/versions",
                content_type="image/png"
            )
            
            if not upload_result.get("success"):
                print(f"Failed to upload to AWS: {upload_result.get('error')}")
                return None
            
            now = datetime.utcnow()
            
            # Create new saved version with AWS URLs
            image_version = ImageVersion(
                version_number=new_version,
                image_key=upload_result.get("key"),
                image_url=upload_result.get("url"),
                prompt_used=session.current_prompt or "",
                parameters=session.current_parameters or StagingParameters(),
                is_saved=True,
                created_at=now,
                saved_at=now
            )
            
            print(f"[SAVE] Uploaded version {new_version} to AWS: {image_version.image_url}")
            
            # Add to saved versions history
            session.saved_versions.append(image_version)
            session.version = new_version
            session.last_saved_version = new_version
            session.updated_at = now
            
            # Move from working to saved in generation history
            if session.generation_history:
                session.generation_history[-1]["saved"] = True
                session.generation_history[-1]["saved_at"] = now.isoformat()
                session.generation_history[-1]["aws_key"] = image_version.image_key
                session.generation_history[-1]["aws_url"] = image_version.image_url
            
            # Update session
            self.repository.update_session(session)
            
            return SaveChangeResponse(
                success=True,
                version=new_version,
                message=f"Version {new_version} saved successfully to AWS",
                image_url=image_version.image_url,
                saved_at=now
            )
        
        except Exception as e:
            print(f"Error saving change: {str(e)}")
            return SaveChangeResponse(
                success=False,
                version=0,
                message=f"Error saving change: {str(e)}",
                image_url="",
                saved_at=datetime.utcnow()
            )
    
    def revert_change(self, session_id: str, version_to_revert_to: int) -> Optional[RevertChangeResponse]:
        """
        Revert to a previous saved version. Sets that version as the current working version.
        
        Args:
            session_id: Session ID
            version_to_revert_to: Version number to revert to
        
        Returns:
            RevertChangeResponse with success status and reverted version, or None if failed
        """
        try:
            session = self.get_session(session_id)
            if not session:
                print(f"Session {session_id} not found")
                return None
            
            # Find the version to revert to
            target_version = None
            for version in session.saved_versions:
                if version.version_number == version_to_revert_to:
                    target_version = version
                    break
            
            if not target_version:
                print(f"Version {version_to_revert_to} not found in session {session_id}")
                return RevertChangeResponse(
                    success=False,
                    version=session.version,
                    message=f"Version {version_to_revert_to} not found",
                    image_url=session.current_image_url or "",
                    reverted_at=datetime.utcnow()
                )
            
            # Set the reverted version as current working version
            session.current_image_key = target_version.image_key
            session.current_image_url = target_version.image_url
            session.current_parameters = target_version.parameters
            session.current_prompt = target_version.prompt_used
            session.updated_at = datetime.utcnow()
            
            # Add revert action to generation history
            session.generation_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "action": "revert",
                "reverted_to_version": version_to_revert_to,
                "image_key": target_version.image_key,
                "image_url": target_version.image_url
            })
            
            self.repository.update_session(session)
            
            return RevertChangeResponse(
                success=True,
                version=version_to_revert_to,
                message=f"Reverted to version {version_to_revert_to} successfully",
                image_url=session.current_image_url,
                reverted_at=datetime.utcnow()
            )
        
        except Exception as e:
            print(f"Error reverting change: {str(e)}")
            return RevertChangeResponse(
                success=False,
                version=0,
                message=f"Error reverting change: {str(e)}",
                image_url="",
                reverted_at=datetime.utcnow()
            )
    
    def get_version_history(self, session_id: str) -> Optional[VersionHistoryResponse]:
        """
        Get complete version history for a session
        
        Args:
            session_id: Session ID
        
        Returns:
            VersionHistoryResponse with all saved versions, or None if failed
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return None
            
            versions = []
            for version in session.saved_versions:
                versions.append(VersionHistoryItem(
                    version_number=version.version_number,
                    image_url=version.image_url,
                    parameters=version.parameters.model_dump() if version.parameters else {},
                    prompt_used=version.prompt_used,
                    created_at=version.created_at,
                    saved_at=version.saved_at or version.created_at,
                    is_current=session.current_image_key == version.image_key
                ))
            
            return VersionHistoryResponse(
                session_id=session_id,
                total_versions=len(session.saved_versions),
                current_version=session.version,
                has_unsaved_changes=session.current_image_url is not None 
                                   and session.current_image_url not in [v.image_url for v in session.saved_versions],
                versions=versions
            )
        
        except Exception as e:
            print(f"Error getting version history: {str(e)}")
            return None
    
    def get_session(self, session_id: str) -> Optional[VirtualStaging]:
        """Get staging session by ID"""
        return self.repository.get_session(session_id)
    
    def get_sessions_by_property(self, property_id: int) -> List[Tuple[str, VirtualStaging]]:
        """Get all sessions for a property"""
        return self.repository.get_sessions_by_property(property_id)
    
    def get_sessions_by_user(self, user_id: int) -> List[Tuple[str, VirtualStaging]]:
        """Get all sessions for a user"""
        return self.repository.get_sessions_by_user(user_id)
    
    def get_session_response(self, session_id: str) -> Optional[StagingSessionResponse]:
        """
        Get staging session as API response with all metadata
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return None
            
            # Get last saved version
            last_saved_url = None
            if session.saved_versions:
                last_saved_url = session.saved_versions[-1].image_url
            
            params_dict = session.current_parameters.model_dump() if session.current_parameters else {}
            
            return StagingSessionResponse(
                session_id=session.session_id,
                property_id=session.property_id,
                user_id=session.user_id,
                room_name=session.room_name,
                original_image_url=session.original_image_url,
                current_image_url=session.current_image_url,
                last_saved_image_url=last_saved_url,
                staging_parameters=params_dict,
                current_version=session.version,
                total_versions=len(session.saved_versions),
                has_unsaved_changes=session.current_image_url is not None 
                                   and session.current_image_url not in [v.image_url for v in session.saved_versions],
                created_at=session.created_at,
                updated_at=session.updated_at,
                completed_at=session.completed_at,
                error_message=session.error_message
            )
        
        except Exception as e:
            print(f"Error getting session response: {str(e)}")
            return None
    
    def refine_staging(self,
                      session_id: str,
                      new_staging_parameters: StagingParameters,
                      user_message: Optional[str] = None) -> Optional[RefinementResponse]:
        """
        Refine existing staging with new parameters.
        Creates a new unsaved working version and persists to chat history.
        
        Args:
            session_id: Session ID to refine
            new_staging_parameters: Updated staging parameters
            user_message: Optional user message for chat history
        
        Returns:
            RefinementResponse with new unsaved working image, or None if failed
        """
        try:
            session = self.get_session(session_id)
            if not session:
                return None
            
            # Ensure we have an image path (with fallback reconstruction)
            image_path = session.original_image_path
            if not image_path or not os.path.exists(image_path):
                # Try to reconstruct path from session_id
                reconstructed_path = str(Path('uploads') / f"original_{session_id}.png")
                if os.path.exists(reconstructed_path):
                    image_path = reconstructed_path
                    print(f"[REFINE] Using reconstructed path: {reconstructed_path}")
                else:
                    print(f"Error: Original image not found for session {session_id}")
                    return None
            
            # Get chat history context
            chat_history_llm_context = None
            if session.chat_history_id:
                chat_history_llm_context = self.chat_history_service.get_llm_context(
                    session.chat_history_id,
                    include_full_history=False,
                    last_n_messages=6
                )
            
            # Build refinement prompt with context
            refinement_context = build_refinement_context(
                previous_parameters=session.current_parameters.model_dump() if session.current_parameters else {},
                new_parameters=new_staging_parameters.model_dump(),
                chat_history=chat_history_llm_context.split('\n') if chat_history_llm_context else []
            )
            
            # Build the new staging prompt
            refinement_prompt = build_staging_prompt(
                role=new_staging_parameters.role,
                style=new_staging_parameters.style,
                furniture_theme=new_staging_parameters.furniture_style or "modern",
                color_scheme=new_staging_parameters.color_scheme,
                specific_request=new_staging_parameters.specific_requests
            )
            
            # Add context
            full_refinement_prompt = f"{refinement_context}\n\n{refinement_prompt}"
            
            # Generate refined image
            refined_image_bytes = self.gemini_service.generate_image_from_image(
                model=self.gemini_model,
                image_path=image_path,
                prompt=full_refinement_prompt
            )
            
            if not refined_image_bytes:
                print(f"Failed to generate refined image for session {session_id}")
                return None
            
            # Save refined image locally (working version - not yet saved to AWS)
            upload_folder = Path('uploads')
            upload_folder.mkdir(parents=True, exist_ok=True)
            
            local_filename = f"generated_{session_id}_v{session.version + 1}.png"
            local_path = upload_folder / local_filename
            
            with open(local_path, 'wb') as f:
                f.write(refined_image_bytes)
            
            print(f"[REFINE] Saved refined image locally: {local_path}")
            
            # Update session with new working version (local only)
            session.current_image_path = str(local_path)
            session.current_image_key = None
            session.current_image_url = None
            session.current_parameters = new_staging_parameters
            session.current_prompt = full_refinement_prompt
            session.updated_at = datetime.utcnow()
            
            # Add to generation history
            session.generation_history.append({
                "timestamp": datetime.utcnow().isoformat(),
                "image_key": None,
                "image_path": str(local_path),
                "prompt": full_refinement_prompt,
                "parameters": new_staging_parameters.model_dump(),
                "saved": False,
                "type": "refinement"
            })
            
            self.repository.update_session(session)
            
            # Persist messages to chat history
            if session.chat_history_id:
                # Add user message if provided
                if user_message:
                    self.chat_history_service.add_user_message(
                        history_id=session.chat_history_id,
                        message_id=f"msg_{uuid.uuid4().hex[:12]}",
                        content=user_message,
                        refinement_iteration=session.version + 1
                    )
                
                # Add assistant message with the refinement prompt and parameters
                self.chat_history_service.add_assistant_message(
                    history_id=session.chat_history_id,
                    message_id=f"msg_{uuid.uuid4().hex[:12]}",
                    content=f"Generated refined image with the following adjustments",
                    refinement_iteration=session.version + 1,
                    staging_parameters=new_staging_parameters.model_dump()
                )
                
                # Increment iteration counter in chat history
                self.chat_history_service.repository.increment_iteration(session.chat_history_id)
                print(f"[REFINE] Chat history updated for session {session_id}")
            
            # Convert local image to base64 for response
            import base64
            with open(session.current_image_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')
            image_data_url = f"data:image/png;base64,{image_data}"
            
            return RefinementResponse(
                image_url=image_data_url,
                version=session.version,
                updated_at=session.updated_at,
                prompt_used=full_refinement_prompt,
                is_saved=False
            )
        
        except Exception as e:
            print(f"Error refining staging: {str(e)}")
            return None
    
    def delete_session(self, session_id: str) -> bool:
        """Delete staging session, associated chat history, and clean up AWS files"""
        try:
            session = self.get_session(session_id)
            if session:
                # Delete chat history if exists
                if session.chat_history_id:
                    self.chat_history_service.delete_history(session.chat_history_id)
                    print(f"[DELETE] Deleted chat history: {session.chat_history_id}")
                
                # Delete all versions from AWS
                for version in session.saved_versions:
                    self.aws_service.delete_file(version.image_key)
                
                # Delete current working image
                if session.current_image_key:
                    self.aws_service.delete_file(session.current_image_key)
                
                # Delete original image
                if session.orignal_image_key:
                    self.aws_service.delete_file(session.orignal_image_key)
            
            return self.repository.delete_session(session_id)
        
        except Exception as e:
            print(f"Error deleting session: {str(e)}")
            return False
    
    def _validate_staging(self, session_id: str, property_id: int, user_id: int,
                         room_name: str) -> bool:
        """Validate staging fields"""
        return (
            session_id and session_id.strip() and
            property_id > 0 and
            user_id > 0 and
            room_name and room_name.strip()
        )

