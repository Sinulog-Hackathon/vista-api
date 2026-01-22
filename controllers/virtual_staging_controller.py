"""
Virtual Staging API Controllers - Image Upload Version
"""
from flask import Blueprint, request, jsonify
from werkzeug.utils import secure_filename
from typing import Tuple, Dict, Any
from models.virtual_staging import StagingParameters, StyleEnum, FurnitureThemeEnum
from service.virtual_staging_service import VirtualStagingService
from datetime import datetime
import uuid
import os
import base64
from pathlib import Path
from PIL import Image

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}

# Create upload folder if it doesn't exist
Path(UPLOAD_FOLDER).mkdir(parents=True, exist_ok=True)


def allowed_file(filename: str) -> bool:
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_uploaded_file(file, session_id: str = None) -> str:
    """Save uploaded file and return file path"""
    if file and allowed_file(file.filename):
        # Use session_id for consistent filename if provided, otherwise use timestamp
        filename = f"original_{session_id}.png"

        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        return filepath
    return None


def encode_image_to_base64(image_path: str) -> str:
    """Encode image file to base64 string"""
    with open(image_path, 'rb') as image_file:
        return base64.standard_b64encode(image_file.read()).decode('utf-8')


def get_image_mime_type(image_path: str) -> str:
    """Get MIME type from image file extension"""
    _, ext = os.path.splitext(image_path.lower())
    mime_types = {
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.png': 'image/png',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    return mime_types.get(ext, 'image/jpeg')


# Create blueprint
virtual_staging_bp = Blueprint('virtual_staging', __name__, url_prefix='/api/virtual-staging')

# Initialize service
vs_service = VirtualStagingService(gemini_model="gemini-2.0-flash")


def generate_session_id() -> str:
    """Generate a unique session ID"""
    return f"vs_{uuid.uuid4().hex[:12]}"


@virtual_staging_bp.route('/session', methods=['POST'])
def create_session() -> Tuple[Dict[str, Any], int]:
    """
    Create a new virtual staging session with image upload
    
    Form data (multipart/form-data):
    {
        "property_id": int,
        "user_id": int,
        "room_name": str,
        "image": file (uploaded image),
        "style": str (StyleEnum),
        "furniture_theme": str (FurnitureThemeEnum),
        "color_scheme": str (hex color, optional),
        "specific_request": str (optional)
    }
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return {'error': 'No image file provided'}, 400
        
        file = request.files['image']
        if file.filename == '':
            return {'error': 'No image file selected'}, 400
        
        # Get form data first to validate required fields
        property_id = request.form.get('property_id', type=int)
        user_id = request.form.get('user_id', type=int)
        room_name = request.form.get('room_name', type=str)
        style = request.form.get('style', type=str)
        furniture_theme = request.form.get('furniture_theme', type=str)
        color_scheme = request.form.get('color_scheme', type=str)
        specific_request = request.form.get('specific_request', type=str)
        
        # Validate required fields before saving file
        if not all([property_id, user_id, room_name, style, furniture_theme]):
            return {'error': 'Missing required fields: property_id, user_id, room_name, style, furniture_theme'}, 400
        
        # Generate session_id FIRST
        session_id = generate_session_id()
        
        # Validate and save image with session_id in filename
        image_path = save_uploaded_file(file, session_id=session_id)
        if not image_path:
            return {'error': 'Invalid image format. Allowed: png, jpg, jpeg, gif, webp'}, 400
        
        # Validate enums
        try:
            style_enum = StyleEnum(style)
            furniture_enum = FurnitureThemeEnum(furniture_theme)
        except ValueError as e:
            return {'error': f'Invalid enum value: {str(e)}'}, 400
        
        # Create staging parameters
        staging_params = StagingParameters(
            role=request.form.get('role', 'professional interior designer'),
            style=style_enum,
            furniture_theme=furniture_enum,
            color_scheme=color_scheme,
            specific_request=specific_request
        )
        
        # Create session with the saved file path
        staging = vs_service.create_staging_session(
            session_id=session_id,
            property_id=property_id,
            user_id=user_id,
            room_name=room_name,
            original_image_path=image_path,
            staging_parameters=staging_params
        )
        
        if not staging:
            return {'error': 'Failed to create staging session'}, 500
        
        return {
            'message': 'Session created successfully',
            'session_id': session_id,
            'image_path': image_path,
            'staging': staging.to_dict()
        }, 201
    
    except Exception as e:
        return {'error': f'Error creating session: {str(e)}'}, 500


@virtual_staging_bp.route('/generate', methods=['POST'])
def generate_staging() -> Tuple[Dict[str, Any], int]:
    """
    Generate virtual staging with Gemini using uploaded image
    
    Form data (multipart/form-data):
    {
        "session_id": str,
        "image": file (uploaded image - required),
        "custom_prompt": str (required - prompt for Gemini to edit the image),
        "image_mask": file (optional - second image to specify specific area/point),
        "style": str (optional),
        "furniture_theme": str (optional),
        "color_scheme": str (optional),
        "specific_request": str (optional),
        "user_message": str (optional - user's request for chat history)
    }
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return {'error': 'No image file provided'}, 400
        
        file = request.files['image']
        if file.filename == '':
            return {'error': 'No image file selected'}, 400
        
        # Get session_id first
        session_id = request.form.get('session_id', type=str)
        if not session_id:
            return {'error': 'Missing required field: session_id'}, 400
        
        # Get custom_prompt (required)
        custom_prompt = request.form.get('custom_prompt', type=str)
        if not custom_prompt or custom_prompt.strip() == '':
            return {'error': 'Missing required field: custom_prompt'}, 400
        
        # Get optional user message for chat history
        user_message = request.form.get('user_message', type=str)
        
        # Validate and save main image with session_id
        image_path = save_uploaded_file(file, session_id)
        if not image_path:
            return {'error': 'Invalid image format. Allowed: png, jpg, jpeg, gif, webp'}, 400
        
        # Get form data
        style = request.form.get('style', type=str)
        furniture_theme = request.form.get('furniture_theme', type=str)
        color_scheme = request.form.get('color_scheme', type=str)
        specific_request = request.form.get('specific_request', type=str)

        
        # Handle optional mask image for specific area/point
        mask_image_path = None
        if 'image_mask' in request.files:
            mask_file = request.files['image_mask']
            if mask_file and mask_file.filename != '':
                mask_image_path = save_uploaded_file(mask_file)
                if not mask_image_path:
                    return {'error': 'Invalid mask image format. Allowed: png, jpg, jpeg, gif, webp'}, 400
        
        # Validate enums if provided
        style_enum = None
        furniture_enum = None
        if style:
            try:
                style_enum = StyleEnum(style)
            except ValueError as e:
                return {'error': f'Invalid style enum value: {str(e)}'}, 400
        
        if furniture_theme:
            try:
                furniture_enum = FurnitureThemeEnum(furniture_theme)
            except ValueError as e:
                return {'error': f'Invalid furniture_theme enum value: {str(e)}'}, 400
        
        # Create staging parameters
        staging_params = None
        if style_enum and furniture_enum:
            staging_params = StagingParameters(
                role=request.form.get('role', 'professional interior designer'),
                style=style_enum,
                furniture_theme=furniture_enum,
                color_scheme=color_scheme,
                specific_request=specific_request
            )
        
        # Generate staging with custom prompt and optional user message
        response = vs_service.generate_staging(
            session_id=session_id,
            staging_parameters=staging_params,
            custom_prompt=custom_prompt,
            mask_image_url=mask_image_path,
            user_message=user_message
        )
        
        if not response:
            return {'error': 'Failed to generate staging'}, 500
        
        # Get the generated image path from response
        generated_image_path = response.image_url
        
        if generated_image_path and os.path.exists(generated_image_path):
            # Read generated image from disk and convert to base64
            with open(generated_image_path, 'rb') as f:
                image_bytes = f.read()
            image_base64 = base64.standard_b64encode(image_bytes).decode('utf-8')
            image_data_url = f"data:image/png;base64,{image_base64}"
        else:
            # Fallback to original image if generation failed
            pil_image = Image.open(image_path)
            from io import BytesIO
            img_buffer = BytesIO()
            pil_image.save(img_buffer, format='PNG')
            img_buffer.seek(0)
            image_base64 = base64.standard_b64encode(img_buffer.getvalue()).decode('utf-8')
            image_data_url = f"data:image/png;base64,{image_base64}"
        
        return {
            'message': 'Virtual staging generated successfully',
            'session_id': session_id,
            'staging_type': 'AI-Generated (gemini-2.5-flash-image)',
            'custom_prompt_used': custom_prompt,
            'prompt_used': response.prompt_used
        }, 200
    
    except Exception as e:
        return {'error': f'Error generating staging: {str(e)}'}, 500


@virtual_staging_bp.route('/session/<session_id>', methods=['GET'])
def get_session(session_id: str) -> Tuple[Dict[str, Any], int]:
    """Get a staging session with all metadata"""
    try:
        response = vs_service.get_session_response(session_id)
        
        if not response:
            return {'error': 'Session not found'}, 404
        
        return {
            'message': 'Session retrieved successfully',
            'session': response.model_dump()
        }, 200
    
    except Exception as e:
        return {'error': f'Error retrieving session: {str(e)}'}, 500


@virtual_staging_bp.route('/refine', methods=['POST'])
def refine_staging() -> Tuple[Dict[str, Any], int]:
    """
    Refine an existing staging with new parameters and uploaded image
    
    Form data (multipart/form-data):
    {
        "session_id": str,
        "image": file (uploaded image),
        "style": str,
        "furniture_theme": str,
        "color_scheme": str (optional),
        "specific_request": str (optional),
        "user_message": str (optional - user's refinement request for chat history)
    }
    """
    try:
        # Check if image file is present
        if 'image' not in request.files:
            return {'error': 'No image file provided'}, 400
        
        file = request.files['image']
        if file.filename == '':
            return {'error': 'No image file selected'}, 400
        
        # Validate and save image
        image_path = save_uploaded_file(file)
        if not image_path:
            return {'error': 'Invalid image format. Allowed: png, jpg, jpeg, gif, webp'}, 400
        
        # Get form data
        session_id = request.form.get('session_id', type=str)
        style = request.form.get('style', type=str)
        furniture_theme = request.form.get('furniture_theme', type=str)
        color_scheme = request.form.get('color_scheme', type=str)
        specific_request = request.form.get('specific_request', type=str)
        user_message = request.form.get('user_message', type=str)
        
        # Validate required fields
        if not all([session_id, style, furniture_theme]):
            return {'error': 'Missing required fields: session_id, style, furniture_theme'}, 400
        
        # Validate enums
        try:
            style_enum = StyleEnum(style)
            furniture_enum = FurnitureThemeEnum(furniture_theme)
        except ValueError as e:
            return {'error': f'Invalid enum value: {str(e)}'}, 400
        
        # Create new staging parameters
        new_params = StagingParameters(
            role=request.form.get('role', 'professional interior designer'),
            style=style_enum,
            furniture_theme=furniture_enum,
            color_scheme=color_scheme,
            specific_request=specific_request
        )
        
        # Refine staging with optional user message for chat history
        response = vs_service.refine_staging(
            session_id=session_id,
            new_staging_parameters=new_params,
            user_message=user_message
        )
        
        if not response:
            return {'error': 'Failed to refine staging'}, 500
        
        return {
            'message': 'Staging refined successfully',
            'image_url': response.image_url,
            'image_path': image_path,
            'version': response.version,
            'updated_at': response.updated_at.isoformat(),
            'prompt_used': response.prompt_used
        }, 200
    
    except Exception as e:
        return {'error': f'Error refining staging: {str(e)}'}, 500


@virtual_staging_bp.route('/property/<int:property_id>', methods=['GET'])
def get_property_sessions(property_id: int) -> Tuple[Dict[str, Any], int]:
    """Get all sessions for a property"""
    try:
        sessions = vs_service.get_sessions_by_property(property_id)
        
        return {
            'message': 'Sessions retrieved successfully',
            'property_id': property_id,
            'sessions': [
                {
                    'session_id': session_id,
                    'data': session.to_dict()
                }
                for session_id, session in sessions
            ]
        }, 200
    
    except Exception as e:
        return {'error': f'Error retrieving sessions: {str(e)}'}, 500


@virtual_staging_bp.route('/user/<int:user_id>', methods=['GET'])
def get_user_sessions(user_id: int) -> Tuple[Dict[str, Any], int]:
    """Get all sessions for a user"""
    try:
        sessions = vs_service.get_sessions_by_user(user_id)
        
        return {
            'message': 'Sessions retrieved successfully',
            'user_id': user_id,
            'sessions': [
                {
                    'session_id': session_id,
                    'data': session.to_dict()
                }
                for session_id, session in sessions
            ]
        }, 200
    
    except Exception as e:
        return {'error': f'Error retrieving sessions: {str(e)}'}, 500


@virtual_staging_bp.route('/session/<session_id>', methods=['DELETE'])
def delete_session(session_id: str) -> Tuple[Dict[str, Any], int]:
    """Delete a staging session"""
    try:
        success = vs_service.delete_session(session_id)
        
        if not success:
            return {'error': 'Failed to delete session'}, 500
        
        return {'message': f'Session {session_id} deleted successfully'}, 200
    
    except Exception as e:
        return {'error': f'Error deleting session: {str(e)}'}, 500


@virtual_staging_bp.route('/styles', methods=['GET'])
def get_available_styles() -> Tuple[Dict[str, Any], int]:
    """Get available style options"""
    styles = [style.value for style in StyleEnum]
    return {
        'message': 'Available styles retrieved successfully',
        'styles': styles
    }, 200


@virtual_staging_bp.route('/furniture-themes', methods=['GET'])
def get_available_furniture_themes() -> Tuple[Dict[str, Any], int]:
    """Get available furniture theme options"""
    themes = [theme.value for theme in FurnitureThemeEnum]
    return {
        'message': 'Available furniture themes retrieved successfully',
        'furniture_themes': themes
    }, 200


@virtual_staging_bp.route('/color-palettes', methods=['GET'])
def get_color_palettes() -> Tuple[Dict[str, Any], int]:
    """Get available color palette options"""
    from config.prompt_config import COLOR_PALETTES
    
    palettes = {
        hex_color: {
            'name': info['name'],
            'description': info['description'],
            'complement': info['complement']
        }
        for hex_color, info in COLOR_PALETTES.items()
    }
    
    return {
        'message': 'Color palettes retrieved successfully',
        'color_palettes': palettes
    }, 200


# ==================== VERSIONING OPERATIONS ====================
@virtual_staging_bp.route('/save-change', methods=['POST'])
def save_change() -> Tuple[Dict[str, Any], int]:
    """
    Save the current working image as a new version in history
    
    JSON body:
    {
        "session_id": str
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return {'error': 'No JSON data provided'}, 400
        
        session_id = data.get('session_id')
        if not session_id:
            return {'error': 'Missing required field: session_id'}, 400
        
        response = vs_service.save_change(session_id)
        
        if not response or not response.success:
            return {'error': response.message if response else 'Failed to save change'}, 500
        
        return {
            'message': response.message,
            'version': response.version,
            'image_url': response.image_url,
            'saved_at': response.saved_at.isoformat()
        }, 200
    
    except Exception as e:
        return {'error': f'Error saving change: {str(e)}'}, 500


@virtual_staging_bp.route('/session/<session_id>/save-change', methods=['POST'])
def save_session_change(session_id: str) -> Tuple[Dict[str, Any], int]:
    """
    Save the current working image as a new version in history and persist to AWS
    
    This is the RESTful endpoint that uses session_id from the URL path.
    It will:
    1. Persist the current working image to AWS S3
    2. Create a new version with versioning
    3. Update the session in Firestore
    4. Return the new version information
    
    URL path:
    /api/virtual-staging/session/<session_id>/save-change
    """
    try:
        # Validate session exists
        session = vs_service.get_session(session_id)
        if not session:
            return {'error': f'Session {session_id} not found'}, 404
        
        # Check if there's a current working image to save
        if not session.current_image_path or not os.path.exists(session.current_image_path):
            return {'error': 'No current working image to save for this session'}, 400
        
        # Save the change (persists to AWS and creates new version)
        response = vs_service.save_change(session_id)
        
        if not response or not response.success:
            return {'error': response.message if response else 'Failed to save change'}, 500

        return {
            'message': response.message,
            'session_id': session_id,
            'version': response.version,
            'image_url': response.image_url,
            'saved_at': response.saved_at.isoformat()
        }, 200
    
    except Exception as e:
        return {'error': f'Error saving session change: {str(e)}'}, 500


@virtual_staging_bp.route('/revert-change', methods=['POST'])
def revert_change() -> Tuple[Dict[str, Any], int]:
    """
    Revert to a previous saved version
    
    JSON body:
    {
        "session_id": str,
        "version_to_revert_to": int
    }
    """
    try:
        data = request.get_json()
        
        if not data:
            return {'error': 'No JSON data provided'}, 400
        
        session_id = data.get('session_id')
        version_to_revert_to = data.get('version_to_revert_to')
        
        if not session_id or version_to_revert_to is None:
            return {'error': 'Missing required fields: session_id, version_to_revert_to'}, 400
        
        response = vs_service.revert_change(session_id, version_to_revert_to)
        
        if not response or not response.success:
            return {'error': response.message if response else 'Failed to revert change'}, 500
        
        return {
            'message': response.message,
            'version': response.version,
            'image_url': response.image_url,
            'reverted_at': response.reverted_at.isoformat()
        }, 200
    
    except Exception as e:
        return {'error': f'Error reverting change: {str(e)}'}, 500


@virtual_staging_bp.route('/version-history/<session_id>', methods=['GET'])
def get_version_history(session_id: str) -> Tuple[Dict[str, Any], int]:
    """Get complete version history for a session"""
    try:
        response = vs_service.get_version_history(session_id)
        
        if not response:
            return {'error': 'Session not found'}, 404
        
        return {
            'message': 'Version history retrieved successfully',
            'session_id': response.session_id,
            'total_versions': response.total_versions,
            'current_version': response.current_version,
            'has_unsaved_changes': response.has_unsaved_changes,
            'versions': [v.model_dump() for v in response.versions]
        }, 200
    
    except Exception as e:
        return {'error': f'Error retrieving version history: {str(e)}'}, 500

@virtual_staging_bp.route('/chat-history/<session_id>', methods=['GET'])
def get_chat_history(session_id: str) -> Tuple[Dict[str, Any], int]:
    """
    Get the persistent chat history for a staging session
    
    Query parameters:
    - include_full_history (bool, optional): Include full conversation or just recent messages. Default: false
    - last_n_messages (int, optional): Number of recent messages to include. Default: 10
    """
    try:
        session = vs_service.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}, 404
        
        if not session.chat_history_id:
            return {'error': 'No chat history found for this session'}, 404
        
        # Get query parameters
        include_full = request.args.get('include_full_history', type=bool, default=False)
        last_n = request.args.get('last_n_messages', type=int, default=10)
        
        # Get chat history
        chat_history = vs_service.chat_history_service.get_history(session.chat_history_id)
        if not chat_history:
            return {'error': 'Chat history not found'}, 404
        
        # Get conversation summary
        summary = vs_service.chat_history_service.get_conversation_summary(session.chat_history_id)
        
        return {
            'message': 'Chat history retrieved successfully',
            'session_id': session_id,
            'chat_history_id': session.chat_history_id,
            'summary': summary,
            'context': vs_service.chat_history_service.get_llm_context(
                session.chat_history_id,
                include_full_history=include_full,
                last_n_messages=last_n
            )
        }, 200
    
    except Exception as e:
        return {'error': f'Error retrieving chat history: {str(e)}'}, 500


@virtual_staging_bp.route('/chat-history/<session_id>/messages', methods=['GET'])
def get_chat_messages(session_id: str) -> Tuple[Dict[str, Any], int]:
    """
    Get all messages from the chat history of a session
    
    Query parameters:
    - role (str, optional): Filter by message role ('user', 'assistant'). Default: all
    """
    try:
        session = vs_service.get_session(session_id)
        if not session:
            return {'error': 'Session not found'}, 404
        
        if not session.chat_history_id:
            return {'error': 'No chat history found for this session'}, 404
        
        # Get chat history
        chat_history = vs_service.chat_history_service.get_history(session.chat_history_id)
        if not chat_history:
            return {'error': 'Chat history not found'}, 404
        
        # Get optional role filter
        role_filter = request.args.get('role', type=str, default=None)
        
        # Filter messages if role is specified
        messages = chat_history.messages
        if role_filter:
            messages = [m for m in messages if m.role.value == role_filter]
        
        return {
            'message': 'Chat messages retrieved successfully',
            'session_id': session_id,
            'total_messages': len(messages),
            'messages': [
                {
                    'message_id': m.message_id,
                    'role': m.role.value,
                    'content': m.content,
                    'refinement_iteration': m.refinement_iteration,
                    'staging_parameters_used': m.staging_parameters_used,
                    'created_at': m.created_at.isoformat(),
                    'metadata': m.metadata
                }
                for m in messages
            ]
        }, 200
    
    except Exception as e:
        return {'error': f'Error retrieving chat messages: {str(e)}'}, 500