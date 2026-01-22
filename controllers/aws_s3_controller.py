"""
AWS S3 Controller - File management endpoints
"""
from flask import Blueprint, request, jsonify, send_file
from typing import Tuple, Dict, Any
from service.aws_service import AWSService
from datetime import datetime
import io

# Create blueprint
aws_s3_bp = Blueprint('aws_s3', __name__, url_prefix='/api/aws-s3')

# Initialize service
s3_service = AWSService()


@aws_s3_bp.route('/upload', methods=['POST'])
def upload_file() -> Tuple[Dict[str, Any], int]:
    """
    Upload a file to AWS S3
    
    Form data (multipart/form-data):
    {
        "file": file (required),
        "folder": str (optional - S3 folder path, e.g., "staging/session123"),
        "filename": str (optional - custom filename, otherwise uses original)
    }
    """
    try:
        if 'file' not in request.files:
            return {'error': 'No file provided'}, 400
        
        file = request.files['file']
        if file.filename == '':
            return {'error': 'No file selected'}, 400
        
        folder = request.form.get('folder', type=str, default='uploads')
        custom_filename = request.form.get('filename', type=str)
        
        # Read file bytes
        file_bytes = file.read()
        
        # Determine filename
        filename = custom_filename if custom_filename else file.filename
        
        # Upload to AWS
        result = s3_service.upload_bytes(
            image_bytes=file_bytes,
            filename=filename,
            folder=folder,
            content_type=file.content_type or 'application/octet-stream'
        )
        
        if not result.get('success'):
            return {'error': f"Upload failed: {result.get('error')}"}, 500
        
        return {
            'message': 'File uploaded successfully',
            'filename': result.get('filename'),
            'key': result.get('key'),
            'url': result.get('url'),
            'size_bytes': len(file_bytes),
            'uploaded_at': datetime.utcnow().isoformat()
        }, 201
    
    except Exception as e:
        return {'error': f'Error uploading file: {str(e)}'}, 500


@aws_s3_bp.route('/list/<path:folder>', methods=['GET'])
def list_files(folder: str) -> Tuple[Dict[str, Any], int]:
    """
    List all files in an S3 folder
    
    URL params:
    - folder: S3 folder path (e.g., "staging/session123")
    """
    try:
        files = s3_service.list_files(folder)
        
        if files is None:
            return {'error': f'Failed to list files in folder: {folder}'}, 500
        
        return {
            'message': 'Files listed successfully',
            'folder': folder,
            'count': len(files),
            'files': files
        }, 200
    
    except Exception as e:
        return {'error': f'Error listing files: {str(e)}'}, 500


@aws_s3_bp.route('/download/<path:key>', methods=['GET'])
def download_file(key: str) -> Any:
    """
    Download a file from S3
    
    URL params:
    - key: S3 object key/path (e.g., "staging/session123/image.png")
    """
    try:
        file_obj = s3_service.get_file(key)
        
        if file_obj is None:
            return {'error': f'File not found: {key}'}, 404
        
        # Get the original filename from key
        filename = key.split('/')[-1]
        
        return send_file(
            file_obj,
            as_attachment=True,
            download_name=filename
        )
    
    except Exception as e:
        return {'error': f'Error downloading file: {str(e)}'}, 500


@aws_s3_bp.route('/delete/<path:key>', methods=['DELETE'])
def delete_file(key: str) -> Tuple[Dict[str, Any], int]:
    """
    Delete a file from S3
    
    URL params:
    - key: S3 object key/path (e.g., "staging/session123/image.png")
    """
    try:
        success = s3_service.delete_file(key)
        
        if not success:
            return {'error': f'Failed to delete file: {key}'}, 500
        
        return {
            'message': f'File deleted successfully: {key}',
            'key': key,
            'deleted_at': datetime.utcnow().isoformat()
        }, 200
    
    except Exception as e:
        return {'error': f'Error deleting file: {str(e)}'}, 500


@aws_s3_bp.route('/exists/<path:key>', methods=['GET'])
def check_file_exists(key: str) -> Tuple[Dict[str, Any], int]:
    """
    Check if a file exists in S3
    
    URL params:
    - key: S3 object key/path (e.g., "staging/session123/image.png")
    """
    try:
        exists = s3_service.file_exists(key)
        
        return {
            'message': 'File existence check completed',
            'key': key,
            'exists': exists
        }, 200
    
    except Exception as e:
        return {'error': f'Error checking file: {str(e)}'}, 500


@aws_s3_bp.route('/url/<path:key>', methods=['GET'])
def get_file_url(key: str) -> Tuple[Dict[str, Any], int]:
    """
    Get the public URL of a file in S3
    
    URL params:
    - key: S3 object key/path (e.g., "staging/session123/image.png")
    """
    try:
        url = s3_service.get_public_url(key)
        
        if not url:
            return {'error': f'Failed to generate URL for: {key}'}, 500
        
        return {
            'message': 'URL generated successfully',
            'key': key,
            'url': url
        }, 200
    
    except Exception as e:
        return {'error': f'Error generating URL: {str(e)}'}, 500


@aws_s3_bp.route('/info/<path:key>', methods=['GET'])
def get_file_info(key: str) -> Tuple[Dict[str, Any], int]:
    """
    Get metadata/info about a file in S3
    
    URL params:
    - key: S3 object key/path (e.g., "staging/session123/image.png")
    """
    try:
        exists = s3_service.file_exists(key)
        
        if not exists:
            return {'error': f'File not found: {key}'}, 404
        
        # Get object metadata from S3
        s3_client = s3_service.s3_client
        try:
            response = s3_client.head_object(Bucket=s3_service.bucket_name, Key=key)
            
            return {
                'message': 'File info retrieved successfully',
                'key': key,
                'exists': True,
                'size_bytes': response.get('ContentLength', 0),
                'content_type': response.get('ContentType', 'application/octet-stream'),
                'last_modified': response.get('LastModified', '').isoformat() if response.get('LastModified') else None,
                'etag': response.get('ETag', '')
            }, 200
        except Exception as metadata_error:
            # Fallback if head_object fails
            return {
                'message': 'File exists but metadata unavailable',
                'key': key,
                'exists': True,
                'error': str(metadata_error)
            }, 200
    
    except Exception as e:
        return {'error': f'Error getting file info: {str(e)}'}, 500


@aws_s3_bp.route('/batch/delete', methods=['POST'])
def batch_delete_files() -> Tuple[Dict[str, Any], int]:
    """
    Delete multiple files from S3
    
    JSON body:
    {
        "keys": [
            "staging/session1/image1.png",
            "staging/session1/image2.png"
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'keys' not in data:
            return {'error': 'Missing required field: keys'}, 400
        
        keys = data.get('keys', [])
        
        if not isinstance(keys, list) or len(keys) == 0:
            return {'error': 'keys must be a non-empty array'}, 400
        
        deleted_keys = []
        failed_keys = []
        
        for key in keys:
            try:
                if s3_service.delete_file(key):
                    deleted_keys.append(key)
                else:
                    failed_keys.append({'key': key, 'error': 'Delete returned False'})
            except Exception as e:
                failed_keys.append({'key': key, 'error': str(e)})
        
        return {
            'message': f'Batch delete completed: {len(deleted_keys)} successful, {len(failed_keys)} failed',
            'deleted': deleted_keys,
            'failed': failed_keys,
            'total_requested': len(keys)
        }, 200
    
    except Exception as e:
        return {'error': f'Error in batch delete: {str(e)}'}, 500


@aws_s3_bp.route('/batch/info', methods=['POST'])
def batch_get_file_info() -> Tuple[Dict[str, Any], int]:
    """
    Get info about multiple files from S3
    
    JSON body:
    {
        "keys": [
            "staging/session1/image1.png",
            "staging/session1/image2.png"
        ]
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'keys' not in data:
            return {'error': 'Missing required field: keys'}, 400
        
        keys = data.get('keys', [])
        
        if not isinstance(keys, list) or len(keys) == 0:
            return {'error': 'keys must be a non-empty array'}, 400
        
        file_infos = []
        
        for key in keys:
            try:
                exists = s3_service.file_exists(key)
                if exists:
                    s3_client = s3_service.s3_client
                    response = s3_client.head_object(Bucket=s3_service.bucket_name, Key=key)
                    file_infos.append({
                        'key': key,
                        'exists': True,
                        'size_bytes': response.get('ContentLength', 0),
                        'content_type': response.get('ContentType', 'application/octet-stream'),
                        'last_modified': response.get('LastModified', '').isoformat() if response.get('LastModified') else None
                    })
                else:
                    file_infos.append({'key': key, 'exists': False})
            except Exception as e:
                file_infos.append({'key': key, 'exists': False, 'error': str(e)})
        
        return {
            'message': 'Batch file info retrieved',
            'count': len(file_infos),
            'files': file_infos
        }, 200
    
    except Exception as e:
        return {'error': f'Error in batch get info: {str(e)}'}, 500
