from config.aws_config import AWSConfig
from botocore.exceptions import ClientError
import uuid
from datetime import datetime
from io import BytesIO

class AWSService:
    """Service for the AWS S3 bucket"""

    @staticmethod
    def upload_file(file, folder: str = "") -> dict:
        """
        Upload a file to S3 bucket.
        
        Args:
            file: File object to upload (e.g., from Flask request.files)
            folder: Optional folder path within the bucket
            
        Returns:
            dict with success status and file URL or error message
        """
        try:
            # Generate unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            original_filename = file.filename
            extension = original_filename.rsplit('.', 1)[-1] if '.' in original_filename else ''
            new_filename = f"{timestamp}_{unique_id}.{extension}" if extension else f"{timestamp}_{unique_id}"
            
            # Build the S3 key (path)
            s3_key = f"{folder}/{new_filename}" if folder else new_filename
            
            # Upload to S3
            AWSConfig.s3.upload_fileobj(
                file,
                AWSConfig.AWS_S3_BUCKET,
                s3_key,
                ExtraArgs={"ContentType": file.content_type}
            )
            
            # Generate the file URL
            file_url = f"https://{AWSConfig.AWS_S3_BUCKET}.s3.{AWSConfig.AWS_REGION}.amazonaws.com/{s3_key}"
            
            return {
                "success": True,
                "url": file_url,
                "key": s3_key,
                "filename": new_filename
            }
        except ClientError as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def upload_bytes(image_bytes: bytes, filename: str, folder: str = "", content_type: str = "image/png") -> dict:
        """
        Upload image bytes directly to S3 bucket.
        
        Args:
            image_bytes: Image data as bytes
            filename: Desired filename (extension should be included)
            folder: Optional folder path within the bucket
            content_type: MIME type of the file (default: image/png)
            
        Returns:
            dict with success status, S3 key, and file URL or error message
        """
        try:
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            
            # Extract extension from filename
            extension = filename.rsplit('.', 1)[-1] if '.' in filename else 'png'
            new_filename = f"{timestamp}_{unique_id}.{extension}"
            
            # Build the S3 key (path)
            s3_key = f"{folder}/{new_filename}" if folder else new_filename
            
            # Convert bytes to BytesIO for upload
            file_obj = BytesIO(image_bytes)
            
            # Upload to S3
            AWSConfig.s3.upload_fileobj(
                file_obj,
                AWSConfig.AWS_S3_BUCKET,
                s3_key,
                ExtraArgs={"ContentType": content_type}
            )
            
            # Generate the file URL
            file_url = f"https://{AWSConfig.AWS_S3_BUCKET}.s3.{AWSConfig.AWS_REGION}.amazonaws.com/{s3_key}"
            
            return {
                "success": True,
                "url": file_url,
                "key": s3_key,
                "filename": new_filename
            }
        except ClientError as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def delete_file(s3_key: str) -> dict:
        """
        Delete a file from S3 bucket.
        
        Args:
            s3_key: The S3 key (path) of the file to delete
            
        Returns:
            dict with success status or error message
        """
        try:
            AWSConfig.s3.delete_object(
                Bucket=AWSConfig.AWS_S3_BUCKET,
                Key=s3_key
            )
            return {"success": True, "message": f"File '{s3_key}' deleted successfully"}
        except ClientError as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def get_file_url(s3_key: str, expiration: int = 3600) -> dict:
        """
        Generate a presigned URL to access a file.
        
        Args:
            s3_key: The S3 key (path) of the file
            expiration: URL expiration time in seconds (default: 1 hour)
            
        Returns:
            dict with success status and presigned URL or error message
        """
        try:
            url = AWSConfig.s3.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': AWSConfig.AWS_S3_BUCKET,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return {"success": True, "url": url}
        except ClientError as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def get_public_url(s3_key: str) -> dict:
        """
        Get the public (non-presigned) URL for a file.
        Note: The file must be publicly accessible for this URL to work.
        
        Args:
            s3_key: The S3 key (path) of the file
            
        Returns:
            dict with success status and public URL
        """
        url = f"https://{AWSConfig.AWS_S3_BUCKET}.s3.{AWSConfig.AWS_REGION}.amazonaws.com/{s3_key}"
        return {"success": True, "url": url}

    @staticmethod
    def list_files(prefix: str = "") -> dict:
        """
        List files in the S3 bucket.
        
        Args:
            prefix: Optional prefix to filter files (folder path)
            
        Returns:
            dict with success status and list of files or error message
        """
        try:
            response = AWSConfig.s3.list_objects_v2(
                Bucket=AWSConfig.AWS_S3_BUCKET,
                Prefix=prefix
            )
            
            files = []
            if 'Contents' in response:
                for obj in response['Contents']:
                    files.append({
                        "key": obj['Key'],
                        "size": obj['Size'],
                        "last_modified": obj['LastModified'].isoformat()
                    })
            
            return {"success": True, "files": files}
        except ClientError as e:
            return {"success": False, "error": str(e)}

    @staticmethod
    def file_exists(s3_key: str) -> bool:
        """
        Check if a file exists in the S3 bucket.
        
        Args:
            s3_key: The S3 key (path) of the file
            
        Returns:
            True if file exists, False otherwise
        """
        try:
            AWSConfig.s3.head_object(
                Bucket=AWSConfig.AWS_S3_BUCKET,
                Key=s3_key
            )
            return True
        except ClientError:
            return False
    
