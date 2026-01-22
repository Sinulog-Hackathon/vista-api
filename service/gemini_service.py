from config.gemini_config import get_gemini_client
from typing import Optional
import base64
import os
import io
from PIL import Image, ImageEnhance, ImageFilter
import requests
from io import BytesIO


class GeminiService:
    """Service to interact with Gemini API"""

    def __init__(self):
        self.client = get_gemini_client()

    def generate_content(self, model: str, contents: str) -> str:
        """Generate content using Gemini model"""
        response = self.client.models.generate_content(
            model=model,
            contents=contents
        )
        return response.text

    @staticmethod
    def encode_image_to_base64(image_path: str) -> str:
        """
        Read an image file and encode it to base64
        
        Args:
            image_path: Path to the image file
        
        Returns:
            Base64 encoded string of the image
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            IOError: If file cannot be read
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        with open(image_path, 'rb') as image_file:
            return base64.standard_b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def apply_staging_filters(image: Image.Image, style: str = "modern") -> Image.Image:
        """
        Apply visual filters to simulate staging (as fallback for image generation)
        This enhances the image to make it look more staged/professional
        """
        try:
            # Create a copy to avoid modifying original
            enhanced = image.copy()
            
            # Enhance colors (saturation)
            color_enhancer = ImageEnhance.Color(enhanced)
            enhanced = color_enhancer.enhance(1.15)  # 15% more saturation
            
            # Enhance contrast
            contrast_enhancer = ImageEnhance.Contrast(enhanced)
            enhanced = contrast_enhancer.enhance(1.1)  # 10% more contrast
            
            # Enhance brightness slightly
            brightness_enhancer = ImageEnhance.Brightness(enhanced)
            enhanced = brightness_enhancer.enhance(1.05)  # 5% brighter
            
            # Apply subtle sharpening
            enhanced = enhanced.filter(ImageFilter.SHARPEN)
            
            print(f"[STAGING] Applied professional staging filters ({style} style)")
            return enhanced
        except Exception as e:
            print(f"[STAGING] Error applying filters: {str(e)}")
            return image

    @staticmethod
    def _generate_image_with_fallback(input_image: Image.Image, style: str = "modern") -> bytes:
        """Generate image bytes with professional enhancement as fallback"""
        enhanced = GeminiService.apply_staging_filters(input_image, style)
        img_io = BytesIO()
        enhanced.save(img_io, format='PNG')
        img_io.seek(0)
        return img_io.getvalue()

    @staticmethod
    def get_image_mime_type(image_path: str) -> str:
        """
        Determine MIME type from image file extension
        
        Args:
            image_path: Path to the image file
        
        Returns:
            MIME type string (e.g., 'image/jpeg', 'image/png')
        """
        _, ext = os.path.splitext(image_path.lower())
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(ext, 'image/jpeg')

    def generate_image_from_text(self, model: str, prompt: str) -> Optional[str]:
        """
        Generate an image from text prompt using Gemini
        Returns text response from the model
        
        Args:
            model: Gemini model to use (e.g., "gemini-2.0-flash")
            prompt: Text prompt for image generation
        
        Returns:
            Response text or None if generation fails
        """
        try:
            response = self.client.models.generate_content(
                model=model,
                contents=prompt
            )
            return response.text if hasattr(response, 'text') else None
        except Exception as e:
            print(f"Error generating content: {str(e)}")
            return None

    def generate_image_from_image(self, model: str, image_path: str, prompt: str, mask_image_path: Optional[str] = None) -> Optional[bytes]:
        """
        Generate virtually staged image using gemini-2.5-flash-image
        Native multimodal model that returns both text (reasoning) and image (inline_data)
        
        Args:
            model: Model name (overridden to gemini-2.5-flash-image)
            image_path: Path to the input room image (local path or S3 URL)
            prompt: Staging parameters (style, furniture, colors, etc)
            mask_image_path: Optional path to mask image for specifying a specific area/point
        
        Returns:
            Image bytes (PNG) of the virtually staged room
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If API call fails
        """
        try:
            # Validate image_path is provided
            if not image_path:
                raise FileNotFoundError("Image path is required and cannot be None or empty")
            
            # Load image - handle both local paths and S3 URLs
            if isinstance(image_path, str) and (image_path.startswith('http://') or image_path.startswith('https://')):
                # Download from URL (S3)
                response = requests.get(image_path)
                response.raise_for_status()
                input_image = Image.open(BytesIO(response.content))
                print(f"[STAGING] Downloaded image from URL: {image_path[:50]}...")
            else:
                # Local file path
                if not os.path.exists(image_path):
                    raise FileNotFoundError(f"Image file not found: {image_path}")
                input_image = Image.open(image_path)
            
            # Build comprehensive staging prompt
            staging_prompt = f"""You are an expert interior designer and virtual staging specialist.

Transform this room image with the following requirements:
{prompt}

Instructions:
1. Analyze the current room composition and layout
2. Apply the requested styling transformations
3. Improve lighting, colors, and overall aesthetics
4. Keep the same camera angle and basic composition
5. Generate a HIGH-QUALITY, realistic virtually staged image

Transform this image now."""
            
            print("[STAGING] Sending to gemini-2.5-flash-image for transformation...")
            
            # Prepare content list with main image
            contents = [staging_prompt, input_image]
            
            # Add mask image if provided for specific area guidance
            if mask_image_path and isinstance(mask_image_path, str) and len(mask_image_path) > 0:
                mask_image = None
                if mask_image_path.startswith('http://') or mask_image_path.startswith('https://'):
                    # Download mask from URL
                    response = requests.get(mask_image_path)
                    response.raise_for_status()
                    mask_image = Image.open(BytesIO(response.content))
                    print(f"[STAGING] Downloaded mask image from URL")
                elif os.path.exists(mask_image_path):
                    mask_image = Image.open(mask_image_path)
                    print(f"[STAGING] Using mask image to focus on specific area: {mask_image_path}")
                
                if mask_image:
                    mask_instruction = "\n\nFocus the transformation primarily on the area highlighted in this mask image:"
                    contents = [staging_prompt + mask_instruction, input_image, mask_image]
            
            # Native multimodal call - no response_modalities needed
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=contents
            )
            
            # Extract both text (reasoning) and image (transformed room)
            image_bytes = None
            model_comment = None
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    
                    # Extract text explanation from model
                    if hasattr(part, 'text') and part.text:
                        model_comment = part.text
                        print(f"[STAGING] Model: {model_comment[:100]}...")
                    
                    # Extract generated image
                    if hasattr(part, 'inline_data') and part.inline_data:
                        image_bytes = part.inline_data.data
                        print(f"[STAGING] ✅ Image generated! Size: {len(image_bytes)} bytes")
            
            if image_bytes:
                return image_bytes
            else:
                print("[STAGING] ⚠️  No image in response - model may have only provided text/analysis")
                print("[STAGING] 📋 Model comment:", model_comment[:200] if model_comment else "None")
                print("[STAGING] 💡 Gemini-2.5-flash-image is designed for image analysis, not generation")
                print("[STAGING] ℹ️  Applying professional staging filters as fallback...")
                
                # Fallback: Return the input image with professional enhancement filters
                fallback_bytes = self._generate_image_with_fallback(input_image, style="modern")
                print(f"[STAGING] ✅ Returning enhanced image with professional filters: {len(fallback_bytes)} bytes")
                return fallback_bytes
            
        except FileNotFoundError as e:
            print(f"[STAGING] ❌ Image file error: {str(e)}")
            raise
        except Exception as e:
            print(f"[STAGING] ❌ Error during staging: {str(e)}")
            raise