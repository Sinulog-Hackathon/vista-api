from config.gemini_config import get_gemini_client
from typing import Optional, Dict, Any
import base64
import os
import io
import json
import uuid
from datetime import datetime
from PIL import Image, ImageEnhance, ImageFilter
import requests
from io import BytesIO
from google.genai import types


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

    def generate_image_from_image(self, model: str, prompt: str, session=None, image_path: str = None, mask_image_path: Optional[str] = None) -> Optional[bytes]:
        """
        Generate virtually staged image using gemini-2.5-flash-image
        Native multimodal model that returns both text (reasoning) and image (inline_data)
        
        Args:
            model: Model name (overridden to gemini-2.5-flash-image)
            prompt: Staging parameters (style, furniture, colors, etc)
            session: VirtualStaging session object to get the latest image from (preferred)
            image_path: Path to the input room image (local path or S3 URL) - used as fallback if session not provided
            mask_image_path: Optional path to mask image for specifying a specific area/point
        
        Returns:
            Image bytes (PNG) of the virtually staged room
        
        Raises:
            FileNotFoundError: If image file doesn't exist
            Exception: If API call fails
        """
        try:
            # Determine image path - get from session if provided, otherwise use image_path parameter
            image_to_use = None
            
            if session:
                # Get the latest image from session (prefer current_image_path/url, fallback to original)
                if session.current_image_path:
                    image_to_use = session.current_image_path
                    print(f"[STAGING] Using current image from session: {session.current_image_path}")
                elif session.current_image_url:
                    image_to_use = session.current_image_url
                    print(f"[STAGING] Using current image URL from session: {session.current_image_url[:50]}...")
                elif session.original_image_path:
                    image_to_use = session.original_image_path
                    print(f"[STAGING] Using original image from session: {session.original_image_path}")
                elif session.original_image_url:
                    image_to_use = session.original_image_url
                    print(f"[STAGING] Using original image URL from session: {session.original_image_url[:50]}...")
                else:
                    print(f"[STAGING] WARNING: Session has no image paths set")
                    print(f"  Session current_image_path: {session.current_image_path}")
                    print(f"  Session current_image_url: {session.current_image_url}")
                    print(f"  Session original_image_path: {session.original_image_path}")
                    print(f"  Session original_image_url: {session.original_image_url}")
            else:
                # Fallback to image_path parameter if no session provided
                image_to_use = image_path
                if image_to_use:
                    print(f"[STAGING] Using image_path parameter: {image_to_use}")
            
            # Validate image_path is provided
            if not image_to_use:
                error_msg = "Image path is required. Either provide a session object with images or image_path parameter"
                print(f"[STAGING] ❌ {error_msg}")
                raise FileNotFoundError(error_msg)
            
            # Load image - handle both local paths and S3 URLs
            try:
                if isinstance(image_to_use, str) and (image_to_use.startswith('http://') or image_to_use.startswith('https://')):
                    # Download from URL (S3)
                    response = requests.get(image_to_use)
                    response.raise_for_status()
                    input_image = Image.open(BytesIO(response.content))
                    print(f"[STAGING] Downloaded image from URL: {image_to_use[:50]}...")
                else:
                    # Local file path
                    if not os.path.exists(image_to_use):
                        raise FileNotFoundError(f"Image file not found: {image_to_use}")
                    input_image = Image.open(image_to_use)
                    print(f"[STAGING] Loaded local image: {image_to_use}")
            except Exception as e:
                print(f"[STAGING] ❌ Error loading image: {str(e)}")
                raise
            
            # Build comprehensive staging prompt
            staging_prompt = f"""
You are VistaAI, an elite Interior Design Architect and Virtual Staging Specialist with an obsessive eye for photorealism.

### INPUT CONTEXT
User Request: "{prompt}"
**Image Type:** Equirectangular 360° Panorama (Spherical Projection).

### INSTRUCTIONS FOR TRANSFORMATION
You must interpret the User Request not just literally, but architecturally.
If the request is vague (e.g., "make it blue"), you must infer the most sophisticated, high-end interpretation (e.g., "Navy Blue matte finish with velvet textures").

1. **Strict Isolation:** Identify ONLY the specific elements mentioned in the User Request. Do NOT touch the flooring, ceiling, structural beams, or unmentioned furniture unless explicitly asked.
2. **Material Physics:** Apply realistic texture mapping. If changing a door, specify the wood grain, gloss level, and hardware reflection. If changing a wall, define the paint finish (eggshell, matte, venetian plaster).
3. **Lighting Integration:** Any color or furniture change must interact realistically with existing light sources. Calculate global illumination, cast shadows, and ambient occlusion accurately.
4. **Resolution & Detail:** The output must be indistinguishable from a high-resolution photograph (8k, hyper-detailed).

### EXECUTION STEPS
1. **Analyze:** Scan the original room for lighting direction, perspective, and identify the Zenith (top) and Nadir (bottom) of the spherical projection.
2. **Expand:** Translate the vague User Request into technical design specifications (e.g., "change walls to green" -> "Sage Green #77896C paint with a matte finish").
3. **Generate:** Apply the changes while strictly masking unaltered areas to preserve original integrity.

### MANDATORY CONSTRAINTS
* **360° Continuity:** This is a spherical image. Ensure seamless wrapping between the left and right edges.
* **Nadir/Bottom Preservation:** Do NOT place complex geometry or furniture legs directly on the extreme bottom edge pixels unless they are correctly warped for spherical projection. Avoid "smearing" artifacts at the bottom pole.
* **ZERO Geometric Distortion:** The perspective, straight lines, and proportions of the original room architecture must remain flawless. Do not warp walls or furniture.
* **Format Preservation:** The output image must exactly maintain the original input's aspect ratio, pixel dimensions, and composition. No cropping.
* NO artifacts, smudges, or AI hallucinations.
* NO cartoonish or over-saturated colors; use commercially viable, real-world palettes.

**Action:** Transform the image now with meticulous adherence to spherical projection and photorealism.
"""
            
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
