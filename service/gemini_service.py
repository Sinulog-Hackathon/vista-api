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
6. STRICT ADHERENCE: Do NOT add any unrequested furniture or decor. Only include exactly what is specified in the requirements above.
7. DO NOT OVERRIDE the room structure or architecture.
8. Provide a brief explanation of the changes made in the image.

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

    def extract_furniture_list_from_staging(self, 
                                           session_id: str, 
                                           version: int, 
                                           image_source: str, 
                                           style: str, 
                                           furniture_theme: str) -> Optional[Any]:
        """
        Extract furniture items from a staged image and find exact matches in Philippine retailers.
        STRICT MODE: Only extracts furniture that is ACTUALLY VISIBLE in the image.
        
        Args:
            session_id: Session ID for reference
            version: Version number
            image_source: Path to image (local or S3 URL)
            style: Interior style
            furniture_theme: Furniture theme
            
        Returns:
            FurnitureList object with extracted items, or None if extraction fails
        """
        try:
            # Load image
            if isinstance(image_source, str) and (image_source.startswith('http://') or image_source.startswith('https://')):
                response = requests.get(image_source)
                response.raise_for_status()
                image = Image.open(BytesIO(response.content))
                print(f"[FURNITURE] Downloaded image from URL for extraction")
            else:
                if not os.path.exists(image_source):
                    raise FileNotFoundError(f"Image file not found: {image_source}")
                image = Image.open(image_source)
            
            # Build strict furniture extraction prompt
            extraction_prompt = """You are an expert Furniture Sourcing Agent for "Vista." Your goal is to identify furniture ACTUALLY VISIBLE in the staged image and find EXACT or CLOSE MATCHES available for purchase in the Philippines.

STRICT EXTRACTION RULES:
1. ONLY extract furniture that is CLEARLY VISIBLE in the image. Do NOT invent or assume furniture.
2. VISUAL ANALYSIS: Identify shape, material, color, leg style, and texture from the image.
3. PHILIPPINE RETAILERS: Search for matches in Mandaue Foam, Blims, Crate & Barrel PH, BoConcept PH, HomeU, Lazada PH, Shopee PH.
4. VERIFICATION: Only include furniture with verified, active product pages and real prices.
5. HONESTY: If no good match exists for an item, DO NOT include it.
6. FORMAT: Return a valid JSON array (not wrapped in markdown code blocks).

For each visible furniture piece, return:
{
  "product_name": "Exact product name from retailer",
  "retailer": "Retailer name",
  "price": "Price with currency (e.g., ₱15,500)",
  "url": "Verified direct product URL",
  "visual_reasoning": "Why this matches the visible item"
}

CRITICAL: Return ONLY valid JSON array. No explanations, no markdown formatting. Start with [ and end with ].
If no furniture is found or visible, return: []"""

            print("[FURNITURE] Analyzing image for furniture extraction...")
            
            # Call Gemini to analyze and extract furniture
            response = self.client.models.generate_content(
            model="gemini-3-pro-preview",
            contents=[extraction_prompt, image],
            config=types.GenerateContentConfig(
                tools=[types.Tool(google_search=types.GoogleSearch())]
            )
        )
            
            response_text = response.text.strip() if hasattr(response, 'text') else ""
            print(f"[FURNITURE] Raw response: {response_text[:200]}...")
            
            # Clean the response - remove markdown code blocks if present
            if response_text.startswith("```"):
                response_text = response_text.split("```")[1]
                if response_text.startswith("json"):
                    response_text = response_text[4:]
            
            response_text = response_text.strip()
            
            furniture_data = json.loads(response_text)
            
            if not isinstance(furniture_data, list):
                print(f"[FURNITURE] Invalid response format: expected list, got {type(furniture_data)}")
                return None
            
            print(f"[FURNITURE] Extracted {len(furniture_data)} furniture items")
            
            # Create FurnitureList
            from models.furniture import FurnitureList, FurnitureItem, SourcingInfo
            
            furniture_list = FurnitureList(
                furniture_id=f"furniture_{session_id}_{version}_{uuid.uuid4().hex[:8]}",
                session_id=session_id,
                currency="PHP",
                location_scope="Philippines"
            )
            
            total_cost = 0
            
            # Process each furniture item
            for idx, item_data in enumerate(furniture_data):
                try:
                    # Extract price as number
                    price_str = item_data.get('price', '0')
                    # Remove currency symbols and commas, keep only digits
                    price_num = float(''.join(filter(lambda x: x.isdigit() or x == '.', price_str)))
                    
                    # Create sourcing info
                    sourcing_info = SourcingInfo(
                        match_type="Exact Match" if "exact" in item_data.get('visual_reasoning', '').lower() else "Closest Available Equivalent",
                        source_shop_name=item_data.get('retailer', 'Unknown'),
                        shop_url=item_data.get('url', ''),
                        notes=item_data.get('visual_reasoning', '')
                    )
                    
                    # Create furniture item (infer category from visual_reasoning or use generic)
                    furniture_item = FurnitureItem(
                        item_id=f"item_{uuid.uuid4().hex[:8]}",
                        name=item_data.get('product_name', 'Unknown'),
                        category="Furniture",  # Default category
                        description=item_data.get('visual_reasoning', ''),
                        estimated_cost=price_num,
                        estimated_cost_currency="PHP",
                        quantity=1,
                        sourcing_info=sourcing_info,
                        notes=f"Sourced from {item_data.get('retailer', 'Unknown')}"
                    )
                    
                    furniture_list.add_item(furniture_item)
                    total_cost += price_num
                    print(f"[FURNITURE] Added: {furniture_item.name} - ₱{price_num}")
                    
                except Exception as e:
                    print(f"[FURNITURE] Error processing item {idx}: {str(e)}")
                    continue
            
            print(f"[FURNITURE] Total furniture cost: ₱{total_cost}")
            
            # Return even if empty - empty list is valid (no furniture found/visible)
            print(f"[FURNITURE] Returning furniture list with {len(furniture_list.furniture_items)} items")
            return furniture_list
            
        except json.JSONDecodeError as e:
            print(f"[FURNITURE] Failed to parse JSON response: {str(e)}")
            return None
        except Exception as e:
            print(f"[FURNITURE] Error extracting furniture: {str(e)}")
            return None

    def get_furniture_simplified(self, furniture_list: Any) -> Optional[Dict]:
        """
        Convert FurnitureList to simplified response format with only:
        - product_name
        - price (with currency)
        - url
        - total_price
        
        Args:
            furniture_list: FurnitureList object to convert
            
        Returns:
            Dictionary with simplified furniture info or None
        """
        try:
            if not furniture_list or len(furniture_list.furniture_items) == 0:
                return None
            
            from models.virtual_staging_response import SimplifiedFurnitureListResponse, SimplifiedFurnitureItemResponse
            
            simplified_items = []
            total_cost = 0
            
            for item in furniture_list.furniture_items:
                # Format price with currency
                price_str = f"₱{item.estimated_cost:,.2f}" if item.estimated_cost else "Price not available"
                
                # Get URL from sourcing info
                url = ""
                if item.sourcing_info and item.sourcing_info.shop_url:
                    url = item.sourcing_info.shop_url
                
                simplified_item = SimplifiedFurnitureItemResponse(
                    product_name=item.name,
                    price=price_str,
                    url=url
                )
                simplified_items.append(simplified_item)
                
                if item.estimated_cost:
                    total_cost += item.estimated_cost * item.quantity
            
            # Format total price
            total_price_str = f"₱{total_cost:,.2f}"
            
            response = SimplifiedFurnitureListResponse(
                furniture_items=simplified_items,
                total_price=total_price_str
            )
            
            print(f"[FURNITURE] Simplified response: {len(simplified_items)} items, Total: {total_price_str}")
            return response.model_dump()
            
        except Exception as e:
            print(f"[FURNITURE] Error creating simplified response: {str(e)}")
            return None

    def get_furniture_by_session_id(self, session_id: str) -> Optional[Dict]:
        """
        Retrieve furniture list for a specific session (requires database query via repository)
        This is a helper method for the service layer.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dictionary with simplified furniture info or None
        """
        try:
            # This method assumes the repository is available through the service layer
            # The virtual_staging_service will call this indirectly via the session's furniture list
            print(f"[FURNITURE] Retrieving furniture for session: {session_id}")
            return None
        except Exception as e:
            print(f"[FURNITURE] Error retrieving furniture: {str(e)}")
            return None