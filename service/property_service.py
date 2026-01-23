"""Service layer for Property entity"""
import re
import uuid
from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from models.property import Property, PropertyImage, NearbyEstablishment
from repositories.property_repository import PropertyRepository
from service.aws_service import AWSService


class ValidationError(Exception):
    def __init__(self, message: str, field: str = None):
        self.message = message
        self.field = field
        super().__init__(message)


class PropertyService:
    """Business logic for property management"""

    def __init__(self):
        self.repository = PropertyRepository()
        self.aws_service = AWSService()

    def create_property(self, data: Dict[str, Any], user_id: str, files: Dict[str, Any] = None) -> Tuple[str, Property, List[PropertyImage]]:
        """
        Create a new property from form data

        Args:
            data: Form data dictionary
            user_id: User creating the property
            files: Uploaded files (regularImages, panoramicImages)

        Returns:
            Tuple of (property_id, Property, List of uploaded images)
        """
        # Validate required fields
        self._validate_required_fields(data)

        # Parse and validate data
        property_data = self._parse_property_data(data, user_id)

        # Generate property ID
        property_id = f"prop_{int(datetime.utcnow().timestamp() * 1000)}_{str(uuid.uuid4())[:9]}"

        # Handle image uploads
        images = []
        main_image = None
        if files:
            images = self._handle_image_uploads(files, property_id)
            # Handle main image separately if provided
            if 'image' in files and files['image']:
                main_image_result = self.aws_service.upload_property_image(files['image'][0], property_id, 'regular')
                if main_image_result.get('success', True):
                    main_image = PropertyImage(**main_image_result)
                else:
                    raise ValidationError(f"Failed to upload main image: {main_image_result.get('error', 'Unknown error')}")

        # Create property model
        property_model = Property(
            **property_data,
            images=images,
            regularImageCount=len([img for img in images if img.imageType == "regular"]),
            panoramicImageCount=len([img for img in images if img.imageType == "panoramic"]),
            image=main_image  # Main thumbnail image
        )

        # Save to repository
        created_property = self.repository.create_property(property_model, property_id)

        return property_id, created_property, images

    def get_property(self, property_id: str) -> Optional[Property]:
        """Get property by ID"""
        return self.repository.get_property(property_id)

    def get_properties_by_user(self, user_id: str) -> List[Tuple[str, Property]]:
        """Get all properties for a user"""
        return self.repository.get_properties_by_user(user_id)

    def get_all_properties(self) -> List[Tuple[str, Property]]:
        """Get all properties"""
        return self.repository.get_all_properties()

    def update_property(self, property_id: str, data: Dict[str, Any], user_id: str) -> Property:
        """
        Update an existing property

        Args:
            property_id: Property ID
            data: Updated data
            user_id: User making the update

        Returns:
            Updated Property
        """
        property_model = self.repository.get_property(property_id)
        if not property_model:
            raise ValidationError("Property not found")

        if property_model.createdBy != user_id:
            raise ValidationError("Unauthorized to update this property")

        # Parse and validate data
        update_data = self._parse_property_data(data, user_id, is_update=True)
        update_data['updatedAt'] = datetime.utcnow()

        # Update model
        for key, value in update_data.items():
            if hasattr(property_model, key):
                setattr(property_model, key, value)

        return self.repository.update_property(property_id, property_model)

    def get_property(self, property_id: str) -> Optional[Property]:
        """
        Get property by ID

        Args:
            property_id: Property ID string

        Returns:
            Property model or None
        """
        property_model = self.repository.get_property(property_id)
        if property_model:
            # Set the propertyId field
            property_model.propertyId = property_id
        return property_model

    def delete_property(self, property_id: str, user_id: str) -> bool:
        """
        Delete a property

        Args:
            property_id: Property ID
            user_id: User deleting the property

        Returns:
            True if deleted
        """
        property_model = self.repository.get_property(property_id)
        if not property_model:
            raise ValidationError("Property not found")

        if property_model.createdBy != user_id:
            raise ValidationError("Unauthorized to delete this property")

        return self.repository.delete_property(property_id)

    def get_property_types(self) -> Dict[str, List[str]]:
        """Get available property and listing types"""
        return {
            "propertyTypes": ["House", "Condo", "Apartment", "Lot", "Commercial"],
            "listingTypes": ["For Sale", "For Rent", "For Lease"]
        }

    def get_amenities_options(self) -> Dict[str, List[str]]:
        """Get available amenities options"""
        return {
            "amenities": ["Swimming Pool", "Gym", "Security (24/7)", "Garden", "Parking", "Elevator"],
            "interiorFeatures": ["Air-conditioning", "Built-in cabinets", "Fireplace", "Hardwood floors"],
            "utilities": ["Water", "Electricity", "Internet readiness", "Cable TV", "Gas"]
        }

    def _validate_required_fields(self, data: Dict[str, Any]):
        """Validate required fields"""
        required_fields = ["name", "propertyType", "listingType", "address", "price"]
        for field in required_fields:
            if not data.get(field):
                raise ValidationError(f"{field} is required", field)

        # Validate price
        try:
            price = float(data.get("price", 0))
            if price <= 0:
                raise ValidationError("Price must be greater than 0", "price")
        except (ValueError, TypeError):
            raise ValidationError("Invalid price format", "price")

        # Validate email if provided
        if data.get("agentEmail"):
            if not self._is_valid_email(data["agentEmail"]):
                raise ValidationError("Invalid agent email format", "agentEmail")

        # Validate developer email if provided
        if data.get("developerEmail"):
            if not self._is_valid_email(data["developerEmail"]):
                raise ValidationError("Invalid developer email format", "developerEmail")

        # Validate URLs
        if data.get("developerWebsite"):
            if not self._is_valid_url(data["developerWebsite"]):
                raise ValidationError("Invalid developer website URL", "developerWebsite")

    def _parse_property_data(self, data: Dict[str, Any], user_id: str, is_update: bool = False) -> Dict[str, Any]:
        """Parse and convert form data to property data"""
        property_data = {}

        # Basic fields
        property_data['name'] = data.get('name', '').strip()
        property_data['propertyType'] = data.get('propertyType')
        property_data['listingType'] = data.get('listingType')
        property_data['address'] = data.get('address', '').strip()

        # Location
        if data.get('latitude'):
            property_data['latitude'] = float(data['latitude'])
        if data.get('longitude'):
            property_data['longitude'] = float(data['longitude'])

        # Pricing
        property_data['price'] = float(data['price'])
        property_data['priceNegotiable'] = data.get('priceNegotiable') == 'true'

        # Specifications
        if data.get('bedrooms'):
            property_data['bedrooms'] = int(data['bedrooms'])
        if data.get('bathrooms'):
            property_data['bathrooms'] = float(data['bathrooms'])
        if data.get('floorArea'):
            property_data['floorArea'] = float(data['floorArea'])
        if data.get('lotArea'):
            property_data['lotArea'] = float(data['lotArea'])

        # Parking
        property_data['parkingAvailable'] = data.get('parkingAvailable') == 'true'
        if data.get('parkingSlots'):
            property_data['parkingSlots'] = int(data['parkingSlots'])

        # Building details
        if data.get('floorLevel'):
            property_data['floorLevel'] = data['floorLevel'].strip()
        if data.get('storeys'):
            property_data['storeys'] = int(data['storeys'])
        if data.get('furnishing'):
            property_data['furnishing'] = data['furnishing']
        if data.get('condition'):
            property_data['condition'] = data['condition']
        if data.get('yearBuilt'):
            property_data['yearBuilt'] = int(data['yearBuilt'])

        # Description
        if data.get('description'):
            property_data['description'] = data['description'].strip()

        # Arrays
        property_data['amenities'] = self._parse_json_array(data.get('amenities', []))
        property_data['interiorFeatures'] = self._parse_json_array(data.get('interiorFeatures', []))
        property_data['buildingAmenities'] = self._parse_json_array(data.get('buildingAmenities', []))
        property_data['utilities'] = self._parse_json_array(data.get('utilities', []))
        property_data['terms'] = self._parse_json_array(data.get('terms', []))

        # Nearby establishments
        property_data['nearbySchools'] = self._parse_nearby_establishments(data.get('nearbySchools', []))
        property_data['nearbyHospitals'] = self._parse_nearby_establishments(data.get('nearbyHospitals', []))
        property_data['nearbyMalls'] = self._parse_nearby_establishments(data.get('nearbyMalls', []))
        property_data['nearbyTransport'] = self._parse_nearby_establishments(data.get('nearbyTransport', []))
        property_data['nearbyOffices'] = self._parse_nearby_establishments(data.get('nearbyOffices', []))

        # Legal & Financial
        if data.get('ownershipStatus'):
            property_data['ownershipStatus'] = data['ownershipStatus'].strip()
        if data.get('taxStatus'):
            property_data['taxStatus'] = data['taxStatus'].strip()
        if data.get('associationDues'):
            property_data['associationDues'] = float(data['associationDues'])

        # Availability
        if data.get('availabilityDate'):
            property_data['availabilityDate'] = data['availabilityDate']
        if data.get('minimumLeasePeriod'):
            property_data['minimumLeasePeriod'] = data['minimumLeasePeriod'].strip()
        if data.get('petPolicy'):
            property_data['petPolicy'] = data['petPolicy'].strip()
        if data.get('smokingPolicy'):
            property_data['smokingPolicy'] = data['smokingPolicy'].strip()

        # Agent info
        if data.get('agentName'):
            property_data['agentName'] = data['agentName'].strip()
        if data.get('agentPhone'):
            property_data['agentPhone'] = data['agentPhone'].strip()
        if data.get('agentEmail'):
            property_data['agentEmail'] = data['agentEmail'].strip()
        if data.get('agentExperience'):
            property_data['agentExperience'] = int(data['agentExperience'])
        if data.get('agentBio'):
            property_data['agentBio'] = data['agentBio'].strip()

        # Developer info
        property_data['hasDeveloper'] = data.get('hasDeveloper') == 'true'
        if data.get('developerName'):
            property_data['developerName'] = data['developerName'].strip()
        if data.get('developerWebsite'):
            property_data['developerWebsite'] = data['developerWebsite'].strip()
        if data.get('developerPhone'):
            property_data['developerPhone'] = data['developerPhone'].strip()
        if data.get('developerEmail'):
            property_data['developerEmail'] = data['developerEmail'].strip()
        if data.get('developerYears'):
            property_data['developerYears'] = int(data['developerYears'])
        if data.get('developerBio'):
            property_data['developerBio'] = data['developerBio'].strip()

        # Metadata
        if not is_update:
            property_data['status'] = 'draft'
            property_data['createdBy'] = user_id

        return property_data

    def _handle_image_uploads(self, files: Dict[str, Any], property_id: str) -> List[PropertyImage]:
        """Handle image uploads to S3"""
        images = []

        # Handle regular images
        if 'regularImages' in files:
            for file in files['regularImages']:
                result = self.aws_service.upload_property_image(file, property_id, 'regular')
                if not result.get('success', True):
                    raise ValidationError(f"Failed to upload regular image: {result.get('error', 'Unknown error')}")
                images.append(PropertyImage(**result))

        # Handle panoramic images
        if 'panoramicImages' in files:
            for file in files['panoramicImages']:
                result = self.aws_service.upload_property_image(file, property_id, 'panoramic')
                if not result.get('success', True):
                    raise ValidationError(f"Failed to upload panoramic image: {result.get('error', 'Unknown error')}")
                images.append(PropertyImage(**result))

        return images

    def _parse_json_array(self, data) -> List[str]:
        """Parse JSON array from form data"""
        if isinstance(data, str):
            try:
                import json
                return json.loads(data)
            except:
                return []
        elif isinstance(data, list):
            return data
        return []

    def _parse_nearby_establishments(self, data) -> List[NearbyEstablishment]:
        """Parse nearby establishments from form data"""
        establishments = self._parse_json_array(data)
        return [NearbyEstablishment(**est) for est in establishments if isinstance(est, dict)]

    def _is_valid_email(self, email: str) -> bool:
        """Validate email format"""
        pattern = r'^[^\s@]+@[^\s@]+\.[^\s@]+$'
        return re.match(pattern, email) is not None

    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format"""
        pattern = r'^https?://[^\s/$.?#].[^\s]*$'
        return re.match(pattern, url) is not None
