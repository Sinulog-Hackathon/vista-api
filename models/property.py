from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime


class NearbyEstablishment(BaseModel):
    name: str
    distance: str


class PropertyImage(BaseModel):
    id: str
    url: str
    thumbnailUrl: Optional[str] = None
    filename: str
    imageType: str  # "regular" or "panoramic"


class Property(BaseModel):
    """Property model matching the frontend PropertyFormData interface"""
    propertyId: Optional[str] = Field(None, description="Unique property ID")
    # Basic Information
    name: str = Field(..., description="Property title/name")
    propertyType: str = Field(..., description="House, Condo, Apartment, Lot, Commercial")
    listingType: str = Field(..., description="For Sale, For Rent, For Lease")
    address: str = Field(..., description="Full street address")

    # Location
    latitude: Optional[float] = Field(None, description="GPS latitude")
    longitude: Optional[float] = Field(None, description="GPS longitude")

    # Pricing
    price: float = Field(..., description="Property price")
    priceNegotiable: bool = Field(False, description="Price negotiation flag")

    # Property Specifications
    bedrooms: Optional[int] = Field(None, description="Number of bedrooms")
    bathrooms: Optional[float] = Field(None, description="Number of bathrooms")
    floorArea: Optional[float] = Field(None, description="Floor area in sqm")
    lotArea: Optional[float] = Field(None, description="Lot area in sqm")

    # Parking
    parkingAvailable: bool = Field(False, description="Parking availability")
    parkingSlots: Optional[int] = Field(None, description="Number of parking slots")

    # Building Details
    floorLevel: Optional[str] = Field(None, description="Floor level (for condos)")
    storeys: Optional[int] = Field(None, description="Number of storeys/floors")
    furnishing: Optional[str] = Field(None, description="Fully furnished, Semi-furnished, Unfurnished")
    condition: Optional[str] = Field(None, description="New, Well-maintained, Renovated, Needs repair")
    yearBuilt: Optional[int] = Field(None, description="Year property was built")

    # Property Description
    description: Optional[str] = Field(None, description="Detailed property description")

    # Features & Amenities
    amenities: List[str] = Field(default_factory=list, description="Property amenities")
    interiorFeatures: List[str] = Field(default_factory=list, description="Interior features")
    buildingAmenities: List[str] = Field(default_factory=list, description="Building amenities")
    utilities: List[str] = Field(default_factory=list, description="Available utilities")

    # Nearby Establishments
    nearbySchools: List[NearbyEstablishment] = Field(default_factory=list)
    nearbyHospitals: List[NearbyEstablishment] = Field(default_factory=list)
    nearbyMalls: List[NearbyEstablishment] = Field(default_factory=list)
    nearbyTransport: List[NearbyEstablishment] = Field(default_factory=list)
    nearbyOffices: List[NearbyEstablishment] = Field(default_factory=list)

    # Legal & Financial
    ownershipStatus: Optional[str] = Field(None, description="Ownership status")
    taxStatus: Optional[str] = Field(None, description="Tax payment status")
    associationDues: Optional[float] = Field(None, description="Monthly HOA dues")

    # Terms & Policies
    terms: List[str] = Field(default_factory=list, description="Property terms & conditions")

    # Availability
    availabilityDate: Optional[str] = Field(None, description="Property availability date")
    minimumLeasePeriod: Optional[str] = Field(None, description="Minimum lease period")
    petPolicy: Optional[str] = Field(None, description="Pet policy")
    smokingPolicy: Optional[str] = Field(None, description="Smoking policy")

    # Agent Information
    agentName: Optional[str] = Field(None, description="Agent full name")
    agentPhone: Optional[str] = Field(None, description="Agent phone number")
    agentEmail: Optional[str] = Field(None, description="Agent email address")
    agentExperience: Optional[int] = Field(None, description="Years of experience")
    agentBio: Optional[str] = Field(None, description="Agent biography")

    # Developer Information
    hasDeveloper: bool = Field(False, description="Whether property has developer")
    developerName: Optional[str] = Field(None, description="Developer/company name")
    developerWebsite: Optional[str] = Field(None, description="Developer website")
    developerPhone: Optional[str] = Field(None, description="Developer phone")
    developerEmail: Optional[str] = Field(None, description="Developer email")
    developerYears: Optional[int] = Field(None, description="Years in business")
    developerBio: Optional[str] = Field(None, description="Developer biography")

    # Metadata
    status: str = Field("draft", description="draft, pending_review, published, rejected")
    createdAt: datetime = Field(default_factory=datetime.utcnow)
    updatedAt: datetime = Field(default_factory=datetime.utcnow)
    createdBy: str = Field(..., description="User ID who created the property")

    # Images
    images: List[PropertyImage] = Field(default_factory=list)
    regularImageCount: int = Field(0)
    panoramicImageCount: int = Field(0)
    image: Optional[PropertyImage] = Field(None, description="Main property image/thumbnail")

    @validator('propertyType')
    def validate_property_type(cls, v):
        valid_types = ["House", "Condo", "Apartment", "Lot", "Commercial"]
        if v not in valid_types:
            raise ValueError(f'propertyType must be one of {valid_types}')
        return v

    @validator('listingType')
    def validate_listing_type(cls, v):
        valid_types = ["For Sale", "For Rent", "For Lease"]
        if v not in valid_types:
            raise ValueError(f'listingType must be one of {valid_types}')
        return v

    @validator('furnishing')
    def validate_furnishing(cls, v):
        if v is None:
            return v
        valid_types = ["Fully furnished", "Semi-furnished", "Unfurnished"]
        if v not in valid_types:
            raise ValueError(f'furnishing must be one of {valid_types}')
        return v

    @validator('condition')
    def validate_condition(cls, v):
        if v is None:
            return v
        valid_types = ["New", "Well-maintained", "Renovated", "Needs repair"]
        if v not in valid_types:
            raise ValueError(f'condition must be one of {valid_types}')
        return v

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for Firestore"""
        data = self.model_dump()
        data['createdAt'] = self.createdAt.isoformat()
        data['updatedAt'] = self.updatedAt.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Property':
        """Create from Firestore dictionary"""
        if isinstance(data.get('createdAt'), str):
            data['createdAt'] = datetime.fromisoformat(data['createdAt'])
        if isinstance(data.get('updatedAt'), str):
            data['updatedAt'] = datetime.fromisoformat(data['updatedAt'])
        return cls(**data)
