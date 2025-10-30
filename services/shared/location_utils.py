"""
Location Utilities for Clinical Trial Matching System
Handles geocoding, distance calculations, and location-based filtering
"""

import math
import time
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, Any
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
try:
    from loguru import logger
except ImportError:
    import logging
    logger = logging.getLogger(__name__)

class LocationUtils:
    """Utility class for location-based operations"""
    
    def __init__(self, cache_enabled: bool = True, user_agent: str = "clinical-trial-matcher"):
        """
        Initialize LocationUtils
        
        Args:
            cache_enabled: Whether to cache geocoded results
            user_agent: User agent string for Nominatim (required)
        """
        self.cache_enabled = cache_enabled
        self.cache_file = Path("persist/location_cache.json")
        self.cache: Dict[str, Dict[str, Any]] = {}
        self.geocoder = Nominatim(user_agent=user_agent)
        
        # Rate limiting: Nominatim allows 1 request per second
        self.last_request_time = 0
        self.min_request_interval = 1.1  # Slightly more than 1 second for safety
        
        # Load cache if enabled
        if self.cache_enabled:
            self.load_cache()
    
    def load_cache(self):
        """Load geocoding cache from file"""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, 'r', encoding='utf-8') as f:
                    self.cache = json.load(f)
                logger.info(f"Loaded {len(self.cache)} cached location entries")
        except Exception as e:
            logger.warning(f"Failed to load location cache: {e}")
            self.cache = {}
    
    def save_cache(self):
        """Save geocoding cache to file"""
        if not self.cache_enabled:
            return
        
        try:
            # Create directory if it doesn't exist
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(self.cache_file, 'w', encoding='utf-8') as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to save location cache: {e}")
    
    def _rate_limit(self):
        """Implement rate limiting for geocoding requests"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            sleep_time = self.min_request_interval - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def geocode_location(self, address: str, max_retries: int = 3) -> Optional[Tuple[float, float]]:
        """
        Geocode an address to latitude and longitude
        
        Args:
            address: Address string to geocode
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (latitude, longitude) or None if geocoding fails
        """
        if not address or not address.strip():
            return None
        
        # Normalize address for cache key
        address_key = address.strip().lower()
        
        # Check cache first
        if address_key in self.cache:
            cached_result = self.cache[address_key]
            if cached_result.get('success'):
                logger.debug(f"Cache hit for address: {address[:50]}...")
                return (cached_result['latitude'], cached_result['longitude'])
        
        # Geocode with retries
        for attempt in range(max_retries):
            try:
                self._rate_limit()
                
                logger.debug(f"Geocoding address (attempt {attempt + 1}/{max_retries}): {address[:50]}...")
                location = self.geocoder.geocode(address, timeout=10)
                
                if location:
                    lat, lon = location.latitude, location.longitude
                    
                    # Cache the result
                    self.cache[address_key] = {
                        'success': True,
                        'latitude': lat,
                        'longitude': lon,
                        'original_address': address
                    }
                    self.save_cache()
                    
                    logger.info(f"Successfully geocoded: {address[:50]}... -> ({lat}, {lon})")
                    return (lat, lon)
                else:
                    logger.warning(f"Geocoding returned no results for: {address[:50]}...")
                    
            except GeocoderTimedOut:
                logger.warning(f"Geocoding timeout for: {address[:50]}... (attempt {attempt + 1}/{max_retries})")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                continue
            
            except Exception as e:
                logger.error(f"Geocoding error for {address[:50]}...: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                continue
        
        # Cache the failure
        self.cache[address_key] = {
            'success': False,
            'original_address': address
        }
        self.save_cache()
        
        return None
    
    def calculate_distance(
        self, 
        lat1: float, 
        lon1: float, 
        lat2: float, 
        lon2: float,
        unit: str = 'km'
    ) -> Optional[float]:
        """
        Calculate distance between two coordinates using Haversine formula
        
        Args:
            lat1, lon1: First point coordinates
            lat2, lon2: Second point coordinates
            unit: 'km' for kilometers, 'miles' for miles
            
        Returns:
            Distance in specified unit, or None if coordinates are invalid
        """
        try:
            # Validate coordinates
            if not all(isinstance(coord, (int, float)) for coord in [lat1, lon1, lat2, lon2]):
                logger.warning("Invalid coordinate types")
                return None
            
            if not (-90 <= lat1 <= 90 and -90 <= lat2 <= 90):
                logger.warning(f"Invalid latitude values: {lat1}, {lat2}")
                return None
            
            if not (-180 <= lon1 <= 180 and -180 <= lon2 <= 180):
                logger.warning(f"Invalid longitude values: {lon1}, {lon2}")
                return None
            
            # Earth radius in kilometers
            R = 6371.0
            
            # Convert to radians
            lat1_rad = math.radians(lat1)
            lat2_rad = math.radians(lat2)
            dlat = math.radians(lat2 - lat1)
            dlon = math.radians(lon2 - lon1)
            
            # Haversine formula
            a = math.sin(dlat / 2) ** 2 + \
                math.cos(lat1_rad) * math.cos(lat2_rad) * \
                math.sin(dlon / 2) ** 2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
            distance_km = R * c
            
            # Convert to miles if requested
            if unit.lower() == 'miles':
                return distance_km * 0.621371
            else:
                return distance_km
                
        except Exception as e:
            logger.error(f"Error calculating distance: {e}")
            return None
    
    def calculate_location_score(
        self, 
        distance_km: float, 
        max_distance_km: Optional[float] = None
    ) -> float:
        """
        Calculate a normalized location score based on distance
        Closer locations get higher scores (0-1 range)
        
        Args:
            distance_km: Distance in kilometers
            max_distance_km: Maximum distance to consider (used for normalization)
            
        Returns:
            Location score between 0 and 1 (higher = closer)
        """
        if max_distance_km is None or max_distance_km <= 0:
            # No max distance specified, use exponential decay
            # Score = e^(-distance/100), so 100km = ~0.37, 200km = ~0.14
            return math.exp(-distance_km / 100.0)
        
        # Normalize based on max distance
        if distance_km >= max_distance_km:
            return 0.0
        
        # Linear normalization: closer = higher score
        normalized = 1.0 - (distance_km / max_distance_km)
        
        # Apply slight exponential curve for better distribution
        return normalized ** 0.8
    
    def filter_by_distance(
        self,
        entities: list,
        reference_lat: float,
        reference_lon: float,
        max_distance_km: Optional[float] = None,
        location_field: str = 'location'
    ) -> list:
        """
        Filter entities by distance from a reference point
        
        Args:
            entities: List of entity dictionaries
            reference_lat, reference_lon: Reference coordinates
            max_distance_km: Maximum allowed distance (None = no filter)
            location_field: Field name in entity dict containing location/lat/lon
            
        Returns:
            Filtered and enriched list with distance and location_score fields
        """
        if not entities:
            return []
        
        filtered_entities = []
        
        for entity in entities:
            # Try to get coordinates from entity
            lat = None
            lon = None
            
            # Check if entity has direct lat/lon fields
            if 'latitude' in entity and 'longitude' in entity:
                try:
                    lat = float(entity['latitude'])
                    lon = float(entity['longitude'])
                except (ValueError, TypeError):
                    pass
            
            # If no direct coordinates, try to geocode location field
            if lat is None or lon is None:
                location = entity.get(location_field)
                if location:
                    coords = self.geocode_location(location)
                    if coords:
                        lat, lon = coords
                        # Store in entity for future use
                        entity['latitude'] = lat
                        entity['longitude'] = lon
            
            # Skip if no valid coordinates
            if lat is None or lon is None:
                logger.debug(f"Skipping entity {entity.get('id', 'unknown')}: no valid coordinates")
                entity['distance_km'] = None
                entity['location_score'] = 0.0
                # Optionally skip entities without location (or include with 0 score)
                if max_distance_km is not None:
                    continue  # Strict filtering: skip if no location
                else:
                    filtered_entities.append(entity)
                continue
            
            # Calculate distance
            distance = self.calculate_distance(reference_lat, reference_lon, lat, lon)
            if distance is None:
                entity['distance_km'] = None
                entity['location_score'] = 0.0
                if max_distance_km is not None:
                    continue
                else:
                    filtered_entities.append(entity)
                continue
            
            # Apply distance filter
            if max_distance_km is not None and distance > max_distance_km:
                continue
            
            # Calculate location score
            location_score = self.calculate_location_score(distance, max_distance_km)
            
            # Add distance and score to entity
            entity['distance_km'] = round(distance, 2)
            entity['location_score'] = round(location_score, 4)
            entity['latitude'] = lat
            entity['longitude'] = lon
            
            filtered_entities.append(entity)
        
        return filtered_entities
    
    def build_address_string(self, location_obj: Dict[str, Any]) -> str:
        """
        Build an address string from a location object
        
        Args:
            location_obj: Dict with 'city', 'state', 'country', 'name' fields
            
        Returns:
            Formatted address string like "City, State, Country"
        """
        parts = []
        
        # Add city if available
        city = location_obj.get('city', '').strip()
        if city:
            parts.append(city)
        
        # Add state if available
        state = location_obj.get('state', '').strip()
        if state:
            parts.append(state)
        
        # Add country if available
        country = location_obj.get('country', '').strip()
        if country:
            parts.append(country)
        
        # If we have any parts, join them with commas
        if parts:
            return ", ".join(parts)
        
        # Fallback: try name field if city/state/country not available
        name = location_obj.get('name', '').strip()
        if name:
            return name
        
        return ""

