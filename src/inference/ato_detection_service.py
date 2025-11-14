"""
Account Takeover (ATO) Detection Service

Comprehensive ATO detection using:
- Geo-velocity analysis (impossible travel)
- Device fingerprinting (new device detection)
- Behavioral profiling (time patterns, spending habits)
- Session anomaly detection (concurrent logins)
- Historical baseline tracking (normal behavior)

Uses Redis for distributed state management across Spark workers.
"""

import logging
import os
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

try:
    import redis
    from redis.exceptions import RedisError, ConnectionError as RedisConnectionError
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    logging.warning("Redis not available. ATO detection will use local fallback.")

logger = logging.getLogger(__name__)


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    """
    Calculate great circle distance between two points on Earth (in km)
    
    Args:
        lon1, lat1: First point coordinates
        lon2, lat2: Second point coordinates
    
    Returns:
        Distance in kilometers
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    
    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    
    # Earth radius in kilometers
    r = 6371
    
    return c * r


class ATODetectionService:
    """
    Comprehensive Account Takeover Detection Service
    
    Detects ATO patterns:
    1. Geographic anomalies (impossible travel)
    2. Device changes (new device usage)
    3. Time anomalies (unusual login hours)
    4. Behavioral deviations (spending pattern changes)
    5. Session anomalies (concurrent sessions)
    """
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        history_days: int = 90,
        connection_timeout: int = 2
    ):
        """
        Initialize ATO detection service
        
        Args:
            redis_url: Redis connection URL
            history_days: Days of history to maintain for profiling
            connection_timeout: Redis connection timeout in seconds
        """
        self.history_days = history_days
        self.history_seconds = history_days * 24 * 3600
        
        # Initialize Redis connection
        if redis_url is None:
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        
        self.redis_client = None
        self.redis_available = False
        
        if REDIS_AVAILABLE:
            try:
                self.redis_client = redis.Redis.from_url(
                    redis_url,
                    socket_connect_timeout=connection_timeout,
                    socket_timeout=connection_timeout,
                    decode_responses=False,
                    max_connections=50,
                    health_check_interval=30
                )
                self.redis_client.ping()
                self.redis_available = True
                logger.info(f"✓ ATO Detection Service connected to Redis: {redis_url}")
            except Exception as e:
                logger.warning(f"Redis connection failed: {e}. ATO detection degraded.")
                self.redis_available = False
        
        # Fallback: Local cache
        self.local_cache = {}
        
        # ATO detection thresholds
        self.thresholds = {
            'max_travel_speed_kmh': 800,  # Faster than plane = impossible
            'suspicious_travel_speed_kmh': 500,  # Very fast = suspicious
            'off_hours_start': 1,  # 1 AM
            'off_hours_end': 6,    # 6 AM
            'max_amount_deviation': 3.0,  # 3x typical amount = suspicious
            'min_transactions_for_profile': 5,  # Need 5 txns to build profile
            'session_timeout_minutes': 30,  # Session expires after 30 min
            'max_concurrent_sessions': 2,  # More than 2 sessions = suspicious
            'device_change_high_amount': 1000.0,  # Device change + high amount = ATO
        }
    
    def create_user_id(self, transaction: Dict) -> str:
        """Create consistent user identifier from transaction"""
        # Use card1 as primary user identifier
        card1 = transaction.get('card1', 'unknown')
        return f"user:{card1}"
    
    # ==================== GEO-VELOCITY DETECTION ====================
    
    def detect_geo_anomaly(
        self,
        user_id: str,
        current_lat: Optional[float],
        current_lon: Optional[float],
        current_timestamp: float
    ) -> Dict[str, any]:
        """
        Detect impossible travel (geo-velocity anomaly)
        
        Returns:
            {
                'geo_anomaly_detected': bool,
                'geo_risk_score': float (0-1),
                'travel_speed_kmh': float,
                'distance_km': float,
                'time_diff_hours': float,
                'reason': str
            }
        """
        result = {
            'geo_anomaly_detected': False,
            'geo_risk_score': 0.0,
            'travel_speed_kmh': 0.0,
            'distance_km': 0.0,
            'time_diff_hours': 0.0,
            'reason': 'no_previous_location'
        }
        
        # Skip if no coordinates provided
        if current_lat is None or current_lon is None:
            return result
        
        # Get previous location from Redis
        prev_location = self._get_last_location(user_id)
        
        if prev_location:
            prev_lat, prev_lon, prev_timestamp = prev_location
            
            # Calculate distance and time
            distance_km = haversine(prev_lon, prev_lat, current_lon, current_lat)
            time_diff_seconds = current_timestamp - prev_timestamp
            time_diff_hours = time_diff_seconds / 3600.0
            
            # Calculate travel speed
            if time_diff_hours > 0:
                travel_speed_kmh = distance_km / time_diff_hours
            else:
                travel_speed_kmh = 0.0
            
            # Determine risk
            if travel_speed_kmh > self.thresholds['max_travel_speed_kmh']:
                # Impossible travel (faster than plane)
                result['geo_anomaly_detected'] = True
                result['geo_risk_score'] = 1.0
                result['reason'] = 'impossible_travel'
            elif travel_speed_kmh > self.thresholds['suspicious_travel_speed_kmh']:
                # Very suspicious (very fast travel)
                result['geo_anomaly_detected'] = True
                result['geo_risk_score'] = 0.8
                result['reason'] = 'suspicious_travel_speed'
            elif distance_km > 1000 and time_diff_hours < 2:
                # Long distance in short time
                result['geo_anomaly_detected'] = True
                result['geo_risk_score'] = 0.7
                result['reason'] = 'rapid_long_distance'
            
            result['travel_speed_kmh'] = travel_speed_kmh
            result['distance_km'] = distance_km
            result['time_diff_hours'] = time_diff_hours
        
        # Store current location
        self._store_location(user_id, current_lat, current_lon, current_timestamp)
        
        return result
    
    def _get_last_location(self, user_id: str) -> Optional[Tuple[float, float, float]]:
        """Get user's last known location from Redis"""
        if self.redis_available:
            try:
                key = f"ato:{user_id}:last_location"
                data = self.redis_client.get(key)
                if data:
                    lat, lon, ts = data.decode('utf-8').split(',')
                    return (float(lat), float(lon), float(ts))
            except Exception as e:
                logger.warning(f"Failed to get last location: {e}")
        return None
    
    def _store_location(self, user_id: str, lat: float, lon: float, timestamp: float):
        """Store user's current location in Redis"""
        if self.redis_available:
            try:
                key = f"ato:{user_id}:last_location"
                value = f"{lat},{lon},{timestamp}"
                self.redis_client.setex(key, 86400, value)  # 24-hour TTL
            except Exception as e:
                logger.warning(f"Failed to store location: {e}")
    
    # ==================== DEVICE FINGERPRINTING ====================
    
    def detect_device_anomaly(
        self,
        user_id: str,
        device_id: Optional[str],
        device_type: Optional[str],
        current_timestamp: float
    ) -> Dict[str, any]:
        """
        Detect new or suspicious device usage
        
        Returns:
            {
                'device_anomaly_detected': bool,
                'device_risk_score': float (0-1),
                'is_new_device': bool,
                'device_count': int,
                'reason': str
            }
        """
        result = {
            'device_anomaly_detected': False,
            'device_risk_score': 0.0,
            'is_new_device': False,
            'device_count': 0,
            'reason': 'no_device_id'
        }
        
        if device_id is None:
            return result
        
        # Get known devices
        known_devices = self._get_known_devices(user_id)
        result['device_count'] = len(known_devices)
        
        if device_id not in known_devices:
            # New device detected
            result['is_new_device'] = True
            result['device_anomaly_detected'] = True
            
            # Risk score based on device count
            if len(known_devices) == 0:
                # First transaction - low risk
                result['device_risk_score'] = 0.3
                result['reason'] = 'first_device'
            elif len(known_devices) < 3:
                # Few devices - medium risk
                result['device_risk_score'] = 0.6
                result['reason'] = 'new_device'
            else:
                # Many devices already - high risk
                result['device_risk_score'] = 0.9
                result['reason'] = 'suspicious_new_device'
            
            # Add device to known list
            self._add_known_device(user_id, device_id, device_type, current_timestamp)
        else:
            # Known device - update last seen
            self._update_device_last_seen(user_id, device_id, current_timestamp)
            result['reason'] = 'known_device'
        
        return result
    
    def _get_known_devices(self, user_id: str) -> List[str]:
        """Get list of known devices for user"""
        if self.redis_available:
            try:
                key = f"ato:{user_id}:devices"
                devices = self.redis_client.smembers(key)
                return [d.decode('utf-8') for d in devices]
            except Exception as e:
                logger.warning(f"Failed to get known devices: {e}")
        return []
    
    def _add_known_device(self, user_id: str, device_id: str, device_type: Optional[str], timestamp: float):
        """Add device to user's known devices"""
        if self.redis_available:
            try:
                # Add to set
                key = f"ato:{user_id}:devices"
                self.redis_client.sadd(key, device_id)
                self.redis_client.expire(key, self.history_seconds)
                
                # Store device metadata
                meta_key = f"ato:{user_id}:device:{device_id}"
                metadata = {
                    'device_type': device_type or 'unknown',
                    'first_seen': timestamp,
                    'last_seen': timestamp
                }
                self.redis_client.setex(meta_key, self.history_seconds, json.dumps(metadata))
            except Exception as e:
                logger.warning(f"Failed to add device: {e}")
    
    def _update_device_last_seen(self, user_id: str, device_id: str, timestamp: float):
        """Update device last seen timestamp"""
        if self.redis_available:
            try:
                meta_key = f"ato:{user_id}:device:{device_id}"
                data = self.redis_client.get(meta_key)
                if data:
                    metadata = json.loads(data.decode('utf-8'))
                    metadata['last_seen'] = timestamp
                    self.redis_client.setex(meta_key, self.history_seconds, json.dumps(metadata))
            except Exception as e:
                logger.warning(f"Failed to update device: {e}")
    
    # ==================== TIME-BASED BEHAVIORAL PROFILING ====================
    
    def detect_time_anomaly(
        self,
        user_id: str,
        current_hour: int,
        current_timestamp: float
    ) -> Dict[str, any]:
        """
        Detect unusual transaction time (off-hours activity)
        
        Returns:
            {
                'time_anomaly_detected': bool,
                'time_risk_score': float (0-1),
                'is_off_hours': bool,
                'is_unusual_hour': bool,
                'reason': str
            }
        """
        result = {
            'time_anomaly_detected': False,
            'time_risk_score': 0.0,
            'is_off_hours': False,
            'is_unusual_hour': False,
            'reason': 'normal_hours'
        }
        
        # Check if off-hours (1 AM - 6 AM)
        if self.thresholds['off_hours_start'] <= current_hour <= self.thresholds['off_hours_end']:
            result['is_off_hours'] = True
            result['time_anomaly_detected'] = True
            result['time_risk_score'] = 0.6
            result['reason'] = 'off_hours_activity'
        
        # Check against user's typical hours
        typical_hours = self._get_typical_hours(user_id)
        
        if len(typical_hours) >= self.thresholds['min_transactions_for_profile']:
            if current_hour not in typical_hours:
                result['is_unusual_hour'] = True
                result['time_anomaly_detected'] = True
                result['time_risk_score'] = max(result['time_risk_score'], 0.5)
                result['reason'] = 'unusual_hour_for_user'
        
        # Store hour
        self._add_typical_hour(user_id, current_hour, current_timestamp)
        
        return result
    
    def _get_typical_hours(self, user_id: str) -> List[int]:
        """Get user's typical transaction hours"""
        if self.redis_available:
            try:
                key = f"ato:{user_id}:typical_hours"
                hours = self.redis_client.smembers(key)
                return [int(h.decode('utf-8')) for h in hours]
            except Exception as e:
                logger.warning(f"Failed to get typical hours: {e}")
        return []
    
    def _add_typical_hour(self, user_id: str, hour: int, timestamp: float):
        """Add hour to user's typical hours"""
        if self.redis_available:
            try:
                key = f"ato:{user_id}:typical_hours"
                self.redis_client.sadd(key, str(hour))
                self.redis_client.expire(key, self.history_seconds)
            except Exception as e:
                logger.warning(f"Failed to add typical hour: {e}")
    
    # ==================== SPENDING BEHAVIOR ANALYSIS ====================
    
    def detect_spending_anomaly(
        self,
        user_id: str,
        current_amount: float,
        current_timestamp: float
    ) -> Dict[str, any]:
        """
        Detect unusual spending patterns
        
        Returns:
            {
                'spending_anomaly_detected': bool,
                'spending_risk_score': float (0-1),
                'amount_deviation': float,
                'typical_amount': float,
                'reason': str
            }
        """
        result = {
            'spending_anomaly_detected': False,
            'spending_risk_score': 0.0,
            'amount_deviation': 0.0,
            'typical_amount': 0.0,
            'reason': 'no_baseline'
        }
        
        # Get spending baseline
        baseline = self._get_spending_baseline(user_id)
        
        if baseline and baseline['count'] >= self.thresholds['min_transactions_for_profile']:
            typical_amount = baseline['mean']
            result['typical_amount'] = typical_amount
            
            if typical_amount > 0:
                deviation = current_amount / typical_amount
                result['amount_deviation'] = deviation
                
                if deviation > self.thresholds['max_amount_deviation']:
                    # Amount significantly higher than typical
                    result['spending_anomaly_detected'] = True
                    result['spending_risk_score'] = min(deviation / 10.0, 1.0)
                    result['reason'] = 'amount_spike'
                elif deviation < 0.1 and typical_amount > 100:
                    # Unusually low amount (testing with stolen card)
                    result['spending_anomaly_detected'] = True
                    result['spending_risk_score'] = 0.4
                    result['reason'] = 'unusually_low_amount'
        
        # Update baseline
        self._update_spending_baseline(user_id, current_amount, current_timestamp)
        
        return result
    
    def _get_spending_baseline(self, user_id: str) -> Optional[Dict]:
        """Get user's spending baseline statistics"""
        if self.redis_available:
            try:
                key = f"ato:{user_id}:spending_baseline"
                data = self.redis_client.get(key)
                if data:
                    return json.loads(data.decode('utf-8'))
            except Exception as e:
                logger.warning(f"Failed to get spending baseline: {e}")
        return None
    
    def _update_spending_baseline(self, user_id: str, amount: float, timestamp: float):
        """Update user's spending baseline with new transaction"""
        if self.redis_available:
            try:
                key = f"ato:{user_id}:spending_baseline"
                
                # Get existing baseline
                baseline = self._get_spending_baseline(user_id)
                
                if baseline:
                    # Update running statistics
                    count = baseline['count']
                    mean = baseline['mean']
                    m2 = baseline.get('m2', 0.0)
                    
                    # Welford's online algorithm for variance
                    count += 1
                    delta = amount - mean
                    mean += delta / count
                    delta2 = amount - mean
                    m2 += delta * delta2
                    
                    baseline = {
                        'count': count,
                        'mean': mean,
                        'm2': m2,
                        'std': (m2 / count) ** 0.5 if count > 1 else 0.0,
                        'last_updated': timestamp
                    }
                else:
                    # Initialize baseline
                    baseline = {
                        'count': 1,
                        'mean': amount,
                        'm2': 0.0,
                        'std': 0.0,
                        'last_updated': timestamp
                    }
                
                self.redis_client.setex(key, self.history_seconds, json.dumps(baseline))
            except Exception as e:
                logger.warning(f"Failed to update spending baseline: {e}")
    
    # ==================== SESSION ANOMALY DETECTION ====================
    
    def detect_session_anomaly(
        self,
        user_id: str,
        session_id: Optional[str],
        current_timestamp: float
    ) -> Dict[str, any]:
        """
        Detect concurrent sessions or session hijacking
        
        Returns:
            {
                'session_anomaly_detected': bool,
                'session_risk_score': float (0-1),
                'active_sessions': int,
                'reason': str
            }
        """
        result = {
            'session_anomaly_detected': False,
            'session_risk_score': 0.0,
            'active_sessions': 0,
            'reason': 'no_session_id'
        }
        
        if session_id is None:
            return result
        
        # Clean expired sessions
        self._clean_expired_sessions(user_id, current_timestamp)
        
        # Get active sessions
        active_sessions = self._get_active_sessions(user_id)
        result['active_sessions'] = len(active_sessions)
        
        # Check for concurrent sessions
        if len(active_sessions) >= self.thresholds['max_concurrent_sessions']:
            result['session_anomaly_detected'] = True
            result['session_risk_score'] = min(len(active_sessions) / 5.0, 1.0)
            result['reason'] = 'concurrent_sessions'
        
        # Add current session
        self._add_active_session(user_id, session_id, current_timestamp)
        
        return result
    
    def _get_active_sessions(self, user_id: str) -> List[str]:
        """Get list of active session IDs"""
        if self.redis_available:
            try:
                key = f"ato:{user_id}:active_sessions"
                sessions = self.redis_client.smembers(key)
                return [s.decode('utf-8') for s in sessions]
            except Exception as e:
                logger.warning(f"Failed to get active sessions: {e}")
        return []
    
    def _add_active_session(self, user_id: str, session_id: str, timestamp: float):
        """Add session to active sessions"""
        if self.redis_available:
            try:
                # Add to set
                key = f"ato:{user_id}:active_sessions"
                self.redis_client.sadd(key, session_id)
                
                # Store session metadata
                meta_key = f"ato:{user_id}:session:{session_id}"
                metadata = {'last_activity': timestamp}
                ttl = self.thresholds['session_timeout_minutes'] * 60
                self.redis_client.setex(meta_key, ttl, json.dumps(metadata))
            except Exception as e:
                logger.warning(f"Failed to add session: {e}")
    
    def _clean_expired_sessions(self, user_id: str, current_timestamp: float):
        """Remove expired sessions from active set"""
        if self.redis_available:
            try:
                key = f"ato:{user_id}:active_sessions"
                sessions = self._get_active_sessions(user_id)
                
                timeout_seconds = self.thresholds['session_timeout_minutes'] * 60
                
                for session_id in sessions:
                    meta_key = f"ato:{user_id}:session:{session_id}"
                    
                    # Check if session metadata exists
                    if not self.redis_client.exists(meta_key):
                        # Session expired, remove from set
                        self.redis_client.srem(key, session_id)
            except Exception as e:
                logger.warning(f"Failed to clean sessions: {e}")
    
    # ==================== COMPREHENSIVE ATO ANALYSIS ====================
    
    def analyze_transaction(
        self,
        transaction: Dict,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        device_id: Optional[str] = None,
        device_type: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, any]:
        """
        Comprehensive ATO analysis combining all detection methods
        
        Args:
            transaction: Transaction data (must include card1, TransactionAmt)
            latitude: Transaction latitude
            longitude: Transaction longitude
            device_id: Device identifier (fingerprint)
            device_type: Device type (mobile/desktop/tablet)
            session_id: Session identifier
        
        Returns:
            {
                'ato_detected': bool,
                'ato_risk_score': float (0-1),
                'ato_confidence': str (LOW/MEDIUM/HIGH/CRITICAL),
                'ato_reasons': List[str],
                'geo_analysis': Dict,
                'device_analysis': Dict,
                'time_analysis': Dict,
                'spending_analysis': Dict,
                'session_analysis': Dict
            }
        """
        user_id = self.create_user_id(transaction)
        current_timestamp = time.time()
        current_hour = datetime.fromtimestamp(current_timestamp).hour
        current_amount = transaction.get('TransactionAmt', 0.0)
        
        # Run all detection methods
        geo_analysis = self.detect_geo_anomaly(user_id, latitude, longitude, current_timestamp)
        device_analysis = self.detect_device_anomaly(user_id, device_id, device_type, current_timestamp)
        time_analysis = self.detect_time_anomaly(user_id, current_hour, current_timestamp)
        spending_analysis = self.detect_spending_anomaly(user_id, current_amount, current_timestamp)
        session_analysis = self.detect_session_anomaly(user_id, session_id, current_timestamp)
        
        # Aggregate risk scores (weighted)
        ato_risk_score = (
            geo_analysis['geo_risk_score'] * 0.35 +          # Geo = 35% (highest weight)
            device_analysis['device_risk_score'] * 0.25 +    # Device = 25%
            spending_analysis['spending_risk_score'] * 0.20 + # Spending = 20%
            time_analysis['time_risk_score'] * 0.10 +        # Time = 10%
            session_analysis['session_risk_score'] * 0.10    # Session = 10%
        )
        
        # Collect reasons
        ato_reasons = []
        if geo_analysis['geo_anomaly_detected']:
            ato_reasons.append(geo_analysis['reason'])
        if device_analysis['device_anomaly_detected']:
            ato_reasons.append(device_analysis['reason'])
        if time_analysis['time_anomaly_detected']:
            ato_reasons.append(time_analysis['reason'])
        if spending_analysis['spending_anomaly_detected']:
            ato_reasons.append(spending_analysis['reason'])
        if session_analysis['session_anomaly_detected']:
            ato_reasons.append(session_analysis['reason'])
        
        # Determine confidence level
        if ato_risk_score >= 0.8:
            ato_confidence = 'CRITICAL'
        elif ato_risk_score >= 0.6:
            ato_confidence = 'HIGH'
        elif ato_risk_score >= 0.4:
            ato_confidence = 'MEDIUM'
        else:
            ato_confidence = 'LOW'
        
        # Overall detection
        ato_detected = ato_risk_score >= 0.5 or len(ato_reasons) >= 2
        
        return {
            'ato_detected': ato_detected,
            'ato_risk_score': float(ato_risk_score),
            'ato_confidence': ato_confidence,
            'ato_reasons': ato_reasons,
            'geo_analysis': geo_analysis,
            'device_analysis': device_analysis,
            'time_analysis': time_analysis,
            'spending_analysis': spending_analysis,
            'session_analysis': session_analysis
        }
    
    def health_check(self) -> Dict[str, any]:
        """Check service health"""
        return {
            'status': 'healthy' if self.redis_available else 'degraded',
            'redis_connected': self.redis_available,
            'history_days': self.history_days
        }


# Singleton instance
_ato_service_instance = None

def get_ato_service() -> ATODetectionService:
    """Get global ATO detection service instance"""
    global _ato_service_instance
    if _ato_service_instance is None:
        _ato_service_instance = ATODetectionService()
    return _ato_service_instance
