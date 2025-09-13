# ==== MODULE CONTRACT =======================================================
# Module: map/spatial_state.py
# Package: MidnightCore.Modules.map.spatial_state
# Location: Production/MidnightCore/Core/Modules/map/spatial_state.py
# Responsibility: Beta's spatial position tracking and world coordinate system
# PUBLIC: BetaSpatialState class, spatial coordinate methods
# DEPENDENCIES: auto_calibrator (Unity constants), tick_engine (position updates)
# POLICY: NO_FALLBACKS=deny, Telemetry: spatial.*
# MIGRATION: New module implementing Phase 1.2 of spatial awareness plan
# ============================================================================

import math
import time
import json
import sys
import os
from typing import Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# Import unified logging system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Common', 'Tools'))
from logging_bus import log_engine

@dataclass
class WorldPosition:
    """Represents a position in VRChat world coordinates"""
    x: float = 0.0          # East-West (positive = east)
    y: float = 0.0          # North-South (positive = north)
    z: float = 0.0          # Up-Down (positive = up, currently unused)
    timestamp: float = 0.0  # When this position was recorded
    
    def distance_to(self, other: 'WorldPosition') -> float:
        """Calculate distance to another position in meters"""
        dx = self.x - other.x
        dy = self.y - other.y
        return math.sqrt(dx*dx + dy*dy)
    
    def bearing_to(self, other: 'WorldPosition') -> float:
        """Calculate bearing to another position in degrees (0=north, 90=east)"""
        dx = other.x - self.x
        dy = other.y - self.y
        bearing_rad = math.atan2(dx, dy)
        bearing_deg = math.degrees(bearing_rad)
        return (bearing_deg + 360) % 360  # Normalize to 0-360

@dataclass 
class MovementValidation:
    """Records movement validation data"""
    expected_distance: float
    expected_bearing: float
    actual_result: Optional[str] = None  # "completed", "blocked", "partial"
    block_reason: Optional[str] = None
    timestamp: float = 0.0

class BetaSpatialState:
    """
    Beta's spatial state tracking system
    Implements Phase 1.2: Position Tracking Foundation from spatial awareness plan
    """
    
    def __init__(self, world_id: str, unity_constants: Optional[Dict[str, float]] = None):
        """
        Initialize spatial state for a specific world
        
        Args:
            world_id: VRChat world identifier
            unity_constants: Movement constants from auto_calibrator
        """
        self.world_id = world_id
        self.position = WorldPosition()  # Start at origin (spawn point)
        self.facing_angle = 0.0  # 0=north, 90=east, 180=south, 270=west
        self.last_movement_validation: Optional[MovementValidation] = None
        
        # Unity constants for calculations (default to confirmed values)
        self.unity_constants = unity_constants or {
            "walk_speed_mps": 2.0,
            "run_speed_mps": 4.0, 
            "strafe_speed_mps": 2.0,
            "beta_height_m": 1.6256,
            "beta_eye_level_m": 1.4734
        }
        
        # Movement tracking
        self.movement_history = []  # Recent movements for validation
        self.spawn_position = WorldPosition()  # Record spawn point as reference
        self.position_confidence = 1.0  # Confidence in current position (0-1)
        
        # Session tracking
        self.session_start = time.time()
        self.total_distance_walked = 0.0
        self.total_distance_run = 0.0
        
        print(f"BetaSpatialState initialized for world: {world_id}")
        
        # Log spatial state initialization
        log_engine("spatial.init", world_id=world_id, spawn_position={"x": 0.0, "y": 0.0}, 
                  unity_constants=self.unity_constants, session_start=self.session_start)
    
    def update_movement(self, action: str, duration: float, intensity: float = 1.0, 
                       is_running: bool = False, was_blocked: bool = False, 
                       block_reason: str = "") -> Dict[str, Any]:
        """
        Track every OSC command as position change
        
        Args:
            action: Movement type ("move_forward", "strafe_left", etc.)
            duration: Duration of movement in seconds
            intensity: Movement intensity (usually 1.0)
            is_running: True if running speed was used
            was_blocked: True if movement was blocked by physics safety
            block_reason: Reason movement was blocked
            
        Returns:
            Movement result with position update
        """
        timestamp = time.time()
        
        # Calculate expected movement
        if "forward" in action or "backward" in action:
            speed = self.unity_constants["run_speed_mps"] if is_running else self.unity_constants["walk_speed_mps"]
        elif "strafe" in action or "left" in action or "right" in action:
            speed = self.unity_constants["strafe_speed_mps"]
        else:
            # Non-movement action (look, jump, etc.)
            return {"action": action, "position_changed": False, "current_position": self.position}
        
        distance = speed * duration * intensity
        
        # Handle blocked movement
        if was_blocked:
            self.last_movement_validation = MovementValidation(
                expected_distance=distance,
                expected_bearing=self.facing_angle,
                actual_result="blocked",
                block_reason=block_reason,
                timestamp=timestamp
            )
            # Position doesn't change for blocked movement
            self.position_confidence = max(0.8, self.position_confidence - 0.1)  # Reduce confidence slightly
            
            # Log blocked movement
            log_engine("spatial.movement.blocked", action=action, duration=duration, 
                      expected_distance=distance, block_reason=block_reason, 
                      position={"x": self.position.x, "y": self.position.y}, 
                      facing_angle=self.facing_angle, confidence=self.position_confidence)
            
            return {
                "action": action,
                "position_changed": False,
                "was_blocked": True,
                "block_reason": block_reason,
                "expected_distance": distance,
                "current_position": self.position
            }
        
        # Calculate new position based on movement type and facing direction
        old_position = WorldPosition(self.position.x, self.position.y, self.position.z)
        
        if action == "move_forward":
            # Move in facing direction
            self.position.x += distance * math.sin(math.radians(self.facing_angle))
            self.position.y += distance * math.cos(math.radians(self.facing_angle))
            
        elif action == "move_backward":
            # Move opposite to facing direction
            self.position.x -= distance * math.sin(math.radians(self.facing_angle))
            self.position.y -= distance * math.cos(math.radians(self.facing_angle))
            
        elif action == "strafe_left":
            # Move perpendicular to facing direction (90 degrees left)
            strafe_angle = (self.facing_angle - 90) % 360
            self.position.x += distance * math.sin(math.radians(strafe_angle))
            self.position.y += distance * math.cos(math.radians(strafe_angle))
            
        elif action == "strafe_right":
            # Move perpendicular to facing direction (90 degrees right)
            strafe_angle = (self.facing_angle + 90) % 360
            self.position.x += distance * math.sin(math.radians(strafe_angle))
            self.position.y += distance * math.cos(math.radians(strafe_angle))
        
        self.position.timestamp = timestamp
        
        # Update distance tracking
        if is_running:
            self.total_distance_run += distance
        else:
            self.total_distance_walked += distance
        
        # Record movement validation
        self.last_movement_validation = MovementValidation(
            expected_distance=distance,
            expected_bearing=self.facing_angle,
            actual_result="completed",
            timestamp=timestamp
        )
        
        # Add to movement history (keep last 20 movements)
        movement_record = {
            "action": action,
            "old_position": old_position,
            "new_position": WorldPosition(self.position.x, self.position.y, self.position.z),
            "distance": distance,
            "duration": duration,
            "facing_angle": self.facing_angle,
            "timestamp": timestamp
        }
        self.movement_history.append(movement_record)
        if len(self.movement_history) > 20:
            self.movement_history.pop(0)
        
        # Log successful movement
        log_engine("spatial.movement.success", action=action, duration=duration, 
                  distance_moved=distance, is_running=is_running,
                  old_position={"x": old_position.x, "y": old_position.y}, 
                  new_position={"x": self.position.x, "y": self.position.y},
                  facing_angle=self.facing_angle, confidence=self.position_confidence,
                  session_distance=self.total_distance_walked + self.total_distance_run)
        
        return {
            "action": action,
            "position_changed": True,
            "old_position": old_position,
            "new_position": self.position,
            "distance_moved": distance,
            "current_facing": self.facing_angle,
            "total_session_distance": self.total_distance_walked + self.total_distance_run
        }
    
    def update_facing(self, turn_action: str, duration: float) -> float:
        """
        Update facing direction based on turn/look actions
        
        Args:
            turn_action: "look_left", "look_right", etc.
            duration: Duration of turn in seconds
            
        Returns:
            New facing angle in degrees
        """
        # Standard VRChat turn rate: ~45 degrees per second for look commands
        turn_rate_deg_per_sec = 45.0
        turn_degrees = turn_rate_deg_per_sec * duration
        
        old_facing = self.facing_angle
        
        if turn_action == "look_left":
            self.facing_angle = (self.facing_angle - turn_degrees) % 360
        elif turn_action == "look_right":
            self.facing_angle = (self.facing_angle + turn_degrees) % 360
        
        # Log facing update
        log_engine("spatial.facing.update", action=turn_action, duration=duration,
                  turn_degrees=turn_degrees, old_facing=old_facing, 
                  new_facing=self.facing_angle, position={"x": self.position.x, "y": self.position.y})
        
        return self.facing_angle
    
    def get_position_relative_to_spawn(self) -> Tuple[float, float, float]:
        """Get current position relative to spawn point in meters"""
        dx = self.position.x - self.spawn_position.x
        dy = self.position.y - self.spawn_position.y
        distance = math.sqrt(dx*dx + dy*dy)
        bearing = math.degrees(math.atan2(dx, dy))
        return dx, dy, distance
    
    def set_spawn_reference(self, x: float = 0.0, y: float = 0.0) -> None:
        """Set current position as spawn reference point"""
        self.spawn_position = WorldPosition(x, y, 0.0, time.time())
        self.position = WorldPosition(x, y, 0.0, time.time())
        print(f"Spawn reference set at ({x:.2f}, {y:.2f})")
    
    def get_spatial_summary(self) -> Dict[str, Any]:
        """Get comprehensive spatial state for logging/debugging"""
        dx, dy, spawn_distance = self.get_position_relative_to_spawn()
        
        return {
            "world_id": self.world_id,
            "position": {
                "x": round(self.position.x, 3),
                "y": round(self.position.y, 3),
                "facing_degrees": round(self.facing_angle, 1)
            },
            "relative_to_spawn": {
                "dx": round(dx, 3),
                "dy": round(dy, 3), 
                "distance": round(spawn_distance, 3)
            },
            "session_stats": {
                "time_elapsed": round(time.time() - self.session_start, 1),
                "distance_walked": round(self.total_distance_walked, 2),
                "distance_run": round(self.total_distance_run, 2),
                "total_distance": round(self.total_distance_walked + self.total_distance_run, 2)
            },
            "confidence": round(self.position_confidence, 2),
            "last_movement": self.last_movement_validation.__dict__ if self.last_movement_validation else None
        }
    
    def save_to_cache(self, cache_dir: str) -> bool:
        """Save spatial state to world-specific cache file"""
        try:
            import os
            os.makedirs(cache_dir, exist_ok=True)
            
            cache_file = os.path.join(cache_dir, f"{self.world_id}_spatial_state.json")
            cache_data = {
                "world_id": self.world_id,
                "current_position": {
                    "x": self.position.x,
                    "y": self.position.y,
                    "z": self.position.z,
                    "facing_angle": self.facing_angle,
                    "timestamp": self.position.timestamp
                },
                "spawn_position": {
                    "x": self.spawn_position.x,
                    "y": self.spawn_position.y,
                    "z": self.spawn_position.z
                },
                "session_stats": {
                    "session_start": self.session_start,
                    "total_distance_walked": self.total_distance_walked,
                    "total_distance_run": self.total_distance_run,
                    "position_confidence": self.position_confidence
                },
                "unity_constants": self.unity_constants,
                "last_updated": datetime.now().isoformat()
            }
            
            with open(cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
            
            # Log cache save
            log_engine("spatial.cache.save", world_id=self.world_id, cache_file=cache_file,
                      position={"x": self.position.x, "y": self.position.y}, 
                      session_distance=self.total_distance_walked + self.total_distance_run)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to save spatial state cache: {e}")
            return False
    
    @classmethod
    def load_from_cache(cls, world_id: str, cache_dir: str) -> Optional['BetaSpatialState']:
        """Load spatial state from world-specific cache file"""
        try:
            import os
            cache_file = os.path.join(cache_dir, f"{world_id}_spatial_state.json")
            
            if not os.path.exists(cache_file):
                return None
            
            with open(cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Create new instance
            spatial_state = cls(world_id, cache_data.get("unity_constants"))
            
            # Restore position data
            pos_data = cache_data["current_position"]
            spatial_state.position = WorldPosition(
                pos_data["x"], pos_data["y"], pos_data["z"], pos_data["timestamp"]
            )
            spatial_state.facing_angle = pos_data["facing_angle"]
            
            spawn_data = cache_data["spawn_position"]
            spatial_state.spawn_position = WorldPosition(
                spawn_data["x"], spawn_data["y"], spawn_data["z"]
            )
            
            # Restore session stats
            stats = cache_data["session_stats"]
            spatial_state.session_start = stats["session_start"]
            spatial_state.total_distance_walked = stats["total_distance_walked"]
            spatial_state.total_distance_run = stats["total_distance_run"]
            spatial_state.position_confidence = stats["position_confidence"]
            
            # Log cache load
            log_engine("spatial.cache.load", world_id=world_id, cache_file=cache_file,
                      position={"x": spatial_state.position.x, "y": spatial_state.position.y},
                      session_distance=spatial_state.total_distance_walked + spatial_state.total_distance_run)
            
            print(f"Spatial state loaded from cache for world: {world_id}")
            return spatial_state
            
        except Exception as e:
            print(f"WARNING: Failed to load spatial state cache: {e}")
            return None


# Integration function for MidnightCore
def create_spatial_state_for_world(world_id: str, auto_calibrator=None) -> BetaSpatialState:
    """
    Create spatial state instance integrated with MidnightCore systems
    
    Args:
        world_id: Current VRChat world ID
        auto_calibrator: Auto-calibrator instance for Unity constants
        
    Returns:
        Initialized spatial state
    """
    unity_constants = None
    if auto_calibrator:
        unity_constants = auto_calibrator.get_unity_constants()
    
    spatial_state = BetaSpatialState(world_id, unity_constants)
    
    # Try to load from cache first
    cache_dir = "G:/Experimental/Production/MidnightCore/Core/Common/cache/spatial_maps"
    cached_state = BetaSpatialState.load_from_cache(world_id, cache_dir)
    
    if cached_state:
        print(f"Using cached spatial state for world: {world_id}")
        return cached_state
    else:
        print(f"Created new spatial state for world: {world_id}")
        return spatial_state


if __name__ == "__main__":
    # Test spatial state system
    print("Testing BetaSpatialState...")
    
    # Create test spatial state
    spatial = BetaSpatialState("test_world_001")
    
    # Test movement tracking
    print("\nTesting movement tracking:")
    result1 = spatial.update_movement("move_forward", 1.0, 1.0, False)
    print(f"Forward 1s: {result1}")
    
    result2 = spatial.update_movement("look_right", 0.5)
    spatial.update_facing("look_right", 0.5)
    
    result3 = spatial.update_movement("strafe_left", 0.5, 1.0, False)
    print(f"Strafe left 0.5s: {result3}")
    
    # Test blocked movement
    result4 = spatial.update_movement("move_forward", 1.0, 1.0, False, True, "wall detected")
    print(f"Blocked movement: {result4}")
    
    # Print summary
    summary = spatial.get_spatial_summary()
    print(f"\nSpatial Summary: {json.dumps(summary, indent=2)}")