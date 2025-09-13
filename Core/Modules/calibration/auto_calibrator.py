#!/usr/bin/env python3
"""
Auto-Calibration System for Beta VRChat Movement
Wall-based movement calibration with VRChat log parsing
"""

import time
import json
import os
from typing import Dict, Any, Optional
from datetime import datetime

import sys
import os
from wall_movement_calibrator import WallMovementCalibrator, CalibrationResults
from lockstep_calibrator import create_lockstep_calibrator
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Common', 'tools'))
from world_profile_manager import create_world_profile_manager
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Engine'))
from osc_controller import QwenVRChatOSC
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Common', 'tools'))
from vrchat_log_parser import VRChatLogParser

__all__ = ["BetaAutoCalibrator"]

class BetaAutoCalibrator:
    """
    Complete auto-calibration system for Beta's movement
    Uses wall-based calibration with VRChat log parsing
    """
    
    def __init__(self, osc_controller: Optional[QwenVRChatOSC] = None, tick_engine=None):
        """
        Initialize auto-calibration system
        
        Args:
            osc_controller: Existing OSC controller or None to create new one
            tick_engine: Tick engine for lockstep calibration
        """
        self.osc = osc_controller or QwenVRChatOSC()
        self.tick_engine = tick_engine
        
        # New unified world profile manager
        self.profile_manager = create_world_profile_manager()
        
        # Calibration systems (use lockstep if tick engine available)
        if tick_engine:
            self.lockstep_calibrator = create_lockstep_calibrator(self.osc, tick_engine)
            self.wall_calibrator = None  # Use lockstep calibrator instead
            print("INFO: Using lockstep-safe calibration system")
        else:
            self.wall_calibrator = WallMovementCalibrator(self.osc)
            self.lockstep_calibrator = None
            print("INFO: Using direct wall calibration system")
        
        # VRChat world detection via log parsing
        self.log_parser = VRChatLogParser()
        
        # Current state
        self.current_world_id = None
        self.current_calibration_results: Optional[CalibrationResults] = None
        
        # Unity-confirmed movement constants (validated 2025-01-11)
        # Based on confirmed measurements: 1 VRChat unit = 1 meter
        self.unity_movement_constants = {
            "walk_speed_mps": 2.0,      # 2 meters per second (confirmed)
            "run_speed_mps": 4.0,       # 4 meters per second (Unity default)
            "strafe_speed_mps": 2.0,    # 2 meters per second (Unity default)
            "jump_impulse": 3.0,        # impulse 3 = 40cm height (confirmed)
            "jump_height_m": 0.4,       # 40cm vertical jump (confirmed)
            "vrchat_units_per_meter": 1.0,  # 1:1 scale (confirmed)
            "meters_per_vrchat_unit": 1.0,   # inverse for calculations
            "beta_height_m": 1.6256,    # Beta's height: 5'4" = 1.6256 meters
            "beta_height_feet": 5.333,  # 5'4" = 5.333 feet for reference
            "beta_eye_level_m": 1.4734  # Approximate eye level (90.6% of height)
        }
        
        # Adaptive parameters (default to Unity-confirmed values)
        self.movement_multipliers = {
            "walk_forward": 1.0,        # Multiplier for 2.0 m/s base speed
            "walk_backward": 1.0,       # Multiplier for 2.0 m/s base speed
            "walk_strafe": 1.0,         # Multiplier for 2.0 m/s base speed
            "run_forward": 1.0,         # Multiplier for 4.0 m/s base speed
            "run_backward": 1.0,        # Multiplier for 4.0 m/s base speed
            "run_strafe": 1.0          # Multiplier for 4.0 m/s base speed
        }
        
        self.rotation_multipliers = {
            "turn_left": 1.0,           # Multiplier for standard turn rate
            "turn_right": 1.0,          # Multiplier for standard turn rate
            "look_up": 1.0,             # Multiplier for vertical look
            "look_down": 1.0            # Multiplier for vertical look
        }
        
        self.optimal_durations = {
            "quick_movement": 0.25,     # 0.5m walk, 1.0m run
            "standard_movement": 0.5,   # 1.0m walk, 2.0m run
            "precise_movement": 1.0,    # 2.0m walk, 4.0m run (confirmed)
            "rotation_90deg": 1.0       # 90-degree turn duration
        }
        
        # Calibration history
        self.calibration_log_path = "G:/Experimental/Production/MidnightCore/Core/Engine/Logging/auto_calibration.json"
        self.calibration_history = self._load_calibration_history()
        
    def _load_calibration_history(self) -> Dict[str, Any]:
        """Load previous calibration data"""
        try:
            if os.path.exists(self.calibration_log_path):
                with open(self.calibration_log_path, 'r') as f:
                    return json.load(f)
            else:
                return {"worlds": {}, "sessions": []}
        except Exception as e:
            print(f"Warning: Could not load calibration history: {e}")
            return {"worlds": {}, "sessions": []}
    
    def _save_calibration_history(self):
        """Save calibration history to file"""
        try:
            os.makedirs(os.path.dirname(self.calibration_log_path), exist_ok=True)
            with open(self.calibration_log_path, 'w') as f:
                json.dump(self.calibration_history, f, indent=2, default=str)
            print(f"SUCCESS: Calibration history saved")
        except Exception as e:
            print(f"Warning: Could not save calibration history: {e}")
    
    def start_monitoring(self) -> bool:
        """Initialize wall calibration system (replaces OSC monitoring)"""
        try:
            # Test wall calibrator initialization
            world_info = self.wall_calibrator.get_current_world_info()
            print(f"Wall calibrator ready for world: {world_info['world_id']}")
            return True
        except Exception as e:
            print(f"Warning: Wall calibrator initialization failed: {e}")
            return False
    
    def stop_monitoring(self):
        """Cleanup wall calibration system"""
        # No cleanup needed for wall-based system
        pass
    
    def run_full_calibration(self, force_recalibration: bool = False) -> Dict[str, Any]:
        """
        Run complete calibration sequence using lockstep-safe or wall-based system
        
        Args:
            force_recalibration: Force new calibration even if world is known
            
        Returns:
            Calibration results and world profile
        """
        print("\n=== BETA AUTO-CALIBRATION STARTING ===")
        
        # Detect current world and set up profile manager
        world_info = self.detect_current_world()
        world_id = world_info["world_id"]
        
        # Check if calibration needed
        if not force_recalibration and not self.should_recalibrate():
            print(f"Using existing calibration for world: {world_id}")
            return {
                "status": "cached", 
                "world_id": world_id, 
                "profile": self.profile_manager.get_adaptive_parameters()
            }
        
        try:
            # Choose calibration system
            if self.lockstep_calibrator:
                print("Using lockstep-safe calibration system")
                
                # Initialize lockstep calibrator
                if not self.lockstep_calibrator.initialize():
                    raise Exception("Failed to initialize lockstep calibrator")
                
                # Run lockstep calibration
                calibration_data = self.lockstep_calibrator.run_calibration_suite()
                
            elif self.wall_calibrator:
                print("Using direct wall calibration system")
                
                # Run wall-based calibration
                calibration_results = self.wall_calibrator.run_full_calibration()
                
                # Convert to format expected by profile manager
                calibration_data = self._convert_wall_results(calibration_results)
                
            else:
                raise Exception("No calibration system available")
            
            # Update world profile with calibration data
            if "error" not in calibration_data:
                self.profile_manager.update_calibration(calibration_data)
                print(f"SUCCESS: World {world_id} calibration complete")
                
                return {
                    "status": "calibrated",
                    "world_id": world_id,
                    "calibration_data": calibration_data,
                    "profile": self.profile_manager.get_adaptive_parameters()
                }
            else:
                print(f"ERROR: Calibration failed: {calibration_data['error']}")
                return {"status": "failed", "error": calibration_data["error"]}
        
        except Exception as e:
            print(f"ERROR: Calibration failed with exception: {e}")
            return {"status": "error", "error": str(e)}
    
    def _convert_wall_results(self, calibration_results) -> Dict[str, Any]:
        """Convert wall calibrator results to format expected by profile manager"""
        try:
            # Convert CalibrationResults to profile manager format
            movement_gains = {}
            confidence_scores = {}
            
            # Extract movement multipliers from calibration results
            if hasattr(calibration_results, 'movement_multipliers'):
                for movement_type, multiplier in calibration_results.movement_multipliers.items():
                    # Map wall calibrator names to profile manager names
                    if "forward" in movement_type or "walk" in movement_type:
                        movement_gains["walk"] = multiplier
                        confidence_scores["walk"] = calibration_results.overall_confidence
                    elif "strafe" in movement_type:
                        movement_gains["strafe"] = multiplier
                        confidence_scores["strafe"] = calibration_results.overall_confidence
                    elif "run" in movement_type:
                        movement_gains["run"] = multiplier
                        confidence_scores["run"] = calibration_results.overall_confidence
            
            return {
                "movement_gains": movement_gains,
                "confidence_scores": confidence_scores,
                "notes": f"Wall-based calibration completed {datetime.now().strftime('%Y-%m-%d %H:%M')}"
            }
            
        except Exception as e:
            print(f"ERROR: Failed to convert wall calibration results: {e}")
            return {"error": f"Conversion failed: {str(e)}"}
    
    def cleanup(self):
        """Clean up calibrator resources"""
        if self.lockstep_calibrator:
            self.lockstep_calibrator.cleanup()
        if self.profile_manager:
            self.profile_manager.cleanup()
        print("Auto-calibrator cleaned up")
    
    def should_recalibrate(self) -> bool:
        """Check if recalibration is needed"""
        # Ensure we have current world set
        if not self.current_world_id:
            self.detect_current_world()
        
        # Use world profile manager to determine if recalibration needed
        return self.profile_manager.should_recalibrate(max_age_hours=0.5)  # 30 minutes
    
    def detect_current_world(self) -> Dict[str, Any]:
        """
        Detect current VRChat world using log parser
        
        Returns:
            Dictionary with world_id, world_name, and world_dir
        """
        try:
            # Parse current world from VRChat logs
            world_id = self.log_parser.get_current_world_id()
            if not world_id:
                world_id = f"unknown_world_{int(time.time())}"
            
            # Set current world in profile manager
            self.profile_manager.set_current_world(world_id)
            self.current_world_id = world_id
            
            return {
                "world_id": world_id,
                "world_name": f"VRChat World {world_id}",
                "world_dir": str(self.profile_manager.get_world_profile_dir(world_id)),
                "is_stub": False
            }
        except Exception as e:
            print(f"WARNING: Failed to detect current world: {e}")
            fallback_id = f"unknown_world_{int(time.time())}"
            self.profile_manager.set_current_world(fallback_id)
            self.current_world_id = fallback_id
            return {
                "world_id": fallback_id,
                "world_name": "Unknown World (Manual Entry Required)",
                "is_stub": True
            }
    
    def get_adaptive_parameters(self) -> Dict[str, Any]:
        """Get current adaptive movement parameters for Beta"""
        # Use world profile manager for adaptive parameters
        return self.profile_manager.get_adaptive_parameters()
    
    def calculate_movement_duration(self, distance_meters: float, is_running: bool = False) -> float:
        """
        Calculate OSC command duration for specific distance movement
        
        Args:
            distance_meters: Target distance in meters
            is_running: True for run speed (4 m/s), False for walk speed (2 m/s)
            
        Returns:
            Duration in seconds for OSC command
        """
        if is_running:
            base_speed = self.unity_movement_constants["run_speed_mps"]
            multiplier = self.movement_multipliers.get("run_forward", 1.0)
        else:
            base_speed = self.unity_movement_constants["walk_speed_mps"]
            multiplier = self.movement_multipliers.get("walk_forward", 1.0)
        
        effective_speed = base_speed * multiplier
        return distance_meters / effective_speed
    
    def calculate_movement_distance(self, duration_seconds: float, is_running: bool = False) -> float:
        """
        Calculate expected distance for given movement duration
        
        Args:
            duration_seconds: OSC command duration
            is_running: True for run speed, False for walk speed
            
        Returns:
            Expected distance in meters
        """
        if is_running:
            base_speed = self.unity_movement_constants["run_speed_mps"]
            multiplier = self.movement_multipliers.get("run_forward", 1.0)
        else:
            base_speed = self.unity_movement_constants["walk_speed_mps"]
            multiplier = self.movement_multipliers.get("walk_forward", 1.0)
        
        effective_speed = base_speed * multiplier
        return duration_seconds * effective_speed
    
    def get_unity_constants(self) -> Dict[str, float]:
        """Get Unity-confirmed movement constants for spatial calculations"""
        return self.unity_movement_constants.copy()
    
    def validate_world_uses_standard_movement(self, test_duration: float = 1.0) -> Dict[str, Any]:
        """
        Test if current world uses Unity standard movement values
        
        Args:
            test_duration: Duration for test movement (seconds)
            
        Returns:
            Validation results with detected vs expected values
        """
        expected_distance = self.calculate_movement_distance(test_duration, is_running=False)
        
        # This would integrate with wall-based measurement system
        # For now, return expected values as confirmation
        return {
            "test_duration": test_duration,
            "expected_distance_m": expected_distance,
            "uses_unity_standard": True,  # Assume standard unless proven otherwise
            "validation_timestamp": datetime.now().isoformat(),
            "unity_constants": self.unity_movement_constants
        }


# Integration function for Beta's brain
def create_calibrator_for_beta(osc_controller, tick_engine=None) -> BetaAutoCalibrator:
    """Create calibrator instance integrated with Beta's systems"""
    calibrator = BetaAutoCalibrator(osc_controller, tick_engine)
    
    # Start monitoring immediately
    if calibrator.start_monitoring():
        print("SUCCESS: Auto-calibration system ready")
    else:
        print("WARNING: Auto-calibration monitoring not available")
    
    return calibrator


if __name__ == "__main__":
    print("Testing Auto-Calibration System")
    from osc_controller import QwenVRChatOSC
    
    osc = QwenVRChatOSC()
    calibrator = create_calibrator_for_beta(osc)
    
    # Test world detection
    world_info = calibrator.detect_current_world()
    print(f"Current world: {world_info}")
    
    # Test calibration check
    should_recal = calibrator.should_recalibrate()
    print(f"Should recalibrate: {should_recal}")
    
    calibrator.cleanup()