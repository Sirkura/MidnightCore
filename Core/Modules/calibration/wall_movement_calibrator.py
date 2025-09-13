# ==== MODULE CONTRACT =======================================================
# Module: calibration/wall_movement_calibrator.py
# Package: MidnightCore.Modules.calibration.wall_movement_calibrator
# Location: Production/MidnightCore/Core/Modules/calibration/wall_movement_calibrator.py
# Responsibility: Wall-referenced movement calibration using depth system measurements
# PUBLIC:
#   Classes: WallMovementCalibrator, CalibrationResults, MovementTestResult
#   Methods: run_full_calibration(), find_suitable_wall(), get_current_world_info()
# DEPENDENCIES:
#   Core: MidnightCore.Engine.config_manager (paths), MidnightCore.Engine.schemas (data types)
#   Modules: MidnightCore.Modules.calibration.depth_system_adapter (DepthSystemAdapter)
#            MidnightCore.Modules.movement.osc_controller (QwenVRChatOSC) - when migrated
#   Common: MidnightCore.Common.adaptors.log_parser_adaptor (VRChatLogParser) - when migrated
#   Legacy: (during migration) vrchat_log_parser, osc_controller
# DEPENDENTS:
#   MidnightCore.Modules.calibration.auto_calibrator
# POLICY: NO_FALLBACKS=deny, Telemetry: wall_calib.*
# DOCS: README anchor "WallMovementCalibrator", File-Map anchor "WALL_CALIBRATION_DOC_ANCHOR"
# MIGRATION: ✅ Migrated | Original: G:\Experimental\Midnight Core\FusionCore\FusionScripts\QwenScripts\wall_movement_calibrator.py
# INVARIANTS:
#   - Wall must be 2-8m away for safe calibration
#   - Wall stability score must be >0.8 before calibration
#   - Calibration files written atomically to prevent corruption
#   - Test durations calculated based on wall distance for safety
# PERFORMANCE:
#   - Complete calibration must finish within 60s budget
#   - Wall detection must complete within 30s
#   - Individual movement tests must complete within 5s each
# ============================================================================

import time
import json
import os
import math
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
from dataclasses import dataclass

# Import from new module structure
try:
    from ...Engine.config_manager import paths, env
    from .depth_system_adapter import DepthSystemAdapter
    from ...Common.adaptors.log_parser_adaptor import VRChatLogParser
    from ...Modules.movement.osc_controller import QwenVRChatOSC
except ImportError:
    # Legacy imports during migration
    print("wall_calib.warn: Using legacy imports during migration")
    
    # Add legacy path for imports
    import sys
    import os
    legacy_path = r"G:\Experimental\Midnight Core\FusionCore\FusionScripts\QwenScripts"
    if legacy_path not in sys.path:
        sys.path.insert(0, legacy_path)
    
    try:
        # Add new module paths
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Common', 'adaptors'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Common', 'tools'))
        sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', 'Engine'))
        from depth_system_adapter import DepthSystemAdapter
        from vrchat_log_parser import VRChatLogParser
        from osc_controller import QwenVRChatOSC
        
        # Mock paths during migration
        class MockPaths:
            storage = {"world_profiles": "G:/Experimental/Midnight Core/FusionCore/World Data"}
        paths = MockPaths()
        
        def env(name: str, default: str = "") -> str:
            return os.environ.get(name, default)
    except ImportError as e:
        print(f"wall_calib.error: Critical dependencies missing: {e}")
        raise

# Check NO_FALLBACKS policy
NO_FALLBACKS = env("NO_FALLBACKS", "0") == "1"

@dataclass
class MovementTestResult:
    """Results from a single movement test"""
    test_name: str
    expected_distance: float
    measured_distance: float
    duration: float
    accuracy_score: float
    world_multiplier: float

@dataclass 
class CalibrationResults:
    """Complete calibration results for a world"""
    world_id: str
    world_name: str
    timestamp: datetime
    
    # Initial wall detection data
    initial_wall_detection: Dict[str, Any]
    
    # Movement test results
    walk_tests: List[MovementTestResult]
    run_tests: List[MovementTestResult] 
    rotation_tests: List[MovementTestResult]
    look_tests: List[MovementTestResult]
    
    # Derived parameters
    movement_multipliers: Dict[str, float]
    rotation_multipliers: Dict[str, float]
    optimal_durations: Dict[str, float]
    
    # Quality metrics
    calibration_confidence: float
    wall_stability_score: float

class WallMovementCalibrator:
    """
    Wall-referenced movement calibration system
    
    Uses a single wall as fixed reference point to measure actual movement
    distances and compare against VRChat default movement speeds
    Respects NO_FALLBACKS policy - fails loud when depth system unavailable
    """
    
    # VRChat default movement parameters (Unity units/second)
    VRCHAT_DEFAULTS = {
        "walkSpeed": 2.0,      # units/second
        "runSpeed": 4.0,       # units/second  
        "strafeSpeed": 2.0,    # units/second (same as walk)
        "runStrafeSpeed": 4.0, # units/second (same as run)
        "rotationSpeed": 90.0, # degrees/second at full intensity
        "lookSpeed": 45.0      # degrees/second vertical look
    }
    
    def __init__(self, osc_controller: QwenVRChatOSC = None):
        print("wall_calib.init: Initializing WallMovementCalibrator")
        
        # Initialize dependencies
        try:
            self.osc = osc_controller or QwenVRChatOSC()
            print("wall_calib.init: OSC controller ready")
        except Exception as e:
            if NO_FALLBACKS:
                print(f"wall_calib.error: Cannot initialize OSC controller and NO_FALLBACKS=1: {e}")
                raise
            else:
                print(f"wall_calib.warn: OSC controller failed, continuing without: {e}")
                self.osc = None
        
        try:
            self.log_parser = VRChatLogParser()
            print("wall_calib.init: Log parser ready")
        except Exception as e:
            if NO_FALLBACKS:
                print(f"wall_calib.error: Cannot initialize log parser and NO_FALLBACKS=1: {e}")
                raise
            else:
                print(f"wall_calib.warn: Log parser failed, using fallback world detection: {e}")
                self.log_parser = None
        
        # Initialize depth system
        try:
            self.depth_system = DepthSystemAdapter()
            if not self.depth_system.initialize():
                error_msg = "Could not initialize depth system"
                print(f"wall_calib.error: {error_msg}")
                if NO_FALLBACKS:
                    raise RuntimeError(error_msg)
                else:
                    print("wall_calib.warn: Depth system unavailable, wall detection disabled")
                    self.depth_system = None
            else:
                print("wall_calib.init: Depth system ready")
        except Exception as e:
            if NO_FALLBACKS:
                print(f"wall_calib.error: Cannot initialize depth system and NO_FALLBACKS=1: {e}")
                raise
            else:
                print(f"wall_calib.warn: Depth system failed: {e}")
                self.depth_system = None
        
        # Use config_manager paths or fallback during migration
        try:
            self.world_data_dir = paths.storage["world_profiles"]
        except:
            # Legacy path during migration
            self.world_data_dir = "G:/Experimental/Midnight Core/FusionCore/World Data"
        
        os.makedirs(self.world_data_dir, exist_ok=True)
        print(f"wall_calib.init: World data directory: {self.world_data_dir}")
        
        # Calibration state
        self.current_wall_bearing = None
        self.current_wall_distance = None
        self.wall_stability_samples = []
        
    def get_current_world_info(self) -> Dict[str, str]:
        """Get current world ID and setup folder structure"""
        print("wall_calib.phase: Detecting current world")
        
        try:
            if self.log_parser:
                world_id = self.log_parser.get_current_world_from_logs()
            else:
                world_id = None
        except Exception as e:
            print(f"wall_calib.warn: Log parser failed: {e}")
            world_id = None
        
        if not world_id:
            if NO_FALLBACKS:
                raise RuntimeError("Cannot detect world and NO_FALLBACKS=1")
            
            # Fallback to timestamp-based ID
            world_id = f"unknown_world_{int(time.time())}"
            world_name = "Unknown World"
            print(f"wall_calib.warn: Could not detect world from VRChat logs, using: {world_id}")
        else:
            world_name = f"VRChat World {world_id[-8:]}"  # Last 8 chars of ID
            print(f"wall_calib.result: Detected world from VRChat logs: {world_id}")
        
        # Create world folder structure
        world_dir = os.path.join(self.world_data_dir, world_id)
        os.makedirs(world_dir, exist_ok=True)
        
        # Create mapping subdirectory
        mapping_dir = os.path.join(world_dir, "mapping")
        os.makedirs(mapping_dir, exist_ok=True)
        
        print(f"wall_calib.result: World directory ready: {world_dir}")
        
        return {
            "world_id": world_id,
            "world_name": world_name,
            "world_dir": world_dir,
            "mapping_dir": mapping_dir
        }
    
    def find_suitable_wall(self, scan_range: Tuple[float, float] = (0, 360), 
                          max_wall_distance: float = 8.0) -> Optional[Dict]:
        """
        Find a suitable wall for calibration using depth measurements
        
        Args:
            scan_range: (start_bearing, end_bearing) to scan
            max_wall_distance: Maximum acceptable wall distance
            
        Returns:
            Dict with wall info or None if no suitable wall found
        """
        print("wall_calib.phase: Scanning for suitable calibration wall")
        
        if not self.depth_system:
            error_msg = "Depth system not available for wall detection"
            print(f"wall_calib.error: {error_msg}")
            if NO_FALLBACKS:
                raise RuntimeError(error_msg)
            else:
                return None
        
        start_bearing, end_bearing = scan_range
        best_wall = None
        best_stability = 0.0
        
        # Scan in 10-degree increments
        scan_start_time = time.time()
        for bearing in range(int(start_bearing), int(end_bearing), 10):
            # Enforce wall detection timeout (30s as per contract)
            if time.time() - scan_start_time > 30:
                print("wall_calib.warn: Wall detection timeout (30s)")
                break
            
            distance = self.depth_system.get_clearance_at_bearing(bearing)
            
            if distance is not None and 2.0 <= distance <= max_wall_distance:
                # Test wall stability by taking multiple samples
                stability_samples = []
                for _ in range(5):
                    sample_distance = self.depth_system.get_clearance_at_bearing(bearing)
                    if sample_distance is not None:
                        stability_samples.append(sample_distance)
                    time.sleep(0.1)
                
                if len(stability_samples) >= 3:  # Need at least 3 valid samples
                    # Calculate stability score
                    avg_distance = sum(stability_samples) / len(stability_samples)
                    variance = sum((d - avg_distance) ** 2 for d in stability_samples) / len(stability_samples)
                    stability_score = 1.0 / (1.0 + variance * 10)  # Higher score = more stable
                    
                    if stability_score > best_stability and stability_score > 0.8:
                        best_wall = {
                            "bearing": bearing,
                            "distance": avg_distance,
                            "stability_score": stability_score,
                            "samples": stability_samples
                        }
                        best_stability = stability_score
        
        if best_wall:
            self.current_wall_bearing = best_wall["bearing"]
            self.current_wall_distance = best_wall["distance"]
            self.wall_stability_samples = best_wall["samples"]
            print(f"wall_calib.result: Found suitable wall: {best_wall['distance']:.1f}m at bearing {best_wall['bearing']}° (stability: {best_wall['stability_score']:.2f})")
            return best_wall
        else:
            error_msg = "No suitable wall found for calibration"
            print(f"wall_calib.error: {error_msg}")
            if NO_FALLBACKS:
                raise RuntimeError(error_msg)
            return None
    
    def measure_movement_distance(self, movement_function, duration: float, 
                                test_name: str = "movement") -> float:
        """
        Measure actual distance traveled using wall reference
        
        Args:
            movement_function: Function to execute movement
            duration: How long to move
            test_name: Name for logging
            
        Returns:
            Measured distance in units
        """
        if not self.current_wall_bearing:
            error_msg = "No wall reference established"
            print(f"wall_calib.error: {error_msg}")
            if NO_FALLBACKS:
                raise RuntimeError(error_msg)
            return 0.0
        
        if not self.depth_system:
            error_msg = f"Depth system unavailable for {test_name}"
            print(f"wall_calib.error: {error_msg}")
            if NO_FALLBACKS:
                raise RuntimeError(error_msg)
            return 0.0
        
        # Get initial wall distance
        initial_distance = self.depth_system.get_clearance_at_bearing(self.current_wall_bearing)
        if initial_distance is None or initial_distance <= 0:
            error_msg = f"Could not measure initial wall distance for {test_name}"
            print(f"wall_calib.error: {error_msg}")
            if NO_FALLBACKS:
                raise RuntimeError(error_msg)
            return 0.0
        
        print(f"wall_calib.phase: Executing {test_name} for {duration}s (initial wall distance: {initial_distance:.2f}m)")
        
        # Execute movement
        movement_function()
        time.sleep(duration)
        
        # Brief settling time for depth system
        time.sleep(0.2)
        
        # Measure final wall distance
        final_distance = self.depth_system.get_clearance_at_bearing(self.current_wall_bearing)
        if final_distance is None or final_distance <= 0:
            error_msg = f"Could not measure final wall distance for {test_name}"
            print(f"wall_calib.error: {error_msg}")
            if NO_FALLBACKS:
                raise RuntimeError(error_msg)
            return 0.0
        
        # Calculate distance moved (change in wall distance)
        distance_traveled = abs(final_distance - initial_distance)
        
        print(f"wall_calib.result: {test_name} - Wall distance changed from {initial_distance:.2f}m to {final_distance:.2f}m (moved {distance_traveled:.2f}m)")
        
        return distance_traveled
    
    def run_movement_test_suite(self) -> List[MovementTestResult]:
        """Run comprehensive movement tests (walk and run modes)"""
        print("wall_calib.phase: Starting movement test suite")
        
        if not self.osc:
            error_msg = "OSC controller not available for movement tests"
            print(f"wall_calib.error: {error_msg}")
            if NO_FALLBACKS:
                raise RuntimeError(error_msg)
            return []
        
        results = []
        
        # Calculate safe test duration based on wall distance
        # Use 40% of wall distance to prevent collisions, minimum 0.2s, maximum 1.0s
        if self.current_wall_distance and self.current_wall_distance > 0:
            safe_distance = self.current_wall_distance * 0.4  # 40% margin
            # Duration = distance / speed, with safety bounds
            test_duration = min(1.0, max(0.2, safe_distance / self.VRCHAT_DEFAULTS["walkSpeed"]))
            print(f"wall_calib.phase: Using dynamic test duration: {test_duration:.2f}s (wall distance: {self.current_wall_distance:.2f}m)")
        else:
            test_duration = 0.5  # Conservative fallback
            print(f"wall_calib.phase: Using fallback test duration: {test_duration:.2f}s (no wall distance available)")
        
        # Movement test matrix - walk tests
        movement_tests = [
            ("walk_forward", lambda: self.osc.move_forward(intensity=1.0, duration=test_duration), 
             self.VRCHAT_DEFAULTS["walkSpeed"] * test_duration),
            ("walk_backward", lambda: self.osc.move_backward(intensity=1.0, duration=test_duration),
             self.VRCHAT_DEFAULTS["walkSpeed"] * test_duration),
            ("walk_strafe_left", lambda: self.osc.strafe_left(intensity=1.0, duration=test_duration),
             self.VRCHAT_DEFAULTS["strafeSpeed"] * test_duration),
            ("walk_strafe_right", lambda: self.osc.strafe_right(intensity=1.0, duration=test_duration),
             self.VRCHAT_DEFAULTS["strafeSpeed"] * test_duration),
        ]
        
        # Run mode tests (shorter duration for faster run speed)
        run_duration = min(test_duration * 0.6, 0.4)  # Even shorter for run tests
        print(f"wall_calib.phase: Using run test duration: {run_duration:.2f}s")
        
        run_tests = [
            ("run_forward", lambda: self._run_movement(lambda: self.osc.move_forward(intensity=1.0, duration=run_duration)),
             self.VRCHAT_DEFAULTS["runSpeed"] * run_duration),
            ("run_strafe_right", lambda: self._run_movement(lambda: self.osc.strafe_right(intensity=1.0, duration=run_duration)),
             self.VRCHAT_DEFAULTS["runStrafeSpeed"] * run_duration),
        ]
        
        all_tests = movement_tests + run_tests
        
        for test_name, test_function, expected_distance in all_tests:
            print(f"wall_calib.phase: Testing {test_name}")
            
            # Allow brief setup time
            time.sleep(0.5)
            
            try:
                # Measure movement
                measured_distance = self.measure_movement_distance(
                    test_function, test_duration, test_name
                )
                
                # Calculate accuracy and world multiplier
                if expected_distance > 0:
                    accuracy_score = min(1.0, measured_distance / expected_distance)
                    
                    # Detect potential wall collision (measured distance much less than expected)
                    if measured_distance < expected_distance * 0.3:
                        print(f"wall_calib.warn: Possible wall collision detected ({measured_distance:.2f}m << {expected_distance:.2f}m)")
                        # Cap multiplier to reasonable range for wall collisions
                        world_multiplier = min(5.0, expected_distance / max(0.1, measured_distance))
                    else:
                        world_multiplier = expected_distance / max(0.1, measured_distance)
                        
                    # Cap extreme multipliers regardless
                    world_multiplier = min(10.0, max(0.1, world_multiplier))
                else:
                    accuracy_score = 0.0
                    world_multiplier = 1.0
                
                result = MovementTestResult(
                    test_name=test_name,
                    expected_distance=expected_distance,
                    measured_distance=measured_distance,
                    duration=test_duration,
                    accuracy_score=accuracy_score,
                    world_multiplier=world_multiplier
                )
                
                results.append(result)
                print(f"wall_calib.result: {test_name} - Expected: {expected_distance:.2f}m, Measured: {measured_distance:.2f}m, Multiplier: {world_multiplier:.2f}")
                
            except Exception as e:
                print(f"wall_calib.error: {test_name} failed: {e}")
                if NO_FALLBACKS:
                    raise
                # Continue with other tests
            
            # Brief pause between tests
            time.sleep(1.0)
        
        print(f"wall_calib.result: Movement test suite complete - {len(results)} tests")
        return results
    
    def _run_movement(self, movement_function):
        """Execute movement with run modifier enabled"""
        if not self.osc:
            return
        # Enable run
        self.osc.client.send_message("/input/Run", 1)
        time.sleep(0.05)  # Brief delay for run to register
        
        # Execute movement
        movement_function()
        
        # Disable run
        self.osc.client.send_message("/input/Run", 0)
    
    def calculate_world_multipliers(self, all_results: List[MovementTestResult]) -> Dict[str, Any]:
        """Calculate world-specific movement multipliers from test results"""
        
        # Group results by movement type
        walk_results = [r for r in all_results if "walk" in r.test_name]
        run_results = [r for r in all_results if "run" in r.test_name]
        
        # Calculate average multipliers by category
        def calc_avg_multiplier(results: List[MovementTestResult]) -> float:
            if not results:
                return 1.0
            return sum(r.world_multiplier for r in results) / len(results)
        
        movement_multipliers = {
            "walk_forward": calc_avg_multiplier([r for r in walk_results if "forward" in r.test_name]),
            "walk_backward": calc_avg_multiplier([r for r in walk_results if "backward" in r.test_name]),
            "walk_strafe": calc_avg_multiplier([r for r in walk_results if "strafe" in r.test_name]),
            "run_forward": calc_avg_multiplier([r for r in run_results if "forward" in r.test_name]),
            "run_strafe": calc_avg_multiplier([r for r in run_results if "strafe" in r.test_name]),
        }
        
        # Mock rotation multipliers (not implemented yet)
        rotation_multipliers = {
            "turn_left": 1.0,
            "turn_right": 1.0,
            "look_up": 1.0,
            "look_down": 1.0,
        }
        
        # Calculate optimal durations based on accuracy
        avg_accuracy = sum(r.accuracy_score for r in all_results) / len(all_results) if all_results else 0.5
        
        optimal_durations = {
            "quick_movement": 0.25 / max(0.5, avg_accuracy),  # Adjust based on world responsiveness
            "standard_movement": 0.5 / max(0.5, avg_accuracy),
            "precise_movement": 1.0 / max(0.5, avg_accuracy),
            "rotation_90deg": 1.0
        }
        
        return {
            "movement_multipliers": movement_multipliers,
            "rotation_multipliers": rotation_multipliers,
            "optimal_durations": optimal_durations
        }
    
    def run_full_calibration(self) -> CalibrationResults:
        """
        Execute complete wall-based calibration sequence
        
        Returns:
            Complete calibration results
        """
        print("wall_calib.phase: Starting full wall-based calibration")
        calibration_start_time = time.time()
        
        # Get world info and setup directories
        world_info = self.get_current_world_info()
        
        # Find calibration wall
        wall_info = self.find_suitable_wall()
        if not wall_info:
            error_msg = "No suitable calibration wall found"
            if NO_FALLBACKS:
                raise RuntimeError(error_msg)
            else:
                # Return minimal results for fallback
                return CalibrationResults(
                    world_id=world_info["world_id"],
                    world_name=world_info["world_name"],
                    timestamp=datetime.now(),
                    initial_wall_detection={},
                    walk_tests=[],
                    run_tests=[],
                    rotation_tests=[],
                    look_tests=[],
                    movement_multipliers={},
                    rotation_multipliers={},
                    optimal_durations={},
                    calibration_confidence=0.0,
                    wall_stability_score=0.0
                )
        
        # Create initial wall detection record
        initial_wall_detection = {
            "wall_bearing": wall_info["bearing"],
            "wall_distance_m": wall_info["distance"],
            "stability_score": wall_info["stability_score"],
            "stability_samples": wall_info["samples"],
            "detection_timestamp": datetime.now().isoformat(),
            "scan_method": "depth_system_bearing_sweep",
            "notes": f"Wall detected at {wall_info['distance']:.2f}m from bearing {wall_info['bearing']}°"
        }
        
        print(f"wall_calib.phase: Starting calibration for world: {world_info['world_id']}")
        print(f"wall_calib.phase: Using wall at {wall_info['distance']:.1f}m, bearing {wall_info['bearing']}°")
        print(f"wall_calib.phase: Wall stability score: {wall_info['stability_score']:.2f}")
        
        # Run movement tests
        movement_results = self.run_movement_test_suite()
        
        # Calculate world multipliers
        multipliers = self.calculate_world_multipliers(movement_results)
        
        # Calculate quality metrics
        if movement_results:
            avg_accuracy = sum(r.accuracy_score for r in movement_results) / len(movement_results)
        else:
            avg_accuracy = 0.0
            
        wall_stability = wall_info.get("stability_score", 0.5)
        calibration_confidence = (avg_accuracy * 0.7) + (wall_stability * 0.3)
        
        # Check calibration timeout (60s as per contract)
        elapsed_time = time.time() - calibration_start_time
        if elapsed_time > 60:
            print(f"wall_calib.warn: Calibration took {elapsed_time:.1f}s (>60s budget)")
        
        # Create results object
        results = CalibrationResults(
            world_id=world_info["world_id"],
            world_name=world_info["world_name"],
            timestamp=datetime.now(),
            initial_wall_detection=initial_wall_detection,
            walk_tests=[r for r in movement_results if "walk" in r.test_name],
            run_tests=[r for r in movement_results if "run" in r.test_name],
            rotation_tests=[],  # Not implemented yet
            look_tests=[],      # Not implemented yet
            movement_multipliers=multipliers["movement_multipliers"],
            rotation_multipliers=multipliers["rotation_multipliers"],
            optimal_durations=multipliers["optimal_durations"],
            calibration_confidence=calibration_confidence,
            wall_stability_score=wall_stability
        )
        
        # Save calibration data
        self.save_calibration_results(results, world_info["world_dir"])
        
        print(f"wall_calib.result: Wall-based calibration complete")
        print(f"wall_calib.result: Wall: {initial_wall_detection['wall_distance_m']:.2f}m at {initial_wall_detection['wall_bearing']}°")
        print(f"wall_calib.result: Confidence: {calibration_confidence:.2f}")
        print(f"wall_calib.result: Stability: {wall_stability:.2f}")
        print(f"wall_calib.result: Elapsed: {elapsed_time:.1f}s")
        
        return results
    
    def save_calibration_results(self, results: CalibrationResults, world_dir: str):
        """Save calibration results to world directory (atomic write)"""
        
        # Create comprehensive calibration file
        calibration_data = {
            # World information
            "world_id": results.world_id,
            "world_name": results.world_name,
            "timestamp": results.timestamp.isoformat(),
            "calibration_version": "2.1",
            "calibration_type": "wall_based",
            
            # Initial wall detection info
            "initial_wall_detection": results.initial_wall_detection,
            
            # VRChat baseline values used
            "vrchat_baseline": self.VRCHAT_DEFAULTS,
            
            # Test results by category
            "test_results": {
                "walk_tests": [
                    {
                        "test_name": t.test_name,
                        "expected_distance": t.expected_distance,
                        "measured_distance": t.measured_distance,
                        "duration": t.duration,
                        "accuracy_score": t.accuracy_score,
                        "world_multiplier": t.world_multiplier
                    } for t in results.walk_tests
                ],
                "run_tests": [
                    {
                        "test_name": t.test_name,
                        "expected_distance": t.expected_distance,
                        "measured_distance": t.measured_distance,
                        "duration": t.duration,
                        "accuracy_score": t.accuracy_score,
                        "world_multiplier": t.world_multiplier
                    } for t in results.run_tests
                ]
            },
            
            # Derived movement parameters
            "movement_multipliers": results.movement_multipliers,
            "rotation_multipliers": results.rotation_multipliers,
            "optimal_durations": results.optimal_durations,
            
            # Quality metrics
            "calibration_confidence": results.calibration_confidence,
            "wall_stability_score": results.wall_stability_score
        }
        
        # Atomic write - save to temp file then rename
        calibration_file = os.path.join(world_dir, "calibration.json")
        temp_file = calibration_file + ".tmp"
        
        try:
            with open(temp_file, 'w') as f:
                json.dump(calibration_data, f, indent=2, default=str)
            
            # Atomic rename
            os.replace(temp_file, calibration_file)
            print(f"wall_calib.result: Calibration saved atomically: {calibration_file}")
        except Exception as e:
            print(f"wall_calib.error: Failed to save calibration: {e}")
            if NO_FALLBACKS:
                raise
    
    def cleanup(self):
        """Clean up calibrator resources"""
        try:
            if self.depth_system:
                self.depth_system.cleanup()
            print("wall_calib.cleanup: WallMovementCalibrator cleaned up")
        except Exception as e:
            print(f"wall_calib.warn: Cleanup error: {e}")


def test_wall_calibrator():
    """Test the wall movement calibrator"""
    print("Testing Wall Movement Calibrator...")
    
    try:
        calibrator = WallMovementCalibrator()
        
        # Test world detection
        world_info = calibrator.get_current_world_info()
        print(f"World: {world_info}")
        
        # Test wall finding
        wall = calibrator.find_suitable_wall()
        if wall:
            print(f"Found wall: {wall}")
            return True
        else:
            print("No wall found")
            return False
    except Exception as e:
        print(f"Test failed: {e}")
        if NO_FALLBACKS:
            raise
        return False

if __name__ == "__main__":
    # Test just the wall calibrator components
    try:
        result = test_wall_calibrator()
        if result:
            print("SUCCESS: Wall calibrator test passed")
        else:
            print("FAILED: Wall calibrator test failed")
    except Exception as e:
        print(f"ERROR: Wall calibrator test exception: {e}")
        if NO_FALLBACKS:
            raise