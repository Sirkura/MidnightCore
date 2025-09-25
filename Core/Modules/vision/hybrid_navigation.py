#!/usr/bin/env python3
"""
Hybrid Navigation System Integration
===================================

PURPOSE: Two-tier navigation safety system for Beta's movement decisions
- Tier 1: Fast Depth Anything V2 threat detection (continuous via DepthWorker)  
- Tier 2: Florence-2 semantic verification (called only when needed)

INTEGRATION: Plugs into existing brain architecture through state_bus
- DepthWorker already provides continuous depth analysis
- StateEstimator already computes clearances  
- This module adds semantic verification layer
- Brain calls hybrid_navigation_check() before movement

CONTRACT:
- get_navigation_safety(vision_state, image_bgr=None) -> safety decision
- verify_depth_threats(threats, image_bgr) -> semantic verification
- integrate with existing get_vision_state() pipeline
"""

import sys
import os
import numpy as np
import cv2
from collections import deque
from typing import Optional, Dict, List, Tuple, Any

# Import existing modules with relative imports
try:
    from .depth_visualizer import save_navigation_decision_image, should_save_decision_image
    DEPTH_VISUALIZATION_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: Depth visualization not available: {e}")
    DEPTH_VISUALIZATION_AVAILABLE = False
    def save_navigation_decision_image(*args, **kwargs):
        pass
    def should_save_decision_image(*args, **kwargs):
        return False

try:
    from .florence_analyzer import get_florence_analyzer
    FLORENCE_AVAILABLE = True
except ImportError:
    print("WARNING: Florence analyzer not available for hybrid navigation")
    FLORENCE_AVAILABLE = False

try:
    from ...Engine.state_bus import get_vision_state
    STATE_BUS_AVAILABLE = True
except ImportError:
    print("WARNING: State bus not available for hybrid navigation")
    STATE_BUS_AVAILABLE = False

# Import logging system
try:
    from ...Common.Tools.logging_bus import (
        log_hybrid_nav_start, log_depth_analysis_tier1, log_florence_verification_tier2,
        log_navigation_decision, log_movement_safety_check, log_temporal_smoothing_buffer,
        log_regional_percentiles
    )
    LOGGING_AVAILABLE = True
except ImportError:
    print("WARNING: Logging bus not available for hybrid navigation")
    LOGGING_AVAILABLE = False

class HybridNavigationSystem:
    """
    Two-tier navigation safety system combining depth + semantic analysis
    """
    
    def __init__(self):
        """Initialize hybrid navigation system"""
        self.florence_analyzer = None
        if FLORENCE_AVAILABLE:
            self.florence_analyzer = get_florence_analyzer()
        
        # Relaxed depth thresholds (less paranoid than pure depth approach)
        self.depth_thresholds = {
            'cliff_ratio': 0.08,      # Only very close objects = potential cliff
            'wall_ratio': 0.25,       # Closer wall detection  
            'clear_ratio': 0.50       # Lower clear threshold
        }
        
        # Cache for avoiding redundant Florence calls
        self.last_florence_call = 0
        self.florence_cooldown = 2.0  # seconds
        self.last_verification = None
        
        # Temporal smoothing buffers (5-frame rolling window)
        self.region_buffers = {}
        self.buffer_size = 5
    
    def roi_stats(self, area: np.ndarray) -> Dict[str, float]:
        """
        Calculate robust regional statistics using percentiles
        More stable than min/max approach, resistant to outliers
        
        Args:
            area: Depth region as numpy array
            
        Returns:
            Dictionary with percentile statistics
        """
        return {
            "p10": float(np.percentile(area, 10)),
            "p25": float(np.percentile(area, 25)),
            "p50": float(np.median(area)),
            "p75": float(np.percentile(area, 75)),
            "p90": float(np.percentile(area, 90)),
            "std": float(np.std(area)),
            "iqr": float(np.percentile(area, 75) - np.percentile(area, 25))
        }
        
    def analyze_depth_threats(self, vision_state: Dict) -> Dict:
        """
        Tier 1: Fast depth threat detection using existing StateEstimator output
        
        Args:
            vision_state: Current vision state from state_bus
            
        Returns:
            Dict with threats, region analysis, and Florence verification needs
        """
        if not vision_state:
            return {
                'threats': [],
                'needs_florence_verification': False,
                'threat_summary': 'No vision data available'
            }
        
        # Extract depth measurements from existing state estimator
        front_m = vision_state.get('front_m', float('inf'))
        left_m = vision_state.get('left_m', float('inf'))  
        right_m = vision_state.get('right_m', float('inf'))
        edge_risk = vision_state.get('edge_risk', 0.0)
        ttc_s = vision_state.get('ttc_s', float('inf'))
        
        # Get detailed clearance data if available (31-bearing analysis)
        clearance_by_bearing = vision_state.get('clearance_by_bearing', {})
        
        threats = []
        needs_florence = False
        
        # Analyze immediate cliff dangers (using edge_risk from StateEstimator)
        if edge_risk > 0.7:  # High edge risk indicates potential cliff
            threats.append({
                'type': 'potential_cliff',
                'region': 'forward',
                'severity': edge_risk,
                'description': f'High edge risk detected: {edge_risk:.2f}'
            })
            needs_florence = True
        
        # Analyze wall proximity using relaxed thresholds
        if front_m < 1.5:  # Closer than 1.5m = potential wall
            if front_m < 0.8:  # Very close = might be obstacle vs wall
                threats.append({
                    'type': 'potential_obstacle',
                    'region': 'forward', 
                    'distance': front_m,
                    'description': f'Very close object ahead: {front_m:.2f}m'
                })
                needs_florence = True
            else:
                threats.append({
                    'type': 'wall_close',
                    'region': 'forward',
                    'distance': front_m, 
                    'description': f'Wall close ahead: {front_m:.2f}m'
                })
        
        # Check side clearances for navigation options
        side_clearances = {'left': left_m, 'right': right_m}
        for side, distance in side_clearances.items():
            if distance < 1.0:  # Side too close for safe movement
                threats.append({
                    'type': 'side_blocked',
                    'region': side,
                    'distance': distance,
                    'description': f'{side.title()} side blocked: {distance:.2f}m'
                })
        
        return {
            'threats': threats,
            'needs_florence_verification': needs_florence,
            'threat_summary': f'{len(threats)} threats detected',
            'measurements': {
                'front_m': front_m,
                'left_m': left_m, 
                'right_m': right_m,
                'edge_risk': edge_risk,
                'ttc_s': ttc_s
            }
        }
    
    def analyze_depth_threats_enhanced(self, depth_map: np.ndarray, tick_id: int = None, frame_id: int = None) -> Dict:
        """
        Enhanced depth threat detection using regional percentiles and temporal smoothing
        More robust than global min/max approach, resistant to lighting changes
        
        Args:
            depth_map: Raw depth map from Depth Anything V2
            
        Returns:
            Dict with enhanced threat analysis
        """
        if depth_map is None:
            return {
                'threats': [],
                'needs_florence_verification': False,
                'threat_summary': 'No depth map available'
            }
        
        height, width = depth_map.shape
        
        print(f"\n=== ENHANCED DEPTH THREAT ANALYSIS ===")
        print(f"Depth map shape: {depth_map.shape}")
        print(f"Depth range: {depth_map.min():.3f} - {depth_map.max():.3f}")
        
        # Define analysis regions (same as before)
        regions = {
            'immediate_forward': depth_map[int(height*0.7):height, int(width*0.3):int(width*0.7)],
            'forward_path': depth_map[int(height*0.4):int(height*0.6), int(width*0.35):int(width*0.65)],
            'forward_wall': depth_map[int(height*0.25):int(height*0.45), int(width*0.4):int(width*0.6)],
            'left_side': depth_map[int(height*0.3):int(height*0.7), :int(width*0.3)],
            'right_side': depth_map[int(height*0.3):int(height*0.7), int(width*0.7):]
        }
        
        # Calculate robust stats for each region with temporal smoothing
        region_stats = {}
        for name, area in regions.items():
            stats = self.roi_stats(area)
            
            # Initialize buffer if needed
            if name not in self.region_buffers:
                self.region_buffers[name] = deque(maxlen=self.buffer_size)
            
            # Add current median to temporal buffer
            self.region_buffers[name].append(stats['p50'])
            
            # Use temporally smoothed median
            smoothed_median = float(np.median(list(self.region_buffers[name])))
            
            region_stats[name] = {
                **stats,
                'smoothed_median': smoothed_median,
                'raw_median': stats['p50']
            }
            
            # Log regional statistics and temporal smoothing (deep trace only)
            if LOGGING_AVAILABLE and tick_id and frame_id:
                log_regional_percentiles(tick_id, frame_id, name, stats)
                log_temporal_smoothing_buffer(tick_id, frame_id, name, stats['p50'], smoothed_median, len(self.region_buffers[name]))
        
        # Enhanced threat detection using robust regional comparisons
        threats = []
        
        # Get reference stats for comparisons
        forward_path = region_stats['forward_path']
        forward_wall = region_stats['forward_wall'] 
        immediate = region_stats['immediate_forward']
        left_side = region_stats['left_side']
        right_side = region_stats['right_side']
        
        print(f"Forward Path (smoothed): {forward_path['smoothed_median']:.3f}")
        print(f"Forward Wall (smoothed): {forward_wall['smoothed_median']:.3f}")
        print(f"Immediate (smoothed): {immediate['smoothed_median']:.3f}")
        
        # CLIFF DETECTION: Immediate area significantly different from expected ground plane
        cliff_threshold = forward_path['smoothed_median'] - (2.0 * forward_path['iqr'])
        
        if immediate['smoothed_median'] < cliff_threshold:
            threats.append({
                'type': 'potential_cliff',
                'region': 'immediate_forward',
                'confidence': 'high',
                'details': f"Immediate depth {immediate['smoothed_median']:.3f} << expected {cliff_threshold:.3f}"
            })
            print(f"[CLIFF THREAT] {immediate['smoothed_median']:.3f} < {cliff_threshold:.3f}")
        
        # WALL DETECTION: Forward wall closer than forward path + std margin
        wall_threshold = forward_path['smoothed_median'] + (0.5 * forward_path['std'])
        
        if forward_wall['smoothed_median'] > wall_threshold:
            threats.append({
                'type': 'wall_close',
                'region': 'forward_wall',
                'confidence': 'high', 
                'details': f"Wall depth {forward_wall['smoothed_median']:.3f} > threshold {wall_threshold:.3f}"
            })
            print(f"[WALL THREAT] {forward_wall['smoothed_median']:.3f} > {wall_threshold:.3f}")
        
        # CLEARANCE CHECK: Forward path should be similar to side clearances
        left_clearance = left_side['smoothed_median']
        right_clearance = right_side['smoothed_median']
        side_average = (left_clearance + right_clearance) / 2.0
        clearance_threshold = side_average - (1.0 * forward_path['std'])
        
        if forward_path['smoothed_median'] > clearance_threshold:
            threats.append({
                'type': 'blocked_path',
                'region': 'forward_path', 
                'confidence': 'medium',
                'details': f"Forward blocked: {forward_path['smoothed_median']:.3f} vs sides avg {side_average:.3f}"
            })
            print(f"[BLOCKED PATH] Forward {forward_path['smoothed_median']:.3f} vs sides {side_average:.3f}")
        
        if not threats:
            print("[NO THREATS] Path appears clear")
        
        # Determine Florence verification needs
        needs_florence = any(t['type'] == 'potential_cliff' for t in threats)
        
        # Build result
        result = {
            'threats': threats,
            'region_stats': region_stats,
            'needs_florence_verification': needs_florence,
            'raw_depth_map': depth_map,  # Include for visualization
            'temporal_buffers_ready': all(len(buf) >= 3 for buf in self.region_buffers.values()),
            'analysis_method': 'regional_percentiles_with_temporal_smoothing',
            'measurements': {
                'front_m': float(forward_path['smoothed_median']),
                'left_m': float(left_side['smoothed_median']),
                'right_m': float(right_side['smoothed_median']),
                'edge_risk': 1.0 if any(t['type'] == 'potential_cliff' for t in threats) else 0.0,
                'ttc_s': float('inf'),  # Not calculated in enhanced mode
                'temporal_buffers_ready': all(len(buf) >= 3 for buf in self.region_buffers.values())
            }
        }
        
        # Log Tier 1 depth analysis
        if LOGGING_AVAILABLE and tick_id and frame_id:
            log_depth_analysis_tier1(
                tick_id, frame_id,
                analysis_method='regional_percentiles_with_temporal_smoothing',
                threats_detected=len(threats),
                measurements=result['measurements'],
                needs_florence=needs_florence
            )
        
        return result
    
    def verify_threats_with_florence(self, threats: List[Dict], image_bgr: np.ndarray = None, tick_id: int = None, frame_id: int = None) -> Dict:
        """
        Tier 2: Florence-2 semantic verification of depth-detected threats
        
        Args:
            threats: List of threats from depth analysis
            image_bgr: Optional current frame for analysis
            
        Returns:
            Dict with verified threats and semantic analysis
        """
        if not FLORENCE_AVAILABLE or not self.florence_analyzer:
            return {
                'verification_available': False,
                'verified_threats': threats,  # Pass through unverified
                'semantic_analysis': 'Florence-2 not available'
            }
        
        # Rate limiting - avoid excessive Florence calls
        current_time = os.times().elapsed
        if current_time - self.last_florence_call < self.florence_cooldown:
            if self.last_verification:
                return self.last_verification
        
        try:
            if image_bgr is None:
                result = {
                    'verification_available': False,
                    'verified_threats': threats,
                    'semantic_analysis': 'No image provided for verification'
                }
                
                # Log Florence verification attempt
                if LOGGING_AVAILABLE and tick_id and frame_id:
                    log_florence_verification_tier2(
                        tick_id, frame_id,
                        verification_available=False,
                        florence_latency_ms=0,
                        threats_verified=0,
                        threats_demoted=0
                    )
                
                return result
            
            florence_start_time = os.times().elapsed
            verified_threats = []
            
            # Check for cliff/drop-off threats
            cliff_threats = [t for t in threats if t['type'] == 'potential_cliff']
            if cliff_threats:
                cliff_prompt = """Look at the floor area directly ahead. Is there a dangerous drop-off, 
                cliff, staircase going down, or major elevation change? Or is it just a floor feature 
                like a mat, logo, small step, or decoration? Focus on actual navigation hazards."""
                
                cliff_analysis = self.florence_analyzer.analyze_image(image_bgr, cliff_prompt)
                
                # Keyword analysis for danger assessment
                danger_words = ['cliff', 'drop', 'stair', 'dangerous', 'fall', 'edge', 'drop-off', 'pit', 'hole']
                safe_words = ['mat', 'logo', 'floor', 'safe', 'small', 'decoration', 'carpet', 'pattern']
                
                cliff_text = cliff_analysis.lower()
                danger_count = sum(1 for word in danger_words if word in cliff_text)
                safe_count = sum(1 for word in safe_words if word in cliff_text)
                
                for threat in cliff_threats:
                    if danger_count > safe_count:
                        verified_threats.append({
                            **threat,
                            'florence_verified': True,
                            'verification': 'real_danger',
                            'florence_analysis': cliff_analysis
                        })
                    else:
                        # Not a real cliff - demote to low priority
                        pass  # Don't add to verified threats
            
            # Check for obstacle vs wall distinction  
            obstacle_threats = [t for t in threats if t['type'] == 'potential_obstacle']
            if obstacle_threats:
                obstacle_prompt = """Look at objects directly ahead. Are they solid walls/barriers that 
                block movement, or are they smaller objects like furniture, decorations, or items that 
                could potentially be navigated around?"""
                
                obstacle_analysis = self.florence_analyzer.analyze_image(image_bgr, obstacle_prompt)
                
                # Classify as wall vs navigable obstacle
                wall_words = ['wall', 'barrier', 'solid', 'block', 'structure']
                object_words = ['furniture', 'decoration', 'item', 'object', 'small', 'around']
                
                obstacle_text = obstacle_analysis.lower()
                wall_count = sum(1 for word in wall_words if word in obstacle_text)
                object_count = sum(1 for word in object_words if word in obstacle_text)
                
                for threat in obstacle_threats:
                    if wall_count > object_count:
                        verified_threats.append({
                            **threat,
                            'florence_verified': True,
                            'verification': 'wall_confirmed',
                            'florence_analysis': obstacle_analysis
                        })
                    else:
                        # Navigable obstacle - lower priority
                        verified_threats.append({
                            **threat,
                            'florence_verified': True,
                            'verification': 'navigable_obstacle',
                            'florence_analysis': obstacle_analysis
                        })
            
            # Pass through non-verifiable threats (walls, side blocks)
            other_threats = [t for t in threats if t['type'] not in ['potential_cliff', 'potential_obstacle']]
            verified_threats.extend(other_threats)
            
            florence_end_time = os.times().elapsed
            florence_latency_ms = (florence_end_time - florence_start_time) * 1000
            
            # Count verification results
            threats_verified = len([t for t in verified_threats if t.get('florence_verified') and t.get('verification') in ['real_danger', 'wall_confirmed']])
            threats_demoted = len(cliff_threats + obstacle_threats) - threats_verified
            
            result = {
                'verification_available': True,
                'verified_threats': verified_threats,
                'semantic_analysis': f'Florence verified {len(cliff_threats + obstacle_threats)} threats'
            }
            
            # Log Florence verification results
            if LOGGING_AVAILABLE and tick_id and frame_id:
                log_florence_verification_tier2(
                    tick_id, frame_id,
                    verification_available=True,
                    florence_latency_ms=florence_latency_ms,
                    threats_verified=threats_verified,
                    threats_demoted=threats_demoted
                )
            
            # Cache result
            self.last_florence_call = current_time
            self.last_verification = result
            
            return result
            
        except Exception as e:
            print(f"ERROR: Florence verification failed: {e}")
            return {
                'verification_available': False,
                'verified_threats': threats,
                'semantic_analysis': f'Florence error: {e}'
            }
    
    def make_navigation_decision(self, depth_analysis: Dict, florence_verification: Dict, tick_id: int = None, frame_id: int = None) -> Dict:
        """
        Final navigation decision combining depth + semantic verification
        
        Args:
            depth_analysis: Output from analyze_depth_threats()
            florence_verification: Output from verify_threats_with_florence()
            
        Returns:
            Navigation decision with movement safety and recommended actions
        """
        verified_threats = florence_verification['verified_threats']
        
        # Categorize verified threats
        real_cliffs = [t for t in verified_threats 
                      if t.get('verification') == 'real_danger']
        confirmed_walls = [t for t in verified_threats 
                          if t.get('verification') == 'wall_confirmed' or t['type'] == 'wall_close']
        navigable_obstacles = [t for t in verified_threats 
                              if t.get('verification') == 'navigable_obstacle']
        side_blocks = [t for t in verified_threats if t['type'] == 'side_blocked']
        
        # Decision logic
        if real_cliffs:
            decision = "STOP_CLIFF_DANGER"
            movement_allowed = False
            safe_actions = ["look_left", "look_right", "turn_around", "back_up"]
            reasoning = f"Florence verified {len(real_cliffs)} real cliff danger(s)"
            priority = "CRITICAL"
            
        elif confirmed_walls and depth_analysis['measurements']['front_m'] < 1.0:
            decision = "STOP_WALL_CLOSE"
            movement_allowed = False
            safe_actions = ["look_left", "look_right", "turn_left", "turn_right", "back_up"]
            reasoning = f"Wall confirmed at {depth_analysis['measurements']['front_m']:.1f}m"
            priority = "HIGH"
            
        elif len(side_blocks) >= 2:  # Both sides blocked
            decision = "STOP_BOXED_IN"
            movement_allowed = False
            safe_actions = ["back_up", "turn_around"]
            reasoning = "Both sides blocked, need to back up"
            priority = "HIGH"
            
        elif navigable_obstacles:
            decision = "NAVIGATE_AROUND"
            movement_allowed = True  # But with caution
            left_clear = depth_analysis['measurements']['left_m'] > 1.5
            right_clear = depth_analysis['measurements']['right_m'] > 1.5
            
            if left_clear and right_clear:
                safe_actions = ["turn_left", "turn_right", "small_step_forward"]
            elif left_clear:
                safe_actions = ["turn_left", "small_step_forward"] 
            elif right_clear:
                safe_actions = ["turn_right", "small_step_forward"]
            else:
                safe_actions = ["back_up", "turn_around"]
                
            reasoning = f"Navigable obstacles detected, {len([s for s in [left_clear, right_clear] if s])} clear sides"
            priority = "MEDIUM"
            
        elif depth_analysis['measurements']['front_m'] > 3.0:
            decision = "MOVE_FORWARD_CLEAR"
            movement_allowed = True
            safe_actions = ["move_forward", "continue_exploration"]
            reasoning = f"Clear path ahead: {depth_analysis['measurements']['front_m']:.1f}m"
            priority = "LOW"
            
        else:
            decision = "ASSESS_CAREFULLY" 
            movement_allowed = False
            safe_actions = ["look_around", "small_step_forward"]
            reasoning = "Ambiguous situation, need careful assessment"
            priority = "MEDIUM"
        
        result = {
            'decision': decision,
            'movement_allowed': movement_allowed,
            'safe_actions': safe_actions,
            'reasoning': reasoning,
            'priority': priority,
            'threat_count': len(verified_threats),
            'florence_used': florence_verification['verification_available'],
            'measurements': depth_analysis['measurements'],
            'temporal_buffers_ready': depth_analysis.get('temporal_buffers_ready', True)
        }
        
        # Log navigation decision
        if LOGGING_AVAILABLE and tick_id and frame_id:
            log_navigation_decision(
                tick_id, frame_id,
                decision=decision,
                movement_allowed=movement_allowed,
                safe_actions=safe_actions,
                reasoning=reasoning,
                priority=priority,
                florence_used=florence_verification['verification_available']
            )
        
        return result
    
    def make_enhanced_navigation_decision(self, depth_analysis: Dict, florence_verification: Dict, tick_id: int = None, frame_id: int = None) -> Dict:
        """
        Enhanced navigation decision with temporal buffer awareness
        
        Args:
            depth_analysis: Output from analyze_depth_threats_enhanced()
            florence_verification: Output from verify_threats_with_florence()
            
        Returns:
            Navigation decision with temporal smoothing status
        """
        # Check if temporal buffers are ready for stable analysis
        buffers_ready = depth_analysis.get('temporal_buffers_ready', False)
        if not buffers_ready:
            return {
                'decision': 'INITIALIZING_TEMPORAL_BUFFERS',
                'movement_allowed': False,
                'safe_actions': ['look_around', 'small_rotation'],
                'reasoning': 'Building temporal smoothing buffers for stable analysis',
                'priority': 'SYSTEM',
                'threat_count': 0,
                'florence_used': False,
                'temporal_buffers_ready': False,
                'measurements': depth_analysis['measurements']
            }
        
        # Use standard decision logic with enhanced threat analysis
        verified_threats = florence_verification['verified_threats']
        
        # Prioritize threats by severity (enhanced categories)
        cliff_threats = [t for t in verified_threats if 'cliff' in t['type']]
        verified_cliff_threats = [t for t in cliff_threats if t.get('florence_verified', False)]
        wall_threats = [t for t in verified_threats if t['type'] == 'wall_close']
        blocked_threats = [t for t in verified_threats if t['type'] == 'blocked_path']
        
        if verified_cliff_threats:
            result = {
                'decision': 'STOP_VERIFIED_CLIFF_DANGER',
                'movement_allowed': False,
                'safe_actions': ['look_left', 'look_right', 'turn_around'],
                'reasoning': f'Florence verified {len(verified_cliff_threats)} cliff danger(s) with temporal smoothing',
                'priority': 'CRITICAL',
                'threat_count': len(verified_cliff_threats),
                'florence_used': florence_verification['verification_available'],
                'temporal_buffers_ready': True,
                'measurements': depth_analysis['measurements']
            }
            
            # Log enhanced navigation decision
            if LOGGING_AVAILABLE and tick_id and frame_id:
                log_navigation_decision(
                    tick_id, frame_id,
                    decision='STOP_VERIFIED_CLIFF_DANGER',
                    movement_allowed=False,
                    safe_actions=['look_left', 'look_right', 'turn_around'],
                    reasoning=f'Florence verified {len(verified_cliff_threats)} cliff danger(s) with temporal smoothing',
                    priority='CRITICAL',
                    florence_used=florence_verification['verification_available']
                )
                
                # Save depth visualization for major cliff danger decision
                if DEPTH_VISUALIZATION_AVAILABLE and should_save_decision_image('STOP_CLIFF_DANGER'):
                    raw_depth_map = depth_analysis.get('raw_depth_map')
                    if raw_depth_map is not None:
                        threat_summary = f"{len(verified_cliff_threats)} verified cliff danger(s) detected"
                        save_navigation_decision_image(
                            raw_depth_map, 'STOP_CLIFF_DANGER', tick_id, frame_id, threat_summary
                        )
            
            return result
        elif cliff_threats:
            return {
                'decision': 'ASSESS_CAREFULLY_POTENTIAL_CLIFF',
                'movement_allowed': False,
                'safe_actions': ['look_around', 'turn_left', 'turn_right'],
                'reasoning': f'{len(cliff_threats)} potential cliff(s) detected, needs Florence verification',
                'priority': 'HIGH',
                'threat_count': len(cliff_threats),
                'florence_used': florence_verification['verification_available'],
                'temporal_buffers_ready': True,
                'measurements': depth_analysis['measurements']
            }
        
        elif wall_threats:
            return {
                'decision': 'STOP_WALL_AHEAD',
                'movement_allowed': False,
                'safe_actions': ['look_left', 'look_right', 'turn_left', 'turn_right'],
                'reasoning': f'Wall detected in forward path with enhanced analysis',
                'priority': 'HIGH',
                'threat_count': len(wall_threats),
                'florence_used': florence_verification['verification_available'],
                'temporal_buffers_ready': True,
                'measurements': depth_analysis['measurements']
            }
        
        elif blocked_threats:
            return {
                'decision': 'CAUTION_PATH_UNCLEAR',
                'movement_allowed': False,
                'safe_actions': ['look_around', 'small_step_forward'],
                'reasoning': f'Forward path may be blocked (regional analysis)',
                'priority': 'MEDIUM',
                'threat_count': len(blocked_threats),
                'florence_used': florence_verification['verification_available'],
                'temporal_buffers_ready': True,
                'measurements': depth_analysis['measurements']
            }
        
        else:
            result = {
                'decision': 'MOVE_FORWARD_ENHANCED_SAFE',
                'movement_allowed': True,
                'safe_actions': ['move_forward'],
                'reasoning': 'No threats detected with enhanced regional analysis and temporal smoothing',
                'priority': 'LOW',
                'threat_count': 0,
                'florence_used': florence_verification['verification_available'],
                'temporal_buffers_ready': True,
                'measurements': depth_analysis['measurements']
            }
            
            # Log enhanced navigation decision
            if LOGGING_AVAILABLE and tick_id and frame_id:
                log_navigation_decision(
                    tick_id, frame_id,
                    decision='MOVE_FORWARD_ENHANCED_SAFE',
                    movement_allowed=True,
                    safe_actions=['move_forward'],
                    reasoning='No threats detected with enhanced regional analysis and temporal smoothing',
                    priority='LOW',
                    florence_used=florence_verification['verification_available']
                )
                
                # Save depth visualization for major forward movement decision
                if DEPTH_VISUALIZATION_AVAILABLE and should_save_decision_image('MOVE_FORWARD_CLEAR'):
                    raw_depth_map = depth_analysis.get('raw_depth_map')
                    if raw_depth_map is not None:
                        threat_summary = "Path clear - no threats detected with enhanced analysis"
                        save_navigation_decision_image(
                            raw_depth_map, 'MOVE_FORWARD_CLEAR', tick_id, frame_id, threat_summary
                        )
            
            return result

# Integration function for brain system
def get_navigation_safety(vision_state: Dict = None, image_bgr: np.ndarray = None, tick_id: int = None, frame_id: int = None) -> Dict:
    """
    Main integration point for Beta's brain system
    
    Args:
        vision_state: Current vision state from get_vision_state() 
        image_bgr: Optional current frame for Florence verification
        
    Returns:
        Navigation safety decision with movement allowance and actions
    """
    # Log start of hybrid navigation analysis
    if LOGGING_AVAILABLE and tick_id and frame_id:
        log_hybrid_nav_start(
            tick_id, frame_id,
            vision_state_available=(vision_state is not None),
            image_available=(image_bgr is not None)
        )
    
    if vision_state is None and STATE_BUS_AVAILABLE:
        from state_bus import get_vision_state
        vision_state = get_vision_state()
    
    if vision_state is None:
        result = {
            'decision': 'NO_VISION_DATA',
            'movement_allowed': False,
            'safe_actions': ['look_around'],
            'reasoning': 'No vision data available',
            'priority': 'CRITICAL'
        }
        
        # Log the no-vision result
        if LOGGING_AVAILABLE and tick_id and frame_id:
            log_navigation_decision(
                tick_id, frame_id,
                decision='NO_VISION_DATA',
                movement_allowed=False,
                safe_actions=['look_around'],
                reasoning='No vision data available',
                priority='CRITICAL',
                florence_used=False
            )
        
        return result
    
    # Initialize hybrid system
    hybrid_nav = HybridNavigationSystem()
    
    # Tier 1: Fast depth threat detection (enhanced with regional percentiles)
    # Try enhanced analysis if raw depth map is available
    raw_depth_map = vision_state.get('depth_map')
    if raw_depth_map is not None:
        depth_analysis = hybrid_nav.analyze_depth_threats_enhanced(raw_depth_map, tick_id, frame_id)
        print("Using enhanced regional percentiles analysis")
    else:
        depth_analysis = hybrid_nav.analyze_depth_threats(vision_state)
        print("Using standard state estimator analysis")
        
        # Log standard depth analysis (enhanced already logs itself)
        if LOGGING_AVAILABLE and tick_id and frame_id:
            log_depth_analysis_tier1(
                tick_id, frame_id,
                analysis_method='standard_state_estimator',
                threats_detected=len(depth_analysis.get('threats', [])),
                measurements=depth_analysis.get('measurements', {}),
                needs_florence=depth_analysis.get('needs_florence_verification', False)
            )
    
    # Tier 2: Florence verification (if needed and available)
    if depth_analysis['needs_florence_verification'] and image_bgr is not None:
        florence_verification = hybrid_nav.verify_threats_with_florence(
            depth_analysis['threats'], image_bgr, tick_id, frame_id)
    else:
        florence_verification = {
            'verification_available': False,
            'verified_threats': depth_analysis['threats'],
            'semantic_analysis': 'Florence verification not needed or unavailable'
        }
        
        # Log that Florence was not used
        if LOGGING_AVAILABLE and tick_id and frame_id:
            log_florence_verification_tier2(
                tick_id, frame_id,
                verification_available=False,
                florence_latency_ms=0,
                threats_verified=0,
                threats_demoted=0
            )
    
    # Final decision - use enhanced decision method if enhanced analysis was used
    if depth_analysis.get('analysis_method') == 'regional_percentiles_with_temporal_smoothing':
        navigation_decision = hybrid_nav.make_enhanced_navigation_decision(
            depth_analysis, florence_verification, tick_id, frame_id)
        print("Using enhanced navigation decision logic")
    else:
        navigation_decision = hybrid_nav.make_navigation_decision(
            depth_analysis, florence_verification, tick_id, frame_id)
        print("Using standard navigation decision logic")
    
    return navigation_decision

# Convenience function for quick safety checks
def is_movement_safe(vision_state: Dict = None) -> bool:
    """
    Quick boolean check for movement safety
    
    Args:
        vision_state: Current vision state
        
    Returns:
        True if movement is safe, False otherwise
    """
    safety_decision = get_navigation_safety(vision_state)
    return safety_decision.get('movement_allowed', False)

# Test function
def test_hybrid_navigation():
    """Test hybrid navigation system with current vision state"""
    print("=== HYBRID NAVIGATION SYSTEM TEST ===")
    
    if STATE_BUS_AVAILABLE:
        vision_state = get_vision_state()
        if vision_state:
            result = get_navigation_safety(vision_state)
            
            print(f"Decision: {result['decision']}")
            print(f"Movement Allowed: {result['movement_allowed']}")
            print(f"Safe Actions: {result['safe_actions']}")
            print(f"Reasoning: {result['reasoning']}")
            print(f"Priority: {result['priority']}")
            print(f"Florence Used: {result['florence_used']}")
            
            return result
        else:
            print("No vision state available")
    else:
        print("State bus not available")
    
    return None

if __name__ == "__main__":
    test_hybrid_navigation()