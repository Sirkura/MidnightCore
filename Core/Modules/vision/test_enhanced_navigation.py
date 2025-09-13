#!/usr/bin/env python3
"""
Test Enhanced Hybrid Navigation System
=====================================

Quick test script to verify regional percentiles and temporal smoothing
enhancements are working correctly in the hybrid navigation system.
"""

import sys
import os
import numpy as np

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from hybrid_navigation import HybridNavigationSystem

def create_test_depth_map(scenario='clear_path'):
    """
    Create synthetic depth maps for testing different scenarios
    
    Args:
        scenario: 'clear_path', 'cliff_ahead', 'wall_close', 'blocked_path'
    
    Returns:
        Synthetic depth map (480x640)
    """
    height, width = 480, 640
    depth_map = np.ones((height, width), dtype=np.float32)
    
    if scenario == 'clear_path':
        # Normal clear path - no threats
        depth_map *= 0.5  # Mid-range depth
        
    elif scenario == 'cliff_ahead':
        # Cliff in immediate forward area
        cliff_area = depth_map[int(height*0.7):height, int(width*0.3):int(width*0.7)]
        cliff_area[:] = 0.05  # Very low depth = cliff
        
    elif scenario == 'wall_close':
        # Wall in forward wall region
        wall_area = depth_map[int(height*0.25):int(height*0.45), int(width*0.4):int(width*0.6)]
        wall_area[:] = 0.9  # High depth = wall
        
    elif scenario == 'blocked_path':
        # Blocked forward path
        path_area = depth_map[int(height*0.4):int(height*0.6), int(width*0.35):int(width*0.65)]
        path_area[:] = 0.8  # Higher than sides
        
        # Make sides clearer
        left_area = depth_map[int(height*0.3):int(height*0.7), :int(width*0.3)]
        left_area[:] = 0.3
        right_area = depth_map[int(height*0.3):int(height*0.7), int(width*0.7):]
        right_area[:] = 0.3
    
    return depth_map

def test_enhanced_navigation():
    """Test enhanced navigation with different scenarios"""
    print("=== TESTING ENHANCED HYBRID NAVIGATION ===\n")
    
    # Initialize system
    nav_system = HybridNavigationSystem()
    
    scenarios = ['clear_path', 'cliff_ahead', 'wall_close', 'blocked_path']
    
    for scenario in scenarios:
        print(f"--- Testing Scenario: {scenario.upper().replace('_', ' ')} ---")
        
        # Create test depth map
        depth_map = create_test_depth_map(scenario)
        
        # Run enhanced analysis multiple times to build temporal buffers
        for i in range(6):  # Build 5-frame buffer + 1 test
            analysis = nav_system.analyze_depth_threats_enhanced(depth_map)
            
            if i == 5:  # Final test with full buffers
                print(f"Analysis Method: {analysis['analysis_method']}")
                print(f"Temporal Buffers Ready: {analysis['temporal_buffers_ready']}")
                print(f"Threats Detected: {len(analysis['threats'])}")
                
                if analysis['threats']:
                    for threat in analysis['threats']:
                        print(f"  - {threat['type']}: {threat.get('details', 'N/A')}")
                
                print(f"Florence Verification Needed: {analysis['needs_florence_verification']}")
                
                # Test enhanced decision making
                florence_verification = {
                    'verification_available': False,
                    'verified_threats': analysis['threats'],
                    'semantic_analysis': 'Mock verification for testing'
                }
                
                decision = nav_system.make_enhanced_navigation_decision(analysis, florence_verification)
                print(f"Decision: {decision['decision']}")
                print(f"Movement Allowed: {decision['movement_allowed']}")
                print(f"Safe Actions: {decision['safe_actions']}")
                print(f"Priority: {decision['priority']}")
                
        print()
    
    # Test temporal smoothing stability
    print("--- Testing Temporal Smoothing Stability ---")
    nav_system_stability = HybridNavigationSystem()
    
    # Create a noisy depth map sequence
    for i in range(10):
        base_depth = create_test_depth_map('clear_path')
        # Add noise to simulate real-world variation
        noise = np.random.normal(0, 0.05, base_depth.shape)
        noisy_depth = np.clip(base_depth + noise, 0.0, 1.0)
        
        analysis = nav_system_stability.analyze_depth_threats_enhanced(noisy_depth)
        
        if i >= 5:  # After buffers are built
            stats = analysis['region_stats']['forward_path']
            print(f"Frame {i}: Raw={stats['raw_median']:.3f}, Smoothed={stats['smoothed_median']:.3f}")
    
    print("\n=== ENHANCED NAVIGATION TEST COMPLETE ===")

if __name__ == "__main__":
    test_enhanced_navigation()