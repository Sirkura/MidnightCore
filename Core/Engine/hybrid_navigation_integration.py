#!/usr/bin/env python3
"""
Hybrid Navigation Integration Patch for Qwen Brain
=================================================

PURPOSE: Integrate two-tier hybrid navigation safety into Beta's movement system
- Adds hybrid navigation safety checks before movement execution
- Integrates with existing depth analysis and Florence systems
- Maintains compatibility with existing brain architecture

INTEGRATION INSTRUCTIONS:
1. Add import to Qwen_Brain_ActiveTesting.py:
   from hybrid_navigation_integration import check_movement_safety, integrate_spatial_context
   
2. Replace movement execution section (around line 2453) with safety-wrapped versions
3. Replace spatial context building (around line 2013) with hybrid analysis

This provides a clean integration path without massive brain system changes.
"""

import sys
import os
import numpy as np

# Import hybrid navigation system
try:
    from ..Modules.vision.hybrid_navigation import get_navigation_safety, is_movement_safe
    HYBRID_NAV_AVAILABLE = True
except ImportError:
    print("WARNING: Hybrid navigation not available")
    HYBRID_NAV_AVAILABLE = False

# Import logging system
try:
    from ..Common.Tools.logging_bus import log_movement_safety_check
    LOGGING_AVAILABLE = True
except ImportError:
    print("WARNING: Logging bus not available for hybrid navigation integration")
    LOGGING_AVAILABLE = False

def check_movement_safety(brain_instance, action_text: str, vision_state: dict = None, tick_id: int = None, frame_id: int = None):
    """
    Safety wrapper for movement commands - integrates hybrid navigation
    
    Args:
        brain_instance: Instance of NaturalQwenVRChatBrain
        action_text: Movement command text (e.g., "move forward", "strafe left")
        vision_state: Current vision state from get_vision_state()
        
    Returns:
        Tuple of (is_safe, safety_decision, modified_action)
    """
    print(f"[HYBRID NAV DEBUG] check_movement_safety called with:")
    print(f"[HYBRID NAV DEBUG]   action_text: {action_text}")
    print(f"[HYBRID NAV DEBUG]   vision_state: {vision_state is not None}")
    print(f"[HYBRID NAV DEBUG]   tick_id: {tick_id}")
    print(f"[HYBRID NAV DEBUG]   frame_id: {frame_id}")
    print(f"[HYBRID NAV DEBUG]   HYBRID_NAV_AVAILABLE: {HYBRID_NAV_AVAILABLE}")
    
    if not HYBRID_NAV_AVAILABLE:
        print(f"[HYBRID NAV DEBUG] Falling back to legacy - hybrid nav not available")
        # Fallback to existing depth analysis
        return _legacy_safety_check(brain_instance, action_text, vision_state)
    
    # Get current screenshot for Florence verification if needed
    current_image = None
    try:
        # Use the brain's existing capture method
        current_image = brain_instance.capture_system.capture_vrchat_for_qwen(filename=None, context="hybrid_nav_check")
        print(f"[HYBRID NAV] Image captured for Florence verification: {current_image is not None}")
        if current_image is not None:
            print(f"[HYBRID NAV] Image type: {type(current_image)}, shape: {getattr(current_image, 'shape', 'no shape attr')}")
    except Exception as e:
        print(f"[HYBRID NAV] Failed to capture image for Florence: {e}")
        import traceback
        print(f"[HYBRID NAV] Exception details: {traceback.format_exc()}")
        pass  # Florence verification will be skipped if no image
    
    # Get hybrid navigation decision
    print(f"[HYBRID NAV] Calling get_navigation_safety - vision_state: {vision_state is not None}, image: {current_image is not None}, tick: {tick_id}, frame: {frame_id}")
    
    try:
        safety_decision = get_navigation_safety(vision_state, current_image, tick_id, frame_id)
        
        # Log hybrid decision for monitoring
        print(f"[HYBRID NAV] Decision received: {safety_decision['decision']} - {safety_decision['reasoning']}")
        print(f"[HYBRID NAV] Decision details: movement_allowed={safety_decision.get('movement_allowed')}, priority={safety_decision.get('priority')}")
        
    except Exception as e:
        print(f"[HYBRID NAV] CRITICAL: get_navigation_safety failed: {e}")
        import traceback
        print(f"[HYBRID NAV] Exception details: {traceback.format_exc()}")
        
        # Fall back to legacy safety check
        print(f"[HYBRID NAV] Falling back to legacy safety check")
        return _legacy_safety_check(brain_instance, action_text, vision_state)
    
    # Movement type classification
    movement_actions = ["move forward", "strafe left", "strafe right", "back up", "approach target"]
    is_movement = any(action in action_text.lower() for action in movement_actions)
    
    if not is_movement:
        # Non-movement actions are always safe (looking, chatting, etc.)
        if LOGGING_AVAILABLE and tick_id and frame_id:
            log_movement_safety_check(tick_id, frame_id, action_text, True, None, "Non-movement action - always safe")
        return True, safety_decision, action_text
    
    # Check specific movement safety
    if "move forward" in action_text.lower() or "approach target" in action_text.lower():
        if safety_decision['decision'] in ['STOP_CLIFF_DANGER', 'STOP_WALL_CLOSE', 'STOP_BOXED_IN']:
            # Block forward movement, suggest alternatives
            safe_action = _suggest_alternative_action(safety_decision['safe_actions'])
            print(f"[HYBRID NAV] BLOCKED forward movement: {safety_decision['reasoning']}")
            print(f"[HYBRID NAV] Suggested alternative: {safe_action}")
            
            if LOGGING_AVAILABLE and tick_id and frame_id:
                log_movement_safety_check(tick_id, frame_id, action_text, False, safe_action, safety_decision['reasoning'])
            
            return False, safety_decision, safe_action
        elif safety_decision['decision'] == 'NAVIGATE_AROUND':
            # Allow forward movement but make it cautious
            cautious_action = action_text.replace("move forward", "move forward cautiously")
            print(f"[HYBRID NAV] MODIFIED to cautious movement")
            
            if LOGGING_AVAILABLE and tick_id and frame_id:
                log_movement_safety_check(tick_id, frame_id, action_text, True, cautious_action, "Modified to cautious movement")
            
            return True, safety_decision, cautious_action
    
    elif "strafe" in action_text.lower():
        # Check if requested strafe direction is blocked
        if "strafe left" in action_text.lower():
            left_clearance = safety_decision.get('measurements', {}).get('left_m', float('inf'))
            if left_clearance < 1.0:
                print(f"[HYBRID NAV] BLOCKED left strafe: insufficient clearance ({left_clearance:.1f}m)")
                safe_action = _suggest_alternative_action(safety_decision['safe_actions'])
                return False, safety_decision, safe_action
        elif "strafe right" in action_text.lower():
            right_clearance = safety_decision.get('measurements', {}).get('right_m', float('inf'))
            if right_clearance < 1.0:
                print(f"[HYBRID NAV] BLOCKED right strafe: insufficient clearance ({right_clearance:.1f}m)")
                safe_action = _suggest_alternative_action(safety_decision['safe_actions'])
                return False, safety_decision, safe_action
    
    # Movement is considered safe
    print(f"[HYBRID NAV] ALLOWED movement: {action_text}")
    
    if LOGGING_AVAILABLE and tick_id and frame_id:
        log_movement_safety_check(tick_id, frame_id, action_text, True, None, "Movement approved by hybrid navigation")
    
    return True, safety_decision, action_text

def _suggest_alternative_action(safe_actions: list) -> str:
    """Convert hybrid nav safe actions to brain action format"""
    action_map = {
        "look_left": "I want to look left to scan the area",
        "look_right": "I want to look right to check my surroundings", 
        "turn_around": "I want to look right to check my surroundings",  # Start of turn around
        "back_up": "I want to back up to get more clearance",
        "small_step_forward": "I want to move forward to explore a few steps",
        "look_around": "I want to look left to scan the area",
        "turn_left": "I want to look left to scan the area",
        "turn_right": "I want to look right to check my surroundings"
    }
    
    # Find first safe action that maps to brain actions
    for safe_action in safe_actions:
        if safe_action in action_map:
            return action_map[safe_action]
    
    # Default fallback
    return "I want to look left to scan the area"

def _legacy_safety_check(brain_instance, action_text: str, vision_state: dict = None):
    """Fallback to existing depth-only safety logic when hybrid nav unavailable"""
    if not vision_state:
        return True, {}, action_text
    
    front_m = vision_state.get('front_m', float('inf'))
    
    # Simple legacy check
    if "move forward" in action_text.lower() and front_m < 0.8:
        return False, {'reasoning': f'Legacy depth check: wall too close ({front_m:.1f}m)'}, "I want to look around to assess"
    
    return True, {}, action_text

def integrate_spatial_context(brain_instance, vision_state: dict = None):
    """
    Enhanced spatial context generation using hybrid navigation analysis
    Replaces the existing spatial context building in the brain
    
    Args:
        brain_instance: Instance of NaturalQwenVRChatBrain
        vision_state: Current vision state
        
    Returns:
        String with enhanced spatial context for LLM
    """
    if not vision_state:
        return "No vision data available for spatial analysis"
    
    spatial_lines = []
    
    # Get hybrid navigation analysis
    if HYBRID_NAV_AVAILABLE:
        try:
            current_image = brain_instance.capture_system.get_screenshot()
            safety_decision = get_navigation_safety(vision_state, current_image)
            
            # Add hybrid analysis to context
            spatial_lines.append(f"NAVIGATION STATUS: {safety_decision['decision']} - {safety_decision['reasoning']}")
            
            # Add specific threat information
            if safety_decision['threat_count'] > 0:
                spatial_lines.append(f"DETECTED THREATS: {safety_decision['threat_count']} navigation hazards")
                if safety_decision['florence_used']:
                    spatial_lines.append("SEMANTIC ANALYSIS: Florence-2 verified threat assessment")
                
            # Add movement recommendations
            safe_actions_str = ", ".join(safety_decision['safe_actions'])
            spatial_lines.append(f"RECOMMENDED ACTIONS: {safe_actions_str}")
            
        except Exception as e:
            print(f"WARNING: Hybrid spatial context failed: {e}")
            # Fall back to legacy analysis
    
    # Legacy depth measurements (still useful context)
    front_m = vision_state.get('front_m', float('inf'))
    left_m = vision_state.get('left_m', float('inf'))
    right_m = vision_state.get('right_m', float('inf'))
    edge_risk = vision_state.get('edge_risk', 0.0)
    
    # Movement safety analysis (enhanced)
    if front_m < 0.5:
        spatial_lines.append(f"CRITICAL: Wall/obstacle {front_m:.1f}m ahead - MOVEMENT BLOCKED!")
    elif front_m < 1.0:
        spatial_lines.append(f"CAUTION: Close obstacle {front_m:.1f}m ahead - approach very carefully")
    elif front_m < 2.0:
        spatial_lines.append(f"AWARE: Object {front_m:.1f}m ahead - room for careful movement")
    else:
        spatial_lines.append(f"CLEAR: Path ahead {front_m:.1f}m - safe for forward movement")
    
    # Directional clearances with movement suggestions  
    movement_options = []
    if left_m > 1.2:
        movement_options.append(f"strafe left ({left_m:.1f}m clearance)")
    if right_m > 1.2:
        movement_options.append(f"strafe right ({right_m:.1f}m clearance)")
    if front_m > 1.2:
        movement_options.append(f"continue forward ({front_m:.1f}m clearance)")
    
    if movement_options:
        spatial_lines.append(f"MOVEMENT OPTIONS: {', '.join(movement_options)}")
    else:
        spatial_lines.append("MOVEMENT RESTRICTED: Limited clearance in all directions")
    
    # Edge risk warning
    if edge_risk > 0.5:
        spatial_lines.append(f"CLIFF WARNING: Edge risk {edge_risk:.1f} - potential drop-off detected")
    
    return "\n".join(spatial_lines)

def execute_movement_with_safety(brain_instance, action_text: str):
    """
    Safe movement execution wrapper that checks hybrid navigation before executing
    
    Args:
        brain_instance: Instance of NaturalQwenVRChatBrain  
        action_text: Action text to execute
        
    Returns:
        Boolean indicating if movement was executed
    """
    # Get current vision state
    vision_state = None
    try:
        from state_bus import get_vision_state
        vision_state = get_vision_state()
    except:
        pass
    
    # Check movement safety
    is_safe, safety_decision, modified_action = check_movement_safety(
        brain_instance, action_text, vision_state)
    
    if not is_safe:
        # Execute alternative safe action instead
        print(f"[HYBRID NAV] Movement blocked, executing alternative: {modified_action}")
        brain_instance.execute_action(modified_action)
        
        # Chat about the safety decision  
        safety_msg = f"Navigation system prevented unsafe movement: {safety_decision.get('reasoning', 'Safety check failed')}"
        brain_instance.chat(safety_msg, "Safety System")
        
        return False
    elif modified_action != action_text:
        # Execute modified (cautious) action
        print(f"[HYBRID NAV] Executing modified action: {modified_action}")
        brain_instance.execute_action(modified_action)
        return True
    else:
        # Execute original action - it's safe
        return None  # Let normal execution continue

# Integration helper functions for easy patching

def patch_brain_movement_safety(brain_instance):
    """
    Monkey patch brain instance to add hybrid navigation safety
    Call this during brain initialization
    """
    # Store original execute_action method
    original_execute = brain_instance.execute_action
    
    def safe_execute_action(action_text):
        """Wrapped execute_action with safety checks"""
        # Check if this needs safety intervention
        safety_result = execute_movement_with_safety(brain_instance, action_text)
        
        if safety_result is None:
            # No intervention needed, use original
            return original_execute(action_text)
        else:
            # Safety system handled it
            return safety_result
    
    # Replace method
    brain_instance.execute_action = safe_execute_action
    print("[HYBRID NAV] Brain movement safety patched successfully")

def test_hybrid_integration():
    """Test hybrid navigation integration"""
    print("=== HYBRID NAVIGATION INTEGRATION TEST ===")
    
    if HYBRID_NAV_AVAILABLE:
        # Test with mock vision state
        test_vision_state = {
            'front_m': 0.5,  # Close wall
            'left_m': 2.0,   # Clear left
            'right_m': 0.8,  # Blocked right
            'edge_risk': 0.3
        }
        
        safety_decision = get_navigation_safety(test_vision_state)
        print(f"Test Decision: {safety_decision['decision']}")
        print(f"Test Reasoning: {safety_decision['reasoning']}")
        print(f"Safe Actions: {safety_decision['safe_actions']}")
        
        return safety_decision
    else:
        print("Hybrid navigation not available for testing")
        return None

if __name__ == "__main__":
    test_hybrid_integration()