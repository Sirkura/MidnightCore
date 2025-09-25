#!/usr/bin/env python3
"""
VRChat OSC Controller for Qwen 3 8B Integration
Direct OSC control optimized for Qwen reasoning and decision making
"""

from pythonosc import udp_client
import time
import logging
import json
import sys
import os
from datetime import datetime

try:
    # Try legacy import for backward compatibility
    from FusionCore.Integration.state_bus import get_vision_state
    DEPTH_SYSTEM_AVAILABLE = True
except ImportError:
    DEPTH_SYSTEM_AVAILABLE = False
    print("Warning: Depth system not available - physics safety disabled")

class QwenVRChatOSC:
    def __init__(self, ip="127.0.0.1", port=9000):
        """Initialize OSC client for VRChat with Qwen integration"""
        self.ip = ip
        self.port = port
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.head_position = {"horizontal": 0, "vertical": 0}  # Track head position
        self.last_action = {"type": None, "timestamp": None, "parameters": None}
        self.action_history = []  # For Qwen context
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Qwen-VRChat OSC Controller initialized: {ip}:{port}")
        
        # Initialize log file for Qwen analysis
        self.osc_log_path = "G:/Experimental/Production/MidnightCore/Core/Engine/Logging/OSC-Log"
        
        # Physics safety thresholds
        self.safety_config = {
            "min_front_clearance": 0.8,    # meters
            "max_edge_risk": 0.35,         # risk score [0-1]
            "min_time_to_contact": 1.0     # seconds
        }
        
        # Track blocked movements for Qwen context
        self.last_blocked_info = {
            "was_blocked": False,
            "reason": "",
            "timestamp": 0
        }
    
    def preflight_allows_forward(self) -> bool:
        """
        Physics safety guard - check if forward movement is safe
        
        Returns:
            True if forward movement is safe, False if blocked
        """
        if not DEPTH_SYSTEM_AVAILABLE:
            return True  # No depth system = no safety guard
        
        try:
            # Get latest vision state
            state = get_vision_state()
            if not state:
                return True  # No depth data available
            
            # Check safety conditions
            front_m = state.get("front_m")
            edge_risk = state.get("edge_risk")
            ttc_s = state.get("ttc_s")
            
            # Block if obstacle too close
            if front_m is not None and front_m < self.safety_config["min_front_clearance"]:
                return False
            
            # Block if edge/cliff detected
            if edge_risk is not None and edge_risk > self.safety_config["max_edge_risk"]:
                return False
            
            # Block if time to contact too short
            if ttc_s is not None and ttc_s < self.safety_config["min_time_to_contact"]:
                return False
            
            return True  # All checks passed
            
        except Exception as e:
            self.logger.warning(f"Physics safety check failed: {e}")
            return True  # Default to allowing movement on error
    
    def _get_block_reason(self) -> str:
        """Get specific reason for blocking forward movement"""
        if not DEPTH_SYSTEM_AVAILABLE:
            return "no depth system"
        
        try:
            state = get_vision_state()
            if not state:
                return "no depth data"
            
            front_m = state.get("front_m")
            edge_risk = state.get("edge_risk") 
            ttc_s = state.get("ttc_s")
            
            reasons = []
            if front_m is not None and front_m < self.safety_config["min_front_clearance"]:
                reasons.append(f"front<{self.safety_config['min_front_clearance']}m")
            if edge_risk is not None and edge_risk > self.safety_config["max_edge_risk"]:
                reasons.append(f"edge>{self.safety_config['max_edge_risk']}")
            if ttc_s is not None and ttc_s < self.safety_config["min_time_to_contact"]:
                reasons.append(f"ttc<{self.safety_config['min_time_to_contact']}s")
            
            return ", ".join(reasons) if reasons else "unknown"
            
        except Exception as e:
            return f"error: {e}"
    
    def get_last_blocked_info(self) -> dict:
        """Get information about last blocked movement for Qwen context"""
        return self.last_blocked_info.copy()
    
    def emergency_stop_all_controls(self):
        """Emergency stop - reset ALL controls to 0 immediately"""
        try:
            print("OSC EMERGENCY STOP: Resetting all controls to 0")
            
            # Reset all axes to 0.0 (CRITICAL - prevents infinite movement)
            # Note: No LookHorizontal axis - desktop mode uses LookLeft/LookRight buttons
            self.client.send_message("/input/Vertical", 0.0)
            self.client.send_message("/input/Horizontal", 0.0)
            self.client.send_message("/input/LookVertical", 0.0)
            
            # Reset all buttons to 0
            buttons_to_reset = [
                "/input/MoveForward", "/input/MoveBackward", "/input/MoveLeft", "/input/MoveRight",
                "/input/LookLeft", "/input/LookRight", "/input/Jump", "/input/Run", "/input/Voice"
            ]
            
            for button in buttons_to_reset:
                self.client.send_message(button, 0)
            
            # Reset head position tracking
            self.head_position = {"horizontal": 0, "vertical": 0}
            
            self.log_action("emergency_stop_all", {}, "Emergency safety reset of all OSC controls")
            self.logger.info("EMERGENCY STOP: All OSC controls reset to 0")
            
        except Exception as e:
            self.logger.error(f"CRITICAL ERROR during emergency stop: {e}")
    
    def log_action(self, action_type, parameters, reasoning=""):
        """Log actions for Qwen context and analysis"""
        action_entry = {
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "action": action_type,
            "parameters": parameters,
            "reasoning": reasoning,
            "head_position": self.head_position.copy()
        }
        
        self.action_history.append(action_entry)
        self.last_action = {
            "type": action_type,
            "timestamp": action_entry["timestamp"],
            "parameters": parameters
        }
        
        # Keep only last 50 actions in memory
        if len(self.action_history) > 50:
            self.action_history.pop(0)
        
        # Write to log file
        try:
            with open(self.osc_log_path, "a", encoding='utf-8') as f:
                f.write(f"{json.dumps(action_entry)}\n")
        except Exception as e:
            self.logger.error(f"Failed to write to OSC log: {e}")
    
    def get_action_context(self, num_actions=10):
        """Get recent action context for Qwen reasoning"""
        recent_actions = self.action_history[-num_actions:] if self.action_history else []
        return {
            "recent_actions": recent_actions,
            "current_head_position": self.head_position,
            "total_actions": len(self.action_history),
            "last_action_time": self.last_action["timestamp"] if self.last_action["timestamp"] else "none"
        }
    
    def chat(self, message, immediate=True, sound=True, reasoning="Qwen generated response"):
        """Send message to VRChat chatbox with Qwen context logging"""
        self.client.send_message("/chatbox/input", [message, immediate, sound])
        self.log_action("chat", {"message": message, "immediate": immediate, "sound": sound}, reasoning)
        self.logger.info(f"Qwen Chat sent: {message}")
    
    def clear_chat(self):
        """Clear chatbox"""
        self.client.send_message("/chatbox/input", ["", True, False])
        self.log_action("clear_chat", {}, "Clearing chatbox for new content")
        self.logger.info("Chatbox cleared")
    
    # MOVEMENT CONTROLS - Enhanced for Qwen decision tracking
    def move_forward(self, intensity=1.0, duration=None, reasoning="Qwen navigation decision"):
        """Move forward with Qwen context and physics safety guard"""
        # Physics safety check
        if not self.preflight_allows_forward():
            # Determine specific reason for blocking
            block_reason = self._get_block_reason()
            
            self.log_action("movement_blocked", {"intensity": intensity, "duration": duration}, 
                          f"Physics guard: {block_reason}")
            self.logger.warning(f"Forward movement blocked by physics safety guard: {block_reason}")
            
            # Track blocked movement for Qwen context
            self.last_blocked_info = {
                "was_blocked": True,
                "reason": block_reason,
                "timestamp": time.time()
            }
            return
        
        # Clear blocked status on successful movement
        self.last_blocked_info = {
            "was_blocked": False,
            "reason": "",
            "timestamp": 0
        }
        
        self.client.send_message("/input/Vertical", intensity)
        self.log_action("move_forward", {"intensity": intensity, "duration": duration}, reasoning)
        self.logger.info(f"Qwen Moving forward: {intensity}")
        if duration:
            time.sleep(duration)
            self.stop_movement()
    
    def move_backward(self, intensity=-1.0, duration=None, reasoning="Qwen navigation decision"):
        """Move backward with Qwen context"""
        self.client.send_message("/input/Vertical", intensity)
        self.log_action("move_backward", {"intensity": intensity, "duration": duration}, reasoning)
        self.logger.info(f"Qwen Moving backward: {intensity}")
        if duration:
            time.sleep(duration)
            self.stop_movement()
    
    def strafe_left(self, intensity=-1.0, duration=None, reasoning="Qwen navigation decision"):
        """Strafe left with Qwen context"""
        self.client.send_message("/input/Horizontal", intensity)
        self.log_action("strafe_left", {"intensity": intensity, "duration": duration}, reasoning)
        self.logger.info(f"Qwen Strafing left: {intensity}")
        if duration:
            time.sleep(duration)
            self.stop_strafe()
    
    def strafe_right(self, intensity=1.0, duration=None, reasoning="Qwen navigation decision"):
        """Strafe right with Qwen context"""
        self.client.send_message("/input/Horizontal", intensity)
        self.log_action("strafe_right", {"intensity": intensity, "duration": duration}, reasoning)
        self.logger.info(f"Qwen Strafing right: {intensity}")
        if duration:
            time.sleep(duration)
            self.stop_strafe()
    
    def stop_movement(self):
        """Stop forward/backward movement"""
        self.client.send_message("/input/Vertical", 0.0)
        self.log_action("stop_movement", {}, "Stopping movement")
        self.logger.info("Movement stopped")
    
    def stop_strafe(self):
        """Stop left/right movement"""
        self.client.send_message("/input/Horizontal", 0.0)
        self.log_action("stop_strafe", {}, "Stopping strafe")
        self.logger.info("Strafe stopped")
    
    def stop_all_movement(self):
        """Stop all movement"""
        self.stop_movement()
        self.stop_strafe()
        self.log_action("stop_all_movement", {}, "Emergency stop - all movement")
        self.logger.info("All movement stopped")
    
    def jump(self, reasoning="Qwen decided to jump"):
        """Jump with Qwen context"""
        self.client.send_message("/input/Jump", 1)
        time.sleep(0.1)  # Brief press
        self.client.send_message("/input/Jump", 0)
        self.log_action("jump", {}, reasoning)
        self.logger.info("Qwen Jump executed")
    
    # LOOK CONTROLS - Enhanced for Qwen spatial awareness
    def look_up(self, intensity=0.25, duration=None, reasoning="Qwen visual scan"):
        """Look up with head position tracking"""
        self.client.send_message("/input/LookVertical", intensity)
        if duration:
            time.sleep(duration)
            self.client.send_message("/input/LookVertical", 0.0)  # Reset axis after duration
            self.log_action("look_up", {"intensity": intensity, "duration": duration}, reasoning)
            self.logger.info(f"Qwen Looking up: {intensity} for {duration}s")
        else:
            # Track position for manual reset later
            self.head_position["vertical"] += intensity
            self.log_action("look_up", {"intensity": intensity}, reasoning)
            self.logger.info(f"Qwen Looking up: {intensity} (manual reset required)")
    
    def look_down(self, intensity=-0.25, duration=None, reasoning="Qwen visual scan"):
        """Look down with head position tracking"""
        self.client.send_message("/input/LookVertical", intensity)
        if duration:
            time.sleep(duration)
            self.client.send_message("/input/LookVertical", 0.0)  # Reset axis after duration
            self.log_action("look_down", {"intensity": intensity, "duration": duration}, reasoning)
            self.logger.info(f"Qwen Looking down: {intensity} for {duration}s")
        else:
            # Track position for manual reset later
            self.head_position["vertical"] += intensity
            self.log_action("look_down", {"intensity": intensity}, reasoning)
            self.logger.info(f"Qwen Looking down: {intensity} (manual reset required)")
    
    def look_left(self, duration=0.5, reasoning="Qwen environmental scan"):
        """Look left for specified duration with enhanced tracking"""
        self.client.send_message("/input/LookLeft", 1)
        self.log_action("look_left", {"duration": duration}, reasoning)
        self.logger.info(f"Qwen Looking left for {duration}s")
        time.sleep(duration)
        self.client.send_message("/input/LookLeft", 0)
        # Track cumulative movement for proper centering (1s = 45°, so duration = degrees/45)
        self.head_position["horizontal"] -= duration  # Store duration for accurate reversal
    
    def look_right(self, duration=0.5, reasoning="Qwen environmental scan"):
        """Look right for specified duration with enhanced tracking"""
        self.client.send_message("/input/LookRight", 1)
        self.log_action("look_right", {"duration": duration}, reasoning)
        self.logger.info(f"Qwen Looking right for {duration}s")
        time.sleep(duration)
        self.client.send_message("/input/LookRight", 0)
        # Track cumulative movement for proper centering (1s = 45°, so duration = degrees/45)
        self.head_position["horizontal"] += duration  # Store duration for accurate reversal
    
    def reset_vertical_look(self):
        """Reset vertical look to center"""
        self.client.send_message("/input/LookVertical", 0.0)
        self.head_position["vertical"] = 0
        self.log_action("reset_vertical_look", {}, "Centering vertical view")
        self.logger.info("Vertical look reset to center")
    
    def return_to_center_horizontal(self):
        """Return horizontal look to center using accurate duration-based reversal"""
        current_offset = self.head_position["horizontal"]
        
        if abs(current_offset) < 0.01:  # Already centered (within margin)
            self.logger.info("Already centered horizontally")
            return
            
        if current_offset > 0:
            # Looking right, need to look left to center
            duration = abs(current_offset)
            self.logger.info(f"Centering: Looking left for {duration}s to compensate for right offset")
            self.client.send_message("/input/LookLeft", 1)
            time.sleep(duration)
            self.client.send_message("/input/LookLeft", 0)
            self.log_action("return_to_center_horizontal", {"correction": "left", "duration": duration}, "Centering from right position")
            
        elif current_offset < 0:
            # Looking left, need to look right to center
            duration = abs(current_offset)
            self.logger.info(f"Centering: Looking right for {duration}s to compensate for left offset")
            self.client.send_message("/input/LookRight", 1)
            time.sleep(duration)
            self.client.send_message("/input/LookRight", 0)
            self.log_action("return_to_center_horizontal", {"correction": "right", "duration": duration}, "Centering from left position")
        
        # Reset position tracking after centering
        self.head_position["horizontal"] = 0
        self.logger.info("Horizontal centering complete - avatar should face original direction")
    
    # QWEN-ENHANCED UTILITY METHODS
    def qwen_scan(self, directions=["left", "right", "up", "down"], scan_duration=0.5, pause=0.5, purpose="environmental_analysis"):
        """Perform an intelligent scan optimized for Qwen analysis"""
        self.chat(f"Qwen initiating {purpose} scan...", True, False)
        
        scan_results = {
            "start_time": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "purpose": purpose,
            "directions": directions,
            "scan_duration": scan_duration
        }
        
        for direction in directions:
            if direction == "left":
                self.look_left(scan_duration, f"Scanning left for {purpose}")
            elif direction == "right":
                self.look_right(scan_duration, f"Scanning right for {purpose}")
            elif direction == "up":
                self.look_up(0.25, scan_duration, f"Scanning up for {purpose}")
            elif direction == "down":
                self.look_down(-0.25, scan_duration, f"Scanning down for {purpose}")
            
            time.sleep(pause)  # Pause for vision processing
        
        # Return to center
        self.reset_vertical_look()
        self.return_to_center_horizontal()
        
        scan_results["end_time"] = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_action("qwen_scan", scan_results, f"Completed {purpose} scan")
        self.logger.info(f"Qwen {purpose} scan completed")
        
        return scan_results
    
    def execute_qwen_action_sequence(self, actions, context="Qwen planned sequence"):
        """Execute a sequence of actions planned by Qwen"""
        self.chat(f"Executing Qwen action sequence: {len(actions)} actions", True, False)
        
        sequence_log = {
            "start_time": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "context": context,
            "planned_actions": actions,
            "executed_actions": []
        }
        
        for i, action in enumerate(actions):
            try:
                action_type = action.get("type")
                params = action.get("parameters", {})
                reasoning = action.get("reasoning", f"Qwen sequence step {i+1}")
                
                # Execute the action based on type
                if action_type == "move_forward":
                    self.move_forward(**params, reasoning=reasoning)
                elif action_type == "look_left":
                    self.look_left(**params, reasoning=reasoning)
                elif action_type == "look_right":
                    self.look_right(**params, reasoning=reasoning)
                elif action_type == "chat":
                    self.chat(**params, reasoning=reasoning)
                elif action_type == "wait":
                    time.sleep(params.get("duration", 1.0))
                    self.log_action("wait", params, reasoning)
                
                sequence_log["executed_actions"].append({
                    "step": i+1,
                    "action": action,
                    "status": "success",
                    "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
                })
                
            except Exception as e:
                self.logger.error(f"Error executing action {i+1}: {e}")
                sequence_log["executed_actions"].append({
                    "step": i+1,
                    "action": action,
                    "status": "error",
                    "error": str(e),
                    "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S')
                })
        
        sequence_log["end_time"] = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_action("execute_action_sequence", sequence_log, context)
        
        return sequence_log

if __name__ == "__main__":
    # Test the Qwen OSC controller
    qwen_osc = QwenVRChatOSC()
    
    # Test basic functionality
    qwen_osc.chat("Qwen OSC Controller online and testing!", True, True, "Initial system test")
    time.sleep(2)
    
    # Test scan functionality
    scan_result = qwen_osc.qwen_scan(purpose="system_test")
    
    # Test action sequence
    test_sequence = [
        {"type": "chat", "parameters": {"message": "Testing Qwen action sequence"}, "reasoning": "Sequence test notification"},
        {"type": "look_left", "parameters": {"duration": 0.25}, "reasoning": "Quick left scan"},
        {"type": "look_right", "parameters": {"duration": 0.5}, "reasoning": "Compensate and scan right"},
        {"type": "wait", "parameters": {"duration": 1.0}, "reasoning": "Processing pause"},
        {"type": "chat", "parameters": {"message": "Qwen sequence test complete!"}, "reasoning": "Test completion notification"}
    ]
    
    qwen_osc.execute_qwen_action_sequence(test_sequence, "Initial system functionality test")