#!/usr/bin/env python3
"""
Qwen-Enhanced Cropped Screenshot Capture for VRChat
Takes full desktop screenshot, crops console area, and prepares images for Qwen + Florence-2 analysis
"""

import pyautogui
from PIL import Image
import time
from datetime import datetime
import os
import json

class QwenVRChatCapture:
    def __init__(self):
        """Initialize Qwen-enhanced VRChat capture system"""
        self.screen_width, self.screen_height = pyautogui.size()
        print(f"Qwen Capture System - Desktop resolution: {self.screen_width}x{self.screen_height}")
        
        # Define crop areas (adjust these based on your setup)
        self.console_width = 600  # Estimated console width
        self.crop_right = self.console_width  # Crop this many pixels from right edge
        
        # Calculate VRChat area (left portion of screen)
        self.vrchat_width = self.screen_width - self.crop_right
        self.vrchat_height = self.screen_height
        
        # Capture metadata for Qwen analysis
        self.capture_history = []
        self.capture_log_path = "G:/Experimental/Midnight Core/Logs/Florence-Log"
        
        print(f"VRChat area: {self.vrchat_width}x{self.vrchat_height}")
        print(f"Console crop area: {self.crop_right} pixels from right edge")
        print("Qwen integration enabled - capture metadata will be logged")
    
    def capture_vrchat_for_qwen(self, filename=None, context="exploration", reasoning="Qwen requested vision analysis"):
        """
        Capture VRChat screenshot optimized for Qwen + Florence-2 processing
        
        Args:
            filename (str, optional): Output filename
            context (str): Context for why this capture was taken
            reasoning (str): Qwen's reasoning for requesting this capture
        
        Returns:
            dict: Comprehensive capture information for Qwen processing
        """
        try:
            # Take full desktop screenshot (no focus changes needed)
            full_screenshot = pyautogui.screenshot()
            
            # Crop out the console area (keep left portion for VRChat)
            vrchat_screenshot = full_screenshot.crop((
                0,                    # Left edge
                0,                    # Top edge  
                self.vrchat_width,    # Right edge (excludes console)
                self.vrchat_height    # Bottom edge
            ))
            
            # Generate timestamp and filename
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            if filename is None:
                filename = f'qwen_capture_{context}_{timestamp}.png'
            
            # Ensure captures directory exists
            captures_dir = "G:/Experimental/Production/MidnightCore/Core/Engine/Captures/"
            os.makedirs(captures_dir, exist_ok=True)
            
            # Save cropped screenshot
            filepath = os.path.join(captures_dir, filename)
            vrchat_screenshot.save(filepath)
            
            # Create capture metadata for Qwen
            capture_info = {
                "timestamp": timestamp,
                "filename": filename,
                "filepath": filepath,
                "context": context,
                "reasoning": reasoning,
                "image_dimensions": {
                    "full_screen": full_screenshot.size,
                    "cropped_vrchat": vrchat_screenshot.size
                },
                "capture_settings": {
                    "console_crop": self.crop_right,
                    "vrchat_area": f"{self.vrchat_width}x{self.vrchat_height}"
                },
                "qwen_ready": True,
                "florence_ready": True
            }
            
            # Log capture for Qwen context
            self.log_capture(capture_info)
            
            print(f"Qwen VRChat capture: {filename}")
            print(f"Context: {context}")
            print(f"Original: {full_screenshot.size}, Cropped: {vrchat_screenshot.size}")
            
            return capture_info
            
        except Exception as e:
            print(f"Error in Qwen capture: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
                "context": context,
                "reasoning": reasoning,
                "qwen_ready": False,
                "florence_ready": False
            }
    
    def log_capture(self, capture_info):
        """Log capture information for Qwen analysis and context building"""
        # Add to memory
        self.capture_history.append(capture_info)
        
        # Keep only last 100 captures in memory
        if len(self.capture_history) > 100:
            self.capture_history.pop(0)
        
        # Write to Florence log file
        try:
            with open(self.capture_log_path, "a", encoding='utf-8') as f:
                f.write(f"CAPTURE: {json.dumps(capture_info)}\n")
        except Exception as e:
            print(f"Failed to write capture log: {e}")
    
    def get_recent_captures(self, count=10):
        """Get recent capture history for Qwen context"""
        recent = self.capture_history[-count:] if self.capture_history else []
        return {
            "recent_captures": recent,
            "total_captures": len(self.capture_history),
            "capture_contexts": list(set([cap.get("context", "unknown") for cap in recent]))
        }
    
    def grab_rgb_frame(self):
        """
        Grab current VRChat frame as RGB numpy array for depth processing
        
        Returns:
            numpy.ndarray: RGB frame [H, W, 3] ready for depth analysis
        """
        try:
            # Take full desktop screenshot
            full_screenshot = pyautogui.screenshot()
            print(f"DEBUG: PyAutoGUI screenshot captured: {full_screenshot.size}")
            
            # Crop to VRChat area (remove console from right edge)
            vrchat_screenshot = full_screenshot.crop((0, 0, self.vrchat_width, self.vrchat_height))
            print(f"DEBUG: VRChat area cropped to: {vrchat_screenshot.size}")
            
            # Convert to RGB numpy array
            import numpy as np
            rgb_frame = np.array(vrchat_screenshot.convert("RGB"))
            print(f"DEBUG: RGB frame converted to numpy: shape={rgb_frame.shape}, dtype={rgb_frame.dtype}")
            return rgb_frame
            
        except Exception as e:
            print(f"ERROR in RGB frame capture: {e}")
            return None
    
    def capture_sequence_for_qwen(self, contexts, delay=1.0, sequence_purpose="multi_angle_analysis"):
        """
        Capture a sequence of screenshots for comprehensive Qwen analysis
        
        Args:
            contexts (list): List of context descriptions for each capture
            delay (float): Delay between captures
            sequence_purpose (str): Overall purpose of this sequence
        
        Returns:
            dict: Information about the entire capture sequence
        """
        sequence_info = {
            "start_time": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "sequence_purpose": sequence_purpose,
            "planned_contexts": contexts,
            "captures": [],
            "total_planned": len(contexts)
        }
        
        print(f"Starting Qwen capture sequence: {sequence_purpose}")
        print(f"Planned captures: {len(contexts)}")
        
        for i, context in enumerate(contexts):
            print(f"Capture {i+1}/{len(contexts)}: {context}")
            
            capture_result = self.capture_vrchat_for_qwen(
                context=context,
                reasoning=f"Sequence step {i+1}: {sequence_purpose}"
            )
            
            sequence_info["captures"].append(capture_result)
            
            # Delay before next capture (except last one)
            if i < len(contexts) - 1:
                time.sleep(delay)
        
        sequence_info["end_time"] = datetime.now().strftime('%Y%m%d_%H%M%S')
        sequence_info["successful_captures"] = len([c for c in sequence_info["captures"] if c.get("qwen_ready", False)])
        sequence_info["sequence_complete"] = sequence_info["successful_captures"] == sequence_info["total_planned"]
        
        # Log the sequence
        self.log_capture({
            "type": "capture_sequence",
            "sequence_info": sequence_info,
            "timestamp": sequence_info["end_time"]
        })
        
        print(f"Sequence complete: {sequence_info['successful_captures']}/{sequence_info['total_planned']} successful")
        return sequence_info
    
    def adjust_crop_area(self, console_width):
        """Adjust the console crop area and log the change"""
        old_width = self.vrchat_width
        self.crop_right = console_width
        self.vrchat_width = self.screen_width - self.crop_right
        
        adjustment_info = {
            "timestamp": datetime.now().strftime('%Y%m%d_%H%M%S'),
            "type": "crop_adjustment",
            "old_vrchat_width": old_width,
            "new_vrchat_width": self.vrchat_width,
            "new_console_crop": console_width,
            "reasoning": "Manual crop area adjustment"
        }
        
        self.log_capture(adjustment_info)
        print(f"Qwen Capture: Adjusted crop area - VRChat now {self.vrchat_width}x{self.vrchat_height}")
    
    def preview_crop_areas(self):
        """Take a test screenshot and show crop boundaries with Qwen integration info"""
        try:
            # Take full screenshot
            full_screenshot = pyautogui.screenshot()
            
            # Create visual guides
            from PIL import ImageDraw, ImageFont
            draw = ImageDraw.Draw(full_screenshot)
            
            # Draw red line where we'll crop (VRChat/Console boundary)
            crop_line = self.vrchat_width
            draw.line([(crop_line, 0), (crop_line, self.screen_height)], fill='red', width=5)
            
            # Add labels with Qwen integration info
            try:
                # Try to use a larger font
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            draw.text((crop_line - 200, 50), "VRChat Area (Qwen Analysis)", fill='green', font=font)
            draw.text((crop_line + 50, 50), "Console (Excluded)", fill='red', font=font)
            draw.text((50, self.screen_height - 100), f"Qwen Capture System Ready", fill='blue', font=font)
            draw.text((50, self.screen_height - 60), f"VRChat: {self.vrchat_width}x{self.vrchat_height}", fill='blue', font=font)
            
            # Save preview
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            preview_path = f"G:/Experimental/Midnight Core/FusionCore/FusionScripts/captures/qwen_crop_preview_{timestamp}.png"
            full_screenshot.save(preview_path)
            
            # Log the preview
            preview_info = {
                "timestamp": timestamp,
                "type": "crop_preview",
                "preview_path": preview_path,
                "crop_settings": {
                    "vrchat_width": self.vrchat_width,
                    "vrchat_height": self.vrchat_height,
                    "console_crop": self.crop_right
                }
            }
            self.log_capture(preview_info)
            
            print(f"Qwen crop preview saved: {preview_path}")
            return preview_path
            
        except Exception as e:
            print(f"Error creating Qwen preview: {e}")
            return None

def qwen_capture_screenshot(context="analysis", reasoning="Qwen vision request"):
    """
    Convenience function for Qwen to capture VRChat screenshot
    
    Args:
        context (str): Context for the capture
        reasoning (str): Why Qwen requested this capture
    
    Returns:
        dict: Capture information optimized for Qwen processing
    """
    capturer = QwenVRChatCapture()
    return capturer.capture_vrchat_for_qwen(context=context, reasoning=reasoning)

if __name__ == "__main__":
    # Test the Qwen-enhanced capture system
    print("Testing Qwen VRChat capture system...")
    
    capturer = QwenVRChatCapture()
    
    # Create a preview showing crop boundaries
    print("Creating Qwen crop preview...")
    capturer.preview_crop_areas()
    
    # Take a test capture
    print("Taking test Qwen capture...")
    capture_result = capturer.capture_vrchat_for_qwen(
        context="system_test",
        reasoning="Initial system test and validation"
    )
    
    if capture_result.get("qwen_ready"):
        print(f"Qwen capture successful: {capture_result['filepath']}")
        print(f"Ready for Florence-2 analysis: {capture_result['florence_ready']}")
        
        # Test sequence capture
        print("\nTesting capture sequence...")
        sequence_result = capturer.capture_sequence_for_qwen(
            contexts=["sequence_test_1", "sequence_test_2", "sequence_test_3"],
            delay=0.5,
            sequence_purpose="system_validation_sequence"
        )
        print(f"Sequence complete: {sequence_result['sequence_complete']}")
    else:
        print("Qwen capture failed")
        if "error" in capture_result:
            print(f"Error: {capture_result['error']}")