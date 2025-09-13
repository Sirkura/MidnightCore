#!/usr/bin/env python3
"""
Florence-2 Image Analysis for VRChat Autonomous Exploration with Qwen Integration
Provides detailed scene analysis for screenshots captured during exploration
Compatible with Qwen 3 8B reasoning system
"""

import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import json
from datetime import datetime
import os

class FlorenceAnalyzer:
    def __init__(self, model_name='microsoft/Florence-2-large'):
        """Initialize Florence-2 model for image analysis"""
        print("Loading Florence-2 model for Qwen integration...")
        
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float32, 
            trust_remote_code=True
        )
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        print(f"Florence-2 loaded on {self.device} - Ready for Qwen integration")
    
    def analyze_image(self, image_path, tasks=None):
        """
        Analyze image with Florence-2 - Optimized for Qwen consumption
        
        Args:
            image_path (str): Path to image file
            tasks (list): Florence-2 tasks to run. Defaults to comprehensive analysis
        
        Returns:
            dict: Analysis results formatted for Qwen processing
        """
        if tasks is None:
            tasks = [
                '<CAPTION>',
                '<DETAILED_CAPTION>', 
                '<MORE_DETAILED_CAPTION>',
                '<OD>',  # Object Detection
                '<DENSE_REGION_CAPTION>'
            ]
        
        # Load image
        image = Image.open(image_path)
        
        results = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'image_path': image_path,
            'image_size': image.size,
            'analysis': {},
            'qwen_ready': True  # Flag for Qwen integration
        }
        
        for task in tasks:
            try:
                inputs = self.processor(text=task, images=image, return_tensors='pt').to(
                    self.device, dtype=torch.float32
                )
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        input_ids=inputs['input_ids'],
                        pixel_values=inputs['pixel_values'],
                        max_new_tokens=1024,
                        do_sample=False,
                        num_beams=3
                    )
                
                generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed_answer = self.processor.post_process_generation(
                    generated_text, task=task, image_size=image.size
                )
                
                results['analysis'][task] = parsed_answer.get(task, parsed_answer)
                
            except Exception as e:
                print(f"Error with {task}: {e}")
                results['analysis'][task] = f"Error: {e}"
        
        return results
    
    def analyze_for_navigation(self, image_path):
        """
        Specialized analysis for VRChat navigation decisions
        Formatted for optimal Qwen 3 8B processing
        """
        analysis = self.analyze_image(image_path, [
            '<MORE_DETAILED_CAPTION>',
            '<OD>'
        ])
        
        # Extract navigation-relevant information
        detailed_caption = analysis['analysis'].get('<MORE_DETAILED_CAPTION>', '')
        objects = analysis['analysis'].get('<OD>', {})
        
        navigation_summary = {
            'scene_description': detailed_caption,
            'detected_objects': objects.get('labels', []) if isinstance(objects, dict) else [],
            'navigation_assessment': self._assess_navigation(detailed_caption, objects),
            'timestamp': analysis['timestamp'],
            'qwen_context': self._create_qwen_context(detailed_caption, objects)
        }
        
        return navigation_summary
    
    def _assess_navigation(self, caption, objects):
        """Assess navigation opportunities from Florence-2 analysis"""
        assessment = {
            'pathways': [],
            'obstacles': [],
            'points_of_interest': [],
            'spatial_context': '',
            'social_elements': []  # Added for VRChat social interaction
        }
        
        # Analyze caption for navigation cues
        caption_lower = caption.lower() if isinstance(caption, str) else ''
        
        # Look for pathways
        pathway_keywords = ['door', 'stairs', 'staircase', 'corridor', 'hallway', 'path', 'entrance', 'exit', 'portal']
        for keyword in pathway_keywords:
            if keyword in caption_lower:
                assessment['pathways'].append(keyword)
        
        # Look for obstacles
        obstacle_keywords = ['wall', 'barrier', 'obstacle', 'furniture', 'blocked']
        for keyword in obstacle_keywords:
            if keyword in caption_lower:
                assessment['obstacles'].append(keyword)
        
        # Look for points of interest
        poi_keywords = ['painting', 'picture', 'window', 'plant', 'display', 'sign', 'light']
        for keyword in poi_keywords:
            if keyword in caption_lower:
                assessment['points_of_interest'].append(keyword)
        
        # Look for social elements (VRChat specific)
        social_keywords = ['character', 'avatar', 'player', 'person', 'friend', 'nameplate']
        for keyword in social_keywords:
            if keyword in caption_lower:
                assessment['social_elements'].append(keyword)
        
        # Extract spatial context
        spatial_keywords = ['left', 'right', 'center', 'front', 'behind', 'above', 'below', 'corner']
        spatial_mentions = [word for word in spatial_keywords if word in caption_lower]
        assessment['spatial_context'] = ', '.join(spatial_mentions)
        
        return assessment
    
    def _create_qwen_context(self, caption, objects):
        """Create structured context for Qwen 3 8B processing"""
        context = {
            'scene_summary': caption if isinstance(caption, str) else 'No caption available',
            'object_count': len(objects.get('labels', [])) if isinstance(objects, dict) else 0,
            'decision_context': 'VRChat exploration and navigation',
            'available_actions': ['move_forward', 'turn_left', 'turn_right', 'approach_target', 'scan_area'],
            'priority': 'navigation_and_social_interaction'
        }
        return context
    
    def save_analysis(self, analysis, output_path=None):
        """Save analysis results to JSON file"""
        if output_path is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = f"G:/Experimental/Production/MidnightCore/Core/Engine/Captures/qwen_florence_analysis_{timestamp}.json"
        
        with open(output_path, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"Qwen-compatible analysis saved to: {output_path}")
        return output_path

# Global analyzer instance
_florence_analyzer = None

def get_florence_analyzer():
    """Get global Florence-2 analyzer instance (singleton pattern)"""
    global _florence_analyzer
    if _florence_analyzer is None:
        _florence_analyzer = FlorenceAnalyzer()
    return _florence_analyzer

def analyze_screenshot(image_path, for_navigation=True):
    """
    Quick function to analyze a screenshot for Qwen integration
    
    Args:
        image_path (str): Path to screenshot
        for_navigation (bool): If True, returns navigation-focused analysis
    
    Returns:
        dict: Analysis results optimized for Qwen processing
    """
    analyzer = get_florence_analyzer()
    
    if for_navigation:
        return analyzer.analyze_for_navigation(image_path)
    else:
        return analyzer.analyze_image(image_path)

if __name__ == "__main__":
    # Test the analyzer with Qwen compatibility
    analyzer = FlorenceAnalyzer()
    
    # Test with recent capture
    import glob
    capture_dir = "G:/Experimental/Production/MidnightCore/Core/Engine/Captures"
    recent_captures = glob.glob(f"{capture_dir}/*.png")
    
    if recent_captures:
        # Use most recent capture
        test_image = sorted(recent_captures)[-1]
        print(f"Testing Florence-2 analyzer with Qwen integration on: {test_image}")
        results = analyzer.analyze_for_navigation(test_image)
        print("\nQwen-Compatible Navigation Analysis Results:")
        print(json.dumps(results, indent=2))
    else:
        print("No test images found in captures directory")