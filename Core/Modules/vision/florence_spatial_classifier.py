#!/usr/bin/env python3
# ==== MODULE CONTRACT =======================================================
# Module: vision/florence_spatial_classifier.py
# Package: MidnightCore.Modules.vision.florence_spatial_classifier
# Location: Production/MidnightCore/Core/Modules/vision/florence_spatial_classifier.py
# Responsibility: Advanced spatial awareness using Florence-2 semantic understanding
# PUBLIC: FlorenceSpatialClassifier class, analyze_spatial_scene() function
# DEPENDENCIES: florence_analyzer, transformers, torch
# POLICY: NO_FALLBACKS=deny, Telemetry: florence.spatial.*
# MIGRATION: Replaces depth-based spatial system with semantic spatial reasoning
# ============================================================================

"""
Florence-2 Enhanced Spatial Awareness System
=============================================

Advanced spatial classification system that uses Florence-2's semantic understanding
to provide VRChat navigation without relying on problematic depth estimation.

Key Features:
- Semantic spatial classification (close/medium/far)
- Directional obstacle detection (left/center/right)
- Movement path identification
- Distance estimation using semantic cues
- Safe zone identification for movement planning
"""

import sys
import os
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import json
import numpy as np
import cv2
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import time

# Add existing Florence analyzer to path
sys.path.insert(0, os.path.dirname(__file__))
from florence_analyzer import FlorenceAnalyzer

class FlorenceSpatialClassifier:
    """
    Advanced spatial awareness system using Florence-2 semantic understanding
    
    Provides spatial navigation capabilities without depth estimation by using
    semantic scene understanding, object relationships, and contextual clues
    """
    
    def __init__(self, model_name='microsoft/Florence-2-large'):
        """Initialize Florence-2 spatial classifier"""
        self.analyzer = FlorenceAnalyzer(model_name)
        self.spatial_history = []
        self.movement_memory = {}
        
        print("INIT: Florence-2 Spatial Classifier - Semantic navigation system")
    
    def analyze_spatial_scene(self, image_path: str) -> Dict[str, Any]:
        """
        Comprehensive spatial analysis of VRChat scene using semantic understanding
        
        Args:
            image_path: Path to screenshot
            
        Returns:
            Spatial analysis with navigation recommendations
        """
        start_time = time.time()
        
        # Get comprehensive Florence-2 analysis
        analysis = self.analyzer.analyze_image(image_path, [
            '<MORE_DETAILED_CAPTION>',
            '<DENSE_REGION_CAPTION>',
            '<OD>'
        ])
        
        # Extract semantic information
        detailed_caption = analysis['analysis'].get('<MORE_DETAILED_CAPTION>', '')
        dense_regions = analysis['analysis'].get('<DENSE_REGION_CAPTION>', {})
        objects = analysis['analysis'].get('<OD>', {})
        
        # Perform spatial classification
        spatial_result = {
            'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
            'image_path': image_path,
            'scene_description': detailed_caption,
            'spatial_classification': self._classify_spatial_zones(detailed_caption, dense_regions, objects),
            'movement_assessment': self._assess_movement_options(detailed_caption, dense_regions),
            'navigation_safety': self._evaluate_navigation_safety(detailed_caption, objects),
            'directional_analysis': self._analyze_directional_context(detailed_caption, dense_regions),
            'distance_estimation': self._estimate_semantic_distances(detailed_caption, dense_regions),
            'recommended_actions': [],
            'confidence_score': 0.0,
            'processing_time_ms': 0
        }
        
        # Generate navigation recommendations
        spatial_result['recommended_actions'] = self._generate_navigation_recommendations(spatial_result)
        spatial_result['confidence_score'] = self._calculate_confidence_score(spatial_result)
        
        processing_time = (time.time() - start_time) * 1000
        spatial_result['processing_time_ms'] = processing_time
        
        # Store in movement history
        self.spatial_history.append(spatial_result)
        if len(self.spatial_history) > 10:
            self.spatial_history.pop(0)  # Keep last 10 analyses
        
        print(f"FLORENCE SPATIAL: {processing_time:.1f}ms analysis complete")
        return spatial_result
    
    def _classify_spatial_zones(self, caption: str, dense_regions: Dict, objects: Dict) -> Dict[str, Any]:
        """Classify spatial zones based on semantic understanding"""
        zones = {
            'immediate_zone': {'distance': 'close', 'objects': [], 'safety': 'unknown'},
            'intermediate_zone': {'distance': 'medium', 'objects': [], 'safety': 'unknown'},
            'far_zone': {'distance': 'far', 'objects': [], 'safety': 'unknown'},
            'dominant_zone': 'unknown'
        }
        
        caption_lower = caption.lower() if caption else ''
        
        # Immediate zone indicators (0-3m equivalent)
        immediate_keywords = ['close', 'near', 'front', 'immediate', 'directly', 'adjacent', 'next to']
        immediate_objects = ['door handle', 'button', 'switch', 'detail', 'texture']
        
        # Intermediate zone indicators (3-10m equivalent) 
        intermediate_keywords = ['across', 'middle', 'center', 'room', 'space', 'area']
        intermediate_objects = ['furniture', 'table', 'chair', 'person', 'avatar']
        
        # Far zone indicators (10m+ equivalent)
        far_keywords = ['distant', 'far', 'background', 'horizon', 'end', 'opposite', 'back']
        far_objects = ['wall', 'ceiling', 'skyline', 'distant']
        
        # Analyze caption for zone indicators
        immediate_count = sum(1 for keyword in immediate_keywords if keyword in caption_lower)
        intermediate_count = sum(1 for keyword in intermediate_keywords if keyword in caption_lower)
        far_count = sum(1 for keyword in far_keywords if keyword in caption_lower)
        
        # Determine dominant zone
        zone_scores = {
            'immediate': immediate_count,
            'intermediate': intermediate_count, 
            'far': far_count
        }
        zones['dominant_zone'] = max(zone_scores, key=zone_scores.get)
        
        # Safety assessment based on obstacles
        obstacle_keywords = ['wall', 'barrier', 'obstacle', 'blocked', 'closed']
        safety_level = 'safe'
        if any(keyword in caption_lower for keyword in obstacle_keywords):
            if 'close' in caption_lower or 'near' in caption_lower:
                safety_level = 'caution'
            elif 'blocked' in caption_lower or 'barrier' in caption_lower:
                safety_level = 'blocked'
        
        zones['immediate_zone']['safety'] = safety_level
        zones['intermediate_zone']['safety'] = 'safe'  # Usually navigable
        zones['far_zone']['safety'] = 'safe'  # Usually distant obstacles
        
        return zones
    
    def _assess_movement_options(self, caption: str, dense_regions: Dict) -> Dict[str, Any]:
        """Assess available movement directions and paths"""
        movement = {
            'forward': {'available': True, 'confidence': 0.5, 'obstacles': []},
            'left': {'available': True, 'confidence': 0.5, 'obstacles': []},
            'right': {'available': True, 'confidence': 0.5, 'obstacles': []},
            'backward': {'available': True, 'confidence': 0.3, 'obstacles': []},  # Usually less info
            'best_direction': 'forward'
        }
        
        caption_lower = caption.lower() if caption else ''
        
        # Forward movement assessment
        forward_blocked = ['wall ahead', 'barrier in front', 'blocked forward', 'door closed']
        forward_clear = ['open space', 'hallway', 'corridor', 'path ahead', 'room extends']
        
        if any(phrase in caption_lower for phrase in forward_blocked):
            movement['forward']['available'] = False
            movement['forward']['confidence'] = 0.1
            movement['forward']['obstacles'] = ['front_obstacle']
        elif any(phrase in caption_lower for phrase in forward_clear):
            movement['forward']['confidence'] = 0.9
        
        # Left/Right movement assessment
        left_keywords = ['left door', 'left corridor', 'left opening', 'left path']
        right_keywords = ['right door', 'right corridor', 'right opening', 'right path']
        
        if any(phrase in caption_lower for phrase in left_keywords):
            movement['left']['confidence'] = 0.8
        if any(phrase in caption_lower for phrase in right_keywords):
            movement['right']['confidence'] = 0.8
        
        # Determine best direction
        direction_scores = {
            'forward': movement['forward']['confidence'],
            'left': movement['left']['confidence'],
            'right': movement['right']['confidence']
        }
        movement['best_direction'] = max(direction_scores, key=direction_scores.get)
        
        return movement
    
    def _evaluate_navigation_safety(self, caption: str, objects: Dict) -> Dict[str, Any]:
        """Evaluate overall navigation safety"""
        safety = {
            'overall_safety': 'safe',
            'hazards': [],
            'safe_zones': [],
            'movement_clearance': 'good',
            'collision_risk': 'low'
        }
        
        caption_lower = caption.lower() if caption else ''
        
        # Hazard detection
        hazard_keywords = [
            'steep', 'drop', 'edge', 'cliff', 'hole', 'pit',
            'blocked', 'barrier', 'obstacle', 'narrow',
            'crowded', 'busy', 'restricted'
        ]
        
        detected_hazards = [hazard for hazard in hazard_keywords if hazard in caption_lower]
        safety['hazards'] = detected_hazards
        
        # Safe zone identification
        safe_keywords = [
            'open space', 'wide', 'clear', 'empty', 'spacious',
            'hallway', 'corridor', 'room', 'area'
        ]
        
        detected_safe = [zone for zone in safe_keywords if zone in caption_lower]
        safety['safe_zones'] = detected_safe
        
        # Overall safety assessment
        if len(detected_hazards) > 2:
            safety['overall_safety'] = 'caution'
            safety['collision_risk'] = 'medium'
        elif len(detected_hazards) > 0:
            safety['overall_safety'] = 'moderate'
            safety['collision_risk'] = 'low-medium'
        
        if len(detected_safe) > 1:
            safety['movement_clearance'] = 'excellent'
        elif len(detected_safe) > 0:
            safety['movement_clearance'] = 'good'
        else:
            safety['movement_clearance'] = 'limited'
        
        return safety
    
    def _analyze_directional_context(self, caption: str, dense_regions: Dict) -> Dict[str, Any]:
        """Analyze directional spatial context"""
        directions = {
            'left_context': {'objects': [], 'navigable': True, 'description': ''},
            'center_context': {'objects': [], 'navigable': True, 'description': ''},
            'right_context': {'objects': [], 'navigable': True, 'description': ''},
            'spatial_layout': 'unknown'
        }
        
        caption_lower = caption.lower() if caption else ''
        
        # Extract directional references
        left_patterns = ['on the left', 'to the left', 'left side', 'left of']
        center_patterns = ['in the center', 'middle', 'center of', 'straight ahead']
        right_patterns = ['on the right', 'to the right', 'right side', 'right of']
        
        # Analyze each direction
        for pattern in left_patterns:
            if pattern in caption_lower:
                # Extract context around the pattern
                start_idx = caption_lower.find(pattern)
                context = caption_lower[max(0, start_idx-20):start_idx+50]
                directions['left_context']['description'] = context
        
        for pattern in center_patterns:
            if pattern in caption_lower:
                start_idx = caption_lower.find(pattern)
                context = caption_lower[max(0, start_idx-20):start_idx+50]
                directions['center_context']['description'] = context
        
        for pattern in right_patterns:
            if pattern in caption_lower:
                start_idx = caption_lower.find(pattern)
                context = caption_lower[max(0, start_idx-20):start_idx+50]
                directions['right_context']['description'] = context
        
        # Determine spatial layout
        if 'hallway' in caption_lower or 'corridor' in caption_lower:
            directions['spatial_layout'] = 'linear_hallway'
        elif 'room' in caption_lower and ('door' in caption_lower or 'opening' in caption_lower):
            directions['spatial_layout'] = 'room_with_exits'
        elif 'open' in caption_lower and 'space' in caption_lower:
            directions['spatial_layout'] = 'open_area'
        else:
            directions['spatial_layout'] = 'complex_environment'
        
        return directions
    
    def _estimate_semantic_distances(self, caption: str, dense_regions: Dict) -> Dict[str, Any]:
        """Estimate distances using semantic cues instead of depth"""
        distances = {
            'closest_obstacle': 'medium',
            'walkable_distance': 'good',
            'visibility_range': 'good',
            'semantic_scale': 'room_scale',
            'movement_estimate': '5-10_steps'
        }
        
        caption_lower = caption.lower() if caption else ''
        
        # Distance indicators from semantic content
        close_indicators = ['close', 'near', 'adjacent', 'next to', 'touching', 'directly']
        medium_indicators = ['across', 'middle', 'center', 'room', 'space']
        far_indicators = ['distant', 'far', 'background', 'end', 'opposite', 'back']
        
        # Closest obstacle assessment
        if any(indicator in caption_lower for indicator in close_indicators):
            distances['closest_obstacle'] = 'close'
            distances['movement_estimate'] = '1-3_steps'
        elif any(indicator in caption_lower for indicator in far_indicators):
            distances['closest_obstacle'] = 'far'
            distances['movement_estimate'] = '10+_steps'
        
        # Walkable distance estimation
        if 'long' in caption_lower or 'extended' in caption_lower:
            distances['walkable_distance'] = 'excellent'
        elif 'short' in caption_lower or 'narrow' in caption_lower:
            distances['walkable_distance'] = 'limited'
        
        # Scale determination
        scale_keywords = {
            'corridor_scale': ['hallway', 'corridor', 'passage'],
            'room_scale': ['room', 'space', 'area', 'chamber'],
            'building_scale': ['large', 'vast', 'expansive', 'huge']
        }
        
        for scale, keywords in scale_keywords.items():
            if any(keyword in caption_lower for keyword in keywords):
                distances['semantic_scale'] = scale
                break
        
        return distances
    
    def _generate_navigation_recommendations(self, spatial_result: Dict) -> List[str]:
        """Generate actionable navigation recommendations"""
        recommendations = []
        
        movement = spatial_result.get('movement_assessment', {})
        safety = spatial_result.get('navigation_safety', {})
        zones = spatial_result.get('spatial_classification', {})
        
        # Primary movement recommendation
        best_direction = movement.get('best_direction', 'forward')
        if movement.get(best_direction, {}).get('available', True):
            confidence = movement.get(best_direction, {}).get('confidence', 0.5)
            if confidence > 0.7:
                recommendations.append(f"RECOMMENDED: Move {best_direction} (high confidence)")
            elif confidence > 0.4:
                recommendations.append(f"ACCEPTABLE: Move {best_direction} (moderate confidence)")
            else:
                recommendations.append(f"CAUTION: Move {best_direction} carefully (low confidence)")
        
        # Safety recommendations
        if safety.get('overall_safety') == 'caution':
            recommendations.append("SAFETY: Reduce movement speed")
        if safety.get('collision_risk') != 'low':
            recommendations.append("SAFETY: Enhanced collision monitoring needed")
        
        # Zone-based recommendations
        dominant_zone = zones.get('dominant_zone', 'unknown')
        if dominant_zone == 'immediate':
            recommendations.append("PROXIMITY: Very close to objects - precise movement required")
        elif dominant_zone == 'far':
            recommendations.append("DISTANCE: Clear long-range view - normal movement OK")
        
        # Default fallback
        if not recommendations:
            recommendations.append("DEFAULT: Scan area and proceed with caution")
        
        return recommendations
    
    def _calculate_confidence_score(self, spatial_result: Dict) -> float:
        """Calculate overall confidence in spatial analysis"""
        confidence_factors = []
        
        # Movement assessment confidence
        movement = spatial_result.get('movement_assessment', {})
        best_dir_confidence = movement.get(movement.get('best_direction', 'forward'), {}).get('confidence', 0.5)
        confidence_factors.append(best_dir_confidence)
        
        # Safety assessment confidence (based on number of detected features)
        safety = spatial_result.get('navigation_safety', {})
        safety_features = len(safety.get('safe_zones', [])) + len(safety.get('hazards', []))
        safety_confidence = min(1.0, safety_features * 0.2 + 0.3)  # 0.3 base + features
        confidence_factors.append(safety_confidence)
        
        # Scene description richness (longer descriptions = more confident)
        description_length = len(spatial_result.get('scene_description', ''))
        desc_confidence = min(1.0, description_length / 100)  # Normalize to max 1.0
        confidence_factors.append(desc_confidence)
        
        # Overall confidence is average of factors
        overall_confidence = sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.5
        return round(overall_confidence, 3)
    
    def get_spatial_history(self) -> List[Dict]:
        """Get recent spatial analysis history"""
        return self.spatial_history.copy()
    
    def compare_spatial_changes(self, previous_analysis: Dict, current_analysis: Dict) -> Dict[str, Any]:
        """Compare two spatial analyses to detect movement/changes"""
        changes = {
            'movement_detected': False,
            'direction_change': None,
            'safety_change': None,
            'zone_transition': None,
            'confidence_change': 0.0
        }
        
        # Compare dominant zones
        prev_zone = previous_analysis.get('spatial_classification', {}).get('dominant_zone')
        curr_zone = current_analysis.get('spatial_classification', {}).get('dominant_zone')
        if prev_zone != curr_zone:
            changes['zone_transition'] = f"{prev_zone} -> {curr_zone}"
            changes['movement_detected'] = True
        
        # Compare best directions
        prev_dir = previous_analysis.get('movement_assessment', {}).get('best_direction')
        curr_dir = current_analysis.get('movement_assessment', {}).get('best_direction')
        if prev_dir != curr_dir:
            changes['direction_change'] = f"{prev_dir} -> {curr_dir}"
        
        # Compare safety levels
        prev_safety = previous_analysis.get('navigation_safety', {}).get('overall_safety')
        curr_safety = current_analysis.get('navigation_safety', {}).get('overall_safety')
        if prev_safety != curr_safety:
            changes['safety_change'] = f"{prev_safety} -> {curr_safety}"
        
        # Compare confidence scores
        prev_conf = previous_analysis.get('confidence_score', 0.5)
        curr_conf = current_analysis.get('confidence_score', 0.5)
        changes['confidence_change'] = curr_conf - prev_conf
        
        return changes

# ============================================================================
# Integration functions for existing MidnightCore system
# ============================================================================

# Global classifier instance
_spatial_classifier = None

def get_spatial_classifier() -> FlorenceSpatialClassifier:
    """Get global spatial classifier instance (singleton pattern)"""
    global _spatial_classifier
    if _spatial_classifier is None:
        _spatial_classifier = FlorenceSpatialClassifier()
    return _spatial_classifier

def analyze_spatial_scene(image_path: str) -> Dict[str, Any]:
    """
    Quick function to analyze spatial scene for navigation
    
    Args:
        image_path: Path to screenshot
        
    Returns:
        Comprehensive spatial analysis for navigation decisions
    """
    classifier = get_spatial_classifier()
    return classifier.analyze_spatial_scene(image_path)

def get_navigation_recommendations(image_path: str) -> List[str]:
    """
    Get immediate navigation recommendations from spatial analysis
    
    Args:
        image_path: Path to screenshot
        
    Returns:
        List of actionable navigation recommendations
    """
    analysis = analyze_spatial_scene(image_path)
    return analysis.get('recommended_actions', ['DEFAULT: Scan area and proceed with caution'])

# ============================================================================
# Testing functions
# ============================================================================

def test_spatial_classifier():
    """Test spatial classifier with sample images"""
    import glob
    
    print("TESTING: Florence-2 Spatial Classifier")
    print("=" * 50)
    
    # Look for test images
    test_dir = r"G:\Experimental\Production\.temp\TestingScripts"
    test_images = glob.glob(f"{test_dir}/*screenshot*.png")
    
    if not test_images:
        print("No test images found. Creating dummy test...")
        # Create a simple test image
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        test_path = os.path.join(test_dir, "spatial_test.png")
        cv2.imwrite(test_path, test_image)
        test_images = [test_path]
    
    classifier = FlorenceSpatialClassifier()
    
    for i, image_path in enumerate(test_images[:3]):  # Test max 3 images
        print(f"\nTesting image {i+1}: {os.path.basename(image_path)}")
        
        try:
            result = classifier.analyze_spatial_scene(image_path)
            
            print(f"  Processing time: {result['processing_time_ms']:.1f}ms")
            print(f"  Confidence score: {result['confidence_score']:.3f}")
            print(f"  Dominant zone: {result['spatial_classification']['dominant_zone']}")
            print(f"  Best direction: {result['movement_assessment']['best_direction']}")
            print(f"  Safety level: {result['navigation_safety']['overall_safety']}")
            print(f"  Recommendations: {len(result['recommended_actions'])}")
            
            for rec in result['recommended_actions']:
                print(f"    - {rec}")
                
        except Exception as e:
            print(f"  ERROR: {e}")
    
    print("\nSpatial classifier test complete")

if __name__ == "__main__":
    test_spatial_classifier()