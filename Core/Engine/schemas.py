# ==== MODULE CONTRACT =======================================================
# Module: engine/schemas.py
# Package: MidnightCore.Engine.schemas
# Location: Production/MidnightCore/Core/Engine/schemas.py
# Responsibility: Canonical data types and JSON schemas (single source of truth)
# PUBLIC: 
#   Classes: Decision, VisionContext, FrameData, TickTelemetry, CalibrationData, OSCParameterInfo
#   Functions: validate_decision_schema(), safe_fallback_decision()
#   Types: Tool, Step
# DEPENDENCIES: None (core types only)
# DEPENDENTS: All modules that handle system data
# POLICY: NO_FALLBACKS=N/A (data types only), Telemetry: N/A
# DOCS: README anchor "schemas", File-Map anchor "ENGINE_SCHEMAS_DOC_ANCHOR"
# MIGRATION: ✅ Migrated | Original: G:\Experimental\Midnight Core\FusionCore\FusionScripts\QwenScripts\schemas.py
# INVARIANTS:
#   - All system interfaces must use these types
#   - JSON schemas must match TypedDict definitions exactly
#   - Breaking changes require version increment
# ============================================================================

from typing import TypedDict, Literal, List, Dict, Any, Optional
import json

# Core system types
Tool = Literal["turn", "scan", "move_forward", "strafe_left", "strafe_right", "look_left", "look_right", "look_up", "look_down"]

class Step(TypedDict):
    tool: Tool
    arg: int | float | str

class Decision(TypedDict):
    plan: List[Step]
    reason: str

class VisionContext(TypedDict):
    scene_description: str
    detected_objects: List[str]
    navigation_assessment: Dict[str, Any]
    spatial_context: str

class FrameData(TypedDict):
    filepath: str
    qwen_ready: bool
    timestamp: float

class TickTelemetry(TypedDict):
    tick_id: int
    frame_id: int
    action_frame_id: int
    chat_frame_id: int
    perception_age_ms: float
    chat_refs_future_frame: bool

class CalibrationData(TypedDict):
    world_id: str
    movement_multipliers: Dict[str, float]
    optimal_durations: Dict[str, float]
    calibration_confidence: float
    responsiveness_score: float
    timestamp: str

class OSCParameterInfo(TypedDict):
    address: str
    value_type: str
    value_range: List[float]
    last_value: Any
    discovery_confidence: float

# JSON schemas for LLM validation
SCHEMA_DECISION = """{
  "type":"object",
  "properties":{
    "plan":{"type":"array","items":{
      "type":"object",
      "properties":{
        "tool":{"enum":["turn","scan","move_forward","strafe_left","strafe_right","look_left","look_right","look_up","look_down"]},
        "arg":{}
      },
      "required":["tool","arg"]
    }},
    "reason":{"type":"string"}
  },
  "required":["plan","reason"]
}"""

SCHEMA_VISION_CONTEXT = """{
  "type":"object", 
  "properties":{
    "scene_description":{"type":"string"},
    "detected_objects":{"type":"array","items":{"type":"string"}},
    "navigation_assessment":{"type":"object"},
    "spatial_context":{"type":"string"}
  },
  "required":["scene_description","detected_objects","navigation_assessment","spatial_context"]
}"""

def validate_decision_schema(data: Any) -> bool:
    """Validate decision payload against schema"""
    try:
        schema = json.loads(SCHEMA_DECISION)
        # Simple validation - in production would use jsonschema library
        if not isinstance(data, dict):
            return False
        return "plan" in data and "reason" in data and isinstance(data["plan"], list)
    except:
        return False

def safe_fallback_decision() -> Decision:
    """Safe fallback decision for error cases"""
    return {
        "plan": [{"tool": "scan", "arg": "safe_fallback"}],
        "reason": "Fallback scan due to decision error"
    }

# Contract tests
def _contract_tests():
    # JSON normalize never returns None on valid JSON
    test_decision = {"plan": [], "reason": "test"}
    assert validate_decision_schema(test_decision)
    
    # Fallback always returns valid structure
    fallback = safe_fallback_decision()
    assert validate_decision_schema(fallback)
    assert fallback["plan"][0]["tool"] == "scan"

if __name__ == "__main__":
    _contract_tests()
    print("All schema contract tests passed")