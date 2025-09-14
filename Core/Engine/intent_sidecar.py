# ==== CONTRACT ==========================================================
# Module: intent_sidecar.py
# Responsibility: Extract action commands from natural language reasoning
# Inputs:  Raw LLM text, spatial context
# Outputs: Normalized action command strings
# Invariants:
#  - Always returns valid action string, never None
#  - Safety vetoes override all other patterns
#  - Defaults to scanning if no clear intent detected
#  - Separates "voice" (natural reasoning) from "controls" (action commands)
# ========================================================================

import re

__all__ = ["extract_intent_sidecar", "clean_natural_response", "get_last_rule_tag"]

# Global for rule tag tracking (for unified logging)
_last_rule_tag = "unknown"

def extract_intent_sidecar(natural_reasoning: str, spatial_context: str = "") -> str:
    """Deterministic intent extraction from natural language (sidecar approach)"""
    global _last_rule_tag  # For logging purposes
    
    text = natural_reasoning.lower()
    print(f"SIDECAR DEBUG: Analyzing text: '{text[:100]}...'")
    
    # Handle explicit choices - look for "I will choose:" or similar patterns
    choice_match = re.search(r'(?:i will choose|i choose|my choice|i\'ll choose)[:\s]*["\']?([^"\'.\n]+)', text)
    if choice_match:
        chosen_text = choice_match.group(1).strip()
        print(f"SIDECAR: Explicit choice detected: '{chosen_text}'")
        text = chosen_text  # Use only the chosen action, not the full option list
    
    # Safety veto - if danger is mentioned, force scanning
    if re.search(r'\b(too close|blocked|collision|0\.\s*3m|<\s*0\.8m|wall|obstacle)\b', text):
        print("SIDECAR: Safety veto triggered - forcing scan")
        _last_rule_tag = "safety_veto"
        return "I want to look left to scan the area"
    
    # Look/scan patterns (high priority)
    if re.search(r'\b(look left|scan left|check left|turn left|investigate left|left side)\b', text):
        print("SIDECAR: Left intent detected")
        _last_rule_tag = "left"
        return "I want to look left to scan the area"
        
    if re.search(r'\b(look right|scan right|check right|turn right|investigate right|right side)\b', text):
        print("SIDECAR: Right intent detected")
        _last_rule_tag = "right"
        return "I want to look right to check my surroundings"
        
    if re.search(r'\b(look up|scan up|check up|above)\b', text):
        print("SIDECAR: Up intent detected")
        _last_rule_tag = "up"
        return "I want to look up to scan above"
        
    if re.search(r'\b(look down|scan down|check down|below)\b', text):
        print("SIDECAR: Down intent detected")
        _last_rule_tag = "down"
        return "I want to look down to check below"
    
    # HIGH PRIORITY: Center intrigue should override general scanning
    if spatial_context and re.search(r'\bCENTER\b', spatial_context) and "HIGHLY INTRIGUING" in spatial_context:
        print("SIDECAR: HIGH PRIORITY - CENTER intrigue detected, overriding scan intent")
        _last_rule_tag = "forward"
        return "I want to move forward to explore"
    
    # General scanning intent (medium priority)
    if re.search(r'\b(look around|scan|check around|examine surroundings|explore area)\b', text):
        print("SIDECAR: General scan intent - choosing left")
        _last_rule_tag = "scan"
        return "I want to look left to scan the area"
    
    # Movement patterns (lower priority) - made less restrictive
    if re.search(r'\b(move forward|step forward|continue forward|walk forward|approach|investigate|explore|go forward|head forward|walk into|go into)\b', text):
        print(f"SIDECAR: Movement pattern matched in text: '{text}'")
        print(f"SIDECAR: Spatial context: '{spatial_context}'")
        # Allow forward movement if there are interesting spatial references or clear exploration intent
        if ("HIGHLY INTRIGUING" in spatial_context) or \
           re.search(r'\b(going|moving|heading|check out|explore|investigate|go to|walk to|into|through|toward)\b', text) or \
           re.search(r'\b(door|entrance|area|sector|room|space|club|building|forward|ahead)\b', text):
            print("SIDECAR: Forward movement allowed - clear exploration intent detected")
            _last_rule_tag = "forward"
            return "I want to move forward to explore"
        else:
            print("SIDECAR: Forward blocked - ambiguous intent, choosing scan instead")
            _last_rule_tag = "scan"
            return "I want to look right to check my surroundings"
    
    # Spatial context fallback - use intrigue to guide direction when reasoning is weak
    if spatial_context:
        # Use word boundaries to prevent substring false matches (e.g., "buiLDing" matching "LEFT")
        if re.search(r'\bLEFT\b', spatial_context) and "HIGHLY INTRIGUING" in spatial_context:
            print("SIDECAR: Spatial context fallback - LEFT intrigue detected")
            _last_rule_tag = "left"
            return "I want to look left to scan the area"
        elif re.search(r'\bRIGHT\b', spatial_context) and "HIGHLY INTRIGUING" in spatial_context:
            print("SIDECAR: Spatial context fallback - RIGHT intrigue detected") 
            _last_rule_tag = "right"
            return "I want to look right to check my surroundings"
        # CENTER intrigue moved to high priority section above
    
    # Final default: safe scanning instead of random forward movement
    print("SIDECAR: No clear intent detected - defaulting to safe scan")
    _last_rule_tag = "scan"
    return "I want to look left to scan the area"

def get_last_rule_tag() -> str:
    """Get the last rule tag used by sidecar for logging purposes"""
    global _last_rule_tag
    return _last_rule_tag

def clean_natural_response(raw_response: str) -> str:
    """Extract pure natural reasoning, removing all system prompt echoes"""
    response = raw_response.strip()
    
    # Remove all system prompt artifacts that she's echoing back
    system_artifacts_to_remove = [
        "You are Beta, an intelligent VRChat explorer",
        "EXPLORATION:",
        "RECENT ACTIONS:",
        "CURRENT SITUATION:",
        "You see:",
        "Objects:",
        "People:",
        "Pathways:",
        "CURIOSITY LEVELS:",
        "CURIOSITY:",
        "RECENTLY INVESTIGATED:",
        "MEMORY:",
        "COOLDOWNS:",
        "EXPLORATION STRATEGY:",
        "Think naturally about",
        "What do you want to do right now?",
        "Focus on novel objects",
        "Move toward what genuinely",
        "What catches your attention",
        "- If you see something HIGHLY INTRIGUING",
        "- If nothing looks particularly interesting",
        "- Only move forward when",
        "- When things become boring"
    ]
    
    # Aggressively remove prompt echoes
    lines = response.split('\n')
    clean_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Skip lines that are clearly prompt echoes
        skip_line = False
        for artifact in system_artifacts_to_remove:
            if artifact.lower() in line.lower():
                skip_line = True
                break
        
        # Skip structured format lines
        if any(pattern in line for pattern in [
            "Objects:", "People:", "Pathways:", 
            "CURIOSITY", "EXPLORATION", "STRATEGY:",
            "(", ")", "[", "]", "Choose from", "Pick one"
        ]):
            skip_line = True
        
        if not skip_line:
            clean_lines.append(line)
    
    # Find her actual natural reasoning (usually starts with "Okay", "I", "Let me", etc.)
    natural_thoughts = []
    for line in clean_lines:
        # Accept lines that sound natural and conversational
        if any(starter in line.lower() for starter in [
            "i see", "i notice", "looking at", "there's", "this looks", 
            "i'm curious", "i want to", "let me", "i think", "seems like",
            "okay", "well", "hmm", "oh", "interesting", "now", "from here"
        ]):
            natural_thoughts.append(line)
    
    # If we found natural thoughts, use those
    if natural_thoughts:
        return ' '.join(natural_thoughts)
    
    # Otherwise, use the first clean line that's substantial
    for line in clean_lines:
        if len(line) > 10:  # Skip very short lines
            return line
    
    # Last resort: return the original response cleaned of obvious artifacts
    cleaned = response
    for artifact in system_artifacts_to_remove:
        cleaned = cleaned.replace(artifact, "")
    
    return cleaned.strip()

# Contract tests
def _contract_tests():
    # Safety veto always works
    dangerous_text = "I see a wall too close, blocked"
    result = extract_intent_sidecar(dangerous_text)
    assert "look left" in result
    
    # Look intents work
    look_text = "I want to look right to see more"
    result = extract_intent_sidecar(look_text)
    assert "look right" in result
    
    # Movement requires CENTER targets
    move_text = "I want to move forward"
    result_no_center = extract_intent_sidecar(move_text, "")
    assert "look" in result_no_center  # Should fallback to scan
    
    result_with_center = extract_intent_sidecar(move_text, "CENTER HIGHLY INTRIGUING object")
    assert "move forward" in result_with_center
    
    # Default fallback works
    unclear_text = "This is confusing"
    result = extract_intent_sidecar(unclear_text)
    assert "look left" in result
    
    # Clean response removes artifacts
    messy_response = "You are Beta, an intelligent VRChat explorer. EXPLORATION: I see something interesting here."
    cleaned = clean_natural_response(messy_response)
    assert "You are Beta" not in cleaned
    assert "EXPLORATION:" not in cleaned

if __name__ == "__main__":
    _contract_tests()
    print("All intent sidecar contract tests passed")