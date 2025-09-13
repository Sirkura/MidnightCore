#!/usr/bin/env python3
"""
Unified NDJSON Logging System for MidnightCore
Consolidates all logging into machine-readable events.ndjson format
Following Telemetry Overhaul 2.md specifications
"""

import json
import time
import threading
import os
import traceback
import hashlib
import uuid
from datetime import datetime
from pathlib import Path

# Thread safety
_lock = threading.RLock()

# Unified log file path
_log_dir = Path("G:/Experimental/Production/MidnightCore/Core/Engine/Logging")
_events_log = _log_dir / "events.ndjson"

# Session tracking
_session_id = str(uuid.uuid4())[:8]

# Configuration from environment variables
class TracingConfig:
    # Unified logging is ALWAYS active when brain starts
    UNIFIED_LOG = True  
    DEEP_TRACE = os.getenv("DEEP_TRACE", "disabled") == "enabled"
    FILE_MAP = os.getenv("FILE_MAP", "disabled") == "enabled"
    MAX_TICKS = int(os.getenv("MAX_TICKS", "999999"))
    LEGACY_MD = os.getenv("LEGACY_MD", "0") == "1"  # Keep markdown logs if explicitly enabled

def _ensure_log_dir():
    """Ensure logging directory exists"""
    _log_dir.mkdir(parents=True, exist_ok=True)

# Core unified logging function
def log_event(event, **payload):
    """
    Central event emitter for unified NDJSON logging
    All events flow through this single function
    """
    if not TracingConfig.UNIFIED_LOG:
        return
    
    _ensure_log_dir()
    
    # Build the canonical record structure
    record = {
        "ts": time.time(),
        "session_id": _session_id,
        "event": event,
        **{k: v for k, v in payload.items() if v is not None}
    }
    
    line = json.dumps(record, ensure_ascii=False)
    
    with _lock:
        with open(_events_log, "a", encoding="utf-8") as f:
            f.write(line + "\n")

# ============================================================================
# TICK ENGINE EVENTS
# ============================================================================

def log_tick_start(tick_id):
    """Log tick engine start"""
    log_event("tick.start", tick_id=tick_id)

def log_capture_done(tick_id, frame_id, capture_ms, action_window=None, debounced_ms=None):
    """Log capture phase completion"""
    log_event("capture.done", 
              tick_id=tick_id, 
              frame_id=frame_id,
              capture_ms=capture_ms, 
              action_window=action_window, 
              debounced_ms=debounced_ms)

def log_decide_done(tick_id, frame_id, decision_ms, frame_age_ms):
    """Log decision phase completion"""
    log_event("decide.done",
              tick_id=tick_id,
              frame_id=frame_id, 
              decision_ms=decision_ms,
              frame_age_ms=frame_age_ms)

def log_speak_done(tick_id, frame_id, chat_ms, chat_frame_id=None):
    """Log speak phase completion"""
    log_event("speak.done",
              tick_id=tick_id,
              frame_id=frame_id,
              chat_ms=chat_ms,
              chat_frame_id=chat_frame_id)

def log_act_done(tick_id, frame_id, action):
    """Log act phase completion"""
    log_event("act.done",
              tick_id=tick_id,
              frame_id=frame_id,
              action=action)

def log_integrate_done(tick_id, frame_id):
    """Log integrate phase completion"""
    log_event("integrate.done",
              tick_id=tick_id,
              frame_id=frame_id)

# ============================================================================
# VISION EVENTS
# ============================================================================

def log_vision_describe(tick_id, frame_id, scene, objects=None, paths=None, people=None):
    """Log Florence vision analysis results"""
    # Top 5 objects only
    objects_limited = (objects or [])[:5]
    log_event("vision.describe",
              tick_id=tick_id,
              frame_id=frame_id,
              scene=scene,
              objects=objects_limited,
              paths=paths or [],
              people=people or [])

def log_vision_detections(tick_id, frame_id, detections):
    """Log vision detection summary"""
    # Top 3 detections with essential info
    top3 = []
    for det in (detections or [])[:3]:
        det_info = {"label": det.get("label", "")}
        if "bbox" in det:
            det_info["bbox"] = det["bbox"]
        if det.get("ocr"):
            det_info["ocr"] = True
        top3.append(det_info)
    
    log_event("vision.detections",
              tick_id=tick_id,
              frame_id=frame_id,
              count=len(detections or []),
              top3=top3)

# ============================================================================
# INTRIGUE / RANKING EVENTS
# ============================================================================

def log_rank_targets(tick_id, frame_id, ranked_targets):
    """Log target ranking results"""
    top3 = []
    for target in (ranked_targets or [])[:3]:
        # Calculate sector from bbox center
        bbox = target.get("bbox", [0, 0, 1, 1])
        if len(bbox) >= 4:
            cx = 0.5 * (bbox[0] + bbox[2])
            if cx < 0.33:
                sector = "LEFT"
            elif cx > 0.67:
                sector = "RIGHT"
            else:
                sector = "CENTER"
        else:
            sector = "UNKNOWN"
        
        top3.append({
            "label": target.get("label", ""),
            "score": float(target.get("score", 0)),
            "recent": bool(target.get("recent", False)),
            "type_blocked": bool(target.get("type_blocked", False)),
            "sector": sector
        })
    
    log_event("rank.targets",
              tick_id=tick_id,
              frame_id=frame_id,
              top3=top3)

# ============================================================================
# HYBRID NAVIGATION EVENTS
# ============================================================================

def log_hybrid_nav_start(tick_id, frame_id, vision_state_available=False, image_available=False):
    """Log start of hybrid navigation analysis"""
    log_event("hybrid_nav.start",
              tick_id=tick_id,
              frame_id=frame_id,
              vision_state_available=vision_state_available,
              image_available=image_available)

def log_depth_analysis_tier1(tick_id, frame_id, analysis_method, threats_detected, measurements, needs_florence=False):
    """Log Tier 1 depth analysis results"""
    log_event("hybrid_nav.depth.tier1",
              tick_id=tick_id,
              frame_id=frame_id,
              analysis_method=analysis_method,
              threats_detected=threats_detected,
              front_m=measurements.get('front_m'),
              left_m=measurements.get('left_m'), 
              right_m=measurements.get('right_m'),
              edge_risk=measurements.get('edge_risk'),
              needs_florence=needs_florence,
              temporal_buffers_ready=measurements.get('temporal_buffers_ready', True))

def log_florence_verification_tier2(tick_id, frame_id, verification_available, florence_latency_ms, threats_verified, threats_demoted):
    """Log Tier 2 Florence semantic verification results"""
    log_event("hybrid_nav.florence.tier2",
              tick_id=tick_id,
              frame_id=frame_id,
              verification_available=verification_available,
              florence_latency_ms=florence_latency_ms,
              threats_verified=threats_verified,
              threats_demoted=threats_demoted)

def log_navigation_decision(tick_id, frame_id, decision, movement_allowed, safe_actions, reasoning, priority, florence_used=False):
    """Log final hybrid navigation decision"""
    log_event("hybrid_nav.decision",
              tick_id=tick_id,
              frame_id=frame_id,
              decision=decision,
              movement_allowed=movement_allowed,
              safe_actions=safe_actions,
              reasoning=reasoning,
              priority=priority,
              florence_used=florence_used)

def log_movement_safety_check(tick_id, frame_id, action_text, is_safe, modified_action=None, safety_reasoning=None):
    """Log movement safety check results"""
    log_event("hybrid_nav.safety_check",
              tick_id=tick_id,
              frame_id=frame_id,
              original_action=action_text,
              is_safe=is_safe,
              modified_action=modified_action,
              safety_reasoning=safety_reasoning)

def log_temporal_smoothing_buffer(tick_id, frame_id, region_name, raw_value, smoothed_value, buffer_size):
    """Log temporal smoothing buffer updates for debugging"""
    if TracingConfig.DEEP_TRACE:  # Only log in deep trace mode to avoid spam
        log_event("hybrid_nav.temporal_buffer",
                  tick_id=tick_id,
                  frame_id=frame_id,
                  region=region_name,
                  raw_value=raw_value,
                  smoothed_value=smoothed_value,
                  buffer_size=buffer_size)

def log_regional_percentiles(tick_id, frame_id, region_name, stats):
    """Log regional percentile statistics for debugging"""
    if TracingConfig.DEEP_TRACE:  # Only log in deep trace mode to avoid spam
        log_event("hybrid_nav.regional_stats",
                  tick_id=tick_id,
                  frame_id=frame_id,
                  region=region_name,
                  p10=stats['p10'],
                  p25=stats['p25'],
                  p50=stats['p50'],
                  p75=stats['p75'],
                  p90=stats['p90'],
                  iqr=stats['iqr'],
                  std=stats['std'])

# ============================================================================
# LLM EVENTS
# ============================================================================

def log_llm_request(tick_id, frame_id, mode, tokens_max, temp, top_p, prompt_text):
    """Log LLM request with hashed prompt"""
    prompt_hash = hashlib.md5(prompt_text.encode()).hexdigest()[:10]
    prompt_preview = prompt_text[:200].replace("\n", " ")
    
    log_event("llm.request",
              tick_id=tick_id,
              frame_id=frame_id,
              mode=mode,
              tokens_max=tokens_max,
              temp=temp,
              top_p=top_p,
              prompt_hash=prompt_hash,
              prompt_preview=prompt_preview)

def log_llm_response(tick_id, frame_id, tokens_used, latency_ms, ended_early=False, cap_hit=False, tier="local", text_preview=""):
    """Log LLM response with timing"""
    text_preview_clean = text_preview[:200].replace("\n", " ")
    
    log_event("llm.response",
              tick_id=tick_id,
              frame_id=frame_id,
              tokens_used=tokens_used,
              latency_ms=latency_ms,
              ended_early=ended_early,
              cap_hit=cap_hit,
              tier=tier,
              text_preview=text_preview_clean)

# ============================================================================
# INTENT SIDECAR EVENTS
# ============================================================================

def log_intent_sidecar_input(tick_id, frame_id, spatial_context, natural_preview):
    """Log sidecar input parameters"""
    spatial_preview = spatial_context[:200] if spatial_context else ""
    natural_clean = natural_preview[:200].replace("\n", " ") if natural_preview else ""
    
    log_event("intent.sidecar.input",
              tick_id=tick_id,
              frame_id=frame_id,
              spatial_context=spatial_preview,
              natural_preview=natural_clean)

def log_intent_sidecar_output(tick_id, frame_id, action, rule):
    """Log sidecar output decision with rule tag"""
    log_event("intent.sidecar.output",
              tick_id=tick_id,
              frame_id=frame_id,
              action=action,
              rule=rule)

# ============================================================================
# OSC EVENTS
# ============================================================================

def log_osc_send(tick_id, frame_id, action, success, latency_ms, args=None):
    """Log OSC command transmission"""
    log_event("osc.send",
              tick_id=tick_id,
              frame_id=frame_id,
              action=action,
              args=args,
              success=bool(success),
              latency_ms=latency_ms)

# ============================================================================
# GPU AND PERFORMANCE EVENTS
# ============================================================================

def log_gpu_sample(util_pct, vram_used_mb, temp_c):
    """Log GPU performance sample"""
    log_event("gpu.sample",
              util_pct=util_pct,
              vram_used_mb=vram_used_mb,
              temp_c=temp_c)

def log_perf_block(name, latency_ms):
    """Log performance timing block"""
    log_event("perf.block",
              name=name,
              latency_ms=latency_ms)

# ============================================================================
# CHAT AND TELEMETRY EVENTS
# ============================================================================

def log_chat_say(tick_id, frame_id, text):
    """Log chat message output"""
    log_event("chat.say",
              tick_id=tick_id,
              frame_id=frame_id,
              text=text)

def log_tick_telemetry(tick_id, frame_id, progress_score, bad_ticks, fl_stage, action, utterance):
    """Log tick-level telemetry summary"""
    utterance_clean = utterance[:120] if utterance else ""
    
    log_event("tick.telemetry",
              tick_id=tick_id,
              frame_id=frame_id,
              progress_score=progress_score,
              bad_ticks=bad_ticks,
              fl_stage=fl_stage,
              action=action,
              utterance=utterance_clean)

# ============================================================================
# BACKWARD COMPATIBILITY FUNCTIONS
# ============================================================================

def log_engine(event, **payload):
    """Backward compatibility: route to unified log"""
    log_event(f"legacy.{event}", **payload)

def log_deep(event, **payload):
    """Backward compatibility: route to unified log with debug flag"""
    if TracingConfig.DEEP_TRACE:
        log_event(f"debug.{event}", **payload)

def log_filemap(file_path, **payload):
    """Backward compatibility: route to unified log"""
    if TracingConfig.FILE_MAP:
        log_event("file.access", file=file_path, **payload)

def initialize_logging():
    """Initialize unified logging system - called when brain starts"""
    _ensure_log_dir()
    
    # Log system initialization with session tracking
    log_event("brain.init.start", 
              session_id=_session_id,
              unified_log=TracingConfig.UNIFIED_LOG,
              deep_trace=TracingConfig.DEEP_TRACE,
              file_map=TracingConfig.FILE_MAP,
              legacy_md=TracingConfig.LEGACY_MD,
              max_ticks=TracingConfig.MAX_TICKS)
    
    print(f"[LOGGING] Unified NDJSON system initialized:")
    print(f"[LOGGING] - Session ID: {_session_id}")
    print(f"[LOGGING] - Unified logging: {'ACTIVE' if TracingConfig.UNIFIED_LOG else 'DISABLED'}")
    print(f"[LOGGING] - Deep trace: {'ACTIVE' if TracingConfig.DEEP_TRACE else 'DISABLED'}")
    print(f"[LOGGING] - File map: {'ACTIVE' if TracingConfig.FILE_MAP else 'DISABLED'}")
    print(f"[LOGGING] - Legacy MD: {'ACTIVE' if TracingConfig.LEGACY_MD else 'DISABLED'}")
    print(f"[LOGGING] - Output file: {_events_log}")

# Integration with existing performance monitor
def log_engine_with_timing(category, operation, duration_ms, **extra_payload):
    """Integrate with existing performance_monitor.py"""
    try:
        # Try to use existing performance monitor
        import sys
        import os
        sys.path.append(os.path.join(os.path.dirname(__file__)))
        from performance_monitor import log_timing
        log_timing(category, operation, duration_ms)
    except ImportError:
        pass  # performance_monitor not available
    
    # Log as performance block
    log_perf_block(f"{category}.{operation}", duration_ms)

# Enhanced function tracing with unified logging
def trace_function(func):
    """Decorator for automatic function tracing"""
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__name__}"
        
        # Use unified logging for function tracing
        log_event("function.enter", name=func_name)
        if TracingConfig.DEEP_TRACE:
            log_event("debug.function.enter", 
                     name=func_name, 
                     args=str(args)[:200], 
                     kwargs=str(kwargs)[:200])
        
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration_ms = (time.time() - start_time) * 1000
            
            log_event("function.exit", 
                     name=func_name, 
                     duration_ms=duration_ms, 
                     success=True)
            if TracingConfig.DEEP_TRACE:
                log_event("debug.function.exit", 
                         name=func_name, 
                         duration_ms=duration_ms, 
                         result_type=type(result).__name__)
            
            return result
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            log_event("function.error", 
                     name=func_name, 
                     duration_ms=duration_ms, 
                     error=str(e)[:200])
            if TracingConfig.DEEP_TRACE:
                log_event("debug.function.error", 
                         name=func_name, 
                         duration_ms=duration_ms, 
                         error=str(e), 
                         traceback=traceback.format_exc()[:500])
            raise
    
    return wrapper

# ============================================================================
# LOG ROTATION AND HYGIENE
# ============================================================================

def rotate_log_if_needed():
    """Rotate events.ndjson if it gets too large (100MB) or daily"""
    try:
        if not _events_log.exists():
            return
        
        file_size = _events_log.stat().st_size
        max_size = 100 * 1024 * 1024  # 100MB
        
        if file_size > max_size:
            # Create timestamped backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = _log_dir / f"events_{timestamp}.ndjson.gz"
            
            # Compress and move
            import gzip
            with open(_events_log, 'rb') as f_in:
                with gzip.open(backup_path, 'wb') as f_out:
                    f_out.writelines(f_in)
            
            # Clear the original file
            _events_log.unlink()
            
            log_event("log.rotated", 
                     old_size_mb=file_size / 1024 / 1024,
                     backup_file=str(backup_path))
            
    except Exception as e:
        # Don't let log rotation break the system
        print(f"[LOGGING] Warning: Log rotation failed: {e}")

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_session_id():
    """Get current session ID for external correlation"""
    return _session_id

def get_events_log_path():
    """Get path to unified events log"""
    return str(_events_log)

if __name__ == "__main__":
    # Test the unified logging system
    initialize_logging()
    
    # Test basic events
    log_event("test.basic", message="Basic unified logging test")
    
    # Test tick engine events
    log_tick_start(1)
    log_capture_done(1, 101, 250.5, action_window=True)
    log_decide_done(1, 101, 1500.2, 1800.7)
    log_speak_done(1, 101, 85.3)
    log_act_done(1, 101, "move forward")
    log_integrate_done(1, 101)
    
    # Test vision events
    log_vision_describe(1, 101, "office room", 
                       objects=[{"label": "door", "cx": 0.5, "cy": 0.6, "area": 0.2}],
                       paths=["forward"], people=[])
    log_vision_detections(1, 101, [
        {"label": "door", "bbox": [0.4, 0.3, 0.6, 0.8], "ocr": False},
        {"label": "poster", "bbox": [0.7, 0.1, 0.9, 0.4], "ocr": True}
    ])
    
    # Test LLM events
    log_llm_request(1, 101, "decision", 200, 0.7, 0.9, "You see a door ahead. What do you do?")
    log_llm_response(1, 101, 95, 450.2, ended_early=False, tier="local", 
                    text_preview="I want to explore the door")
    
    # Test sidecar events
    log_intent_sidecar_input(1, 101, "door CENTER HIGHLY INTRIGUING", "I want to explore")
    log_intent_sidecar_output(1, 101, "move forward", "forward")
    
    # Test OSC and performance
    log_osc_send(1, 101, "move forward", True, 15.2)
    log_gpu_sample(65, 4096, 72)
    log_perf_block("vision", 320.5)
    
    # Test backward compatibility
    log_engine("test.legacy", message="Legacy compatibility test")
    
    print(f"Unified logging test complete - check {_events_log}")