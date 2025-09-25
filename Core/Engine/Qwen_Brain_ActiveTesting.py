#!/usr/bin/env python3
"""
Natural Qwen VRChat Brain - Threaded Architecture for Reliability
Revolutionary natural decisions with battle-tested reliability patterns
"""

import subprocess
import json
import time
import threading
import hashlib
import traceback
from collections import defaultdict, deque, Counter
import re
import random
import queue
import uuid
from datetime import datetime
from pathlib import Path
from pythonosc import udp_client, dispatcher, osc_server

# Use relative imports for portability
from ..Modules.vision.florence_analyzer import get_florence_analyzer, analyze_screenshot
from .cropped_capture import QwenVRChatCapture
from ..Common.Tools.performance_monitor import get_gpu_monitor, log_timing

# Import enhanced logging system
from ..Common.Tools.logging_bus import (
    log_engine, log_deep, log_filemap, initialize_logging, TracingConfig, trace_function,
    log_llm_request, log_llm_response,
    log_vision_describe, log_vision_detections, log_rank_targets, log_osc_send,
    log_chat_say, log_tick_telemetry
)

# Import modular components
from .tick_engine import TickEngine
from .intent_sidecar import clean_natural_response
from .schemas import validate_decision_schema, safe_fallback_decision, CalibrationData
from ..Common.Tools.vrchat_log_parser import VRChatLogParser
from ..Modules.map.spatial_state import BetaSpatialState, create_spatial_state_for_world

# Import hybrid navigation system
from .hybrid_navigation_integration import check_movement_safety, integrate_spatial_context

# Import new vision system components
try:
    # Import state bus functions from Engine
    from .state_bus import get_vision_state, get_vision_facts

    # Import vision workers from our migrated locations
    from ..Modules.vision.workers.depth_worker import start_depth_worker, stop_depth_worker
    from ..Modules.vision.workers.event_router import start_event_router, stop_event_router
    from ..Modules.vision.workers.florence_worker import start_florence_worker, stop_florence_worker
    
    VISION_SYSTEM_AVAILABLE = True
    print("SUCCESS: New vision system components loaded")
except ImportError as e:
    VISION_SYSTEM_AVAILABLE = False
    print(f"WARNING: New vision system not available: {e}")

# TickEngine class extracted to tick_engine.py module

class NaturalQwenVRChatBrain:
    def __init__(self):
        """Initialize Natural Qwen VRChat Brain with threaded reliability"""
        print("Initializing Natural Qwen VRChat Brain...")
        
        # Initialize enhanced logging system FIRST
        initialize_logging()
        log_engine("brain.init.start", version="lockstep", 
                  max_ticks=TracingConfig.MAX_TICKS if hasattr(TracingConfig, 'MAX_TICKS') else "unlimited")
        
        # Initialize tick engine for proper sequencing
        self.tick_engine = TickEngine()
        
        # Core systems
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
        self.capture_system = QwenVRChatCapture()
        self.florence_analyzer = get_florence_analyzer()
        
        # Initialize GPU monitoring for stability verification
        self.gpu_monitor = get_gpu_monitor()
        self.gpu_monitor.start_monitoring()
        print("GPU monitoring started - tracking VRAM and performance telemetry")
        
        # Initialize OSC controller and world detection (skip broken auto-calibration)
        from .osc_controller import QwenVRChatOSC
        self.osc_controller = QwenVRChatOSC()
        self.world_detector = VRChatLogParser()
        print("World detection system initialized - using VRChat log parser")
        
        # Initialize spatial state system
        self.beta_spatial_state = None  # Will be initialized when world is detected
        print("Spatial state system ready - will initialize on world detection")
        
        # Qwen 3 8B configuration (from proper documentation)
        self.llama_path = "G:/Experimental/llama.cpp/build/bin/Release/llama-cli.exe"
        self.model_path = "G:/Experimental/Production/MidnightCore/Core/models/llm_gguf/Qwen3-8B-Q4_K_M.gguf"
        self.qwen_config = {
            "ngl": 25,
            "ctx": 4096,         # Increased context window
            "temp": 0.7,
            "top_p": 0.9,
            "repeat_penalty": 1.1,
            "max_tokens": 200,   # Increased to match decision mode (was 120)
            "timeout_s": 45
        }
        
        # Model family detection for chat templates
        self.model_family = "qwen" if "qwen" in self.model_path.lower() else \
                           ("mistral" if "mistral" in self.model_path.lower() else "generic")
        
        # Thread timing configuration (no rate limits)
        self.vision_fast_hz = 2.0   # 0.5s quick pass
        self.vision_slow_hz = 0.5   # 2.0s full pass  
        self.decision_hz = 0.5      # 2.0s per decision
        self.watchdog_idle_s = 4.0  # if no action executed in 4s, do keepalive
        
        # OSC calibrations from documentation
        self.look_button_90s = 0.25     # 0.25s = 90°
        self.look_axis_vel = 0.25       # Vertical axis velocity
        self.look_axis_deg_per_s = 45.0 # 45°/s for vertical
        
        # Threaded architecture components
        self._llm_lock = threading.Lock()
        self._vision_q = queue.Queue(maxsize=1)  # Always keep latest only
        self._action_q = queue.Queue(maxsize=4)
        self._last_action_ts = time.time()
        
        # Safe actions for fallback
        self._safe_actions = [
            "I want to look right to scan the area.",
            "I want to look left to check my surroundings.",
            "I want to move forward to explore a few steps.", 
            "I want to strafe left to avoid obstacles.",
            "I want to strafe right to navigate around objects.",
            "I want to turn right toward something interesting.",
            "I want to turn left toward something interesting.",
            "I want to back up to get more clearance.",
            "I want to say Hello there!"
        ]
        
        # Natural AI state
        self.consciousness_active = True
        self.emergency_stop = False
        self.environment_memory = {}
        self.recent_actions = []
        
        # Enhanced look position and target tracking
        self.look_state = {
            "horizontal_offset": 0.0,
            "vertical_offset": 0.0,
            "needs_centering": False,
            "target_of_interest": None,
            "target_angle": 0.0,
            "target_description": "",
            "investigating_target": False
        }
        
        # Action history and reasoning memory
        self.action_history = []  # List of (timestamp, action, reasoning, scene_context)
        self.max_history_length = 8  # Keep last 8 actions for context
        self.recent_scenes = []  # Track scene changes for spatial reasoning
        self.exploration_goals = []  # Stack of current exploration objectives
        
        # LLM failure retry system
        self.consecutive_llm_failures = 0
        self.max_llm_failures = 3
        self.llm_failure_pause = False

        # OSC client subprocess management
        self.osc_client_process = None
        self.osc_client_running = False
        
        # Proportional target facing system (from suggestion)
        self.hfov_deg = 90.0          # Horizontal field of view
        self._yaw_err_ema = 0.0       # Exponential moving average of yaw error
        self._spin_guard_deg = 0.0    # Track total rotation to prevent endless spinning
        self._last_target_bbox = None # Store bounding box of current target
        
        # Token budgets and chat system
        self.chat_char_cap = 144
        self.chat_split_pause_s = 5.0  # seconds between overflow messages
        
        # Per-mode token budgets - optimized for lockstep synchronization
        self.n_by_mode = {
            "decision": 200,   # increased for better spatial reasoning (was 120)
            "chat": 320,       # reduced for faster, more focused responses (was 512)  
            "fallback": 150    # improved fallback responses (increased from 100)
        }
        
        # Target memory and cooldown system (from Target Updates.md)
        self.targets = {}  # tid -> {"last_seen": t, "seen": int, "last_area": float,
                          #        "inst_cooldown_s": 20, "notes": str}
        self.type_cooldown = {"person": 0.0, "poster": 0.0, "door": 0.0}
        self.type_cooldown_default = {"person": 15.0, "poster": 12.0, "door": 10.0}
        
        # Remove movement cooldown system - using target-based boredom/intrigue instead
        
        # Investigation micro-FSM
        self.invest = {"tid": None, "phase": "idle", "ticks": 0}
        
        # Spatial Memory System
        self.spatial_memory = {
            "visited_sectors": set(),  # Track 45-degree sectors that have been explored
            "current_sector": 0,       # Current facing direction (0-360 degrees)
            "movement_history": deque(maxlen=10),  # Recent movement commands for pattern detection
            "exploration_goals": [],   # Priority list of unexplored areas
            "landmark_map": {},        # Memorable objects and their approximate sectors
            "last_position_check": time.time()
        }
        
        # Telemetry tracking
        self.telemetry = {
            "bad_ticks": 0,
            "progress_score": 0.0,
            "fl_stage": "unknown",
            "last_executed_action": "none",
            "last_utterance": ""
        }
        
        # Explain status buffers (from Explain Command.txt)
        self._explain_last = deque(maxlen=20)   # (ts, canonical_action)
        self._last_vision = {"objects": [], "paths": [], "scene": ""}
        self._last_llm_meta = {"tier": "n/a", "lat": 0.0}
        
        # Logging paths (preserved)
        # All log files as .md in the Logs directory
        self.log_dir = Path("G:/Experimental/Production/MidnightCore/Core/Engine/Logging")
        self.chat_log_path = "G:/Experimental/Production/MidnightCore/Core/Engine/Logging/Chat-Log.md"
        self.florence_call_log_path = "G:/Experimental/Production/MidnightCore/Core/Engine/Logging/Florence-Log.md"
        self.fvision_log_path = "G:/Experimental/Production/MidnightCore/Core/Engine/Logging/FVision-Log.md"
        self.osc_log_path = "G:/Experimental/Production/MidnightCore/Core/Engine/Logging/OSC-Log.md"
        self.qwen_log_path = "G:/Experimental/Production/MidnightCore/Core/Engine/Logging/Qwen-Log.md"
        self.telemetry_decision_log_path = "G:/Experimental/Production/MidnightCore/Core/Engine/Logging/Telemetry-Decision.md"
        self.telemetry_tick_log_path = "G:/Experimental/Production/MidnightCore/Core/Engine/Logging/Telemetry-Tick.md"
        
        # Ensure log directory exists and initialize log files
        self.log_dir.mkdir(exist_ok=True)
        self._initialize_log_files()
        
        # Initialize systems
        self.setup_vrchat_receiver()
        self.start_vision_system()
        
        # Engine mode selection - prevent dual brain conflict
        import os
        if os.getenv("QWEN_ENGINE", "lockstep") == "threaded":
            self.start_threaded_consciousness()   # legacy
            print("Natural Qwen VRChat Brain initialized!")
            print("Architecture: Threaded (Vision/Decision/Executor)")
            print("Emergency Stop: Type 'qwen stop' in VRChat chat")
        else:
            print("Skipping legacy threads (lockstep mode).")
            print("Natural Qwen VRChat Brain initialized!")
            print("Architecture: Lockstep Tick Engine")
            print("Emergency Stop: Type 'qwen stop' in VRChat chat")
    
    def start_vision_system(self):
        """Initialize the new fast vision system"""
        if not VISION_SYSTEM_AVAILABLE:
            print("WARNING: Depth vision system not available - using legacy Florence only")
            return
        
        try:
            print("STARTING: Enhanced vision system...")
            
            # Start depth worker with unified capture pipeline
            if start_depth_worker(
                get_frame_rgb=self.capture_system.grab_rgb_frame,
                capture_mode="screen", 
                depth_fps=3.0, 
                target_fps=15.0
            ):
                print("SUCCESS: Depth worker started with unified capture (3 Hz depth, 15 Hz flow)")
            else:
                print("FAILED: Failed to start depth worker")
                return
            
            # TEMPORARILY DISABLED: Event router and Florence worker to reduce VRAM usage
            print("SKIPPED: Event router and Florence worker (VRAM optimization)")
            # if start_event_router():
            #     print("SUCCESS: Event router started")
            # else:
            #     print("FAILED: Failed to start event router")
            #     return
            # 
            # if start_florence_worker():
            #     print("SUCCESS: Florence worker started (ROI + escalation)")
            # else:
            #     print("FAILED: Failed to start Florence worker")
            #     return
            
            print("ACTIVE: Enhanced vision system - expecting 7-10x performance improvement")
            
        except Exception as e:
            error_msg = f"ERROR: Vision system initialization failed: {e}"
            print(error_msg)
            log_engine("vision.init.error", error=str(e), traceback=traceback.format_exc()[:500])
            log_filemap(__file__, event="error_in_init", error_type="vision_init_failed")
    
    def stop_vision_system(self):
        """Stop the vision system"""
        if not VISION_SYSTEM_AVAILABLE:
            return
        
        try:
            print("STOPPING: Vision system...")
            stop_florence_worker()
            stop_event_router() 
            stop_depth_worker()
            print("SUCCESS: Vision system stopped")
        except Exception as e:
            print(f"Warning: Vision system stop error: {e}")
    
    def _n_for_mode(self, mode: str) -> int:
        """Get token budget for specific mode"""
        return self.n_by_mode.get(mode, self.n_by_mode["decision"])
    
    def _tick_type_cooldowns(self, dt: float):
        """Decay type cooldowns over time"""
        for k in list(self.type_cooldown.keys()):
            self.type_cooldown[k] = max(0.0, self.type_cooldown[k] - dt)
    
    def _initialize_log_files(self):
        """Initialize all log files with proper headers if they don't exist"""
        log_files = {
            self.chat_log_path: self._create_chat_log_header,
            self.florence_call_log_path: self._create_florence_log_header, 
            self.fvision_log_path: self._create_fvision_log_header,
            self.osc_log_path: self._create_osc_log_header,
            self.qwen_log_path: self._create_qwen_log_header,
            self.telemetry_decision_log_path: self._create_telemetry_decision_log_header,
            self.telemetry_tick_log_path: self._create_telemetry_tick_log_header,
        }
        
        for log_path, header_func in log_files.items():
            if not Path(log_path).exists():
                try:
                    with open(log_path, "w") as f:
                        f.write(header_func())
                    print(f"Created log file: {log_path}")
                except Exception as e:
                    print(f"Failed to create log file {log_path}: {e}")
    
    def _log_decision_telemetry(self, tokens_used, ended_early, cap_hit, tier, latency, retried):
        """Log detailed telemetry for each decision"""
        try:
            with open(self.telemetry_decision_log_path, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} | DECISION | tokens:{tokens_used} | early:{ended_early} | cap:{cap_hit} | tier:{tier} | latency:{latency:.2f}s | retry:{retried}\n")
        except Exception as e:
            print(f"Failed to write decision telemetry: {e}")
    
    def _log_tick_telemetry(self, progress_score, bad_ticks, fl_stage, executed_action, utterance):
        """Log detailed telemetry for each tick"""
        self.telemetry["progress_score"] = progress_score
        self.telemetry["bad_ticks"] = bad_ticks  
        self.telemetry["fl_stage"] = fl_stage
        self.telemetry["last_executed_action"] = executed_action
        self.telemetry["last_utterance"] = utterance
        
        try:
            with open(self.telemetry_tick_log_path, "a") as f:
                f.write(f"{time.strftime('%H:%M:%S')} | TICK | progress:{progress_score:.2f} | bad_ticks:{bad_ticks} | fl_stage:{fl_stage} | action:{executed_action} | utterance:'{utterance[:50]}...'\n")
        except Exception as e:
            print(f"Failed to write tick telemetry: {e}")
    
    def _calculate_progress_score(self, analysis, targets):
        """Calculate a progress score based on exploration activity"""
        score = 0.0
        
        # Base score for successful vision analysis
        if analysis:
            score += 10.0
            
        # Bonus for finding new targets
        if targets:
            novel_targets = sum(1 for _, tid, _, recent, _ in targets if not recent)
            score += novel_targets * 5.0
            
        # Penalty for repeated failed attempts
        score -= self.telemetry["bad_ticks"] * 2.0
        
        return max(0.0, min(100.0, score))  # Clamp between 0-100
    
    def _create_chat_log_header(self):
        """Create header for Chat-Log.md"""
        return f"""# Natural Qwen VRChat Chat Log
**Session Started:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Purpose:** All chat messages from Natural Qwen

---

"""

    def _create_florence_log_header(self):
        """Create header for Florence-Log.md"""
        return f"""# Florence-2 Analysis Log
**Session Started:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Purpose:** Florence-2 vision analysis results

---

"""

    def _create_fvision_log_header(self):
        """Create header for FVision-Log.md"""
        return f"""# Florence Vision Processing Log
**Session Started:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Purpose:** Florence vision pipeline processing details

---

"""

    def _create_osc_log_header(self):
        """Create header for OSC-Log.md"""
        return f"""# Natural Qwen OSC Command Log
**Session Started:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Purpose:** All OSC commands executed by Natural Qwen

---

"""

    def _create_qwen_log_header(self):
        """Create header for Qwen-Log.md"""
        return f"""# Qwen Decision Log
**Session Started:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Purpose:** All Qwen LLM decisions, thoughts, and reasoning

---

"""

    def _create_telemetry_decision_log_header(self):
        """Create header for Telemetry-Decision.md"""
        return f"""# Telemetry - Decision Level
**Session Started:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Purpose:** Per-decision telemetry: tokens, early stop, cap hit, tier, latency

---

"""

    def _create_telemetry_tick_log_header(self):
        """Create header for Telemetry-Tick.md"""
        return f"""# Telemetry - Tick Level
**Session Started:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Purpose:** Per-tick telemetry: progress score, bad ticks, florence stage, action, utterance

---

"""
    
    
    def _target_id(self, label: str, bbox: list, ocr: str = None) -> str:
        """Generate stable target ID from label, bbox, and optional OCR"""
        # coarse center & area make IDs stable across small jitters
        cx = int(((bbox[0]+bbox[2])*0.5)*20)
        cy = int(((bbox[1]+bbox[3])*0.5)*12)
        area = int(((bbox[2]-bbox[0])*(bbox[3]-bbox[1]))*100)
        key = f"{label}|{ocr or ''}|{cx},{cy}|{area}"
        return hashlib.md5(key.encode()).hexdigest()[:10]
    
    def _rank_targets(self, detections):
        """
        Enhanced reward-based target ranking system
        detections: list of {"label": "person|poster|door|...", "bbox": [x1,y1,x2,y2], "ocr": Optional[str]}
        returns: list[(score, tid, det, recent, type_blocked)] sorted desc by score
        """
        now = time.time()
        ranked = []
        for d in detections:
            tid = self._target_id(d["label"], d["bbox"], d.get("ocr"))
            seen = self.targets.get(tid)
            recent = bool(seen and now - seen["last_seen"] < seen.get("inst_cooldown_s", 20))
            type_blocked = self.type_cooldown.get(d["label"], 0.0) > 0.0

            area = (d["bbox"][2]-d["bbox"][0]) * (d["bbox"][3]-d["bbox"][1])
            cx_off = abs(((d["bbox"][0]+d["bbox"][2])*0.5) - 0.5)  # central bias
            
            # Enhanced reward-based scoring
            base_score = 0
            
            # MASSIVE novelty bonus for completely new targets
            if not seen:
                base_score += 15  # Big intrigue bonus for never-seen targets
            elif not recent and not type_blocked:
                base_score += 8   # Medium interest for targets that have "refreshed"
            
            # Salience based on size and type
            type_multiplier = {
                "person": 1.5,    # People are inherently interesting
                "door": 1.3,      # Doors lead to new areas
                "stairs": 1.4,    # Stairs are pathways to explore
                "window": 0.8,    # Windows are less interactive
                "poster": 0.9     # Posters are moderate interest
            }.get(d["label"], 1.0)
            
            salience = area * 4 * type_multiplier
            base_score += salience
            
            # Centrality bonus (things in center of view are easier to approach)
            centrality_bonus = 2 * (1 - cx_off)  # Higher bonus for centered objects
            base_score += centrality_bonus
            
            # Boredom penalty for recently investigated targets
            if recent:
                base_score *= 0.1  # Dramatic reduction for boring targets
            elif type_blocked:
                base_score *= 0.3  # Moderate reduction for type-blocked targets
            
            # Curiosity bonus for targets with interesting text/descriptions
            if d.get("ocr") and len(d["ocr"]) > 5:
                base_score += 2  # Bonus for targets with readable text
            
            ranked.append((base_score, tid, d, recent, type_blocked))

        ranked.sort(reverse=True, key=lambda r: r[0])
        return ranked
    
    def _bbox_area(self, bbox):
        """Calculate bounding box area"""
        return (bbox[2]-bbox[0]) * (bbox[3]-bbox[1])

    def _bbox_area_grew(self, tid, bbox_now):
        """Check if target bbox grew (indicating we're getting closer)"""
        area_now = self._bbox_area(bbox_now)
        last = self.targets.get(tid, {}).get("last_area")
        self.targets.setdefault(tid, {})["last_area"] = area_now
        return last is not None and area_now > last * 1.08  # >8% bigger -> closer
    
    def _investigate_step(self, tid, det):
        """
        Investigation micro-FSM step
        det: {"label", "bbox", "ocr"?}
        returns: a canonical action string to execute this tick
        """
        self.invest["tid"] = tid
        self.invest["ticks"] += 1
        phase = self.invest.get("phase", "idle")

        if phase == "idle":
            self.invest["phase"] = "acquire"; self.invest["ticks"] = 0
            return "I want to face target."

        if phase == "acquire":
            self.invest["phase"] = "center"
            return "I want to face target."

        if phase == "center":
            state = self.face_target_step(det["bbox"])  # "turning" | "aligned"
            if state == "aligned":
                self.invest["phase"] = "approach"; self.invest["ticks"] = 0
            elif self.invest["ticks"] > 6:  # give up
                return self._finish_investigation(tid, det["label"], moved=False)
            return "I want to face target."

        if phase == "approach":
            if self._bbox_area_grew(tid, det["bbox"]):
                self.invest["phase"] = "inspect"; self.invest["ticks"] = 0
                return "I want to approach target."
            if self.invest["ticks"] <= 4:
                return "I want to approach target."
            return self._finish_investigation(tid, det["label"], moved=False)

        if phase == "inspect":
            if self.invest["ticks"] <= 3:
                return "I want to peek right."
            return self._finish_investigation(tid, det["label"], moved=True)

    def _build_target_memory_context(self, ranked):
        """Build context string about target investigation history with reward/intrigue levels"""
        if not ranked:
            return "CURIOSITY: No notable objects detected - look around for something interesting!\n"
        
        context = "CURIOSITY LEVELS:\n"
        for score, tid, det, recent, type_blocked in ranked[:4]:  # Top 4 targets
            # Convert score to intrigue description
            if score >= 15:
                intrigue = "[HIGHLY INTRIGUING]"
            elif score >= 10:
                intrigue = "[Very interesting]"  
            elif score >= 5:
                intrigue = "[Moderately interesting]"
            elif score >= 2:
                intrigue = "[Somewhat boring]"
            else:
                intrigue = "[Very boring]"
            
            # Add status context
            status_notes = []
            if recent:
                status_notes.append("just investigated")
            elif type_blocked:
                status_notes.append("type recently explored")
            elif not self.targets.get(tid):
                status_notes.append("never explored!")
            
            status_str = f" ({', '.join(status_notes)})" if status_notes else ""
            context += f"- {det['ocr']}: {intrigue} [{det['label']}]{status_str}\n"
        
        return context + "\n"
    
    def _build_cooldown_context(self):
        """Build context string about what types are on cooldown"""
        active_cooldowns = [(label, time_left) for label, time_left in self.type_cooldown.items() if time_left > 0]
        if not active_cooldowns:
            return ""
        
        cooldown_strs = [f"{label} ({time_left:.1f}s)" for label, time_left in active_cooldowns]
        return f"RECENTLY INVESTIGATED: {', '.join(cooldown_strs)}\n"
    
    def _extract_action_from_reasoning(self, reasoning, ranked):
        """Extract executable action from natural language reasoning using LLM"""
        # Update target memory based on what was mentioned
        r = reasoning.lower()
        mentioned_target = None
        for score, tid, det, recent, type_blocked in ranked:
            if det['ocr'].lower() in r or det['label'] in r:
                # Update last_seen time for mentioned target
                now = time.time()
                self.targets.setdefault(tid, {"seen": 0, "inst_cooldown_s": 20})
                self.targets[tid]["last_seen"] = now
                self.targets[tid]["seen"] = self.targets[tid].get("seen", 0) + 1
                mentioned_target = det
                break
        
        # Apply type cooldown if investigating something
        if mentioned_target and any(word in r for word in ["investigate", "explore", "look at", "examine", "face"]):
            label = mentioned_target['label']
            self.type_cooldown[label] = max(
                self.type_cooldown.get(label, 0.0),
                self.type_cooldown_default.get(label, 10.0)
            )
        
        # Use sidecar intent extraction (deterministic) instead of unreliable LLM
        spatial_context_str = ""
        if ranked:
            for score, tid, det, recent, type_blocked in ranked[:3]:
                bbox = det['bbox']
                center_x = (bbox[0] + bbox[2]) / 2
                direction = "LEFT" if center_x < 0.3 else ("RIGHT" if center_x > 0.7 else "CENTER")
                curiosity = "HIGHLY INTRIGUING" if not recent and not type_blocked else "recently seen"
                spatial_context_str += f"{det['label']} {direction} {curiosity} "
        
        # STEP 3: File-based OSC command system - Beta writes structured commands to file
        print(f"[STEP 3] Beta's reasoning: {reasoning[:100]}...")
        print(f"[STEP 3] Spatial context: {spatial_context_str}")
        
        # Have Beta generate structured commands in the expected format
        structured_output = self._generate_structured_commands(reasoning, spatial_context_str)
        
        # Write commands to file for external execution
        self._write_command_file(structured_output)
        
        # No return needed - command execution handled by external OSC client
        # Beta's commands are written to beta_commands.json for execution
        return None

    def _generate_structured_commands(self, reasoning, spatial_context_str):
        """Generate structured <CONTROL_JSON> and <SAY> format output using LLM"""
        try:
            system_prompt = """You are Beta, an autonomous VRChat AI. Based on your reasoning and spatial context, choose ONE command and output EXACTLY this format:

<CONTROL_JSON>
```json
{"commands": [{"name": "CHOOSE_COMMAND", "params": {"PARAM_NAME": VALUE}}]}
```
</CONTROL_JSON>

<SAY>
Write your own unique message about what you actually see and want to do.
</SAY>

Available commands (choose ONE):
- move_forward: {"distance_m": 0.5-3.0}
- turn: {"angle_deg": -180 to 180}
- look: {"pitch_deg": -80 to 80}
- stop: {} (no params)
- interact: {} (no params)
- describe: {"note": "text"} (optional)

Command examples (use actual values):
- {"name": "move_forward", "params": {"distance_m": 1.5}}
- {"name": "turn", "params": {"angle_deg": 45}}
- {"name": "look", "params": {"pitch_deg": -20}}
- {"name": "stop", "params": {}}

Output ONLY the CONTROL_JSON and SAY blocks. Write your SAY message about what you actually see in the scene, not any example text."""

            user_prompt = f"""REASONING: {reasoning}

SPATIAL CONTEXT: {spatial_context_str}

Current situation: Standing in a VRChat world, depth sensors show front clearance ~1.8m, detecting various objects around me. 

Generate structured command output now:"""

            # Use existing LLM call infrastructure
            response = self._call_llm_simple(user_prompt, system_prompt, max_tokens=200, temperature=0.3)
            
            if response and response.strip():
                print(f"[STRUCTURED OUTPUT] Beta generated: {response[:200]}...")
                return response.strip()
            else:
                print("[STRUCTURED OUTPUT] LLM returned empty response, using fallback")
                return self._get_fallback_structured_output()
                
        except Exception as e:
            print(f"[STRUCTURED OUTPUT] Error generating structured commands: {e}")
            return self._get_fallback_structured_output()

    def _get_fallback_structured_output(self):
        """Fallback structured output if LLM fails - use safe look command instead of movement"""
        return """<CONTROL_JSON>
```json
{"commands": [{"name": "look", "params": {"direction": "left", "duration": 0.5}}]}
```
</CONTROL_JSON>

<SAY>
Looking around to understand my surroundings better.
</SAY>"""

    def _write_command_file(self, structured_output):
        """Write structured commands to file for external execution"""
        import json
        import re
        import os
        from datetime import datetime

        try:
            # Parse the structured output - find LAST occurrence (Beta's actual response, not system prompt)
            control_matches = list(re.finditer(r'<CONTROL_JSON>\s*```json\s*(\{.*?\})\s*```\s*</CONTROL_JSON>', structured_output, re.DOTALL | re.IGNORECASE))
            say_matches = list(re.finditer(r'<SAY>\s*(.*?)\s*</SAY>', structured_output, re.DOTALL | re.IGNORECASE))

            control_match = control_matches[-1] if control_matches else None
            say_match = say_matches[-1] if say_matches else None

            commands_json = None
            say_text = None

            # Debug parsing results
            print(f"[COMMAND FILE DEBUG] Control matches found: {len(control_matches)}")
            print(f"[COMMAND FILE DEBUG] SAY matches found: {len(say_matches)}")
            print(f"[COMMAND FILE DEBUG] Using last control match: {control_match is not None}")
            print(f"[COMMAND FILE DEBUG] Using last SAY match: {say_match is not None}")

            if control_match:
                try:
                    raw_json = control_match.group(1)
                    print(f"[COMMAND FILE DEBUG] Extracted JSON: {raw_json}")
                    commands_json = json.loads(raw_json)
                    print(f"[COMMAND FILE DEBUG] Parsed commands: {commands_json}")
                except json.JSONDecodeError as e:
                    print(f"[COMMAND FILE] JSON parse error: {e}")
                    print(f"[COMMAND FILE] Raw JSON that failed: {control_match.group(1)}")
                    # NO FALLBACKS - let it fail
                    raise ValueError(f"JSON parsing failed: {e}")
            else:
                print(f"[COMMAND FILE] No CONTROL_JSON match found in output")
                print(f"[COMMAND FILE] Raw output: {structured_output[:500]}...")
                # NO FALLBACKS - let it fail
                raise ValueError("No CONTROL_JSON found in LLM output")

            if say_match:
                say_text = say_match.group(1).strip().replace('\n', ' ')
                print(f"[COMMAND FILE DEBUG] Extracted say text: {say_text}")
            else:
                print(f"[COMMAND FILE] No SAY match found in output")
                # NO FALLBACKS - let it fail
                raise ValueError("No SAY block found in LLM output")

            # Create command file structure
            command_data = {
                "timestamp": datetime.now().isoformat(),
                "frame_id": getattr(self.tick_engine, 'frame_id', 0),
                "tick_id": getattr(self.tick_engine, 'tick_id', 0),
                "commands": commands_json.get("commands", []),
                "say": say_text,
                "raw_output": structured_output
            }
            
            # Write to commands directory
            commands_dir = os.path.join(os.path.dirname(__file__), "..", "Common", "Cache", "Commands")
            os.makedirs(commands_dir, exist_ok=True)
            
            command_file = os.path.join(commands_dir, "beta_commands.json")
            
            with open(command_file, 'w', encoding='utf-8') as f:
                json.dump(command_data, f, indent=2)
            
            print(f"[COMMAND FILE] Commands written to: {command_file}")
            print(f"[COMMAND FILE] Commands: {commands_json}")
            print(f"[COMMAND FILE] Say: {say_text}")
            
        except Exception as e:
            print(f"[COMMAND FILE] Error writing command file: {e}")
    
    def _extract_action_llm(self, reasoning, ranked=None):
        """Use LLM to naturally translate reasoning to OSC commands"""
        try:
            system_prompt = "You are Beta's action translator. Analyze her reasoning and spatial context, then output EXACTLY one action command."
            
            # Build spatial context from ranked targets
            spatial_context = ""
            if ranked and len(ranked) > 0:
                spatial_context = "\nCurrent spatial context:\n"
                for i, (score, tid, det, recent, type_blocked) in enumerate(ranked[:3]):  # Top 3 targets
                    bbox = det['bbox']
                    # Calculate rough position from bounding box center
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    
                    # Convert to directional guidance
                    if center_x < 0.3:
                        direction = "LEFT"
                    elif center_x > 0.7:
                        direction = "RIGHT" 
                    else:
                        direction = "CENTER"
                        
                    object_desc = det.get('ocr', det['label'])
                    curiosity_level = "HIGHLY INTRIGUING" if not recent and not type_blocked else "recently seen"
                    spatial_context += f"- {object_desc} ({det['label']}) is {direction} - {curiosity_level}\n"
                
                spatial_context += "\nIf Beta mentions wanting to scan/look at specific objects, choose the correct directional look command to face them first.\n"
            
            user_prompt = f'''Beta reasoned: "{reasoning}"{spatial_context}

Available VRChat actions (choose exactly ONE):
- "I want to move forward to explore" (toward interesting objects/targets)
- "I want to look left to scan the area" (turn left with button press)
- "I want to look right to check my surroundings" (turn right with button press) 
- "I want to look up to scan above" (vertical axis up)
- "I want to look down to check below" (vertical axis down)
- "I want to back up to get more clearance" (if too close/dangerous)
- "I want to strafe left to avoid obstacles" (sidestep left)
- "I want to strafe right to navigate around objects" (sidestep right)
- "I want to say Hello there!" (if social)

Beta is naturally curious and should look toward objects based on their location:
- If she mentions "left side", "on the left", or objects marked as LEFT: choose "I want to look left to scan the area" 
- If she mentions "right side", "on the right", or objects marked as RIGHT: choose "I want to look right to check my surroundings"
- If she says "look around", "scan", "check around": choose a look command (left/right/up) NOT move forward
- If she mentions "mobile phone" and it's on LEFT: choose "I want to look left to scan the area"
- If she mentions looking up/down: choose appropriate look commands
- Only choose "move forward" if ALL interesting objects are in CENTER and she wants to approach them
- If she mentions danger/obstacles/cliffs/too close: choose look commands to find safe alternatives
- CRITICAL: Don't default to move forward - match her directional language to directional actions!

CRITICAL OUTPUT FORMAT:
Think through your decision, then respond with EXACTLY:
"I want to [action]"

NO other text allowed. Examples:
"I want to look left to scan the area"
"I want to move forward to explore" 
"I want to look right to check my surroundings"'''
            
            # Use decision mode for fast, crisp action selection - use proper mode token limit
            result = self._qwen_chat(system_prompt, user_prompt, max_tokens=self.n_by_mode["decision"], mode="decision")
            
            if result["success"]:
                raw_response = result["text"].strip()
                print(f"*** RAW LLM RESPONSE: '{raw_response}' ***")
                
                # Parse action from potentially verbose response using robust regex
                action = self._parse_action_from_response(raw_response)
                
                if action:
                    print(f"*** EXTRACTED ACTION: '{action}' ***")
                    return action
                else:
                    print(f"Warning: Could not extract valid action from LLM response: '{raw_response}'")
            
            # No fallback - let the calling function handle failure
            return None
            
        except Exception as e:
            print(f"Warning: LLM action extraction failed: {e}")
            return None
    
    def _parse_action_from_response(self, response):
        """Extract 'I want to [action]' from verbose LLM responses using robust regex"""
        import re
        
        print(f"PARSING DEBUG: Input response length: {len(response)} chars")
        print(f"PARSING DEBUG: First 200 chars: '{response[:200]}'")
        
        # Define valid action patterns - match the exact actions we expect
        valid_actions = [
            r"I want to move forward to explore",
            r"I want to look left to scan the area", 
            r"I want to look right to check my surroundings",
            r"I want to look up to scan above",
            r"I want to look down to check below", 
            r"I want to back up to get more clearance",
            r"I want to strafe left to avoid obstacles",
            r"I want to strafe right to navigate around objects",
            r"I want to say Hello there!"
        ]
        
        # Try to find exact matches first
        for action_pattern in valid_actions:
            if re.search(action_pattern, response, re.IGNORECASE):
                # Return the standardized form
                action_map = {
                    "move forward": "I want to move forward to explore",
                    "look left": "I want to look left to scan the area",
                    "look right": "I want to look right to check my surroundings", 
                    "look up": "I want to look up to scan above",
                    "look down": "I want to look down to check below",
                    "back up": "I want to back up to get more clearance",
                    "strafe left": "I want to strafe left to avoid obstacles",
                    "strafe right": "I want to strafe right to navigate around objects",
                    "say hello": "I want to say Hello there!"
                }
                
                # Find which action was matched and return standard form
                for key, standard_action in action_map.items():
                    if key in action_pattern.lower():
                        return standard_action
        
        # Try broader pattern matching for "I want to [verb]" - prioritize look commands
        broad_patterns = [
            (r"I want to look left", "I want to look left to scan the area"),
            (r"I want to look right", "I want to look right to check my surroundings"),  
            (r"I want to look up", "I want to look up to scan above"), 
            (r"I want to look down", "I want to look down to check below"),
            (r"I want to back up", "I want to back up to get more clearance"),
            (r"I want to strafe left", "I want to strafe left to avoid obstacles"),
            (r"I want to strafe right", "I want to strafe right to navigate around objects"),
            (r"I want to move forward", "I want to move forward to explore"),  # Moved to end - less priority
            (r"I want to say", "I want to say Hello there!")
        ]
        
        for pattern, standard_action in broad_patterns:
            if re.search(pattern, response, re.IGNORECASE):
                print(f"REGEX MATCH: Found '{pattern}' in response, returning '{standard_action}'")
                return standard_action
        
        # Last resort: look for any "I want to" phrase and try to extract meaningful action
        # Find ALL occurrences and take the LAST one (LLM's final answer)
        want_matches = re.findall(r"I want to ([^.!?\n\"]*)", response, re.IGNORECASE)
        if want_matches:
            # Take the LAST occurrence - this is likely the LLM's final decision
            action_text = want_matches[-1].strip().lower()
            print(f"REGEX DEBUG: Found {len(want_matches)} 'I want to' matches, using last one: '{action_text}'")
            
            # Map common action variants to standard forms
            if any(word in action_text for word in ["look left", "scan left", "turn left"]):
                return "I want to look left to scan the area"
            elif any(word in action_text for word in ["look right", "scan right", "turn right"]):
                return "I want to look right to check my surroundings"
            elif any(word in action_text for word in ["look up", "scan above", "scan up"]):
                return "I want to look up to scan above"
            elif any(word in action_text for word in ["look down", "scan below", "scan down"]):
                return "I want to look down to check below"
            elif any(word in action_text for word in ["move", "go", "approach", "explore", "investigate"]):
                return "I want to move forward to explore"
            else:
                print(f"REGEX DEBUG: No action match found for: '{action_text}'")
        
        print("REGEX DEBUG: No 'I want to' patterns found in response")
        return None
    
    # REMOVED: _extract_intent_sidecar method - no longer needed with file-based command system
    # Beta's natural reasoning flows directly to structured command generation
    
    def _clean_natural_response(self, raw_response):
        """Extract pure natural reasoning, removing all system prompt echoes"""
        return clean_natural_response(raw_response)
    
    def _deliver_natural_thought(self, raw_thought):
        """Deliver natural language thought with smart multi-part system"""
        # Clean the response first
        clean_thought = self._clean_natural_response(raw_thought)
        
        # VRChat chat limit is 144 characters
        max_chars = 140  # Leave buffer for safety
        
        if len(clean_thought) <= max_chars:
            # Short enough to send in one message
            self.chat(clean_thought, "Natural exploration reasoning")
        else:
            # Split into multiple parts and pause exploration
            self._pause_exploration_for_chat(clean_thought, max_chars)
    
    def _pause_exploration_for_chat(self, thought, max_chars):
        """Pause exploration and deliver thought in chunks"""
        # Split thought into sentences for better chunking
        sentences = [s.strip() + '.' for s in thought.replace('.', '.|||').split('|||') if s.strip()]
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk + " " + sentence) <= max_chars:
                current_chunk += (" " + sentence if current_chunk else sentence)
            else:
                if current_chunk:
                    chunks.append(current_chunk)
                current_chunk = sentence
        
        if current_chunk:
            chunks.append(current_chunk)
        
        # Deliver chunks with delays
        for i, chunk in enumerate(chunks):
            if i == 0:
                self.chat(chunk, "Natural exploration reasoning")
            else:
                # Schedule delayed message (we'll implement this as immediate for now)
                time.sleep(3)  # 3-second pause between chunks
                self.chat(f"...{chunk}", "Natural exploration reasoning")
    
    # Explain command system (from Explain Command.txt)
    def _update_vision_summary(self, scene_text, objs, paths=None):
        """Update vision summary for explain system"""
        self._last_vision = {
            "scene": (scene_text or "")[:80],
            "objects": (objs or [])[:3],
            "paths": (paths or [])[:3],
        }
    
    def _rle(self, seq):
        """Run-length encode a sequence"""
        if not seq: return []
        out = []
        cur, c = seq[0], 1
        for s in seq[1:]:
            if s == cur: c += 1
            else: out.append((cur, c)); cur, c = s, 1
        out.append((cur, c))
        return out

    def _explain_compact(self) -> str:
        """Generate compact status explanation"""
        # last 6 actions → run-length compress → 3 most recent groups
        actions = [a for (_, a) in list(self._explain_last)[-6:]]
        r = self._rle(actions)[-3:]
        if not r:
            return "Idle; waiting for context."
        steps = "; ".join(
            f"{a.replace('I want to ','')}{('x'+str(c)) if c>1 else ''}"
            for (a, c) in r
        )

        objs = ", ".join(self._last_vision.get("objects") or []) or "nothing"
        tier = self._last_llm_meta.get("tier","n/a")
        lat  = self._last_llm_meta.get("lat",0.0)
        stuck = "yes" if getattr(self, "_stuck_score", 0) >= 2 else "no"
        landmark = getattr(self, "current_landmark", None)
        loc = f" @ {landmark}" if landmark else ""

        # Add telemetry info to compact explain
        progress = self.telemetry.get("progress_score", 0.0)
        stage = self.telemetry.get("fl_stage", "unknown")
        
        # Keep this under ~120 chars typically (your chat splitter handles worst case)
        return f"Doing: {steps}; see {objs}{loc}. LLM {tier} {lat:.1f}s; stuck:{stuck}; progress:{progress:.0f}; stage:{stage}"

    def _explain_full(self) -> str:
        """Generate detailed status explanation"""
        actions = [a for (_, a) in list(self._explain_last)[-10:]]
        r = self._rle(actions)
        r_txt = " | ".join(f"{a.replace('I want to ','')}x{c}" for a, c in r) if r else "(none)"
        objs = ", ".join(self._last_vision.get("objects") or []) or "none"
        paths = ", ".join(self._last_vision.get("paths") or []) or "none"
        tier = self._last_llm_meta.get("tier","n/a")
        lat  = self._last_llm_meta.get("lat",0.0)
        stuck = getattr(self, "_stuck_score", 0)

        # Add telemetry info to full explain
        progress = self.telemetry.get("progress_score", 0.0)
        stage = self.telemetry.get("fl_stage", "unknown")
        bad_ticks = self.telemetry.get("bad_ticks", 0)
        last_action = self.telemetry.get("last_executed_action", "none")
        
        lines = [
            "=== EXPLAIN (full) ===",
            f"Recent actions: {r_txt}",
            f"Seeing: {objs} | Paths: {paths}",
            f"LLM: tier={tier} latency={lat:.2f}s",
            f"StuckScore={stuck}",
            f"Telemetry: progress={progress:.1f} stage={stage} bad_ticks={bad_ticks}",
            f"Last action: {last_action}",
        ]
        # include landmark/sector if you have them
        if hasattr(self, "current_landmark") and self.current_landmark:
            lines.append(f"Landmark: {self.current_landmark}")
        if hasattr(self, "sectors"):
            visits = [s.get('visits',0) for s in self.sectors]
            lines.append(f"Sectors visits: {visits}")
        return "\n".join(lines)
    
    def _handle_command(self, msg: str):
        """Handle chat commands like !explain"""
        m = msg.strip().lower()
        if m == "!explain":
            one_line = self._explain_compact()
            # send to VRChat chat (auto-splits if needed)
            self.chat(one_line, "Explain")
            return True
        if m == "!explain full":
            # log only (don't spam VRChat chat)
            self.log_chat_message(self._explain_full(), "Explain")
            self.chat("Posted a detailed status to the log.", "Explain")
            return True
        # ...your other commands...
        return False
    
        
    def setup_vrchat_receiver(self):
        """Setup OSC server to receive data from VRChat (preserved)"""
        try:
            disp = dispatcher.Dispatcher()
            disp.map("/avatar/parameters/*", self.handle_avatar_parameter)
            disp.map("/avatar/change", self.handle_avatar_change)
            disp.map("/chatbox/input", self.handle_chat_input)
            
            self.server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", 9001), disp)
            self.server_thread = threading.Thread(target=self.server.serve_forever)
            self.server_thread.daemon = True
            self.server_thread.start()
            print("VRChat OSC receiver active on port 9001")
        except Exception as e:
            print(f"Warning: Could not start OSC receiver: {e}")
    
    def start_threaded_consciousness(self):
        """LEGACY: Start all consciousness threads (replaced by lockstep)"""
        # Vision processing thread
        self.vision_thread = threading.Thread(target=self._vision_loop)
        self.vision_thread.daemon = True
        self.vision_thread.start()
        
        # Decision making thread
        self.decision_thread = threading.Thread(target=self._decision_loop)
        self.decision_thread.daemon = True
        self.decision_thread.start()
        
        # Action executor thread
        self.executor_thread = threading.Thread(target=self._executor_loop)
        self.executor_thread.daemon = True
        self.executor_thread.start()
        
        print("All consciousness threads started")
    
    def detect_world_and_initialize(self):
        """Clean world detection without broken auto-calibration"""
        import os
        
        # Detect current world using VRChat logs
        world_id = self.world_detector.get_current_world_from_logs()
        if not world_id:
            world_id = f"unknown_world_{int(time.time())}"
            print(f"WARNING: Could not detect world from VRChat logs, using fallback: {world_id}")
        else:
            print(f"SUCCESS: Detected VRChat world: {world_id}")
        
        # Create world-specific cache directory structure
        cache_base = "G:/Experimental/Production/MidnightCore/Core/Common/Cache"
        world_data_dir = os.path.join(cache_base, "World_Data")
        world_specific_dir = os.path.join(world_data_dir, world_id)
        
        # Ensure both base and world-specific directories exist
        if not os.path.exists(world_data_dir):
            os.makedirs(world_data_dir, exist_ok=True)
            print(f"Created base cache directory: {world_data_dir}")
        
        if not os.path.exists(world_specific_dir):
            os.makedirs(world_specific_dir, exist_ok=True)
            print(f"Created world-specific directory: {world_specific_dir}")
        else:
            print(f"Using existing world directory: {world_specific_dir}")
        
        # Initialize spatial state for detected world
        # Note: 'Just B' world uses Unity defaults, no calibration needed
        self.beta_spatial_state = create_spatial_state_for_world(world_id, auto_calibrator=None)
        print(f"Spatial state initialized for world: {world_id}")
        
        return world_id
    
    def wait_for_depth_system_ready(self, timeout_seconds=10):
        """Wait for depth system to start publishing real data"""
        import time
        import os
        print("Phase 2: Waiting for depth system to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout_seconds:
            try:
                # Import the state bus to check if depth data is available
                from .state_bus import get_vision_state
                
                # Check if depth system is publishing real data
                vision_state = get_vision_state()
                print(f"DEBUG: vision_state = {vision_state}")
                if vision_state is not None:
                    # Check if we have valid depth data (not simulated)
                    front_m = vision_state.get('front_m', 0.0)
                    print(f"DEBUG: front_m = {front_m}")
                    if front_m > 0.1:  # Valid depth measurement
                        print(f"SUCCESS: Depth system ready after {time.time() - start_time:.1f}s (front_m={front_m:.2f})")
                        return True

                elapsed = time.time() - start_time
                print(f"Waiting for depth system... ({elapsed:.1f}s) - vision_state: {vision_state is not None}, front_m: {vision_state.get('front_m', 'N/A') if vision_state else 'N/A'}")
                time.sleep(0.5)
                
            except Exception as e:
                elapsed = time.time() - start_time
                print(f"Depth system check failed ({elapsed:.1f}s): {e}")
                time.sleep(0.5)
        
        print(f"WARNING: Depth system not ready after {timeout_seconds}s, proceeding anyway")
        return False
    
    def sequential_startup(self):
        """Sequential startup system - wait for each phase before proceeding"""
        print("\n=== SEQUENTIAL STARTUP INITIATED ===")
        
        # Phase 1: Base systems (already initialized in __init__)
        print("Phase 1: Base systems ready (logging, GPU monitoring)")
        
        # Phase 2: Wait for vision system to be ready
        self.wait_for_depth_system_ready()
        
        # Phase 3: World detection (now depth is ready)
        print("Phase 3: Detecting world and initializing cache...")
        world_id = self.detect_world_and_initialize()
        
        # Phase 4: All systems ready
        print("Phase 4: All systems ready for consciousness")
        print("=== SEQUENTIAL STARTUP COMPLETE ===\n")

        return world_id

    def start_osc_client(self):
        """Start the OSC client as a managed subprocess"""
        if self.osc_client_running:
            print("OSC client already running")
            return True

        try:
            # Get the path to the Python executable and OSC client module
            import sys
            python_exe = sys.executable

            # Get the project root directory
            import os
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))

            # Start OSC client subprocess
            self.osc_client_process = subprocess.Popen(
                [python_exe, "-m", "Core.Engine.osc_client"],
                cwd=project_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                creationflags=subprocess.CREATE_NEW_CONSOLE if os.name == 'nt' else 0
            )

            self.osc_client_running = True
            print(f"SUCCESS: OSC client started as subprocess (PID: {self.osc_client_process.pid})")

            # Give it a moment to initialize
            import time
            time.sleep(2)

            return True

        except Exception as e:
            print(f"ERROR: Failed to start OSC client subprocess: {e}")
            self.osc_client_running = False
            return False

    def stop_osc_client(self):
        """Stop the OSC client subprocess"""
        if not self.osc_client_running or not self.osc_client_process:
            return

        try:
            print("Stopping OSC client subprocess...")
            self.osc_client_process.terminate()

            # Wait up to 5 seconds for graceful shutdown
            try:
                self.osc_client_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                print("OSC client didn't stop gracefully, forcing termination...")
                self.osc_client_process.kill()
                self.osc_client_process.wait()

            print("OSC client subprocess stopped")
            self.osc_client_running = False
            self.osc_client_process = None

        except Exception as e:
            print(f"Warning: Error stopping OSC client: {e}")

    def start_lockstep_consciousness(self):
        """NEW: Lockstep main loop with speak-before-act sequencing"""
        print("Starting lockstep consciousness loop...")
        
        # Sequential startup - wait for each system to be ready
        world_id = self.sequential_startup()

        # Start OSC client subprocess
        if not self.start_osc_client():
            print("WARNING: OSC client failed to start - continuing without external OSC client")

        # Send welcome message with world status
        world_info_str = f" World: {world_id}" if world_id else ""
        self.chat(f"Natural Qwen Brain activated with lockstep synchronization! Using Unity defaults for movement.{world_info_str} Frame-perfect exploration ready.", "System activation")
        
        while self.consciousness_active and not self.emergency_stop and not self.llm_failure_pause:
            try:
                # Check tick limits for auto-stop (batch file control)
                current_tick = getattr(self.tick_engine, 'tick_id', 0)
                if hasattr(TracingConfig, 'MAX_TICKS') and current_tick >= TracingConfig.MAX_TICKS:
                    log_engine("brain.auto_stop", reason="tick_limit_reached", 
                              current_tick=current_tick, max_ticks=TracingConfig.MAX_TICKS)
                    print(f"[BRAIN] Tick limit reached ({TracingConfig.MAX_TICKS}) - stopping brain for tracing")
                    self.consciousness_active = False
                    break
                
                # Start new tick cycle
                self.tick_engine.start_tick()
                
                # PHASE 1: CAPTURE - Get vision data with fresh frame ID
                vision_context = self.tick_engine.capture_phase(
                    capture_fn=self._lockstep_capture,
                    vision_fn=self._lockstep_vision_analysis
                )
                
                # PHASE 2: DECIDE - Make decision using only current frame
                raw = self.tick_engine.decide_phase(
                    decision_fn=self._lockstep_decision,
                    vision_context=vision_context
                )
                decision = self._to_decision(raw)
                if not decision:
                    current_tick = getattr(self.tick_engine, 'tick_id', 0)
                    log_engine("decision.invalid_payload", 
                              tick_id=current_tick, 
                              raw_response=str(raw)[:500] if raw else "None",
                              fallback_triggered=True)
                    self.chat("[lockstep] invalid decision payload; scanning instead.", "System error")
                    decision = {"action": "I want to look left to scan the area", "reasoning": "Fallback scan"}

                action_command = decision.get("action", "")
                reasoning_text = decision.get("reasoning", "")
                
                # Log successful decision
                if decision and action_command:
                    current_tick = getattr(self.tick_engine, 'tick_id', 0)
                    log_engine("decision.success", 
                              tick_id=current_tick,
                              action_preview=action_command[:100],
                              reasoning_preview=reasoning_text[:100])
                
                # PHASE 3: SPEAK - Generate narration BEFORE action
                self.tick_engine.speak_phase(
                    chat_fn=self._lockstep_chat,
                    reasoning_text=reasoning_text
                )
                
                # PHASE 4: ACT - Execute action with frame stamping
                if action_command:
                    self.tick_engine.act_phase(
                            action_fn=self._lockstep_action,
                            action_command=action_command
                        )
                    
                    # PHASE 5: INTEGRATE - Update internal state including spatial memory
                    self.tick_engine.integrate_phase()
                    if action_command:
                        self._update_spatial_memory(action_command, vision_context)
                
                # Print enhanced telemetry for debugging
                telemetry = self.tick_engine.get_telemetry()
                print(f"[TELEMETRY] engine=lockstep tick_id={telemetry['tick_id']} frame_id={telemetry['frame_id']} "
                      f"action_frame={telemetry['action_frame_id']} chat_frame={telemetry['chat_frame_id']} "
                      f"age={telemetry['perception_age_ms']:.1f}ms future_ref={telemetry['chat_refs_future_frame']} "
                      f"payload_type={type(raw).__name__}")
                
                # Fixed tick interval - increased to prevent overlapping decisions
                time.sleep(6.5)  # >= p50 LLM decision latency from logs
                
            except Exception as e:
                error_msg = f"[TICK ERROR] {e}"
                print(error_msg)
                log_engine("tick.error", tick_id=getattr(self.tick_engine, 'tick_id', 0),
                          error=str(e), traceback=traceback.format_exc()[:500])
                self.chat(f"Tick engine error: {str(e)}", "System error")
                time.sleep(2.0)  # Continue with fixed interval
    
    def _lockstep_capture(self):
        """LOCKSTEP: Capture screenshot for current frame"""
        return self.capture_system.capture_vrchat_for_qwen(filename=None, context="lockstep_frame")
    
    def _lockstep_vision_analysis(self, frame_data):
        """LOCKSTEP: Analyze vision data using only current frame"""
        if not frame_data or not frame_data.get("qwen_ready"):
            return "No vision data available"
        
        # Time Florence analysis
        florence_start = time.time()
        analysis = analyze_screenshot(frame_data["filepath"], for_navigation=True)
        florence_latency = (time.time() - florence_start) * 1000
        log_timing("florence", "lockstep_analysis", florence_latency, details="Lockstep=True")
        
        if not analysis:
            return "Vision analysis failed"
        
        # Build vision context
        scene = analysis.get("scene_description", "Unknown environment")
        objs = analysis.get("detected_objects", [])[:3]
        social = analysis.get("navigation_assessment", {}).get("social_elements", [])[:2]
        paths = analysis.get("navigation_assessment", {}).get("pathways", [])[:2]
        
        # Get depth/spatial data from new vision system
        depth_context = ""
        vision_facts_context = ""
        if VISION_SYSTEM_AVAILABLE:
            try:
                vision_state = get_vision_state()
                if vision_state:
                    # Build clearance summary from available depth data
                    front = vision_state.get('front_m', 'N/A')
                    left = vision_state.get('left_m', 'N/A')
                    right = vision_state.get('right_m', 'N/A')
                    edge_risk = vision_state.get('edge_risk', 0.0)
                    ttc = vision_state.get('ttc_s', 'N/A')
                    
                    if isinstance(front, float):
                        front = f"{front:.1f}m"
                    if isinstance(left, float):
                        left = f"{left:.1f}m"  
                    if isinstance(right, float):
                        right = f"{right:.1f}m"
                    if isinstance(ttc, float):
                        ttc = f"{ttc:.1f}s"
                        
                    clearance_summary = f"Front:{front} Left:{left} Right:{right} Risk:{edge_risk:.2f} TTC:{ttc}"
                    depth_context = f"\\nDEPTH: {clearance_summary}"
                else:
                    depth_context = f"\\nDEPTH: No depth data available"
                
                vision_facts = get_vision_facts()
                if vision_facts:
                    vision_facts_context = f"\\nVISION: {vision_facts}"
            except Exception as e:
                depth_context = f"\\nDEPTH: Error accessing depth data: {str(e)}"
        
        # Get spatial memory context
        spatial_context = self._get_spatial_context()
        
        # Build comprehensive context with spatial memory (objects are strings, not dictionaries)
        vision_context = f"""SCENE: {scene}
OBJECTS: {', '.join(objs) if objs else 'None detected'}
PEOPLE: {', '.join(social) if social else 'None detected'}
PATHWAYS: {', '.join(paths) if paths else 'None detected'}{depth_context}{vision_facts_context}
{spatial_context}"""
        
        return vision_context
    
    def _parse_vision_context_for_detections(self, vision_context):
        """Parse vision context string to extract object detections for ranking"""
        detections = []
        try:
            # Parse vision context to extract object information
            lines = vision_context.split('\n')
            for line in lines:
                if line.startswith('OBJECTS: '):
                    objects_str = line.replace('OBJECTS: ', '').strip()
                    if objects_str and objects_str != 'None detected':
                        objects = [obj.strip() for obj in objects_str.split(',')]
                        for i, obj in enumerate(objects):
                            # Create detection with position based on order (left to right)
                            x_pos = 0.2 + (i * 0.3)  # Distribute objects from left to right
                            detections.append({
                                "label": obj,
                                "bbox": [x_pos, 0.4, x_pos + 0.2, 0.8],  # Standard size box
                                "ocr": obj
                            })
                        break
            
            # Parse people separately
            for line in lines:
                if line.startswith('PEOPLE: '):
                    people_str = line.replace('PEOPLE: ', '').strip()
                    if people_str and people_str != 'None detected':
                        people = [person.strip() for person in people_str.split(',')]
                        for i, person in enumerate(people):
                            # Position people more to the right
                            x_pos = 0.6 + (i * 0.2)
                            detections.append({
                                "label": "person",
                                "bbox": [x_pos, 0.2, x_pos + 0.2, 0.9],  # Tall person box
                                "ocr": person
                            })
                        break
                        
        except Exception as e:
            print(f"[VISION_PARSE_ERROR] {e}")
            
        return detections
    
    def _lockstep_decision(self, vision_context):
        """LOCKSTEP: Make decision using only provided vision context"""
        try:
            current_tick = getattr(self.tick_engine, 'tick_id', 0)
            log_engine("decision.start", tick_id=current_tick, vision_context_length=len(vision_context))
            
            # Tick down type cooldowns
            self._tick_type_cooldowns(2.0)  # Fixed 2s interval
            
            # Build target ranking from vision context - FIXED from empty stub
            detections = self._parse_vision_context_for_detections(vision_context)
            ranked = self._rank_targets(detections) if detections else []
            
            # Create system prompt for natural reasoning
            system_prompt = """You are Beta, a curious AI exploring VRChat worlds. You can see the environment and decide what to do next.
            
Express your thoughts naturally about what you see and what interests you. Then state your intended action clearly."""
            
            # Use vision context as user prompt
            user_prompt = vision_context
            
            # Get natural reasoning from Qwen
            q = self._qwen_chat(system_prompt, user_prompt, max_tokens=self.n_by_mode["chat"], mode="chat")
            
            if q["success"]:
                raw_response = q["text"].strip()
                
                # Clean the response to extract pure reasoning
                natural_thought = self._clean_natural_response(raw_response)
                
                # Extract action from natural reasoning (returns string action command)
                action_decision = self._extract_action_from_reasoning(natural_thought, ranked)
                
                print(f"*** LOCKSTEP REASONING: {natural_thought} ***")
                print(f"*** LOCKSTEP ACTION: {action_decision} ***")
                
                return {
                    'action': action_decision,  # action_decision is already a string action command
                    'reasoning': natural_thought
                }
            else:
                print("[LOCKSTEP] Qwen chat failed")
                return None
                
        except Exception as e:
            error_msg = f"[DECISION ERROR] {e}"
            print(error_msg)
            log_engine("decision.error", tick_id=getattr(self.tick_engine, 'tick_id', 0),
                      error=str(e), traceback=traceback.format_exc()[:500])
            return None
    
    def _lockstep_action(self, action_command, tick_id, frame_id):
        """LOCKSTEP: Execute action with frame ID stamping"""
        print(f"[ACTION] tick={tick_id} frame={frame_id} executing: {action_command}")
        
        # DISABLED: Legacy OSC execution replaced by file-based command system
        # Commands are now written to beta_commands.json for external execution
        # self._perform_action(action_command)  # Removed for Session 24
        
        # Log action with frame ID (using existing OSC logging method)
        self.log_osc_command("lockstep_action", {"command": action_command, "tick_id": tick_id, "frame_id": frame_id})
    
    def _lockstep_chat(self, reasoning_text):
        """LOCKSTEP: Send chat using frame-stamped reasoning - BLOCKS until all chunks sent"""
        if reasoning_text:
            self.chat_blocking(reasoning_text, "lockstep_reasoning")
    
    def _to_decision(self, result):
        """Normalize decision payloads before calling .get()"""
        if isinstance(result, dict):
            return result
        if isinstance(result, str):
            try:
                import json
                return json.loads(result)
            except Exception:
                # treat raw string as a direct canonical action
                return {"action": result.strip(), "reasoning": "Direct action command"}
        return None
    
    def _update_spatial_memory(self, action_command, vision_context):
        """Update spatial memory based on actions and observations"""
        try:
            # Enhanced spatial tracking with BetaSpatialState
            if self.beta_spatial_state:
                # Parse action details for precise tracking
                duration = self._extract_action_duration(action_command)
                intensity = 1.0  # Default intensity
                is_running = "run" in action_command.lower()
                
                # Check if movement was blocked (from OSC controller)
                was_blocked = False
                block_reason = ""
                if hasattr(self.osc_controller, 'get_last_blocked_info'):
                    block_info = self.osc_controller.get_last_blocked_info()
                    was_blocked = block_info.get("was_blocked", False)
                    block_reason = block_info.get("reason", "")
                
                # Determine action type for spatial tracking
                spatial_action = self._map_to_spatial_action(action_command)
                
                if spatial_action:
                    if "look" in spatial_action:
                        # Update facing direction for look commands
                        self.beta_spatial_state.update_facing(spatial_action, duration)
                        print(f"[SPATIAL] Look action: {spatial_action}, new facing: {self.beta_spatial_state.facing_angle:.1f}°")
                    else:
                        # Update position for movement commands
                        result = self.beta_spatial_state.update_movement(
                            action=spatial_action,
                            duration=duration,
                            intensity=intensity,
                            is_running=is_running,
                            was_blocked=was_blocked,
                            block_reason=block_reason
                        )
                        
                        if result["position_changed"]:
                            pos = result["new_position"]
                            print(f"[SPATIAL] Position updated: ({pos.x:.2f}, {pos.y:.2f}) facing {result['current_facing']:.1f}°")
                            # Log brain-level spatial update
                            log_engine("brain.spatial.position_update", action_command=action_command,
                                      old_pos={"x": result["old_position"].x, "y": result["old_position"].y},
                                      new_pos={"x": pos.x, "y": pos.y}, facing=result['current_facing'])
                        elif was_blocked:
                            print(f"[SPATIAL] Movement blocked: {block_reason}")
                            # Log brain-level blocked movement
                            log_engine("brain.spatial.movement_blocked", action_command=action_command,
                                      block_reason=block_reason)
                
                # Save spatial state periodically
                if hasattr(self.beta_spatial_state, 'position') and self.beta_spatial_state.position.timestamp:
                    cache_dir = "G:/Experimental/Production/MidnightCore/Core/Common/cache/spatial_maps"
                    self.beta_spatial_state.save_to_cache(cache_dir)
            
            # Legacy spatial memory tracking (keep for compatibility)
            self.spatial_memory["movement_history"].append(action_command)
            
            # Update current sector based on look commands (sync with BetaSpatialState)
            if self.beta_spatial_state:
                self.spatial_memory["current_sector"] = self.beta_spatial_state.facing_angle
            else:
                # Fallback to legacy tracking
                if "look left" in action_command.lower():
                    self.spatial_memory["current_sector"] = (self.spatial_memory["current_sector"] - 45) % 360
                elif "look right" in action_command.lower():
                    self.spatial_memory["current_sector"] = (self.spatial_memory["current_sector"] + 45) % 360
            
            # Mark current sector as visited when looking around
            if "look" in action_command.lower():
                sector = int(self.spatial_memory["current_sector"] // 45)  # Convert to 8 sectors
                self.spatial_memory["visited_sectors"].add(sector)
                print(f"[SPATIAL] Visited sector {sector} (facing {self.spatial_memory['current_sector']:.1f}°)")
            
            # Extract landmarks from vision context
            if "OBJECTS:" in vision_context:
                objects_line = vision_context.split("OBJECTS:")[1].split("\n")[0]
                if "None detected" not in objects_line:
                    # Add interesting objects to landmark map
                    current_sector = int(self.spatial_memory["current_sector"] // 45)
                    if objects_line.strip() and current_sector not in self.spatial_memory["landmark_map"]:
                        self.spatial_memory["landmark_map"][current_sector] = objects_line.strip()[:50]  # Brief description
                        print(f"[SPATIAL] Landmark recorded in sector {current_sector}: {objects_line.strip()[:50]}")
            
            # Generate exploration goals for unvisited sectors
            all_sectors = set(range(8))  # 8 sectors (45° each)
            unvisited = all_sectors - self.spatial_memory["visited_sectors"]
            self.spatial_memory["exploration_goals"] = list(unvisited)
            
        except Exception as e:
            print(f"[SPATIAL ERROR] {e}")
    
    def _extract_action_duration(self, action_command):
        """Extract duration from action command string"""
        import re
        # Look for patterns like "0.5 seconds", "1s", "2.0s"
        duration_match = re.search(r'(\d+\.?\d*)\s*(?:second|sec|s)', action_command.lower())
        if duration_match:
            return float(duration_match.group(1))
        else:
            # Default durations based on action type
            if "look" in action_command.lower():
                return 0.5  # Standard look duration
            else:
                return 1.0  # Standard movement duration
    
    def _map_to_spatial_action(self, action_command):
        """Map natural action command to spatial tracking action"""
        cmd_lower = action_command.lower()
        
        if "move forward" in cmd_lower or "walk forward" in cmd_lower:
            return "move_forward"
        elif "move backward" in cmd_lower or "walk backward" in cmd_lower:
            return "move_backward"
        elif "strafe left" in cmd_lower or "move left" in cmd_lower:
            return "strafe_left"
        elif "strafe right" in cmd_lower or "move right" in cmd_lower:
            return "strafe_right"
        elif "look left" in cmd_lower:
            return "look_left"
        elif "look right" in cmd_lower:
            return "look_right"
        elif "look up" in cmd_lower:
            return "look_up"
        elif "look down" in cmd_lower:
            return "look_down"
        else:
            return None  # Non-spatial action
    
    def _get_spatial_context(self):
        """Get spatial memory context for decision making"""
        # Enhanced spatial context with BetaSpatialState
        if self.beta_spatial_state:
            summary = self.beta_spatial_state.get_spatial_summary()
            position = summary["position"]
            relative = summary["relative_to_spawn"]
            session = summary["session_stats"]
            
            # Position information
            pos_info = f"Position: ({position['x']:.1f}, {position['y']:.1f})m facing {position['facing_degrees']:.0f}°"
            
            # Distance from spawn
            spawn_info = f"Distance from spawn: {relative['distance']:.1f}m"
            
            # Movement statistics
            movement_info = f"Session: {session['distance_walked']:.1f}m walked"
            if session['distance_run'] > 0:
                movement_info += f", {session['distance_run']:.1f}m run"
            
            # Confidence indicator
            confidence_info = f"Position confidence: {summary['confidence']:.1f}"
            
            spatial_context = f"SPATIAL STATE: {pos_info}. {spawn_info}. {movement_info}. {confidence_info}."
            
            # Log spatial context generation
            log_engine("brain.spatial.context_generated", 
                      position=position, relative_spawn=relative, session_stats=session,
                      confidence=summary["confidence"], context_length=len(spatial_context))
        else:
            # Legacy spatial memory
            visited_count = len(self.spatial_memory["visited_sectors"])
            unvisited = 8 - visited_count
            spatial_context = f"SPATIAL MEMORY: Explored {visited_count}/8 sectors. {unvisited} unexplored areas remain."
        
        # Recent movement pattern analysis
        recent_moves = list(self.spatial_memory["movement_history"])[-3:]
        pattern_info = ""
        if len(set(recent_moves)) == 1 and len(recent_moves) >= 2:
            pattern_info = f" (repeating {recent_moves[0]})"
        
        # Landmark info
        landmark_info = ""
        if self.spatial_memory["landmark_map"]:
            landmark_count = len(self.spatial_memory["landmark_map"])
            landmark_info = f" Found {landmark_count} landmark areas."
        
        spatial_context += f"{pattern_info}{landmark_info}"
        
        # Add exploration suggestions
        if self.spatial_memory["exploration_goals"]:
            next_goal = self.spatial_memory["exploration_goals"][0]
            direction_name = ["North", "NE", "East", "SE", "South", "SW", "West", "NW"][next_goal]
            spatial_context += f" Priority: explore {direction_name} sector."
        
        return spatial_context
    
    def _vision_loop(self):
        """Vision processing with fast/slow passes (non-blocking)"""
        fast_dt = 1.0 / self.vision_fast_hz
        slow_dt = 1.0 / self.vision_slow_hz
        t_last_slow = 0.0
        
        while self.consciousness_active and not self.emergency_stop:
            try:
                # Quick pass (center crop → lightweight describe)
                quick = self.capture_system.capture_vrchat_for_qwen(filename=None, context="fast_centered")
                analysis = {}
                
                if quick.get("qwen_ready"):
                    self.log_florence_call("quick_capture", "Fast vision pass", "centered crop")
                    # Time Florence analysis
                    florence_start = time.time()
                    analysis = analyze_screenshot(quick["filepath"], for_navigation=False)
                    florence_latency = (time.time() - florence_start) * 1000
                    log_timing("florence", "fast_analysis", florence_latency, details="Navigation=False")
                    
                    if analysis:
                        self.log_florence_results(analysis, "Fast vision analysis")
                
                # Full pass periodically
                now = time.time()
                if now - t_last_slow >= slow_dt:
                    full = self.capture_system.capture_vrchat_for_qwen(filename=None, context="full_analysis")
                    if full.get("qwen_ready"):
                        self.log_florence_call("full_capture", "Full vision pass", "complete scene")
                        # Time full Florence analysis  
                        florence_start = time.time()
                        analysis = analyze_screenshot(full["filepath"], for_navigation=True)
                        florence_latency = (time.time() - florence_start) * 1000
                        log_timing("florence", "full_analysis", florence_latency, details="Navigation=True")
                        
                        if analysis:
                            self.log_florence_results(analysis, "Full vision analysis")
                    t_last_slow = now
                
                # Push latest (non-blocking)
                try:
                    if self._vision_q.full():
                        self._vision_q.get_nowait()
                    self._vision_q.put_nowait(analysis or {})
                except queue.Full:
                    pass
                
            except Exception as e:
                print(f"[VISION] error: {e}")
            time.sleep(fast_dt)
    
    def _render_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Render proper chat template based on model family"""
        fam = self.model_family
        if fam == "qwen":
            # Qwen3 - use simple concatenation (chat templates cause empty output)
            return f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
        if fam == "mistral":
            # Mistral - use simple prompt without INST tags (they cause hangs)
            return f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
        # Generic fallback: just concatenate
        return f"{system_prompt.strip()}\n\n{user_prompt.strip()}"

    def _sanitize_decision(self, text: str) -> str:
        """Enhanced decision sanitizer for executor compatibility"""
        t = text.strip().lower()
        if "look right" in t or "turn right" in t:
            return "I want to look right to scan the area."
        if "look left" in t or "turn left" in t:
            return "I want to look left to check my surroundings."
        if any(w in t for w in ["move forward","walk forward","go forward","explore"]):
            return "I want to move forward to explore a few steps."
        if "look up" in t:   return "I want to look up to scan above."
        if "look down" in t: return "I want to look down to check below."
        if "jump" in t:      return "I want to jump once."
        if any(w in t for w in ["say ","chat ","hello"]):
            return "I want to say Hello there!"
        return random.choice([
            "I want to look right to scan the area.",
            "I want to look left to check my surroundings.",
            "I want to move forward to explore a few steps.",
            "I want to say Hello there!",
        ])

    def _record_action(self, action, reasoning, scene_context):
        """Record an action in history for contextual reasoning"""
        timestamp = time.time()
        self.action_history.append((timestamp, action, reasoning, scene_context))
        
        # Trim history to max length
        if len(self.action_history) > self.max_history_length:
            self.action_history = self.action_history[-self.max_history_length:]

    def _build_action_context(self):
        """Build contextual summary of recent actions for LLM reasoning"""
        if not self.action_history:
            return ""
        
        # Get last 4 actions for immediate context
        recent_actions = self.action_history[-4:]
        
        context_parts = ["RECENT ACTIONS:"]
        for i, (timestamp, action, reasoning, scene) in enumerate(recent_actions, 1):
            # Simplified action description
            action_desc = action.replace("I want to ", "").replace(" to scan the area", "").replace(" to check my surroundings", "")
            context_parts.append(f"{i}. {action_desc} - {reasoning[:50]}...")
        
        # Add spatial reasoning if we have movement patterns
        movement_actions = [a for a in recent_actions if any(move in a[1].lower() for move in ["look", "turn", "move", "face"])]
        if len(movement_actions) >= 2:
            context_parts.append("\nSPATIAL CONTEXT:")
            last_two = movement_actions[-2:]
            if "look left" in last_two[0][1] and "look right" in last_two[1][1]:
                context_parts.append("- You turned left then right, likely scanning")
            elif "face target" in last_two[0][1] and "face target" in last_two[1][1]:
                context_parts.append("- You faced the same target twice, consider approaching or exploring elsewhere")
            elif any("move forward" in a[1] for a in last_two):
                context_parts.append("- You've been moving forward, consider looking around")
        
        return "\n".join(context_parts) + "\n"

    def _update_exploration_goals(self, current_scene, detected_objects):
        """Update exploration goals based on current situation"""
        goal_text = ""
        
        if self.look_state["investigating_target"]:
            target = self.look_state["target_description"]
            # Check if target is still visible
            target_visible = any(target.lower() in obj.lower() or obj.lower() in target.lower() 
                               for obj in detected_objects)
            if target_visible:
                goal_text = f"CURRENT GOAL: Investigate {target} (visible)"
            else:
                goal_text = f"CURRENT GOAL: Find {target} (lost sight of it)"
        elif detected_objects:
            primary_object = detected_objects[0] if detected_objects else "area"
            goal_text = f"EXPLORATION: Examining {primary_object}"
        else:
            goal_text = "EXPLORATION: General area scanning"
            
        return goal_text + "\n"

    def _detect_target_loss(self, detected_objects):
        """Detect if we've lost sight of our target and provide reasoning context"""
        if not self.look_state["investigating_target"]:
            return ""
        
        target = self.look_state["target_description"]
        target_visible = any(target.lower() in obj.lower() or obj.lower() in target.lower() 
                           for obj in detected_objects)
        
        if not target_visible and len(self.action_history) > 0:
            last_action = self.action_history[-1][1]
            if "look" in last_action or "turn" in last_action:
                return f"TARGET LOST: You were looking for {target} but turned away. Consider reversing direction.\n"
        
        return ""

    def _qwen_chat(self, system_prompt: str, user_prompt: str, max_tokens=80, mode="decision"):
        """
        WORKING llama.cpp wrapper with venv fixes
        Returns {success, text, tier, latency}
        """
        n_decision = self._n_for_mode("decision")
        n_chat     = self._n_for_mode("chat")
        n_fallback = self._n_for_mode("fallback")
        # pick the request cap based on mode (caller can still override via max_tokens)
        req_n = min(max_tokens, n_chat if mode == "chat" else n_decision)
        
        # Log LLM request with unified logging
        current_tick = getattr(self.tick_engine, 'tick_id', 0)
        current_frame = getattr(self.tick_engine, 'frame_id', 0)
        full_prompt = self._render_chat_prompt(system_prompt, user_prompt)
        
        log_llm_request(
            tick_id=current_tick,
            frame_id=current_frame,
            mode=mode,
            tokens_max=req_n,
            temp=self.qwen_config.get("temp", 0.7),
            top_p=self.qwen_config.get("top_p", 0.9),
            prompt_text=full_prompt
        )
        tiers = [
            ("full_context", 
             self._render_chat_prompt(system_prompt, user_prompt),
             dict(temp=0.7, top_p=0.9, n=req_n)),
            ("minimal_fallback",
             self._render_chat_prompt(
                 "Choose ONE short action.",
                 'Return ONE of:\n'
                 '"I want to move forward to explore a few steps."\n'
                 '"I want to look left to scan the area."\n'
                 '"I want to look right to check my surroundings."\n'
                 '"I want to say Hello there!"'
             ),
             dict(temp=0.4, top_p=0.8, n=n_fallback))
        ]

        with self._llm_lock:
            for tier_name, rendered_prompt, samp in tiers:
                cmd = [
                    self.llama_path, "-m", self.model_path,
                    "-ngl", str(self.qwen_config.get("ngl", 25)),
                    "-c",   str(self.qwen_config.get("ctx", 4096)),
                    "--temp", str(samp["temp"]), "--top-p", str(samp["top_p"]),
                    "--repeat-penalty", str(self.qwen_config.get("repeat_penalty", 1.1)),
                    "--ignore-eos",  # CRITICAL FIX - prevents early EOS termination
                    "-n", str(samp["n"]), "-no-cnv",  # REMOVED --log-disable (breaks in venv)
                    "-p", rendered_prompt
                ]
                try:
                    t0 = time.time()
                    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    llm_start_time = time.time()
                    out, err = proc.communicate(timeout=120)  # Increased to 2 minutes for Qwen-3 reasoning
                    llm_duration = time.time() - llm_start_time
                    print(f"[LLM TIMING] Qwen-3 reasoning completed in {llm_duration:.1f}s")
                    if proc.returncode != 0:
                        print(f"Tier {tier_name}: returncode {proc.returncode}")
                        continue
                    # Extract only the generated text (before any performance stats)
                    text = (out or "").replace("<s>", "").replace("</s>", "").strip()
                    
                    # Early-stop detection and retry logic
                    should_retry = False
                    if tier_name == "full_context":  # Only apply to main tier
                        # Check if we likely hit token cap without completing a thought
                        has_sentence_ending = any(punct in text for punct in ['.', '!', '?'])
                        appears_truncated = not has_sentence_ending and len(text.split()) >= 35  # ~80% of typical response
                        
                        if appears_truncated and not hasattr(self, '_retry_attempted'):
                            print(f"Response appears truncated ({len(text.split())} words, no sentence ending) - retrying with higher limit")
                            self._retry_attempted = True
                            should_retry = True
                        else:
                            # Reset retry flag for next decision
                            if hasattr(self, '_retry_attempted'):
                                delattr(self, '_retry_attempted')
                    
                    if should_retry:
                        # Retry with 50% more tokens
                        retry_tokens = int(samp["n"] * 1.5)
                        retry_cmd = cmd[:-2] + ["-n", str(retry_tokens), "-p", rendered_prompt]  # Replace -n value
                        print(f"Retrying with {retry_tokens} tokens...")
                        
                        try:
                            retry_proc = subprocess.Popen(retry_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                            retry_start_time = time.time()
                            retry_out, retry_err = retry_proc.communicate(timeout=120)
                            retry_duration = time.time() - retry_start_time
                            print(f"[LLM TIMING] Qwen-3 retry completed in {retry_duration:.1f}s")
                            if retry_proc.returncode == 0:
                                text = (retry_out or "").replace("<s>", "").replace("</s>", "").strip()
                                print(f"Retry successful: {len(text.split())} words")
                        except Exception as e:
                            print(f"Retry failed: {e}, using original response")
                    
                    # Remove llama performance logs that may appear at end
                    if "llama_perf_sampler_print:" in text:
                        text = text.split("llama_perf_sampler_print:")[0].strip()
                    if "llama_perf_context_print:" in text:
                        text = text.split("llama_perf_context_print:")[0].strip()
                    
                    text = re.sub(r"\\s+", " ", text)
                    latency = time.time() - t0
                    
                    # Log GPU telemetry for Qwen inference
                    log_timing("qwen", f"{mode}_{tier_name}", latency * 1000, details=f"{len(text.split())} tokens")
                    
                    if len(text) >= 3:
                        # Calculate telemetry metrics
                        tokens_used = len(text.split()) if text else 0
                        ended_early = bool(text and any(punct in text for punct in ['.', '!', '?']) and tokens_used < samp["n"] * 0.8)
                        cap_hit = tokens_used >= samp["n"] * 0.9 and not ended_early
                        
                        # Log decision telemetry
                        self._log_decision_telemetry(tokens_used, ended_early, cap_hit, tier_name, latency, should_retry)
                        
                        # Log LLM response with unified logging
                        log_llm_response(
                            tick_id=current_tick,
                            frame_id=current_frame,
                            tokens_used=tokens_used,
                            latency_ms=latency * 1000,
                            ended_early=ended_early,
                            cap_hit=cap_hit,
                            tier=tier_name,
                            text_preview=text
                        )
                        
                        print(f"SUCCESS on tier {tier_name}: {len(text)} chars, {tokens_used} tokens - '{text[:50]}...'")
                        return dict(success=True, text=text, tier=tier_name, latency=latency, 
                                    tokens_used=tokens_used, ended_early=ended_early, cap_hit=cap_hit)
                    else:
                        print(f"Tier {tier_name}: empty output (len={len(text)})")
                except subprocess.TimeoutExpired:
                    print(f"Tier {tier_name}: timeout after 120s")
                    proc.kill(); proc.communicate()
                except Exception as e:
                    print(f"Tier {tier_name}: exception {e}")
                    pass
            return dict(success=False, text="")

    def _call_llm_simple(self, user_prompt: str, system_prompt: str, max_tokens=200, temperature=0.3):
        """Simple LLM call for structured output generation"""
        try:
            result = self._qwen_chat(system_prompt, user_prompt, max_tokens=max_tokens, mode="decision")
            if result.get("success", False):
                return result.get("text", "").strip()
            else:
                print(f"[STRUCTURED OUTPUT] LLM call failed: {result}")
                return None
        except Exception as e:
            print(f"[STRUCTURED OUTPUT] LLM call exception: {e}")
            return None

    def _detect_targets_of_interest(self, analysis):
        """Detect and prioritize targets of interest from Florence analysis"""
        if not analysis:
            return None
            
        targets = []
        
        # Check for people (highest priority)
        social_elements = analysis.get("navigation_assessment", {}).get("social_elements", [])
        for person in social_elements:
            if any(word in person.lower() for word in ["person", "people", "user", "avatar", "player"]):
                targets.append({"type": "person", "description": person, "priority": 10})
        
        # Check for interactive objects
        objects = analysis.get("detected_objects", [])
        for obj in objects:
            obj_lower = obj.lower()
            if any(word in obj_lower for word in ["door", "doorway", "entrance", "exit"]):
                targets.append({"type": "door", "description": obj, "priority": 8})
            elif any(word in obj_lower for word in ["chair", "table", "desk", "computer"]):
                targets.append({"type": "furniture", "description": obj, "priority": 6})
            elif any(word in obj_lower for word in ["stairs", "stairway", "elevator", "escalator"]):
                targets.append({"type": "pathway", "description": obj, "priority": 7})
        
        # Check for pathways
        pathways = analysis.get("navigation_assessment", {}).get("pathways", [])
        for pathway in pathways:
            if any(word in pathway.lower() for word in ["hallway", "corridor", "path", "walkway"]):
                targets.append({"type": "pathway", "description": pathway, "priority": 5})
        
        # Return highest priority target
        if targets:
            best_target = max(targets, key=lambda x: x["priority"])
            return best_target
        
        return None
    
    def _estimate_target_angle(self, target, analysis):
        """Estimate the angle of a target based on context clues"""
        # This is a simple heuristic - in reality we'd need Florence bounding boxes
        # For now, use description keywords to estimate rough position
        desc = target["description"].lower()
        
        # Look for directional keywords
        if any(word in desc for word in ["left", "west"]):
            return -45  # Left side
        elif any(word in desc for word in ["right", "east"]):
            return 45   # Right side
        elif any(word in desc for word in ["ahead", "forward", "front", "north"]):
            return 0    # Straight ahead
        elif any(word in desc for word in ["behind", "back", "south"]):
            return 180  # Behind
        else:
            # No clear directional info - assume roughly to the side for scanning
            return random.choice([-30, 30, -60, 60])  # Various scan angles
    
    def _sanitize_decision(self, t):
        """Convert any text to canonical action - legacy fallback"""
        s = t.lower()
        
        # Standard actions
        if "look right" in s or "turn right" in s: return self._safe_actions[0]
        if "look left" in s or "turn left" in s: return self._safe_actions[1]
        if any(w in s for w in ["move forward","walk forward","go forward","explore"]): return self._safe_actions[2]
        if any(w in s for w in ["say ","chat ","hello"]): return self._safe_actions[3]
        if "look up" in s: return "I want to look up to scan above."
        if "look down" in s: return "I want to look down to check below."
        if "jump" in s: return "I want to jump once."
        
        return random.choice(self._safe_actions)
    
    def _decision_loop(self):
        """Natural decision making with target memory and cooldown system"""
        dt = 1.0 / self.decision_hz
        
        # Send welcome message
        self.chat("Natural Qwen Brain activated! I can see and explore freely now.", "System activation")
        
        while self.consciousness_active and not self.emergency_stop and not self.llm_failure_pause:
            try:
                # Tick down type cooldowns
                now = time.time()
                self._tick_type_cooldowns(dt)
                
                # Get latest vision snapshot (non-blocking)
                analysis = {}
                fl_stage = "no_vision"
                try:
                    analysis = self._vision_q.get_nowait()
                    fl_stage = "vision_ready" if analysis else "vision_empty"
                except:
                    fl_stage = "vision_waiting"
                
                scene = analysis.get("scene_description", "Unknown environment")
                objs = analysis.get("detected_objects", [])[:3]
                social = analysis.get("navigation_assessment", {}).get("social_elements", [])[:2]
                paths = analysis.get("navigation_assessment", {}).get("pathways", [])[:2]
                
                # Get depth/spatial data from new vision system
                depth_context = ""
                vision_facts_context = ""
                if VISION_SYSTEM_AVAILABLE:
                    try:
                        # Get depth state
                        depth_state = get_vision_state()
                        if depth_state:
                            # Get depth measurements
                            front_m = depth_state.get('front_m', float('inf'))
                            left_m = depth_state.get('left_m', float('inf'))  
                            right_m = depth_state.get('right_m', float('inf'))
                            edge_risk = depth_state.get('edge_risk', 0.0)
                            ttc_s = depth_state.get('ttc_s', float('inf'))
                            
                            # Build enhanced spatial context using hybrid navigation
                            try:
                                enhanced_spatial_context = integrate_spatial_context(self, depth_state)
                                spatial_lines = [enhanced_spatial_context]
                            except Exception as e:
                                print(f"[HYBRID NAV] Enhanced spatial context failed, using legacy: {e}")
                                # Fallback to legacy spatial analysis
                                spatial_lines = []
                                
                                # Movement safety analysis
                                if front_m < 0.5:
                                    spatial_lines.append(f"DANGER: Wall/obstacle {front_m:.1f}m ahead - TOO CLOSE!")
                                elif front_m < 1.0:
                                    spatial_lines.append(f"CAUTION: Obstacle {front_m:.1f}m ahead - approach carefully")
                                elif front_m < 2.0:
                                    spatial_lines.append(f"AWARE: Object {front_m:.1f}m ahead - room to move")
                                else:
                                    spatial_lines.append(f"CLEAR: Path ahead {front_m:.1f}m - safe to move forward")
                                
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
                                    spatial_lines.append("MOVEMENT RESTRICTED: Consider looking around or backing up")
                            
                            # Time-to-contact warning
                            if ttc_s < 1.0:
                                spatial_lines.append(f"COLLISION WARNING: Impact in {ttc_s:.1f}s if moving forward!")
                            elif ttc_s < 3.0:
                                spatial_lines.append(f"TIMING ALERT: {ttc_s:.1f}s to reach obstacle ahead")
                            
                            # Edge/cliff detection
                            if edge_risk > 0.3:
                                spatial_lines.append(f"CLIFF RISK: Edge detected (risk={edge_risk:.1f}) - avoid forward movement!")
                            
                            depth_line = "\n".join(spatial_lines)
                            
                            # Add bearing-based directional analysis  
                            clearances = depth_state.get('clearance_by_bearing', [])
                            if clearances:
                                # Find key directional clearances
                                directional_info = []
                                
                                # Analyze major directions
                                directions = [
                                    ("straight ahead", 0, 8),     # ±8° center
                                    ("slight left", -20, 8),      # -28° to -12°  
                                    ("slight right", 20, 8),      # 12° to 28°
                                    ("left side", -45, 8),        # -53° to -37°
                                    ("right side", 45, 8),        # 37° to 53°
                                ]
                                
                                for direction_name, center_bearing, half_width in directions:
                                    # Find clearances in this direction
                                    relevant_clearances = []
                                    for bearing, clearance in clearances:
                                        if abs(bearing - center_bearing) <= half_width:
                                            relevant_clearances.append(clearance)
                                    
                                    if relevant_clearances:
                                        min_clearance = min(relevant_clearances)
                                        avg_clearance = sum(relevant_clearances) / len(relevant_clearances)
                                        
                                        if min_clearance < 0.5:
                                            directional_info.append(f"{direction_name}: BLOCKED ({min_clearance:.1f}m)")
                                        elif min_clearance < 1.0:
                                            directional_info.append(f"{direction_name}: TIGHT ({min_clearance:.1f}m)")  
                                        elif min_clearance > 2.0:
                                            directional_info.append(f"{direction_name}: OPEN ({avg_clearance:.1f}m)")
                                
                                # Find best navigation options
                                navigation_suggestions = []
                                
                                # Look for wide safe passages
                                safe_runs = []
                                current_start = None
                                current_width = 0
                                
                                for bearing, clearance in clearances:
                                    if clearance >= 1.2:  # Safe threshold
                                        if current_start is None:
                                            current_start = bearing
                                            current_width = 4  # 4° per bin
                                        else:
                                            current_width += 4
                                    else:
                                        if current_start is not None and current_width >= 20:
                                            center = current_start + current_width / 2
                                            if abs(center) <= 30:  # Within easy reach
                                                turn_direction = "turn left" if center < -10 else "turn right" if center > 10 else "continue forward"
                                                navigation_suggestions.append(f"{turn_direction} toward {center:+.0f}° ({current_width:.0f}° wide passage)")
                                        current_start = None
                                        current_width = 0
                                
                                # Handle run at end
                                if current_start is not None and current_width >= 20:
                                    center = current_start + current_width / 2
                                    if abs(center) <= 30:
                                        turn_direction = "turn left" if center < -10 else "turn right" if center > 10 else "continue forward"  
                                        navigation_suggestions.append(f"{turn_direction} toward {center:+.0f}° ({current_width:.0f}° wide passage)")
                                
                                # Add directional context to spatial understanding
                                if directional_info:
                                    spatial_lines.append(f"DIRECTIONS: {', '.join(directional_info[:3])}")  # Max 3
                                    
                                if navigation_suggestions:
                                    spatial_lines.append(f"NAVIGATION: {', '.join(navigation_suggestions[:2])}")  # Max 2
                            
                            depth_context = "\n".join(spatial_lines) + "\n"
                        
                        # Get vision facts from Florence ROI analysis
                        vision_facts = get_vision_facts()
                        if vision_facts:
                            fact_lines = []
                            for fact in vision_facts[:3]:  # Max 3 facts
                                bearing = fact.get('bearing_deg', 0)
                                label = fact.get('label', 'object')
                                conf = fact.get('conf', 0)
                                dist = fact.get('dist_m', 0)
                                moving = fact.get('moving', False)
                                
                                # Format: obstacle(+2°,0.6m) label="person" conf=0.81 moving=true
                                fact_line = f"obstacle({bearing:+.0f}°,{dist:.1f}m) label=\"{label}\" conf={conf:.2f} moving={str(moving).lower()}"
                                fact_lines.append(f"VISION: {fact_line}")
                            
                            if fact_lines:
                                vision_facts_context = '\n'.join(fact_lines) + '\n'
                    
                        # Add NOTE context for blocked movement
                        note_context = ""
                        if hasattr(self, 'osc_controller') and self.osc_controller:
                            blocked_info = self.osc_controller.get_last_blocked_info()
                            if blocked_info['was_blocked'] and time.time() - blocked_info['timestamp'] < 10:  # Within last 10s
                                note_context = f"NOTE: last forward blocked ({blocked_info['reason']}). choose next action.\n"
                        
                        # Combine all vision contexts
                        if note_context:
                            vision_facts_context = vision_facts_context + note_context
                        
                    except Exception as e:
                        print(f"Warning: Depth data integration error: {e}")
                
                # NEW TARGET MEMORY SYSTEM - Provide context for LLM decisions
                detections = []  # Convert analysis to detection format
                for obj in objs + social:
                    # Create detection with dummy bbox for now (Florence-2 integration needed)
                    obj_lower = obj.lower()
                    if any(word in obj_lower for word in ["person", "people", "user", "avatar"]):
                        label = "person"
                    elif any(word in obj_lower for word in ["door", "doorway", "entrance"]):
                        label = "door"
                    elif any(word in obj_lower for word in ["poster", "sign", "text"]):
                        label = "poster"
                    else:
                        label = "object"
                    
                    # Dummy centered bbox - will be replaced with Florence-2 bounding boxes
                    detections.append({
                        "label": label,
                        "bbox": [0.4, 0.4, 0.6, 0.6],  # Center area
                        "ocr": obj  # Use object description as OCR text
                    })
                
                # Generate memory-aware context for LLM
                ranked = self._rank_targets(detections)
                memory_context = self._build_target_memory_context(ranked)
                cooldown_context = self._build_cooldown_context()
                
                # Build contextual memory for intelligent reasoning
                action_context = self._build_action_context()
                goal_context = self._update_exploration_goals(scene, objs)
                target_loss_context = self._detect_target_loss(objs)
                
                # Split into system and user prompts with memory-aware context  
                system_prompt = "You are Beta, an intelligent VRChat explorer. Reason naturally about what you see and want to do. Speak your thoughts as you explore. Use your memory to avoid repeating the same investigations."
                
                user_prompt = (
                    f"{goal_context}"
                    f"{action_context}"
                    f"CURRENT SITUATION:\n"
                    f"{depth_context}"
                    f"{vision_facts_context}"
                    f"You see: {scene[:200]}\n\n"
                    f"Objects: {', '.join(objs) if objs else 'none'}\n"
                    f"People: {', '.join(social) if social else 'none'}\n" 
                    f"Pathways: {', '.join(paths) if paths else 'none'}\n\n"
                    f"{memory_context}"
                    f"{cooldown_context}\n"
                    f"{target_loss_context}"
                    "EXPLORATION STRATEGY:\n"
                    "- If you see something HIGHLY INTRIGUING or novel, move toward it to investigate\n" 
                    "- If nothing looks particularly interesting, look around (left/right/up/down) to scan for new things\n"
                    "- Only move forward when you have a specific target that genuinely interests you\n"
                    "- When things become boring, scan in a different direction to find something new\n"
                    "- Use depth data to avoid obstacles: don't move forward if front clearance < 0.8m"
                    "\n\nWhat do you want to do right now?"
                )
                
                print(f"\n=== QWEN DECISION REQUEST ===")
                print(f"Scene: {scene[:100]}...")
                print(f"Objects: {objs}")
                print(f"Querying Qwen...")
                
                q = self._qwen_chat(system_prompt, user_prompt, max_tokens=self.n_by_mode["chat"], mode="chat")
                natural_thought = ""  # Initialize
                decision = None
                
                if q["success"]:
                    # Reset failure counter on success
                    self.consecutive_llm_failures = 0
                    raw_response = q["text"].strip()
                    
                    # Clean the response to extract pure reasoning
                    natural_thought = self._clean_natural_response(raw_response)
                    
                    # Extract action from natural reasoning and update target memory
                    decision = self._extract_action_from_reasoning(natural_thought, ranked)
                    reasoning = f"Natural reasoning from {q['tier']} tier"
                    print(f"*** RAW RESPONSE ({q['tier']} tier): {raw_response[:200]}...***")
                    print(f"*** CLEAN REASONING: {natural_thought} ***")
                    print(f"*** EXTRACTED ACTION: {decision} ***")
                    
                    # Log Qwen decision
                    scene_summary = f"{scene[:100]} | Objects: {', '.join(objs[:3])}"
                    self.log_qwen_decision(user_prompt, natural_thought, q["tier"], True, scene_summary)
                else:
                    # Handle LLM failure with retry system
                    self.consecutive_llm_failures += 1
                    print(f"!!! QWEN FAILED (attempt {self.consecutive_llm_failures}/{self.max_llm_failures}) !!!")
                    
                    # Log Qwen failure
                    scene_summary = f"{scene[:100]} | Objects: {', '.join(objs[:3])}"
                    self.log_qwen_decision(user_prompt, "FAILED", "none", False, scene_summary)
                    
                    if self.consecutive_llm_failures >= self.max_llm_failures:
                        # After 3 failures, reset OSC and pause (waiting for instructions)
                        print("*** MAXIMUM LLM FAILURES REACHED - ENTERING SAFE PAUSE MODE ***")
                        self.chat("Please Tell Devaliah I'm lost!", "Help request")
                        self.emergency_reset_all()  # Reset all OSC controls
                        self.llm_failure_pause = True
                        print("Brain is now paused and waiting for instructions...")
                        break  # Exit decision loop
                    else:
                        # Retry - continue to next cycle
                        retry_msg = f"Thinking system retry {self.consecutive_llm_failures}/{self.max_llm_failures}"
                        print(f"Retrying LLM in next cycle...")
                        continue  # Skip this cycle, retry next time
                
                # Record action in history for future context
                scene_summary = f"{', '.join(objs[:2]) if objs else 'empty area'}"
                self._record_action(decision, reasoning, scene_summary)
                
                # Handle chat commands in natural thought
                if natural_thought and "!explain" in natural_thought.lower():
                    if "full" in natural_thought.lower():
                        self._explain_full()
                    else:
                        self._explain_compact()
                    continue  # Skip this cycle after explain
                
                # Only proceed if we have a valid decision
                if decision:
                    self._enqueue_action(decision)
                    # Deliver natural reasoning with smart chat system
                    self._deliver_natural_thought(natural_thought)
                    
                    # Calculate and log tick telemetry
                    progress_score = self._calculate_progress_score(analysis, ranked)
                    bad_ticks = self.telemetry["bad_ticks"] if not decision else 0  # Reset on success
                    executed_action = decision.replace("I want to ", "").replace(" to explore a few steps", "").replace(" to scan the area", "").replace(" to check my surroundings", "")
                    utterance = natural_thought[:50] if natural_thought else ""
                    
                    self._log_tick_telemetry(progress_score, bad_ticks, fl_stage, executed_action, utterance)
                else:
                    # Increment bad ticks counter
                    self.telemetry["bad_ticks"] += 1
                
            except Exception as e:
                print(f"[DECISION] error: {e}")
                self.telemetry["bad_ticks"] += 1
            time.sleep(dt)
    
    def _enqueue_action(self, decision):
        """Add action to queue (non-blocking)"""
        try:
            self._action_q.put_nowait(decision)
        except queue.Full:
            try:
                self._action_q.get_nowait()
                self._action_q.put_nowait(decision)
            except:
                pass
    
    def _watchdog(self):
        """Inject keepalive action if system is idle - DISABLED FOR TESTING"""
        # DISABLED: Testing pure LLM behavior without fallbacks
        # if time.time() - self._last_action_ts > self.watchdog_idle_s:
        #     choice = random.choice(self._safe_actions)
        #     print(f"[WATCHDOG] injecting: {choice}")
        #     self._perform_action(choice)
        pass
    
    def _executor_loop(self):
        """Execute actions with watchdog protection"""
        while self.consciousness_active and not self.emergency_stop:
            try:
                try:
                    decision = self._action_q.get(timeout=15.0)  # Increased from 0.25s to match Qwen processing time
                    self._perform_action(decision)
                    self._last_action_ts = time.time()
                except queue.Empty:
                    self._watchdog()
            except Exception as e:
                print(f"[EXEC] error: {e}")
            time.sleep(0.01)
    
    def _yaw_error_deg(self, bbox):
        """Calculate yaw error from bounding box - bbox [x1,y1,x2,y2] in 0..1"""
        cx = (bbox[0] + bbox[2]) * 0.5  # Center X of target
        return (cx - 0.5) * self.hfov_deg  # +right / -left
    
    def _turn_body_deg(self, deg):
        """Turn body proportionally with clamps + deadband"""
        # ~90° ≈ 0.25s in calibration → scale and clamp
        secs = max(0.05, min(0.40, abs(deg) * (0.25/90.0)))
        if deg > 0: 
            self.osc_look_right(secs)  # Turn right
        else:       
            self.osc_look_left(secs)   # Turn left
        self._spin_guard_deg += deg
    
    def face_target_step(self, bbox):
        """Proportional controller for facing target"""
        err = self._yaw_error_deg(bbox)
        self._yaw_err_ema = 0.7*self._yaw_err_ema + 0.3*err
        e = self._yaw_err_ema
        
        if abs(e) <= 4.0:  # Aligned within 4 degrees
            self._spin_guard_deg = 0.0
            return "aligned"
        
        # Smaller steps near center, larger steps when far off
        step = max(3.0, min(20.0, abs(e)*0.6))
        self._turn_body_deg(step if e>0 else -step)
        return "turning"
    
    def spin_guard_check(self):
        """Prevent endless spinning - bailout after ~1.5 turns"""
        if abs(self._spin_guard_deg) >= 540:  # ~1.5 turns
            self._spin_guard_deg = 0.0
            self.osc_strafe_right(0.35)  # Small sidestep
            self.osc_look_left(0.15); self.osc_look_right(0.15)  # Quick re-scan
            self.chat("Spin-guard: re-scanning.", "System")
            return True
        return False

    def _perform_action(self, decision):
        """Execute canonical action using auto-calibrated OSC parameters"""
        s = decision.lower()
        
        # Track this decision
        self.recent_actions.append(decision[:50])
        if len(self.recent_actions) > 10:
            self.recent_actions.pop(0)
        
        # Use Unity default movement parameters (no calibration needed for 'Just B' world)
        movement_mult = {
            "walk_forward": 1.0,
            "walk_backward": 1.0, 
            "walk_strafe": 1.0,
            "run_forward": 1.0,
            "run_backward": 1.0,
            "run_strafe": 1.0
        }
        durations = {
            "quick_movement": 0.25,
            "standard_movement": 0.5,
            "precise_movement": 1.0,
            "rotation_90deg": 1.0
        }
        
        print(f"\n=== EXECUTING MOVEMENT DECISION ===")
        print(f"Action: {decision}")
        # Always use Unity defaults - no calibration needed
        print(f"Using Unity default parameters - Movement scale: {movement_mult.get('walk_forward', 1.0):.2f}, Rotation: standard")
        
        # Check if centering needed first - DISABLED FOR TESTING
        if "center" in s or "return to center" in s: # Removed auto-centering: or self.look_state["needs_centering"]:
            self.return_to_look_center()
            return
        
        # Target-oriented commands with proportional controller
        if "face target" in s:
            if self.look_state["investigating_target"]:
                print("-> Executing: FACE TARGET (proportional)")
                # Use small incremental turns instead of big rotations
                small_turn = 15.0  # Small 15-degree turns
                current_offset = abs(self.look_state["horizontal_offset"])
                
                if current_offset < 5.0:  # Close enough - stop spinning
                    print("-> Target aligned (within 5 degrees)")
                    self.chat("Target centered.", "System")
                    self.clear_target_of_interest("target aligned")
                    return
                elif current_offset > 180:  # Prevent endless spinning
                    print("-> Spin guard activated - re-scanning")
                    self.spin_guard_check()
                    return
                else:
                    # Small proportional turn
                    turn_size = min(small_turn, current_offset * 0.3)
                    turn_duration = turn_size / 90.0 * 0.25  # Scale to timing
                    print(f"-> Small turn: {turn_size}° ({turn_duration:.3f}s)")
                    
                    if self.look_state["target_angle"] > 0:
                        self.osc_look_right(turn_duration)
                    else:
                        self.osc_look_left(turn_duration)
                    return
            else:
                print("-> No target to face, quick re-scan")
                self.osc_look_left(0.12)
                self.osc_look_right(0.12)
                return
        
        if "approach target" in s:
            if self.look_state["investigating_target"]:
                # Check hybrid navigation safety before approaching target
                try:
                    vision_state = get_vision_state()
                    is_safe, safety_decision, modified_action = check_movement_safety(
                        self, "move forward", vision_state,  # Treat as forward movement
                        tick_id=getattr(self.tick_engine, 'tick_id', None),
                        frame_id=getattr(self.tick_engine, 'frame_id', None)
                    )
                    
                    if not is_safe:
                        print(f"[HYBRID NAV] Target approach blocked: {safety_decision.get('reasoning', 'Safety check failed')}")
                        print(f"[HYBRID NAV] Executing alternative: {modified_action}")
                        self._perform_action(modified_action)
                        return
                        
                except Exception as e:
                    print(f"[HYBRID NAV] Target approach safety check failed, proceeding with caution: {e}")
                
                print("-> Executing: APPROACH TARGET with hybrid navigation check")
                target_desc = self.look_state["target_description"]
                self.chat(f"Moving toward {target_desc}", "Target approach")
                self.osc_move_forward(3.0)  # Longer approach movement
                return
            else:
                print("-> No target to approach, exploring forward")
                self.osc_move_forward(2.0)
                return
        
        # Standard movement commands with hybrid navigation safety
        if "move forward" in s:
            # Check hybrid navigation safety before executing movement
            try:
                vision_state = get_vision_state()
                is_safe, safety_decision, modified_action = check_movement_safety(
                    self, s, vision_state, 
                    tick_id=getattr(self.tick_engine, 'tick_id', None),
                    frame_id=getattr(self.tick_engine, 'frame_id', None)
                )
                
                if not is_safe:
                    print(f"[HYBRID NAV] Movement blocked: {safety_decision.get('reasoning', 'Safety check failed')}")
                    print(f"[HYBRID NAV] Executing alternative: {modified_action}")
                    # Execute the safer alternative action instead
                    self._perform_action(modified_action)
                    return
                elif modified_action != s:
                    print(f"[HYBRID NAV] Movement modified for safety: {modified_action}")
                    s = modified_action  # Use the modified action
                    
            except Exception as e:
                print(f"[HYBRID NAV] Safety check failed, proceeding with caution: {e}")
            
            print("-> Executing: MOVE FORWARD with hybrid navigation check")
            duration = durations.get("movement_step", 2.0) * movement_mult.get("forward", 1.0)
            self.osc_move_forward(duration)
            return
        if "back up" in s or "retreat" in s:
            print("-> Executing: BACK UP (calibrated)")
            duration = (durations.get("movement_step", 1.5) * 0.75) * movement_mult.get("backward", 1.0)  # Shorter for safety
            self.osc_move_backward(duration)
            return
        if "strafe left" in s:
            print("-> Executing: STRAFE LEFT (calibrated)")
            duration = durations.get("movement_step", 1.0) * movement_mult.get("strafe", 1.0)
            self.osc_strafe_left(duration)
            return
        if "strafe right" in s:
            print("-> Executing: STRAFE RIGHT (calibrated)")
            duration = durations.get("movement_step", 1.0) * movement_mult.get("strafe", 1.0)
            self.osc_strafe_right(duration)
            return
        if "look left" in s or "turn left" in s:
            print("-> Executing: LOOK LEFT (calibrated)")
            duration = durations.get("standard_scan", self.look_button_90s) * movement_mult.get("rotation", 1.0)
            self.osc_look_left(duration)
            return
        if "look right" in s or "turn right" in s:
            print("-> Executing: LOOK RIGHT (calibrated)")
            duration = durations.get("standard_scan", self.look_button_90s) * movement_mult.get("rotation", 1.0)
            self.osc_look_right(duration)
            return
        if "look up" in s:
            print("-> Executing: LOOK UP (calibrated)")
            duration = durations.get("quick_scan", 0.5) * movement_mult.get("rotation", 1.0)
            self.osc_look_up(duration)
            return
        if "look down" in s:
            print("-> Executing: LOOK DOWN (calibrated)")
            duration = durations.get("quick_scan", 0.5) * movement_mult.get("rotation", 1.0)
            self.osc_look_down(duration)
            return
        if "jump" in s:
            print("-> Executing: JUMP")
            self.osc_jump()
            return
        
        # Say/hello/default
        msg = "Hello there!"
        m = re.search(r'say\\s+(.+)$', s)
        if m:
            msg = re.sub(r'^["\\'']|["\\'']$', '', m.group(1).strip())
        self.chat(msg, "Natural conversation")
    
    # OSC Command Functions (preserved from original with proper logging)
    def osc_move_forward(self, duration):
        """Move forward for specified duration - MUST reset to 0"""
        self.log_osc_command("move_forward", {"duration": duration})
        self.osc_client.send_message("/input/Vertical", 1.0)
        time.sleep(duration)
        self.osc_client.send_message("/input/Vertical", 0.0)
        print(f"Moved forward for {duration}s")
    
    def osc_move_backward(self, duration):
        """Move backward for specified duration - MUST reset to 0"""
        self.log_osc_command("move_backward", {"duration": duration})
        self.osc_client.send_message("/input/Vertical", -1.0)
        time.sleep(duration)
        self.osc_client.send_message("/input/Vertical", 0.0)
        print(f"Moved backward for {duration}s")
    
    def osc_strafe_left(self, duration):
        """Strafe left for specified duration - MUST reset to 0"""
        self.log_osc_command("strafe_left", {"duration": duration})
        self.osc_client.send_message("/input/Horizontal", -1.0)
        time.sleep(duration)
        self.osc_client.send_message("/input/Horizontal", 0.0)
        print(f"Strafed left for {duration}s")
    
    def osc_strafe_right(self, duration):
        """Strafe right for specified duration - MUST reset to 0"""
        self.log_osc_command("strafe_right", {"duration": duration})
        self.osc_client.send_message("/input/Horizontal", 1.0)
        time.sleep(duration)
        self.osc_client.send_message("/input/Horizontal", 0.0)
        print(f"Strafed right for {duration}s")
    
    def osc_look_left(self, duration):
        """Look left button - duration controls angle (0.25s = 90°)"""
        angle_degrees = (duration / 0.25) * 90
        self.log_osc_command("look_left", {"duration": duration, "angle_degrees": angle_degrees})
        self.osc_client.send_message("/input/LookLeft", 1)
        time.sleep(duration)
        self.osc_client.send_message("/input/LookLeft", 0)
        
        # Track position for centering
        self.look_state["horizontal_offset"] -= angle_degrees
        self.look_state["needs_centering"] = abs(self.look_state["horizontal_offset"]) > 5 or abs(self.look_state["vertical_offset"]) > 5
        
        print(f"Looked left {angle_degrees}° (total offset: {self.look_state['horizontal_offset']}°)")
    
    def osc_look_right(self, duration):
        """Look right button - duration controls angle (0.25s = 90°)"""
        angle_degrees = (duration / 0.25) * 90
        self.log_osc_command("look_right", {"duration": duration, "angle_degrees": angle_degrees})
        self.osc_client.send_message("/input/LookRight", 1)
        time.sleep(duration)
        self.osc_client.send_message("/input/LookRight", 0)
        
        # Track position for centering
        self.look_state["horizontal_offset"] += angle_degrees
        self.look_state["needs_centering"] = abs(self.look_state["horizontal_offset"]) > 5 or abs(self.look_state["vertical_offset"]) > 5
        
        print(f"Looked right {angle_degrees}° (total offset: {self.look_state['horizontal_offset']}°)")
    
    def osc_look_up(self, duration=0.5):
        """Look up - 0.25 velocity, duration controls angle (1s = 45°)"""
        angle_degrees = duration * 45
        self.log_osc_command("look_up", {"duration": duration, "angle_degrees": angle_degrees})
        self.osc_client.send_message("/input/LookVertical", 0.25)
        time.sleep(duration)
        self.osc_client.send_message("/input/LookVertical", 0.0)
        
        # Track position for centering
        self.look_state["vertical_offset"] += angle_degrees
        self.look_state["needs_centering"] = abs(self.look_state["horizontal_offset"]) > 5 or abs(self.look_state["vertical_offset"]) > 5
        
        print(f"Looked up {angle_degrees}° (total offset: {self.look_state['vertical_offset']}°)")
        
        # AUTO-RESET: Give vision time to process, then return to horizontal
        time.sleep(1.0)  # Allow vision system to capture what's above
        print("Auto-centering view after looking up...")
        self.osc_client.send_message("/input/LookVertical", -0.25)
        time.sleep(duration)  # Same duration to return to horizontal
        self.osc_client.send_message("/input/LookVertical", 0.0)
        
        # Reset vertical offset since we returned to center
        self.look_state["vertical_offset"] = 0
        self.look_state["needs_centering"] = abs(self.look_state["horizontal_offset"]) > 5
        print("View reset to horizontal")
    
    def osc_look_down(self, duration=0.5):
        """Look down - 0.25 velocity, duration controls angle (1s = 45°)"""
        angle_degrees = duration * 45
        self.log_osc_command("look_down", {"duration": duration, "angle_degrees": angle_degrees})
        self.osc_client.send_message("/input/LookVertical", -0.25)
        time.sleep(duration)
        self.osc_client.send_message("/input/LookVertical", 0.0)
        
        # Track position for centering
        self.look_state["vertical_offset"] -= angle_degrees
        self.look_state["needs_centering"] = abs(self.look_state["horizontal_offset"]) > 5 or abs(self.look_state["vertical_offset"]) > 5
        
        print(f"Looked down {angle_degrees}° (total offset: {self.look_state['vertical_offset']}°)")
        
        # AUTO-RESET: Give vision time to process, then return to horizontal
        time.sleep(1.0)  # Allow vision system to capture what's below
        print("Auto-centering view after looking down...")
        self.osc_client.send_message("/input/LookVertical", 0.25)
        time.sleep(duration)  # Same duration to return to horizontal
        self.osc_client.send_message("/input/LookVertical", 0.0)
        
        # Reset vertical offset since we returned to center
        self.look_state["vertical_offset"] = 0
        self.look_state["needs_centering"] = abs(self.look_state["horizontal_offset"]) > 5
        print("View reset to horizontal")
    
    def osc_jump(self):
        """Make avatar jump - MUST reset button to 0"""
        self.log_osc_command("jump", {})
        self.osc_client.send_message("/input/Jump", 1)
        time.sleep(0.1)
        self.osc_client.send_message("/input/Jump", 0)
        print("Jumped")
    
    def osc_run_mode(self, enable):
        """Enable or disable run mode"""
        self.log_osc_command("run_mode", {"enable": enable})
        self.osc_client.send_message("/input/Run", 1 if enable else 0)
    
    def osc_emote(self, emote_num):
        """Trigger avatar emote"""
        self.log_osc_command("emote", {"emote_number": emote_num})
        self.osc_client.send_message("/avatar/parameters/VRCEmote", emote_num)
    
    def osc_rotate_to_angle(self, target_angle_degrees):
        """Rotate avatar body to face specific angle"""
        current_offset = self.look_state["horizontal_offset"]
        total_rotation_needed = target_angle_degrees - current_offset
        
        # Determine rotation direction and duration
        if abs(total_rotation_needed) < 5:  # Already close enough
            return
            
        # Use movement to rotate body instead of head
        rotation_time = abs(total_rotation_needed) / 90.0 * 0.5  # Roughly 0.5s per 90°
        
        if total_rotation_needed > 0:  # Need to turn right
            print(f"Rotating body right {abs(total_rotation_needed)}° to face target")
            self.log_osc_command("body_rotate_right", {"degrees": abs(total_rotation_needed), "duration": rotation_time})
            
            # Short strafe right while turning look right to rotate body
            self.osc_client.send_message("/input/Horizontal", 0.3)  # Gentle strafe
            self.osc_client.send_message("/input/LookRight", 1)
            time.sleep(rotation_time)
            self.osc_client.send_message("/input/Horizontal", 0.0)  # Stop strafe
            self.osc_client.send_message("/input/LookRight", 0)
            
        else:  # Need to turn left
            print(f"Rotating body left {abs(total_rotation_needed)}° to face target")
            self.log_osc_command("body_rotate_left", {"degrees": abs(total_rotation_needed), "duration": rotation_time})
            
            # Short strafe left while turning look left to rotate body  
            self.osc_client.send_message("/input/Horizontal", -0.3)  # Gentle strafe
            self.osc_client.send_message("/input/LookLeft", 1)
            time.sleep(rotation_time)
            self.osc_client.send_message("/input/Horizontal", 0.0)  # Stop strafe
            self.osc_client.send_message("/input/LookLeft", 0)
        
        # Update tracking - body is now facing the target, so reset head offset
        self.look_state["horizontal_offset"] = 0.0
        self.look_state["target_angle"] = target_angle_degrees
        
        print(f"Body rotation complete - now facing target at {target_angle_degrees}°")
    
    def set_target_of_interest(self, target_info, estimated_angle=None):
        """Set a new target of interest for investigation"""
        self.look_state["target_of_interest"] = target_info
        self.look_state["target_description"] = target_info.get("description", "unknown target")
        self.look_state["investigating_target"] = True
        
        if estimated_angle is not None:
            self.look_state["target_angle"] = estimated_angle
        
        print(f"New target of interest: {self.look_state['target_description']} at ~{estimated_angle}°")
    
    def clear_target_of_interest(self, reason="investigation complete"):
        """Clear current target of interest"""
        if self.look_state["target_of_interest"]:
            target_desc = self.look_state["target_description"]
            print(f"Clearing target: {target_desc} ({reason})")
            
            self.look_state["target_of_interest"] = None
            self.look_state["target_description"] = ""
            self.look_state["investigating_target"] = False
            self.look_state["target_angle"] = 0.0
            
            self.chat(f"Finished investigating {target_desc}", reason)
    
    def _split_for_vrc(self, text: str, cap: int = None):
        """Split text into VRChat-safe chunks (word-aware)"""
        cap = cap or self.chat_char_cap
        t = " ".join((text or "").strip().split())
        if len(t) <= cap:
            return [t]
        parts, cur = [], []
        for word in t.split(" "):
            if (len(" ".join(cur)) + (1 if cur else 0) + len(word)) <= cap:
                cur.append(word)
            else:
                parts.append(" ".join(cur))
                cur = [word]
        if cur: parts.append(" ".join(cur))
        return parts

    def _send_chat_part(self, msg: str):
        """Send one message to VRChat, with logging (single part)"""
        self.osc_client.send_message("/chatbox/input", [msg, True, True])
        print(f"CHAT: {msg}")

    def chat(self, message: str, reasoning="Natural conversation", pause_s: float = None):
        """
        Non-blocking multi-send: logs full text once, splits into 144-char
        chunks for VRChat chatbox, and sends sequentially with a short pause.
        """
        try:
            # Log the full message once (even if we split on output)
            self.log_chat_message(message, reasoning)

            parts = self._split_for_vrc(message, self.chat_char_cap)
            if len(parts) == 1:
                # short → send immediately
                self._send_chat_part(parts[0])
                return

            # long → send in a background thread so we don't block decision loop
            pause = self.chat_split_pause_s if pause_s is None else pause_s

            def _bg_send(parts_local):
                for i, p in enumerate(parts_local):
                    self._send_chat_part(p)
                    # pause between parts, but not after the last one
                    if i < len(parts_local) - 1:
                        time.sleep(pause)

            th = threading.Thread(target=_bg_send, args=(parts,), daemon=True)
            th.start()

        except Exception as e:
            print(f"CHAT error: {e}")
    
    def chat_blocking(self, message: str, reasoning="Natural conversation", pause_s: float = None):
        """
        BLOCKING multi-send for lockstep mode: waits for all chunks to be sent before returning.
        Used in lockstep speak phase to ensure Beta finishes talking before moving.
        """
        try:
            # Log the full message once (even if we split on output)
            self.log_chat_message(message, reasoning)

            parts = self._split_for_vrc(message, self.chat_char_cap)
            pause = self.chat_split_pause_s if pause_s is None else pause_s
            
            print(f"CHAT_BLOCKING: Sending {len(parts)} message parts with {pause}s pauses")
            
            # Send all parts sequentially and wait for each
            for i, p in enumerate(parts):
                self._send_chat_part(p)
                
                # pause between parts, but not after the last one
                if i < len(parts) - 1:
                    print(f"CHAT_BLOCKING: Part {i+1}/{len(parts)} sent, waiting {pause}s")
                    time.sleep(pause)
                else:
                    print(f"CHAT_BLOCKING: Final part {i+1}/{len(parts)} sent - message complete")

        except Exception as e:
            print(f"CHAT_BLOCKING error: {e}")
    
    def return_to_look_center(self):
        """Return look position to center - now target-oriented"""
        h_offset = self.look_state["horizontal_offset"]
        v_offset = self.look_state["vertical_offset"]
        
        # If investigating a target, face the target instead of just forward
        if self.look_state["investigating_target"] and self.look_state["target_of_interest"]:
            target_desc = self.look_state["target_description"]
            print(f"Recentering on target: {target_desc}")
            
            # Rotate body to face the target
            self.osc_rotate_to_angle(self.look_state["target_angle"])
            
            # Reset vertical look to center
            if abs(v_offset) > 1:
                if v_offset > 0:
                    duration = abs(v_offset) / 45
                    self.osc_client.send_message("/input/LookVertical", -0.25)
                    time.sleep(duration)
                    self.osc_client.send_message("/input/LookVertical", 0.0)
                else:
                    duration = abs(v_offset) / 45
                    self.osc_client.send_message("/input/LookVertical", 0.25)
                    time.sleep(duration)
                    self.osc_client.send_message("/input/LookVertical", 0.0)
            
            self.look_state["vertical_offset"] = 0.0
            self.look_state["needs_centering"] = False
            self.chat(f"Now facing {target_desc} directly", "Target-oriented centering")
            print(f"Successfully centered on target: {target_desc}")
            
        else:
            # Standard centering behavior when no target
            print(f"Standard centering from offset: {h_offset}° horizontal, {v_offset}° vertical")
            
            # Return horizontal to center
            if abs(h_offset) > 1:
                if h_offset > 0:
                    duration = abs(h_offset) / 90 * 0.25
                    self.osc_client.send_message("/input/LookLeft", 1)
                    time.sleep(duration)
                    self.osc_client.send_message("/input/LookLeft", 0)
                else:
                    duration = abs(h_offset) / 90 * 0.25
                    self.osc_client.send_message("/input/LookRight", 1)
                    time.sleep(duration)
                    self.osc_client.send_message("/input/LookRight", 0)
            
            # Return vertical to center
            if abs(v_offset) > 1:
                if v_offset > 0:
                    duration = abs(v_offset) / 45
                    self.osc_client.send_message("/input/LookVertical", -0.25)
                    time.sleep(duration)
                    self.osc_client.send_message("/input/LookVertical", 0.0)
                else:
                    duration = abs(v_offset) / 45
                    self.osc_client.send_message("/input/LookVertical", 0.25)
                    time.sleep(duration)
                    self.osc_client.send_message("/input/LookVertical", 0.0)
            
            # Reset tracking
            self.look_state["horizontal_offset"] = 0.0
            self.look_state["vertical_offset"] = 0.0
            self.look_state["needs_centering"] = False
            
            self.chat("Returned to center view", "Standard look reset")
            print("Standard look centering complete")
    
    # Event Handlers (preserved from original)
    def handle_avatar_parameter(self, address, *args):
        """Handle avatar parameter changes"""
        pass
    
    def handle_avatar_change(self, address, avatar_id):
        """Handle avatar changes"""
        self.chat(f"Avatar changed! New look activated.", "Avatar change response")
    
    def handle_chat_input(self, address, message):
        """Handle chat input - check for emergency commands (preserved)"""
        try:
            if isinstance(message, list) and len(message) > 0:
                chat_text = str(message[0]).lower().strip()
                
                if "qwen stop" in chat_text:
                    print("EMERGENCY STOP COMMAND RECEIVED")
                    self.emergency_stop = True
                    self.emergency_reset_all()
                    self.stop_vision_system()
                    return
                
                if "qwen reset" in chat_text:
                    print("RESET COMMAND RECEIVED")
                    self.emergency_reset_all()
                    self.chat("All controls reset to safe state", "Manual reset")
                    return
                    
        except Exception as e:
            print(f"Error processing chat command: {e}")
    
    def emergency_reset_all(self):
        """Emergency reset all OSC controls (preserved)"""
        try:
            print("EMERGENCY RESET: Setting all controls to 0")
            
            # Reset all movement axes to 0.0 (no LookHorizontal - desktop mode uses buttons)
            self.osc_client.send_message("/input/Vertical", 0.0)
            self.osc_client.send_message("/input/Horizontal", 0.0)
            self.osc_client.send_message("/input/LookVertical", 0.0)
            
            # Reset all buttons to 0
            buttons = ["/input/MoveForward", "/input/MoveBackward", "/input/MoveLeft", 
                      "/input/MoveRight", "/input/LookLeft", "/input/LookRight", 
                      "/input/Jump", "/input/Run", "/input/Voice",
                      "/input/ComfortLeft", "/input/ComfortRight", "/input/UseRight", "/input/GrabRight"]
            
            for button in buttons:
                self.osc_client.send_message(button, 0)
            
            print("Emergency reset complete")

            # Give OSC client time to process any pending commands
            print("Waiting 3 seconds for OSC client to process pending commands...")
            import time
            time.sleep(3.0)

            # Stop OSC client subprocess
            self.stop_osc_client()

        except Exception as e:
            print(f"Error during emergency reset: {e}")
    
    # Logging Functions (ALL preserved from original)
    def initialize_log_files(self):
        """Initialize all log files (preserved)"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        try:
            with open(self.chat_log_path, "w", encoding='utf-8') as f:
                f.write(f"# Natural Qwen VRChat Chat Log\n")
                f.write(f"**Session Started:** {timestamp}\n")
                f.write(f"**Purpose:** All chat messages from Natural Qwen\n\n")
                f.write(f"---\n\n")
        except Exception as e:
            print(f"Failed to initialize chat log: {e}")
        
        try:
            with open(self.florence_call_log_path, "w", encoding='utf-8') as f:
                f.write(f"# Florence-2 Call Log\n")
                f.write(f"**Session Started:** {timestamp}\n")
                f.write(f"**Purpose:** All calls to Florence-2 for troubleshooting\n\n")
                f.write(f"---\n\n")
        except Exception as e:
            print(f"Failed to initialize Florence call log: {e}")
        
        try:
            with open(self.fvision_log_path, "w", encoding='utf-8') as f:
                f.write(f"# Florence-2 Vision Results Log\n")
                f.write(f"**Session Started:** {timestamp}\n")
                f.write(f"**Purpose:** All vision analysis results for troubleshooting\n\n")
                f.write(f"---\n\n")
        except Exception as e:
            print(f"Failed to initialize vision results log: {e}")
        
        try:
            with open(self.osc_log_path, "w", encoding='utf-8') as f:
                f.write(f"# Natural Qwen OSC Command Log\n")
                f.write(f"**Session Started:** {timestamp}\n")
                f.write(f"**Purpose:** All OSC commands executed by Natural Qwen\n\n")
                f.write(f"---\n\n")
        except Exception as e:
            print(f"Failed to initialize OSC log: {e}")
        
        try:
            with open(self.qwen_log_path, "w", encoding='utf-8') as f:
                f.write(f"# Qwen Decision Log\n")
                f.write(f"**Session Started:** {timestamp}\n")
                f.write(f"**Purpose:** All Qwen LLM decisions, thoughts, and reasoning\n\n")
                f.write(f"---\n\n")
        except Exception as e:
            print(f"Failed to initialize Qwen log: {e}")
    
    def log_chat_message(self, message, reasoning=""):
        """Log chat messages (preserved)"""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            with open(self.chat_log_path, "a", encoding='utf-8') as f:
                f.write(f"## {timestamp}\n")
                f.write(f"**Message:** {message}\n")
                if reasoning:
                    f.write(f"**Reasoning:** {reasoning}\n")
                f.write(f"\n")
        except Exception as e:
            print(f"Failed to log chat: {e}")
    
    def log_florence_call(self, filepath, context, parameters):
        """Log Florence-2 calls (preserved)"""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            with open(self.florence_call_log_path, "a", encoding='utf-8') as f:
                f.write(f"## {timestamp} - Florence Call\n")
                f.write(f"**Filepath:** {filepath}\n")
                f.write(f"**Context:** {context}\n")
                f.write(f"**Parameters:** {parameters}\n")
                f.write(f"\n")
        except Exception as e:
            print(f"Failed to log Florence call: {e}")
    
    def log_florence_results(self, analysis, context):
        """Log Florence-2 results (preserved)"""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            with open(self.fvision_log_path, "a", encoding='utf-8') as f:
                f.write(f"## {timestamp} - Vision Results\n")
                f.write(f"**Context:** {context}\n")
                
                if 'scene_description' in analysis:
                    f.write(f"**Scene:** {analysis['scene_description']}\n")
                
                if 'detected_objects' in analysis:
                    objects = analysis['detected_objects']
                    f.write(f"**Objects:** {', '.join(objects) if objects else 'None'}\n")
                
                if 'navigation_assessment' in analysis:
                    nav = analysis['navigation_assessment']
                    if nav.get('social_elements'):
                        f.write(f"**Social:** {', '.join(nav['social_elements'])}\n")
                    if nav.get('pathways'):
                        f.write(f"**Pathways:** {', '.join(nav['pathways'])}\n")
                
                f.write(f"\n")
        except Exception as e:
            print(f"Failed to log Florence results: {e}")
    
    def log_osc_command(self, command, parameters):
        """Log OSC commands (preserved)"""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            with open(self.osc_log_path, "a", encoding='utf-8') as f:
                f.write(f"## {timestamp} - {command.upper()}\n")
                f.write(f"**Command:** {command}\n")
                f.write(f"**Parameters:** {json.dumps(parameters)}\n")
                f.write(f"\n")
        except Exception as e:
            print(f"Failed to log OSC command: {e}")
    
    def log_qwen_decision(self, prompt, response, tier, success, scene_context=""):
        """Log Qwen LLM decisions and reasoning"""
        try:
            timestamp = datetime.now().strftime('%H:%M:%S')
            with open(self.qwen_log_path, "a", encoding='utf-8') as f:
                f.write(f"## {timestamp} - Qwen Decision\n")
                f.write(f"**Success:** {success}\n")
                f.write(f"**Tier:** {tier}\n")
                if scene_context:
                    f.write(f"**Scene:** {scene_context}\n")
                f.write(f"**Prompt:** {prompt[:400]}...\n")
                f.write(f"**Response:** {response}\n")
                f.write(f"\n")
        except Exception as e:
            print(f"Failed to log Qwen decision: {e}")

def main():
    """Main function"""
    print("Starting Natural Qwen VRChat Brain...")
    print("Architecture: Lockstep Tick Engine with Frame-Perfect Synchronization")
    
    try:
        brain = NaturalQwenVRChatBrain()
        
        print("\\n=== NATURAL QWEN VRCHAT BRAIN ACTIVE (LOCKSTEP) ===")
        print("Natural consciousness: ACTIVE (Lockstep Synchronization)")
        print("Frame-perfect timing: ENABLED")
        print("Decision making: PURE LLM with frame ID tracking")
        print("Emergency stop: 'qwen stop' in VRChat chat")
        print("Vision-action sync: GUARANTEED (no race conditions)")
        print("===============================================\\n")
        
        # Start lockstep main loop instead of threaded approach
        brain.start_lockstep_consciousness()
        
        if brain.emergency_stop:
            print("\\nNatural Qwen Brain stopped via emergency command")
        elif brain.llm_failure_pause:
            print("\\nNatural Qwen Brain is paused after repeated LLM failures - waiting for instructions")
            
    except KeyboardInterrupt:
        print("\\nKeyboard interrupt - shutting down...")
        if 'brain' in locals():
            brain.emergency_stop = True
            brain.emergency_reset_all()
    except Exception as e:
        print(f"Critical error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Always reset OSC on exit
        if 'brain' in locals():
            brain.emergency_reset_all()
        print("OSC controls reset on exit")

if __name__ == "__main__":
    main()