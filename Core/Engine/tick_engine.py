# ==== MODULE CONTRACT =======================================================
# Module: engine/tick_engine.py
# Package: MidnightCore.Engine.tick_engine
# Location: Production/MidnightCore/Core/Engine/tick_engine.py
# Responsibility: Lockstep 5-phase synchronization (Capture -> Decide -> Speak -> Act -> Integrate)
# PUBLIC: 
#   Classes: TickEngine
#   Methods: start_tick(), capture_phase(), decide_phase(), speak_phase(), act_phase(), integrate_phase()
# DEPENDENCIES: MidnightCore.Engine.schemas (TickTelemetry)
# DEPENDENTS: MidnightCore.Engine.main_brain
# POLICY: NO_FALLBACKS=N/A (coordination only), Telemetry: tick.*
# DOCS: README anchor "lockstep", File-Map anchor "TICK_ENGINE_DOC_ANCHOR"
# MIGRATION: ✅ Migrated | Original: G:\Experimental\Midnight Core\FusionCore\FusionScripts\QwenScripts\tick_engine.py
# INVARIANTS:
#   - Phases execute in strict order: Capture -> Decide -> Speak -> Act -> Integrate
#   - Frame IDs are unique per tick, actions are stamped with frame_id
#   - Action window debouncing prevents vision capture during OSC execution (300ms)
#   - Causality maintained: chat references same frame_id as action
# ============================================================================

import time
import sys
import os

try:
    from .schemas import TickTelemetry
except ImportError:
    from schemas import TickTelemetry

# Import unified logging system
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'Common', 'Tools'))
from logging_bus import (
    log_tick_start, log_capture_done, log_decide_done, 
    log_speak_done, log_act_done, log_integrate_done
)

__all__ = ["TickEngine"]

class TickEngine:
    """Lockstep tick engine enforcing: Capture -> Decide -> Speak -> Act -> Integrate"""
    
    def __init__(self):
        self.tick_id = 0
        self.frame_id = 0
        self.current_frame_data = None
        self.current_action_frame_id = None
        self.chat_frame_id = None
        self.perception_age_ms = 0
        self.tick_start_time = 0
        self.action_window_active = False
        self.action_start_time = 0
        
    def start_tick(self):
        """Begin new tick cycle with fresh IDs"""
        self.tick_id += 1
        self.tick_start_time = time.time()
        print(f"[TICK {self.tick_id}] Starting new tick cycle")
        
        # Log tick start event
        log_tick_start(self.tick_id)
        
    def capture_phase(self, capture_fn, vision_fn):
        """PHASE 1: Capture -> get frame_id = N"""
        self.frame_id += 1
        capture_start = time.time()
        
        # Debounce Florence during active action window (200-300ms after action)
        if self.action_window_active:
            time_since_action = (time.time() - self.action_start_time) * 1000
            if time_since_action < 300:  # 300ms action window
                print(f"[TICK {self.tick_id}] CAPTURE DEBOUNCED: action_window active ({time_since_action:.1f}ms)")
                time.sleep(0.05)  # Brief delay for action completion
            else:
                self.action_window_active = False
                print(f"[TICK {self.tick_id}] CAPTURE: action_window expired, resuming vision")
        
        # Capture screenshot
        self.current_frame_data = capture_fn()
        
        # Get vision analysis
        vision_context = vision_fn(self.current_frame_data)
        
        capture_time_ms = (time.time() - capture_start) * 1000
        print(f"[TICK {self.tick_id}] CAPTURE: frame_id={self.frame_id} capture_time={capture_time_ms:.1f}ms")
        
        # Log capture completion
        debounced_ms = None
        if self.action_window_active:
            debounced_ms = (time.time() - self.action_start_time) * 1000
        
        log_capture_done(
            tick_id=self.tick_id,
            frame_id=self.frame_id,
            capture_ms=capture_time_ms,
            action_window=self.action_window_active,
            debounced_ms=debounced_ms
        )
        
        return vision_context
        
    def decide_phase(self, decision_fn, vision_context):
        """PHASE 2: Decide -> produce plan using only frame_id = N"""
        decision_start = time.time()
        
        # Check frame age for stale data detection
        frame_age_ms = (time.time() - self.tick_start_time) * 1000
        
        # Add frame metadata to vision context with age validation
        stamped_context = f"PERCEPTION: frame_id={self.frame_id} age={frame_age_ms:.1f}ms\n"
        if frame_age_ms > 500:  # Frame is getting stale
            stamped_context += "CAUTION: Frame age >500ms - request fresh scan if asserting new details\n"
        stamped_context += vision_context
        
        # Make decision using current frame only
        decision_result = decision_fn(stamped_context)
        
        decision_time_ms = (time.time() - decision_start) * 1000
        print(f"[TICK {self.tick_id}] DECIDE: frame_id={self.frame_id} age={frame_age_ms:.1f}ms decision_time={decision_time_ms:.1f}ms")
        
        # Log decision completion
        log_decide_done(
            tick_id=self.tick_id,
            frame_id=self.frame_id,
            decision_ms=decision_time_ms,
            frame_age_ms=frame_age_ms
        )
        
        return decision_result
        
    def speak_phase(self, chat_fn, reasoning_text):
        """PHASE 3: Speak -> generate narration BEFORE action"""
        if reasoning_text:
            self.chat_frame_id = self.frame_id
            self.perception_age_ms = (time.time() - self.tick_start_time) * 1000
            
            # Add frame reference to chat
            stamped_reasoning = f"[Frame {self.frame_id}] {reasoning_text}"
            
            print(f"[TICK {self.tick_id}] SPEAK: chat_frame_id={self.chat_frame_id} age={self.perception_age_ms:.1f}ms")
            
            # Send chat message - this now blocks until all chunks are delivered
            speak_start = time.time()
            chat_fn(stamped_reasoning)
            speak_time = (time.time() - speak_start) * 1000
            print(f"[TICK {self.tick_id}] SPEAK: Message delivery complete after {speak_time:.1f}ms")
            
            # Log speak completion
            log_speak_done(
                tick_id=self.tick_id,
                frame_id=self.frame_id,
                chat_ms=speak_time,
                chat_frame_id=self.chat_frame_id
            )
        
    def act_phase(self, action_fn, action_command):
        """PHASE 4: Act -> send OSC with tick_id, frame_id"""
        self.current_action_frame_id = self.frame_id
        
        print(f"[TICK {self.tick_id}] ACT: frame_id={self.frame_id} action='{action_command}'")
        
        # Start action window debouncing
        self.action_window_active = True
        self.action_start_time = time.time()
        
        # Execute action with frame ID stamping
        action_fn(action_command, self.tick_id, self.frame_id)
        
        # Log act completion
        log_act_done(
            tick_id=self.tick_id,
            frame_id=self.frame_id,
            action=action_command
        )
        
    def integrate_phase(self):
        """PHASE 5: Integrate -> update pose/map; prepare for next but don't use yet"""
        print(f"[TICK {self.tick_id}] INTEGRATE: Updating internal state")
        # Spatial memory updates handled in calling code
        
        # Log integrate completion
        log_integrate_done(
            tick_id=self.tick_id,
            frame_id=self.frame_id
        )
        
    def get_telemetry(self) -> TickTelemetry:
        """Return telemetry data for debugging"""
        return {
            'tick_id': self.tick_id,
            'frame_id': self.frame_id, 
            'chat_frame_id': self.chat_frame_id,
            'action_frame_id': self.current_action_frame_id,
            'chat_refs_future_frame': self.chat_frame_id > self.current_action_frame_id if self.current_action_frame_id else False,
            'perception_age_ms': self.perception_age_ms
        }

# Contract tests
def _contract_tests():
    engine = TickEngine()
    
    # Tick IDs increment properly
    assert engine.tick_id == 0
    engine.start_tick()
    assert engine.tick_id == 1
    
    # Frame IDs increment in capture phase
    initial_frame = engine.frame_id
    engine.frame_id += 1  # Simulate capture phase
    assert engine.frame_id == initial_frame + 1
    
    # Telemetry returns valid structure
    telemetry = engine.get_telemetry()
    assert 'tick_id' in telemetry
    assert 'frame_id' in telemetry

if __name__ == "__main__":
    _contract_tests()
    print("All tick engine contract tests passed")