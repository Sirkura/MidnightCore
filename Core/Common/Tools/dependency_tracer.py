#!/usr/bin/env python3
"""
Enhanced Dependency Tracer for MidnightCore
Integrates with three-tier logging bus for comprehensive file tracking
"""
import sys
import os
import time
import threading
from datetime import datetime
from pathlib import Path

# Import our logging bus
try:
    from logging_bus import log_filemap, log_engine, log_deep, TracingConfig
except ImportError:
    # Fallback if logging_bus not available
    def log_filemap(*args, **kwargs): pass
    def log_engine(*args, **kwargs): pass
    def log_deep(*args, **kwargs): pass
    class TracingConfig:
        FILE_MAP = False
        DEEP_TRACE = False

class DependencyTracer:
    """Enhanced real-time file access tracer with logging bus integration"""
    
    def __init__(self):
        self.log_dir = Path("G:/Experimental/Production/MidnightCore/Core/Engine/Logging/.engine_log")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking sets
        self.files_accessed = set()
        self.imports_used = set()
        self.lock = threading.Lock()
        
        # Original hooks
        self._original_import = __builtins__.__import__
        self._original_trace = sys.gettrace()
        
        print(f"[TRACER] Enhanced dependency tracer initialized - integrating with logging bus")
    
    def start_tracing(self):
        """Start enhanced real-time tracing with logging bus integration"""
        # Hook import system
        __builtins__.__import__ = self._trace_import
        
        # Hook execution tracing for deep mode
        if TracingConfig.DEEP_TRACE:
            sys.settrace(self._trace_calls)
        
        log_engine("tracer.started", 
                  file_map_active=TracingConfig.FILE_MAP,
                  deep_trace_active=TracingConfig.DEEP_TRACE)
        
        print("[TRACER] Enhanced tracing started with logging bus integration")
    
    def _trace_import(self, name, globals=None, locals=None, fromlist=(), level=0):
        """Enhanced import tracing with logging bus integration"""
        try:
            # Get calling file
            frame = sys._getframe(1)
            caller_file = frame.f_code.co_filename
            caller_short = os.path.relpath(caller_file, ".")
            
            # Only trace MidnightCore imports
            if "MidnightCore" in caller_file or "MidnightCore" in name:
                with self.lock:
                    self.imports_used.add(name)
                    
                    # Log to file map if active
                    log_filemap(caller_file, event="import", module=name, caller=caller_short)
                    
                    # Log to engine for basic tracking
                    log_engine("import.traced", module=name, caller=caller_short)
                    
                    # Deep trace if enabled
                    if TracingConfig.DEEP_TRACE:
                        log_deep("import.detailed", 
                               module=name, 
                               caller=caller_short,
                               fromlist=list(fromlist) if fromlist else [],
                               level=level)
        
        except Exception as e:
            log_engine("tracer.error", location="_trace_import", error=str(e)[:200])
        
        return self._original_import(name, globals, locals, fromlist, level)
    
    def _trace_calls(self, frame, event, arg):
        """Enhanced call tracing with logging bus integration - DEEP TRACE ONLY"""
        try:
            if event == 'call':
                filename = frame.f_code.co_filename
                function_name = frame.f_code.co_name
                
                # Only trace MidnightCore files
                if "MidnightCore" in filename:
                    file_short = os.path.relpath(filename, ".")
                    
                    # Track new files for file map
                    if filename not in self.files_accessed:
                        with self.lock:
                            self.files_accessed.add(filename)
                            log_filemap(filename, event="first_access", function=function_name)
                    
                    # Deep trace every function call
                    if TracingConfig.DEEP_TRACE:
                        log_deep("function.call", 
                               file=file_short, 
                               function=function_name,
                               line=frame.f_lineno)
        
        except Exception as e:
            log_engine("tracer.error", location="_trace_calls", error=str(e)[:200])
        
        return self._trace_calls
    
    def stop_tracing(self):
        """Stop enhanced tracing and generate summary"""
        # Restore hooks
        __builtins__.__import__ = self._original_import
        sys.settrace(self._original_trace)
        
        # Log summary to engine
        log_engine("tracer.stopped", 
                  files_accessed=len(self.files_accessed),
                  imports_used=len(self.imports_used))
        
        # Generate human-readable summary
        summary_file = self.log_dir / f"tracer_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(summary_file, 'w') as f:
            f.write(f"# Enhanced Dependency Trace Summary - {datetime.now()}\n\n")
            f.write(f"## Files Actually Used ({len(self.files_accessed)} files)\n")
            for file_path in sorted([os.path.relpath(f, ".") for f in self.files_accessed]):
                f.write(f"- {file_path}\n")
            
            f.write(f"\n## Imports Used ({len(self.imports_used)} imports)\n")
            for import_name in sorted(self.imports_used):
                f.write(f"- {import_name}\n")
        
        print(f"[TRACER] Enhanced tracing stopped")
        print(f"[TRACER] Summary: {summary_file}")
        print(f"[TRACER] Files accessed: {len(self.files_accessed)}")
        print(f"[TRACER] Imports used: {len(self.imports_used)}")

# Global tracer
_global_tracer = None

def start_dependency_tracing():
    """Start global dependency tracing"""
    global _global_tracer
    if _global_tracer is None:
        _global_tracer = DependencyTracer()
        _global_tracer.start_tracing()
    return _global_tracer

def stop_dependency_tracing():
    """Stop global dependency tracing"""
    global _global_tracer
    if _global_tracer is not None:
        _global_tracer.stop_tracing()
        _global_tracer = None

# Auto-start dependency tracing when imported (if file map is enabled)
if TracingConfig.FILE_MAP and _global_tracer is None:
    _global_tracer = DependencyTracer()
    _global_tracer.start_tracing()
    log_engine("tracer.auto_started", reason="FILE_MAP_enabled")

if __name__ == "__main__":
    # Test the enhanced tracer
    print("Testing Enhanced Dependency Tracer...")
    
    tracer = start_dependency_tracing()
    
    # Simulate some activity
    import time
    import json  # This should be traced
    
    log_engine("test.activity", message="Testing tracer functionality")
    time.sleep(1)
    
    stop_dependency_tracing()
    print("Test complete - check .engine_log directory")