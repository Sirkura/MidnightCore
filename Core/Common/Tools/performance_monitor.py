#!/usr/bin/env python3
"""
GPU Monitoring and Stability Verification System
Real-time VRAM tracking and telemetry for Midnight Core vision systems
"""

import subprocess
import json
import time
import threading
import csv
import os
from datetime import datetime
from pathlib import Path
from collections import defaultdict, deque
import logging

class GPUMonitor:
    def __init__(self, log_dir="G:/Experimental/Production/MidnightCore/Core/Engine/Logging", warning_threshold=14.8, critical_threshold=15.5):
        """Initialize GPU monitoring system"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Thresholds (GB)
        self.warning_threshold = warning_threshold  # 14.8GB - warn when approaching limit
        self.critical_threshold = critical_threshold  # 15.5GB - critical warning
        
        # Monitoring state
        self.is_monitoring = False
        self.monitor_thread = None
        self.gpu_stats = {}
        self.last_stats_time = 0
        self.warning_count = 0
        self.critical_count = 0
        
        # Telemetry storage (recent history)
        self.vram_history = deque(maxlen=300)  # 5 minutes at 1Hz
        self.telemetry_history = deque(maxlen=300)
        
        # Performance tracking
        self.subsystem_timers = defaultdict(list)
        self.subsystem_counters = defaultdict(int)
        
        # Log files - consolidate everything to Telemetry-GPU.md
        self.telemetry_log_path = self.log_dir / "Telemetry-GPU.md"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize CSV files
        self._init_log_files()
        
        self.logger.info(f"GPU Monitor initialized - Warning: {warning_threshold}GB, Critical: {critical_threshold}GB")
    
    def _init_log_files(self):
        """Initialize consolidated telemetry log"""
        try:
            if not self.telemetry_log_path.exists():
                with open(self.telemetry_log_path, 'w', encoding='utf-8') as f:
                    f.write("# GPU & Performance Telemetry Log\n")
                    f.write("**Real-time GPU performance and stability monitoring**\n\n")
                    f.write("## System Status\n\n")
                    f.write("## VRAM Usage\n\n")
                    f.write("## Performance Timing\n\n")
                    f.write("---\n\n")
                    
        except Exception as e:
            self.logger.error(f"Failed to initialize log files: {e}")
    
    def get_gpu_stats(self):
        """Get current GPU statistics via nvidia-smi"""
        try:
            # Run nvidia-smi with JSON output
            cmd = [
                'nvidia-smi', 
                '--query-gpu=memory.total,memory.used,memory.free,utilization.gpu,utilization.memory,temperature.gpu,power.draw',
                '--format=csv,noheader,nounits'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                line = result.stdout.strip()
                values = [val.strip() for val in line.split(',')]
                
                stats = {
                    'total_mb': float(values[0]),
                    'used_mb': float(values[1]),
                    'free_mb': float(values[2]),
                    'gpu_util': float(values[3]),
                    'mem_util': float(values[4]),
                    'temperature': float(values[5]),
                    'power': float(values[6]),
                    'timestamp': time.time()
                }
                
                # Convert to GB for easier reading
                stats.update({
                    'total_gb': stats['total_mb'] / 1024.0,
                    'used_gb': stats['used_mb'] / 1024.0,
                    'free_gb': stats['free_mb'] / 1024.0
                })
                
                return stats
            else:
                self.logger.error(f"nvidia-smi failed: {result.stderr}")
                return None
                
        except subprocess.TimeoutExpired:
            self.logger.error("nvidia-smi timeout")
            return None
        except Exception as e:
            self.logger.error(f"GPU stats error: {e}")
            return None
    
    def check_stability(self, stats):
        """Check GPU stability and issue warnings if needed"""
        if not stats:
            return "unknown"
        
        used_gb = stats['used_gb']
        temperature = stats['temperature']
        
        # Check VRAM usage
        if used_gb >= self.critical_threshold:
            self.critical_count += 1
            self.logger.error(f"CRITICAL: VRAM usage {used_gb:.1f}GB >= {self.critical_threshold}GB")
            return "critical"
        elif used_gb >= self.warning_threshold:
            self.warning_count += 1
            if self.warning_count % 5 == 1:  # Log every 5th warning to avoid spam
                self.logger.warning(f"HIGH VRAM: {used_gb:.1f}GB >= {self.warning_threshold}GB")
            return "warning"
        else:
            # Reset warning count when back to normal
            if self.warning_count > 0:
                self.warning_count = max(0, self.warning_count - 1)
            return "stable"
        
        # TODO: Add temperature checks if needed
    
    def log_vram_stats(self, stats):
        """Log VRAM statistics to telemetry file"""
        if not stats:
            return
        
        try:
            timestamp = datetime.fromtimestamp(stats['timestamp']).strftime("%H:%M:%S")
            stability_icon = {
                'stable': '[OK]',
                'warning': '[WARN]',
                'critical': '[CRIT]',
                'unknown': '❓'
            }.get(stats.get('stability', 'unknown'), '❓')
            
            vram_entry = (f"**{timestamp}** VRAM: {stats['used_gb']:.1f}/{stats['total_gb']:.1f}GB "
                         f"({stats['mem_util']:.0f}%) | GPU: {stats['gpu_util']:.0f}% | "
                         f"Temp: {stats['temperature']:.0f}°C | Power: {stats['power']:.0f}W {stability_icon}\n\n")
            
            with open(self.telemetry_log_path, 'a', encoding='utf-8') as f:
                f.write(vram_entry)
                
        except Exception as e:
            self.logger.error(f"Failed to log VRAM stats: {e}")
    
    def monitor_loop(self):
        """Main monitoring loop (runs in separate thread)"""
        self.logger.info("GPU monitoring started")
        
        while self.is_monitoring:
            try:
                # Get current stats
                stats = self.get_gpu_stats()
                self.gpu_stats = stats
                self.last_stats_time = time.time()
                
                if stats:
                    # Check stability
                    stability = self.check_stability(stats)
                    stats['stability'] = stability
                    
                    # Store in history
                    self.vram_history.append(stats)
                    
                    # Print periodic status (every 10 seconds) and log VRAM data (every 5 seconds)
                    if len(self.vram_history) % 10 == 0:
                        self._print_status_summary(stats)
                    
                    # Log VRAM stats every 5 seconds to reduce spam
                    if len(self.vram_history) % 5 == 0:
                        self.log_vram_stats(stats)
                    
                    # Log performance summary every 30 seconds
                    if len(self.vram_history) % 30 == 0 and self.subsystem_timers:
                        self.log_performance_summary()
                
                time.sleep(1.0)  # 1Hz monitoring
                
            except Exception as e:
                self.logger.error(f"Monitor loop error: {e}")
                time.sleep(2.0)
        
        self.logger.info("GPU monitoring stopped")
    
    def _print_status_summary(self, stats):
        """Print compact status summary"""
        if not stats:
            return
        
        stability_icon = {
            'stable': '[OK]',
            'warning': '[WARN]',
            'critical': '[CRIT]',
            'unknown': '❓'
        }.get(stats['stability'], '❓')
        
        print(f"GPU: {stats['used_gb']:.1f}/{stats['total_gb']:.1f}GB "
              f"({stats['mem_util']:.0f}%) "
              f"{stats['temperature']:.0f}°C "
              f"{stability_icon} {stats['stability'].upper()}")
    
    def start_monitoring(self):
        """Start GPU monitoring in background thread"""
        if self.is_monitoring:
            self.logger.warning("GPU monitoring already running")
            return
        
        self.is_monitoring = True
        self.monitor_thread = threading.Thread(target=self.monitor_loop, daemon=True)
        self.monitor_thread.start()
        self.logger.info("GPU monitoring thread started")
    
    def stop_monitoring(self):
        """Stop GPU monitoring"""
        self.is_monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        self.logger.info("GPU monitoring stopped")
    
    def get_current_status(self):
        """Get current GPU status for external systems"""
        if not self.gpu_stats:
            return {"status": "unknown", "message": "No GPU data available"}
        
        stats = self.gpu_stats.copy()
        age = time.time() - self.last_stats_time
        
        if age > 5.0:
            return {"status": "stale", "message": f"GPU data is {age:.1f}s old"}
        
        return {
            "status": stats.get('stability', 'unknown'),
            "used_gb": stats['used_gb'],
            "total_gb": stats['total_gb'],
            "utilization": stats['mem_util'],
            "temperature": stats['temperature'],
            "age_seconds": age
        }
    
    def log_subsystem_timing(self, subsystem, operation, latency_ms, memory_mb=None, details=None):
        """Log timing data for subsystem operations"""
        try:
            # Store in memory for analysis
            self.subsystem_timers[f"{subsystem}_{operation}"].append(latency_ms)
            self.subsystem_counters[f"{subsystem}_{operation}"] += 1
            
            # Keep only recent data
            if len(self.subsystem_timers[f"{subsystem}_{operation}"]) > 100:
                self.subsystem_timers[f"{subsystem}_{operation}"] = \
                    self.subsystem_timers[f"{subsystem}_{operation}"][-50:]
            
            # Determine warning level
            warning_level = "[OK] Normal"
            if latency_ms > 2000:
                warning_level = "[CRIT] Critical"
            elif latency_ms > 1000:
                warning_level = "[WARN] Slow"
            
            # Log to markdown file
            timestamp = datetime.now().strftime("%H:%M:%S")
            memory_text = f" | {memory_mb:.1f}MB" if memory_mb else ""
            details_text = f" | {details}" if details else ""
            
            with open(self.telemetry_log_path, 'a', encoding='utf-8') as f:
                f.write(f"**{timestamp}** `{subsystem}_{operation}` {latency_ms:.1f}ms{memory_text}{details_text} {warning_level}\n\n")
                
        except Exception as e:
            self.logger.error(f"Failed to log subsystem timing: {e}")
    
    def get_performance_summary(self):
        """Get performance summary for all tracked subsystems"""
        summary = {}
        
        for key, timings in self.subsystem_timers.items():
            if timings:
                summary[key] = {
                    'count': len(timings),
                    'avg_ms': sum(timings) / len(timings),
                    'p50_ms': sorted(timings)[len(timings) // 2],
                    'p90_ms': sorted(timings)[int(len(timings) * 0.9)],
                    'p99_ms': sorted(timings)[int(len(timings) * 0.99)],
                    'max_ms': max(timings)
                }
        
        return summary
    
    def print_performance_report(self):
        """Print compact performance report"""
        summary = self.get_performance_summary()
        
        if not summary:
            print("No performance data available")
            return
        
        print("\n=== PERFORMANCE SUMMARY ===")
        for subsystem, stats in summary.items():
            print(f"{subsystem}: "
                  f"p50 {stats['p50_ms']:.0f}ms | "
                  f"p90 {stats['p90_ms']:.0f}ms | "
                  f"max {stats['max_ms']:.0f}ms "
                  f"({stats['count']} ops)")
    
    def log_performance_summary(self):
        """Log performance summary to telemetry file"""
        summary = self.get_performance_summary()
        
        if not summary:
            return
            
        try:
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            with open(self.telemetry_log_path, 'a', encoding='utf-8') as f:
                f.write(f"**{timestamp}** PERFORMANCE SUMMARY:\n\n")
                
                for subsystem, stats in summary.items():
                    perf_icon = "[CRIT]" if stats['p90_ms'] > 2000 else "[WARN]" if stats['p90_ms'] > 1000 else "[OK]"
                    f.write(f"- `{subsystem}`: p50 {stats['p50_ms']:.0f}ms | p90 {stats['p90_ms']:.0f}ms | max {stats['max_ms']:.0f}ms ({stats['count']} ops) {perf_icon}\n")
                
                f.write("\n---\n\n")
                
        except Exception as e:
            self.logger.error(f"Failed to log performance summary: {e}")

# Global monitor instance for easy access
_gpu_monitor = None

def get_gpu_monitor():
    """Get or create global GPU monitor instance"""
    global _gpu_monitor
    if _gpu_monitor is None:
        _gpu_monitor = GPUMonitor()
    return _gpu_monitor

def start_monitoring():
    """Convenience function to start GPU monitoring"""
    monitor = get_gpu_monitor()
    monitor.start_monitoring()
    return monitor

def log_timing(subsystem, operation, latency_ms, memory_mb=None, details=None):
    """Convenience function to log timing data"""
    monitor = get_gpu_monitor()
    monitor.log_subsystem_timing(subsystem, operation, latency_ms, memory_mb, details)

if __name__ == "__main__":
    # Test monitoring
    print("Starting GPU monitoring test...")
    
    monitor = GPUMonitor()
    monitor.start_monitoring()
    
    try:
        # Run for 30 seconds
        for i in range(30):
            time.sleep(1)
            status = monitor.get_current_status()
            if i % 5 == 0:
                print(f"Test {i}: {status}")
    
    except KeyboardInterrupt:
        print("\nTest interrupted")
    
    finally:
        monitor.stop_monitoring()
        print("GPU monitoring test complete")