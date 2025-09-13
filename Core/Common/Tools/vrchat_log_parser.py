#!/usr/bin/env python3
"""
VRChat Log Parser - Extract World Information and Events
=========================================================

PURPOSE: Parse VRChat output logs to extract real-time world IDs, transitions, 
         OSC events, and other calibration-relevant data for the auto-calibration system.

CONTRACT:
- Parses timestamped VRChat log files for world transitions and OSC data
- Provides real VRChat world IDs (wrld_...) for accurate calibration file naming
- Monitors current world state and detects world changes
- Extracts OSC configuration and parameter information

INTEGRATION: 
- Used by auto_calibrator.py to get real world IDs instead of fingerprinting
- Replaces mock world detection with actual VRChat log parsing
"""

import os
import re
import glob
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import json

@dataclass
class WorldTransition:
    """Represents a world change event from VRChat logs"""
    timestamp: datetime
    world_id: str
    world_name: Optional[str] = None
    instance_details: Optional[str] = None
    
@dataclass
class OSCEvent:
    """Represents OSC-related log entries"""
    timestamp: datetime
    event_type: str  # 'enabled', 'service_advertised', 'parameter_changed'
    details: str

class VRChatLogParser:
    """
    Parses VRChat output logs to extract world transitions and OSC events
    
    Key Features:
    - Real-time world ID extraction from logs
    - World transition detection with timestamps
    - OSC configuration monitoring
    - Current world state tracking
    """
    
    def __init__(self, log_directory: str = r"C:\Users\sable\AppData\LocalLow\VRChat\VRChat"):
        self.log_directory = log_directory
        self.current_world_id: Optional[str] = None
        self.current_log_file: Optional[str] = None
        
        # Regex patterns for parsing
        self.world_patterns = {
            'entering': re.compile(r'Joining (wrld_[a-f0-9-]+)'),
            'destination': re.compile(r'Going to.*?Location: (wrld_[a-f0-9-]+)'),
            'fetching': re.compile(r'Fetching world information for (wrld_[a-f0-9-]+)'),
            'loaded': re.compile(r'after world loaded \[(wrld_[a-f0-9-]+)\]')
        }
        
        self.osc_patterns = {
            'enabled': re.compile(r'OSC enabled: (True|False)'),
            'service': re.compile(r'Advertising Service.*OSC on (\d+)'),
            'query': re.compile(r'Advertising Service.*OSCQuery on (\d+)')
        }
        
    def get_latest_log_file(self) -> Optional[str]:
        """Find the most recent VRChat output log file"""
        pattern = os.path.join(self.log_directory, "output_log_*.txt")
        log_files = glob.glob(pattern)
        
        if not log_files:
            return None
            
        # Sort by modification time, most recent first
        log_files.sort(key=os.path.getmtime, reverse=True)
        return log_files[0]
    
    def parse_timestamp(self, line: str) -> Optional[datetime]:
        """Extract timestamp from VRChat log line"""
        # VRChat format: "2025.09.01 16:15:06 Debug      -  "
        match = re.match(r'^(\d{4}\.\d{2}\.\d{2} \d{2}:\d{2}:\d{2})', line)
        if match:
            try:
                return datetime.strptime(match.group(1), "%Y.%m.%d %H:%M:%S")
            except ValueError:
                pass
        return None
    
    def extract_world_transitions(self, log_file: str, 
                                since: Optional[datetime] = None) -> List[WorldTransition]:
        """
        Extract all world transitions from a log file
        
        Args:
            log_file: Path to VRChat log file
            since: Only return transitions after this timestamp
            
        Returns:
            List of WorldTransition objects
        """
        transitions = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    timestamp = self.parse_timestamp(line)
                    if not timestamp or (since and timestamp < since):
                        continue
                    
                    # Check for world transition patterns
                    for pattern_name, pattern in self.world_patterns.items():
                        match = pattern.search(line)
                        if match:
                            world_id = match.group(1)
                            
                            # Extract instance details for joining events
                            instance_details = None
                            if pattern_name == 'entering' and ':' in line:
                                # Full instance string like: wrld_xxx:12345~private(usr_yyy)~region(us)
                                instance_match = re.search(r'(wrld_[a-f0-9-]+:[^)]+\))', line)
                                if instance_match:
                                    instance_details = instance_match.group(1)
                            
                            transition = WorldTransition(
                                timestamp=timestamp,
                                world_id=world_id,
                                instance_details=instance_details
                            )
                            transitions.append(transition)
                            break
                            
        except Exception as e:
            print(f"ERROR: Failed to parse log file {log_file}: {e}")
            
        return transitions
    
    def get_current_world_from_logs(self) -> Optional[str]:
        """
        Determine the current world ID from the latest log entries
        
        Returns:
            Current world ID or None if not found
        """
        log_file = self.get_latest_log_file()
        if not log_file:
            return None
        
        # Look for recent world transitions (last 10 minutes)
        since = datetime.now() - timedelta(minutes=10)
        transitions = self.extract_world_transitions(log_file, since)
        
        if transitions:
            # Return the most recent world transition
            latest = max(transitions, key=lambda t: t.timestamp)
            self.current_world_id = latest.world_id
            return latest.world_id
        
        # Fallback: scan entire latest log for the last world loaded
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                
                # Search backwards for world loaded messages
                for line in reversed(lines):
                    match = self.world_patterns['loaded'].search(line)
                    if match:
                        world_id = match.group(1)
                        self.current_world_id = world_id
                        return world_id
                        
        except Exception as e:
            print(f"ERROR: Failed to scan log file for current world: {e}")
            
        return None
    
    def get_current_world_id(self) -> Optional[str]:
        """
        Get current world ID (convenience method matching auto_calibrator expectations)
        
        Returns:
            Current world ID or None if not found
        """
        return self.get_current_world_from_logs()
    
    def extract_osc_events(self, log_file: str, 
                          since: Optional[datetime] = None) -> List[OSCEvent]:
        """Extract OSC-related events from log file"""
        events = []
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    timestamp = self.parse_timestamp(line)
                    if not timestamp or (since and timestamp < since):
                        continue
                    
                    # Check for OSC patterns
                    for pattern_name, pattern in self.osc_patterns.items():
                        match = pattern.search(line)
                        if match:
                            event = OSCEvent(
                                timestamp=timestamp,
                                event_type=pattern_name,
                                details=line.strip()
                            )
                            events.append(event)
                            break
                            
        except Exception as e:
            print(f"ERROR: Failed to extract OSC events from {log_file}: {e}")
            
        return events
    
    def monitor_world_changes(self, callback_fn=None) -> bool:
        """
        Check if the current world has changed since last check
        
        Args:
            callback_fn: Optional function to call with (old_world, new_world) when change detected
            
        Returns:
            True if world changed, False otherwise
        """
        previous_world = self.current_world_id
        current_world = self.get_current_world_from_logs()
        
        if current_world != previous_world:
            if callback_fn:
                callback_fn(previous_world, current_world)
            return True
        return False
    
    def get_world_info_summary(self) -> Dict:
        """Get summary of current world state and recent activity"""
        current_world = self.get_current_world_from_logs()
        log_file = self.get_latest_log_file()
        
        summary = {
            'current_world_id': current_world,
            'log_file': os.path.basename(log_file) if log_file else None,
            'last_updated': datetime.now().isoformat()
        }
        
        if log_file:
            # Get recent transitions
            since = datetime.now() - timedelta(minutes=30)
            recent_transitions = self.extract_world_transitions(log_file, since)
            summary['recent_transitions'] = [
                {
                    'timestamp': t.timestamp.isoformat(),
                    'world_id': t.world_id,
                    'instance_details': t.instance_details
                }
                for t in recent_transitions[-5:]  # Last 5 transitions
            ]
            
            # Get OSC status
            recent_osc = self.extract_osc_events(log_file, since)
            summary['osc_events'] = [
                {
                    'timestamp': e.timestamp.isoformat(),
                    'event_type': e.event_type,
                    'details': e.details
                }
                for e in recent_osc[-3:]  # Last 3 OSC events
            ]
        
        return summary

def test_vrchat_log_parser():
    """Test the VRChat log parser with current log files"""
    parser = VRChatLogParser()
    
    print("VRChat Log Parser Test")
    print("=" * 50)
    
    # Test latest log file detection
    latest_log = parser.get_latest_log_file()
    print(f"Latest log file: {latest_log}")
    
    if latest_log:
        # Test current world detection
        current_world = parser.get_current_world_from_logs()
        print(f"Current world ID: {current_world}")
        
        # Test world transitions
        since = datetime.now() - timedelta(hours=1)
        transitions = parser.extract_world_transitions(latest_log, since)
        print(f"Recent transitions ({len(transitions)}):")
        for t in transitions[-3:]:
            print(f"  {t.timestamp}: {t.world_id}")
            if t.instance_details:
                print(f"    Instance: {t.instance_details}")
        
        # Test OSC events
        osc_events = parser.extract_osc_events(latest_log, since)
        print(f"Recent OSC events ({len(osc_events)}):")
        for e in osc_events[-3:]:
            print(f"  {e.timestamp}: {e.event_type} - {e.details}")
        
        # Test summary
        summary = parser.get_world_info_summary()
        print(f"World info summary:")
        print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    test_vrchat_log_parser()