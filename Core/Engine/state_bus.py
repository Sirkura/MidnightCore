"""
State Bus - Pub/Sub Communication System
Handles inter-process communication for vision system components
"""

import json
import threading
import time
import queue
from typing import Dict, Any, Optional, List, Callable
import zmq

class InProcStateBus:
    """
    In-process state bus using threading locks
    Fast communication for components in same process
    """
    
    def __init__(self):
        self._state = {}
        self._lock = threading.RLock()
        self._subscribers = []
        self._message_queues = {}
    
    def set_state(self, topic: str, data: Dict[str, Any]) -> None:
        """
        Set state for a topic
        
        Args:
            topic: Topic name (e.g., 'vision.state')
            data: State data dictionary
        """
        with self._lock:
            self._state[topic] = {
                **data,
                '_timestamp': time.time()
            }
            
            # Notify subscribers
            for callback in self._subscribers:
                try:
                    callback(topic, data)
                except Exception as e:
                    print(f"Warning: subscriber callback failed: {e}")
    
    def get_state(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Get latest state for a topic
        
        Args:
            topic: Topic name
            
        Returns:
            State dictionary or None if not found
        """
        with self._lock:
            return self._state.get(topic)
    
    def get_all_topics(self) -> List[str]:
        """Get list of all available topics"""
        with self._lock:
            return list(self._state.keys())
    
    def subscribe(self, callback: Callable[[str, Dict], None]) -> None:
        """
        Subscribe to state updates
        
        Args:
            callback: Function called on updates (topic, data)
        """
        with self._lock:
            self._subscribers.append(callback)
    
    def publish_message(self, topic: str, message: Dict[str, Any]) -> None:
        """
        Publish message to topic queue
        
        Args:
            topic: Topic name 
            message: Message dictionary
        """
        with self._lock:
            if topic not in self._message_queues:
                self._message_queues[topic] = queue.Queue(maxsize=10)
            
            try:
                self._message_queues[topic].put_nowait({
                    **message,
                    '_timestamp': time.time()
                })
            except queue.Full:
                # Drop oldest message
                try:
                    self._message_queues[topic].get_nowait()
                    self._message_queues[topic].put_nowait({
                        **message,
                        '_timestamp': time.time()
                    })
                except queue.Empty:
                    pass
    
    def get_message(self, topic: str, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
        """
        Get next message from topic queue
        
        Args:
            topic: Topic name
            timeout: Timeout in seconds (0 = non-blocking)
            
        Returns:
            Message dictionary or None
        """
        with self._lock:
            if topic not in self._message_queues:
                return None
            
            try:
                if timeout > 0:
                    return self._message_queues[topic].get(timeout=timeout)
                else:
                    return self._message_queues[topic].get_nowait()
            except queue.Empty:
                return None

class ZMQStateBus:
    """
    ZMQ-based state bus for cross-process communication
    More robust for distributed components
    """
    
    def __init__(self, pub_addr: str = "tcp://127.0.0.1:5557", 
                 sub_addr: str = "tcp://127.0.0.1:5558"):
        self.pub_addr = pub_addr
        self.sub_addr = sub_addr
        
        # ZMQ context and sockets
        self.context = zmq.Context()
        self.pub_socket = None
        self.sub_socket = None
        
        # Local state cache
        self._state_cache = {}
        self._cache_lock = threading.RLock()
        
        # Subscriber thread
        self._sub_thread = None
        self._running = False
    
    def start_publisher(self) -> bool:
        """
        Start publisher socket
        
        Returns:
            True if successful
        """
        try:
            self.pub_socket = self.context.socket(zmq.PUB)
            self.pub_socket.bind(self.pub_addr)
            time.sleep(0.1)  # Let socket establish
            print(f"SUCCESS: ZMQ Publisher started on {self.pub_addr}")
            return True
        except Exception as e:
            print(f"FAILED: Failed to start ZMQ publisher: {e}")
            return False
    
    def start_subscriber(self, topics: List[str] = None) -> bool:
        """
        Start subscriber with background thread
        
        Args:
            topics: List of topics to subscribe to (None = all)
            
        Returns:
            True if successful
        """
        try:
            self.sub_socket = self.context.socket(zmq.SUB)
            self.sub_socket.connect(self.sub_addr)
            
            # Subscribe to topics
            if topics is None:
                self.sub_socket.setsockopt(zmq.SUBSCRIBE, b'')  # All topics
            else:
                for topic in topics:
                    self.sub_socket.setsockopt(zmq.SUBSCRIBE, topic.encode('utf-8'))
            
            # Start subscriber thread
            self._running = True
            self._sub_thread = threading.Thread(target=self._subscriber_loop, daemon=True)
            self._sub_thread.start()
            
            print(f"SUCCESS: ZMQ Subscriber started on {self.sub_addr}")
            return True
            
        except Exception as e:
            print(f"FAILED: Failed to start ZMQ subscriber: {e}")
            return False
    
    def _subscriber_loop(self):
        """Background thread for receiving messages"""
        while self._running:
            try:
                # Non-blocking receive with timeout
                topic_bytes = self.sub_socket.recv(zmq.NOBLOCK)
                data_bytes = self.sub_socket.recv(zmq.NOBLOCK)
                
                topic = topic_bytes.decode('utf-8')
                data = json.loads(data_bytes.decode('utf-8'))
                
                # Update cache
                with self._cache_lock:
                    self._state_cache[topic] = data
                    
            except zmq.Again:
                time.sleep(0.001)  # 1ms sleep
                continue
            except Exception as e:
                if self._running:
                    print(f"Warning: ZMQ subscriber error: {e}")
                time.sleep(0.01)
    
    def publish_state(self, topic: str, data: Dict[str, Any]) -> bool:
        """
        Publish state to topic
        
        Args:
            topic: Topic name
            data: State data
            
        Returns:
            True if successful
        """
        if self.pub_socket is None:
            return False
        
        try:
            # Add timestamp
            state_with_ts = {
                **data,
                '_timestamp': time.time()
            }
            
            # Send multipart message: topic, data
            self.pub_socket.send_string(topic, zmq.SNDMORE)
            self.pub_socket.send_string(json.dumps(state_with_ts))
            return True
            
        except Exception as e:
            print(f"Warning: ZMQ publish failed: {e}")
            return False
    
    def get_state(self, topic: str) -> Optional[Dict[str, Any]]:
        """
        Get latest cached state for topic
        
        Args:
            topic: Topic name
            
        Returns:
            State dictionary or None
        """
        with self._cache_lock:
            return self._state_cache.get(topic)
    
    def stop(self):
        """Stop subscriber and cleanup"""
        self._running = False
        
        if self._sub_thread and self._sub_thread.is_alive():
            self._sub_thread.join(timeout=1.0)
        
        if self.sub_socket:
            self.sub_socket.close()
        if self.pub_socket:
            self.pub_socket.close()
        
        self.context.term()

# Global instances
_inproc_bus = InProcStateBus()
_zmq_bus = None

# Configuration
USE_ZMQ = True  # Set to False for in-process only
ZMQ_PUB_ADDR = "tcp://127.0.0.1:5557"
ZMQ_SUB_ADDR = "tcp://127.0.0.1:5557"  # Same as pub for simplicity

def init_state_bus(use_zmq: bool = USE_ZMQ) -> bool:
    """
    Initialize state bus system
    
    Args:
        use_zmq: Whether to use ZMQ for cross-process communication
        
    Returns:
        True if successful
    """
    global _zmq_bus
    
    if use_zmq:
        _zmq_bus = ZMQStateBus(ZMQ_PUB_ADDR, ZMQ_SUB_ADDR)
        if not _zmq_bus.start_publisher():
            print("Warning: ZMQ publisher failed, falling back to in-process only")
            _zmq_bus = None
            return False
        
        # Start subscriber for topics we care about
        topics = ['vision.state', 'vision.facts', 'vision.inspect']
        if not _zmq_bus.start_subscriber(topics):
            print("Warning: ZMQ subscriber failed")
            _zmq_bus.stop()
            _zmq_bus = None
            return False
    
    print(f"SUCCESS: State bus initialized ({'ZMQ' if _zmq_bus else 'in-process'})")
    return True

def set_state(topic: str, data: Dict[str, Any]) -> None:
    """
    Set/publish state for a topic
    
    Args:
        topic: Topic name (e.g., 'vision.state')
        data: State data dictionary
    """
    # Always update in-process bus
    _inproc_bus.set_state(topic, data)
    
    # Also publish via ZMQ if available
    if _zmq_bus:
        _zmq_bus.publish_state(topic, data)

def get_state(topic: str) -> Optional[Dict[str, Any]]:
    """
    Get latest state for a topic
    
    Args:
        topic: Topic name
        
    Returns:
        State dictionary or None if not found
    """
    # Try ZMQ cache first (more up-to-date for cross-process)
    if _zmq_bus:
        state = _zmq_bus.get_state(topic)
        if state is not None:
            return state
    
    # Fallback to in-process
    return _inproc_bus.get_state(topic)

def publish_message(topic: str, message: Dict[str, Any]) -> None:
    """
    Publish message to topic queue
    
    Args:
        topic: Topic name (e.g., 'vision.inspect')
        message: Message dictionary
    """
    _inproc_bus.publish_message(topic, message)
    
    # For ZMQ, treat as state update
    if _zmq_bus:
        _zmq_bus.publish_state(topic, message)

def get_message(topic: str, timeout: float = 0.0) -> Optional[Dict[str, Any]]:
    """
    Get next message from topic queue
    
    Args:
        topic: Topic name
        timeout: Timeout in seconds (0 = non-blocking)
        
    Returns:
        Message dictionary or None
    """
    return _inproc_bus.get_message(topic, timeout)

def shutdown_state_bus():
    """Clean shutdown of state bus"""
    global _zmq_bus
    
    if _zmq_bus:
        _zmq_bus.stop()
        _zmq_bus = None
    
    print("SUCCESS: State bus shutdown complete")

# Convenience functions for common topics

def publish_vision_state(front_m: float, left_m: float, right_m: float, 
                        edge_risk: float, tilt_deg: float, mean_flow: float, 
                        ttc_s: float, **kwargs) -> None:
    """Publish vision state with required fields"""
    state = {
        'front_m': front_m,
        'left_m': left_m, 
        'right_m': right_m,
        'edge_risk': edge_risk,
        'tilt_deg': tilt_deg,
        'mean_flow': mean_flow,
        'ttc_s': ttc_s,
        **kwargs
    }
    set_state('vision.state', state)

def get_vision_state() -> Optional[Dict[str, Any]]:
    """Get latest vision state"""
    return get_state('vision.state')

def publish_vision_facts(facts: List[Dict[str, Any]]) -> None:
    """Publish Florence analysis facts"""
    set_state('vision.facts', {'facts': facts, 'count': len(facts)})

def get_vision_facts() -> List[Dict[str, Any]]:
    """Get latest vision facts"""
    state = get_state('vision.facts')
    return state.get('facts', []) if state else []

# Test function
def test_state_bus():
    """Test state bus functionality"""
    print("TESTING: State Bus...")
    
    # Initialize
    init_state_bus(use_zmq=False)  # Test in-process first
    
    # Test vision state
    publish_vision_state(
        front_m=2.5, left_m=1.8, right_m=3.2,
        edge_risk=0.15, tilt_deg=2.1, mean_flow=0.05, ttc_s=12.3
    )
    
    state = get_vision_state()
    assert state is not None
    assert state['front_m'] == 2.5
    
    # Test facts
    facts = [
        {'label': 'person', 'bearing_deg': -28, 'conf': 0.83, 'dist_m': 0.9},
        {'label': 'chair', 'bearing_deg': 15, 'conf': 0.76, 'dist_m': 2.1}
    ]
    publish_vision_facts(facts)
    
    retrieved_facts = get_vision_facts()
    assert len(retrieved_facts) == 2
    assert retrieved_facts[0]['label'] == 'person'
    
    print("SUCCESS: State bus test successful!")
    return True

if __name__ == "__main__":
    test_state_bus()