# ==== MODULE CONTRACT =======================================================
# Module: social/chat_system/qwen_llm_adapter.py
# Package: MidnightCore.Modules.Social.chat_system.qwen_llm_adapter
# Location: Production/MidnightCore/Core/Modules/Social/chat_system/qwen_llm_adapter.py
# Responsibility: Local Qwen 3 LLM integration for VRChat chat bot
# PUBLIC: QwenLLMAdapter class, chat() method
# DEPENDENCIES: subprocess, llama-cli.exe
# POLICY: NO_FALLBACKS=deny, Telemetry: chat.*
# MIGRATION: Adapted from MidnightCore brain _qwen_chat function
# ============================================================================

import subprocess
import time
import re
import threading
import random
from typing import Dict, List, Optional, Any

class QwenLLMAdapter:
    """
    Local Qwen 3 LLM adapter using llama-cli.exe subprocess calls
    Extracted and simplified from MidnightCore brain system for chat use
    """
    
    def __init__(self):
        """Initialize Qwen LLM adapter with MidnightCore-compatible settings"""
        # Use exact same paths as MidnightCore brain
        self.llama_path = "G:/Experimental/llama.cpp/build/bin/Release/llama-cli.exe"
        self.model_path = "G:/Experimental/ComfyUI_windows_portable/ComfyUI/models/llm_gguf/Qwen3-8B-Q4_K_M.gguf"
        
        # Qwen 3 8B configuration (from MidnightCore)
        self.qwen_config = {
            "ngl": 25,           # GPU layers
            "ctx": 4096,         # Context window
            "temp": 0.7,         # Temperature
            "top_p": 0.9,        # Top-p sampling
            "repeat_penalty": 1.1
        }
        
        # Chat-specific token budgets
        self.token_budgets = {
            "chat": 512,         # Full conversation capacity
            "fallback": 150      # Emergency responses
        }
        
        # Model family detection
        self.model_family = "qwen"  # Force Qwen for this adapter
        
        # Thread safety
        self._llm_lock = threading.Lock()
        
        # Conversation management
        self.conversation_history = []
        self.max_history_length = 10  # Keep last 10 exchanges
        
        print(f"QwenLLMAdapter initialized with model: {self.model_path}")
        print(f"Using llama-cli: {self.llama_path}")
    
    def _render_chat_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Render proper chat template for Qwen 3 (simple concatenation)"""
        return f"{system_prompt.strip()}\n\n{user_prompt.strip()}"
    
    def _build_conversation_context(self, new_message: str) -> str:
        """Build conversation context from history"""
        if not self.conversation_history:
            return new_message
        
        # Build context with recent history
        context_parts = ["RECENT CONVERSATION:"]
        
        # Add last few exchanges for context
        for user_msg, bot_response in self.conversation_history[-3:]:
            context_parts.append(f"User: {user_msg}")
            context_parts.append(f"Bot: {bot_response}")
        
        context_parts.append(f"\nCurrent message: {new_message}")
        return "\n".join(context_parts)
    
    def chat(self, message: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        Main chat method - simplified from MidnightCore _qwen_chat
        
        Args:
            message: User's chat message
            system_prompt: Optional system prompt (defaults to chat bot personality)
        
        Returns:
            Dict with success, text, tokens_used, latency
        """
        if system_prompt is None:
            system_prompt = (
                "You are a friendly AI assistant in VRChat. "
                "Respond naturally and conversationally in 1-3 sentences. "
                "Keep responses concise and engaging. "
                "You can mention emotes like *waves* or *dances* when appropriate."
            )
        
        # Build conversation context  
        user_prompt = self._build_conversation_context(message)
        max_tokens = 120  # Reduced for more concise responses
        
        # Tiered approach (simplified from MidnightCore)
        tiers = [
            ("full_context", 
             self._render_chat_prompt(system_prompt, f"User says: {message}\nRespond briefly:"),
             dict(temp=0.7, top_p=0.9, n=max_tokens)),
            ("fallback",
             self._render_chat_prompt(
                 "You are a friendly VRChat bot. Respond in 1-2 sentences.",
                 f"User says: {message}\nBot responds:"
             ),
             dict(temp=0.4, top_p=0.8, n=80))
        ]
        
        with self._llm_lock:
            for tier_name, rendered_prompt, samp in tiers:
                result = self._call_llama_subprocess(rendered_prompt, samp, tier_name)
                
                if result["success"]:
                    # Add to conversation history
                    self._add_to_history(message, result["text"])
                    return result
            
            # All tiers failed
            return {"success": False, "text": "I'm having trouble responding right now.", 
                   "tokens_used": 0, "latency": 0}
    
    def _call_llama_subprocess(self, prompt: str, sampling_params: Dict, tier_name: str) -> Dict[str, Any]:
        """Call llama-cli.exe subprocess (extracted from MidnightCore)"""
        cmd = [
            self.llama_path, "-m", self.model_path,
            "-ngl", str(self.qwen_config["ngl"]),
            "-c", str(self.qwen_config["ctx"]),
            "--temp", str(sampling_params["temp"]),
            "--top-p", str(sampling_params["top_p"]),
            "--repeat-penalty", str(self.qwen_config["repeat_penalty"]),
            "--ignore-eos",  # Prevents early termination
            "-n", str(sampling_params["n"]),
            "-no-cnv",       # No conversation mode
            "-p", prompt
        ]
        
        try:
            t0 = time.time()
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                   text=True, encoding='utf-8', errors='replace')
            out, err = proc.communicate(timeout=45)  # Longer timeout for chat
            
            if proc.returncode != 0:
                print(f"LLM tier {tier_name}: returncode {proc.returncode}")
                print(f"Error: {err}")
                return {"success": False, "text": ""}
            
            # Clean response (same as MidnightCore)
            text = (out or "").replace("<s>", "").replace("</s>", "").strip()
            
            # Remove performance logs
            if "llama_perf_sampler_print:" in text:
                text = text.split("llama_perf_sampler_print:")[0].strip()
            if "llama_perf_context_print:" in text:
                text = text.split("llama_perf_context_print:")[0].strip()
            
            # Clean whitespace and emojis for Windows CMD compatibility
            text = re.sub(r"\s+", " ", text)
            text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII characters
            
            # Extract only the actual response (after "Respond briefly:")
            if "Respond briefly:" in text:
                response_parts = text.split("Respond briefly:")
                if len(response_parts) > 1:
                    # Take the part after "Respond briefly:" and clean it
                    text = response_parts[1].strip()
                    # Remove any meta-discussion that might follow
                    text = text.split("User says:")[0].strip()
                    text = text.split("Okay")[0].strip()
            
            latency = time.time() - t0
            
            if len(text) >= 3:
                tokens_used = len(text.split())
                print(f"LLM SUCCESS on tier {tier_name}: {len(text)} chars, {tokens_used} tokens - '{text[:50]}...'")
                return {
                    "success": True, 
                    "text": text, 
                    "tier": tier_name, 
                    "latency": latency,
                    "tokens_used": tokens_used
                }
            else:
                print(f"LLM tier {tier_name}: empty output")
                return {"success": False, "text": ""}
                
        except subprocess.TimeoutExpired:
            print(f"LLM tier {tier_name}: timeout after 45s")
            proc.kill()
            proc.communicate()
            return {"success": False, "text": ""}
        except Exception as e:
            print(f"LLM tier {tier_name}: exception {e}")
            return {"success": False, "text": ""}
    
    def _add_to_history(self, user_message: str, bot_response: str):
        """Add exchange to conversation history"""
        self.conversation_history.append((user_message, bot_response))
        
        # Trim to max length
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared")
    
    def get_history_summary(self) -> str:
        """Get summary of conversation history"""
        if not self.conversation_history:
            return "No conversation history"
        
        return f"Conversation history: {len(self.conversation_history)} exchanges"

# Test function
if __name__ == "__main__":
    print("Testing Qwen LLM Adapter...")
    adapter = QwenLLMAdapter()
    
    # Test basic chat
    test_messages = [
        "Hello! How are you?",
        "What can you do in VRChat?", 
        "Can you dance?"
    ]
    
    for message in test_messages:
        print(f"\n--- Testing message: '{message}' ---")
        response = adapter.chat(message)
        
        if response["success"]:
            print(f"[SUCCESS] Bot response: {response['text']}")
            print(f"  Tokens: {response['tokens_used']}, Latency: {response['latency']:.2f}s")
        else:
            print("[FAILED] No response received")
        
    print("\nTest completed!")