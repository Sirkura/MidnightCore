# Qwen Chat Bot - Local VRChat AI Assistant

A VRChat AI chat bot powered by local Qwen 3 LLM, integrated with the MidnightCore ecosystem.

## Features

- **Local AI**: Uses your local Qwen 3 8B model via llama-cli.exe
- **No External APIs**: No CharacterAI, OpenAI, or Ollama server required
- **Full VRChat Integration**: OSC communication, friend requests, emotes
- **Speech Recognition**: Google STT for voice input
- **Text-to-Speech**: gTTS with VB-Audio Cable routing
- **Content Filtering**: Built-in inappropriate content detection
- **Conversation Memory**: Maintains context across exchanges

## Quick Start

1. **Install Dependencies**
   ```bash
   pip install -r requirements_local.txt
   ```

2. **Configure Credentials**
   - Edit `credentials_local.py`
   - Add your VRChat username and password

3. **Verify Paths**
   - Ensure llama-cli.exe is at: `G:/Experimental/llama.cpp/build/bin/Release/llama-cli.exe`
   - Ensure Qwen 3 model is at: `G:/Experimental/ComfyUI_windows_portable/ComfyUI/models/llm_gguf/Qwen3-8B-Q4_K_M.gguf`

4. **Setup Audio**
   - Install VB-Audio Virtual Cable
   - Set VRChat audio input to "CABLE Output"

5. **Run the Bot**
   ```bash
   python qwen_chat_bot.py
   ```

## Architecture

### Core Components

- **qwen_llm_adapter.py**: Local LLM integration using MidnightCore's proven approach
- **qwen_chat_bot.py**: Main bot class with VRChat integration
- **credentials_local.py**: Configuration (VRChat credentials only)

### Integration with MidnightCore

- Uses same Qwen 3 model and llama-cli.exe as main brain system
- Compatible with MidnightCore file structure and conventions
- Can be extended with MidnightCore logging system

## Commands

### Voice Commands
- **Movement**: "move forward", "move backward", "look left", "look right"
- **Utility**: "clear history" - clears conversation memory
- **Control**: "pause movement", "unpause movement"

### Emote Triggers
The bot automatically triggers VRChat emotes based on conversation content:
- "wave", "hi", "hello" → Wave emote
- "clap", "congratulations" → Clap emote  
- "point", "look", "!" → Point emote
- "cheer" → Cheer emote
- "dance" → Dance emote
- "flip", "backflip" → Backflip emote
- "kick" → Kick emote

## Configuration

### Bot Personality
Edit the system prompt in `QwenLLMAdapter.chat()` to customize personality:

```python
system_prompt = (
    "You are a friendly AI assistant in VRChat. "
    "Respond naturally and conversationally. "
    "Keep responses concise but engaging."
)
```

### Performance Tuning
Adjust token budgets in `QwenLLMAdapter.__init__()`:

```python
self.token_budgets = {
    "chat": 512,         # Full conversation capacity
    "fallback": 150      # Emergency responses
}
```

## Troubleshooting

### Common Issues

1. **"Model not found"**
   - Verify Qwen 3 model path in `qwen_llm_adapter.py`
   - Check that model file exists and is accessible

2. **"llama-cli.exe not found"** 
   - Verify llama.cpp path in `qwen_llm_adapter.py`
   - Ensure llama.cpp is built with your GPU support

3. **"No audio output"**
   - Install VB-Audio Virtual Cable
   - Check VRChat audio settings
   - Verify mixer device name in bot code

4. **"Speech recognition failing"**
   - Check microphone permissions
   - Verify internet connection (Google STT requires online access)
   - Test microphone input levels

5. **"VRChat login failed"**
   - Check VRChat credentials in `credentials_local.py`
   - Handle 2FA prompts if enabled on your account
   - Verify VRChat API access

### Performance Tips

- **GPU Acceleration**: Ensure llama.cpp is built with CUDA/OpenCL support
- **Memory Management**: Monitor system RAM usage during long conversations
- **Network**: Stable internet required for speech recognition and VRChat API

## Development

### Extending the Bot

1. **Add New Commands**: Modify `check_for_commands()` in `qwen_chat_bot.py`
2. **Custom Emotes**: Extend `check_for_emotes()` with new trigger words
3. **Advanced AI**: Enhance system prompts in `qwen_llm_adapter.py`
4. **MidnightCore Integration**: Add logging calls using MidnightCore's logging system

### Testing

Test the LLM adapter independently:
```bash
python qwen_llm_adapter.py
```

### File Structure
```
chat_system/
├── qwen_llm_adapter.py      # Core LLM integration
├── qwen_chat_bot.py         # Main bot application  
├── credentials_local.py     # Configuration
├── requirements_local.txt   # Dependencies
└── README.md               # This file
```

## License

Integrated with MidnightCore production system. Based on VRChat-AI-Bot by tuckerisapizza.