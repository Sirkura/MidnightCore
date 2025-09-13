#!/usr/bin/env python3
# ==== MODULE CONTRACT =======================================================
# Module: social/chat_system/launch_qwen_bot.py
# Package: MidnightCore.Modules.Social.chat_system.launch_qwen_bot
# Location: Production/MidnightCore/Core/Modules/Social/chat_system/launch_qwen_bot.py
# Responsibility: Launcher script for Qwen Chat Bot with system checks
# PUBLIC: main() function, system verification
# DEPENDENCIES: qwen_chat_bot, system path verification
# POLICY: NO_FALLBACKS=deny, Telemetry: none
# MIGRATION: New launcher for integrated chat system
# ============================================================================

import os
import sys
from pathlib import Path

def verify_system_requirements():
    """Verify all system requirements before launching bot"""
    print("Qwen Chat Bot System Verification")
    print("=================================")
    
    errors = []
    warnings = []
    
    # Check Python version
    if sys.version_info < (3, 8):
        errors.append(f"Python 3.8+ required, found {sys.version}")
    else:
        print(f"✓ Python version: {sys.version}")
    
    # Check llama-cli.exe
    llama_path = "G:/Experimental/llama.cpp/build/bin/Release/llama-cli.exe"
    if not os.path.exists(llama_path):
        errors.append(f"llama-cli.exe not found at: {llama_path}")
    else:
        print(f"✓ llama-cli.exe found: {llama_path}")
    
    # Check Qwen 3 model
    model_path = "G:/Experimental/ComfyUI_windows_portable/ComfyUI/models/llm_gguf/Qwen3-8B-Q4_K_M.gguf"
    if not os.path.exists(model_path):
        errors.append(f"Qwen 3 model not found at: {model_path}")
    else:
        print(f"✓ Qwen 3 model found: {model_path}")
    
    # Check required modules
    required_modules = [
        'speech_recognition',
        'gtts', 
        'pygame',
        'pythonosc',
        'vrchatapi',
        'mutagen',
        'syllables'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
            print(f"✓ Module available: {module}")
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        errors.append(f"Missing required modules: {', '.join(missing_modules)}")
        print("\nInstall missing modules with:")
        print("pip install -r requirements_local.txt")
    
    # Check credentials
    try:
        from credentials_local import LocalCredentials
        creds = LocalCredentials()
        
        if creds.VRCHAT_USER == 'your_vrchat_username_here':
            warnings.append("VRChat credentials not configured in credentials_local.py")
        else:
            print(f"✓ VRChat credentials configured for user: {creds.VRCHAT_USER}")
            
    except ImportError:
        errors.append("credentials_local.py not found or has errors")
    
    # Check audio device (warning only)
    try:
        from pygame import mixer, _sdl2 as devices
        mixer.init()
        audio_devices = devices.audio.get_audio_device_names(False)
        mixer.quit()
        
        if "CABLE Input (VB-Audio Virtual Cable)" in audio_devices:
            print("✓ VB-Audio Virtual Cable detected")
        else:
            warnings.append("VB-Audio Virtual Cable not detected - TTS may not work in VRChat")
            print(f"Available audio devices: {audio_devices}")
    except Exception as e:
        warnings.append(f"Could not check audio devices: {e}")
    
    # Display results
    print("\nVerification Results:")
    print("====================")
    
    if errors:
        print("❌ ERRORS (must fix before running):")
        for error in errors:
            print(f"   - {error}")
    
    if warnings:
        print("\n⚠️  WARNINGS (optional but recommended):")
        for warning in warnings:
            print(f"   - {warning}")
    
    if not errors and not warnings:
        print("✅ All checks passed! Ready to launch bot.")
    elif not errors:
        print("✅ Core requirements met. Bot can launch with warnings.")
    else:
        print("❌ Cannot launch bot until errors are resolved.")
        return False
    
    return len(errors) == 0

def launch_bot():
    """Launch the Qwen Chat Bot"""
    print("\nLaunching Qwen Chat Bot...")
    print("==========================")
    print("Starting bot systems...")
    print("- Speech recognition active")  
    print("- VRChat API connection")
    print("- Local Qwen 3 LLM ready")
    print("\nBot is running. Press Ctrl+C to stop.")
    print("Speak near your microphone to chat!")
    
    try:
        # Import and run the bot
        from qwen_chat_bot import QwenChatBot
        from credentials_local import LocalCredentials
        
        credentials = LocalCredentials()
        bot = QwenChatBot(credentials)
        bot.run()
        
    except KeyboardInterrupt:
        print("\nBot stopped by user.")
    except ImportError as e:
        print(f"\nImport error: {e}")
        print("Make sure all required files are present and modules are installed.")
    except Exception as e:
        print(f"\nBot error: {e}")
        print("Check the error message above for troubleshooting.")

def main():
    """Main launcher entry point"""
    print("MidnightCore Qwen Chat Bot Launcher")
    print("===================================\n")
    
    # Add current directory to path for local imports
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Verify system requirements
    if verify_system_requirements():
        print("\nPress Enter to launch the bot, or Ctrl+C to cancel...")
        try:
            input()
            launch_bot()
        except KeyboardInterrupt:
            print("\nLaunch cancelled.")
    else:
        print("\nPlease resolve the errors above before launching the bot.")
        print("\nCommon fixes:")
        print("1. Install dependencies: pip install -r requirements_local.txt")
        print("2. Configure VRChat credentials in credentials_local.py")
        print("3. Verify Qwen 3 model and llama-cli.exe paths")
        print("4. Install VB-Audio Virtual Cable for audio routing")

if __name__ == "__main__":
    main()