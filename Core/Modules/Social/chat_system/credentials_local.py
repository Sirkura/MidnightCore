# ==== MODULE CONTRACT =======================================================
# Module: social/chat_system/credentials_local.py
# Package: MidnightCore.Modules.Social.chat_system.credentials_local
# Location: Production/MidnightCore/Core/Modules/Social/chat_system/credentials_local.py
# Responsibility: Local VRChat credentials configuration (no external AI services)
# PUBLIC: LocalCredentials class
# DEPENDENCIES: None
# POLICY: NO_FALLBACKS=deny, Telemetry: none
# MIGRATION: Simplified from original VRChat-AI-Bot credentials for local-only use
# ============================================================================

class LocalCredentials:
    """
    Local credentials configuration for Qwen Chat Bot
    Only VRChat API credentials needed - no external AI services
    """
    
    # VRChat API credentials
    VRCHAT_USER = 'your_vrchat_username_here'
    VRCHAT_PASSWORD = 'your_vrchat_password_here'
    USER_AGENT = 'QwenChatBot/1.0 MidnightCore/Production'
    
    # Optional: VRChat group ID for auto-inviting friends
    # GROUP_ID = "grp_your_group_id_here"
    
    # Bot configuration
    BOT_NAME = "Qwen Bot"
    BOT_TITLE = "🤖 Qwen Bot 🤖"
    
    # Audio configuration
    AUDIO_DEVICE = "VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)"
    TTS_LANGUAGE = "en"
    
    # Content filtering enabled by default
    CONTENT_FILTERING = True
    
    # Conversation settings
    MAX_CONVERSATION_HISTORY = 10
    IDLE_MESSAGE = "Hi, I'm Qwen Bot!\vCome talk with me!\v(Powered by local AI)"

# Example configuration - copy this to configure your bot
EXAMPLE_CONFIG = """
# Copy this example and update with your actual VRChat credentials

class MyBotCredentials:
    # Your actual VRChat login credentials
    VRCHAT_USER = 'myusername'
    VRCHAT_PASSWORD = 'mypassword'
    USER_AGENT = 'MyBot/1.0 MidnightCore'
    
    # Optional: Auto-invite friends to a group
    GROUP_ID = "grp_12345678-1234-1234-1234-123456789abc"
    
    # Customize your bot
    BOT_NAME = "My AI Assistant"
    BOT_TITLE = "🎭 My Bot 🎭"
    IDLE_MESSAGE = "Hello! I'm your friendly AI assistant!"
"""

if __name__ == "__main__":
    print("Qwen Chat Bot Local Credentials")
    print("===============================")
    print()
    print("This file contains the credentials configuration for your VRChat bot.")
    print("Update the LocalCredentials class with your actual VRChat login details.")
    print()
    print("Required:")
    print("- VRCHAT_USER: Your VRChat username")  
    print("- VRCHAT_PASSWORD: Your VRChat password")
    print()
    print("Optional:")
    print("- GROUP_ID: VRChat group ID for auto-inviting friends")
    print("- BOT_NAME, BOT_TITLE: Customize your bot's identity")
    print()
    print("Security note: Keep your credentials secure and never commit them to version control!")