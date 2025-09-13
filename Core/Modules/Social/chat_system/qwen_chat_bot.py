# ==== MODULE CONTRACT =======================================================
# Module: social/chat_system/qwen_chat_bot.py
# Package: MidnightCore.Modules.Social.chat_system.qwen_chat_bot
# Location: Production/MidnightCore/Core/Modules/Social/chat_system/qwen_chat_bot.py
# Responsibility: VRChat AI chat bot using local Qwen 3 LLM
# PUBLIC: QwenChatBot class, main() function
# DEPENDENCIES: qwen_llm_adapter, speech_recognition, gtts, pygame, pythonosc, vrchatapi
# POLICY: NO_FALLBACKS=deny, Telemetry: chat.*
# MIGRATION: Adapted from VRChat-AI-Bot ollama branch to use local Qwen 3
# ============================================================================

import speech_recognition as sr
from gtts import gTTS
from pygame import mixer, _sdl2 as devices
import threading
import random
import time
import asyncio
from pythonosc import udp_client
import vrchatapi
from vrchatapi.api import authentication_api, notifications_api, groups_api
from vrchatapi.exceptions import UnauthorizedException
from vrchatapi.models.two_factor_auth_code import TwoFactorAuthCode
from vrchatapi.models.two_factor_email_code import TwoFactorEmailCode
from vrchatapi.models.create_group_invite_request import CreateGroupInviteRequest

# Import our local Qwen adapter
from qwen_llm_adapter import QwenLLMAdapter

class QwenChatBot:
    """VRChat AI Chat Bot using local Qwen 3 LLM"""
    
    def __init__(self, credentials):
        """Initialize the chat bot with credentials"""
        self.credentials = credentials
        
        # Initialize Qwen LLM adapter
        self.llm_adapter = QwenLLMAdapter()
        
        # Bot configuration
        self.bot_title = "🤖 Qwen Bot 🤖"
        self.idle_message = "Hi, I'm Qwen Bot!\vCome talk with me!\v(Powered by local AI)"
        
        # State management
        self.speech_recognized = False
        self.listen_count = 0
        self.is_emoting = False
        self.movement_paused = False
        self.audio_counter = 1
        
        # OSC client for VRChat communication
        self.osc_client = udp_client.SimpleUDPClient("127.0.0.1", 9000)
        
        print(f"QwenChatBot initialized with title: {self.bot_title}")
    
    def filter_content(self, text: str) -> bool:
        """Content filtering - simplified version"""
        bad_words = [
            " edging ", " penis ", "mein kampf", " cult ", " touch ", " rape ", 
            "daddy", "jew", "porn", "p hub", "9/11", "9:11", "hitler", "911", 
            "nazi", "1940", "drug", "methan", "serial killer", "kill myself", 
            "cannibalism", "columbine", "minstrel", "blackface", "standoff", 
            "murder", "bombing", "suicide", "massacre", "genocide", "zoophil", 
            "knot", "canna", "nigg", "fag", "adult content", "nsfw"
        ]
        
        for word in bad_words:
            if word in text.lower():
                print(f"BAD WORD FOUND: {word}")
                return True
        return False
    
    def send_chatbox(self, message: str):
        """Send message to VRChat chatbox via OSC"""
        formatted_message = f"{self.bot_title}\v╔═══════╗\v{message}\v╚═══════╝"
        
        # Truncate if too long
        if len(formatted_message) > 144:
            formatted_message = formatted_message[:140] + "..."
        
        self.osc_client.send_message("/chatbox/input", [formatted_message, True, False])
        print(f"Chatbox: {formatted_message}")
    
    def speak_text(self, text: str):
        """Convert text to speech and play it"""
        try:
            self.listen_count = 0
            self.send_chatbox(text)
            
            # Initialize mixer with VoiceMeeter
            mixer.init(devicename="VoiceMeeter Input (VB-Audio VoiceMeeter VAIO)", frequency=48510)
            
            # Generate TTS
            tts = gTTS(text.replace(":", " colon "), lang='en')
            filename = f"{self.audio_counter}.mp3"
            tts.save(filename)
            
            # Play audio
            mixer.music.load(filename)
            mixer.music.play()
            
            # Wait for audio to finish
            while mixer.music.get_busy():
                time.sleep(0.1)
            
            mixer.stop()
            self.audio_counter += 1
            
        except Exception as e:
            print(f"TTS error: {e}")
            self.send_chatbox("Text to speech failed")
    
    def check_for_emotes(self, text: str):
        """Check for emote keywords and trigger VRChat emotes"""
        text_lower = text.lower()
        emote = 0
        self.is_emoting = True
        
        # Map keywords to emote numbers
        if "wave" in text_lower or "hi " in text_lower or "hello" in text_lower:
            emote = 1
        elif "clap" in text_lower or "congrat" in text_lower:
            emote = 2
        elif "point" in text_lower or "look" in text_lower or "!" in text_lower:
            emote = 3
        elif "cheer" in text_lower:
            emote = 4
        elif "dance" in text_lower:
            emote = 5
        elif "backflip" in text_lower or "flip" in text_lower:
            emote = 6
        elif "kick" in text_lower:
            emote = 7
        elif "die" in text_lower or "dead" in text_lower:
            emote = 8
        
        if emote != 0:
            self.osc_client.send_message("/avatar/parameters/VRCEmote", [emote])
            print(f"Triggered emote #{emote}")
            time.sleep(2)
            self.osc_client.send_message("/avatar/parameters/VRCEmote", [0])
        
        self.is_emoting = False
    
    def check_for_commands(self, text: str, original_prompt: str):
        """Check for movement commands"""
        text_lower = text.lower()
        self.is_emoting = True
        
        # Movement commands
        if "forward" in original_prompt.lower() and "move" in original_prompt.lower():
            self.osc_client.send_message("/input/MoveForward", [1])
            self.speak_text("Moving forward")
            time.sleep(2)
            self.osc_client.send_message("/input/MoveForward", [0])
            return True
            
        elif "backward" in original_prompt.lower() and "move" in original_prompt.lower():
            self.osc_client.send_message("/input/MoveBackward", [1])
            self.speak_text("Moving backward")
            time.sleep(2)
            self.osc_client.send_message("/input/MoveBackward", [0])
            return True
            
        elif "left" in original_prompt.lower() and "look" in original_prompt.lower():
            self.osc_client.send_message("/input/LookLeft", [1])
            self.speak_text("Looking left")
            time.sleep(0.45)
            self.osc_client.send_message("/input/LookLeft", [0])
            return True
            
        elif "right" in original_prompt.lower() and "look" in original_prompt.lower():
            self.osc_client.send_message("/input/LookRight", [1])
            self.speak_text("Looking right")
            time.sleep(0.45)
            self.osc_client.send_message("/input/LookRight", [0])
            return True
        
        # Other commands
        if "clear" in text_lower and "history" in text_lower:
            self.llm_adapter.clear_history()
            self.speak_text("Chat history cleared")
            return True
        
        # Movement pause/unpause
        if "pause" in text_lower and "move" in text_lower:
            self.movement_paused = True
            self.bot_title = "Movement paused.\vSay 'unpause movement'."
            return True
        elif "unpause" in text_lower and "move" in text_lower:
            self.movement_paused = False
            self.bot_title = "🤖 Qwen Bot 🤖"
            return True
        
        self.is_emoting = False
        return False
    
    def speech_recognition_loop(self):
        """Continuous speech recognition loop"""
        recognizer = sr.Recognizer()
        
        while True:
            try:
                with sr.Microphone() as source:
                    self.listen_count += 1
                    
                    # Listen for audio with timeout
                    audio = recognizer.listen(source, timeout=2.5, phrase_time_limit=8)
                    print("Recognizing...")
                    
                    # Use Google Speech Recognition
                    sentence = recognizer.recognize_google(audio)
                    print(f"Recognized: {sentence}")
                    
                    # Process the recognized speech
                    self.process_speech_input(sentence)
                    
            except sr.WaitTimeoutError:
                if self.listen_count > 10:
                    self.send_chatbox(self.idle_message)
                    self.listen_count = 0
                    
            except sr.UnknownValueError:
                print("Speech recognition could not understand audio")
                
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition: {e}")
    
    def process_speech_input(self, sentence: str):
        """Process recognized speech input"""
        # Check for inappropriate content
        if self.filter_content(sentence):
            self.speak_text("Prompt is inappropriate. Please try again.")
            return
        
        # Check for commands first
        if self.check_for_commands(sentence, sentence):
            return  # Command was processed, don't send to LLM
        
        # Send to Qwen LLM
        self.send_chatbox("Thinking...")
        response = self.llm_adapter.chat(sentence)
        
        if response["success"]:
            response_text = response["text"]
            
            # Check if response is appropriate
            if self.filter_content(response_text):
                self.speak_text("Response is inappropriate. Please try again.")
                return
            
            print(f"Qwen: {response_text}")
            self.speak_text(response_text)
            
            # Check for emotes and commands in the response
            combined_text = response_text + " " + sentence
            self.check_for_emotes(combined_text)
            
        else:
            self.speak_text("I'm having trouble thinking right now. Please try again.")
    
    def random_movement_loop(self):
        """Random movement when idle"""
        while True:
            time.sleep(0.13)
            
            if not self.speech_recognized and not self.is_emoting and not self.movement_paused:
                num = random.randrange(1, 160)
                
                # Random jump
                if num == 10:
                    self.osc_client.send_message("/input/Jump", [1])
                    time.sleep(random.randrange(1, 2))
                    self.osc_client.send_message("/input/Jump", [0])
                
                # Random forward movement
                elif num == 60:
                    self.osc_client.send_message("/input/MoveForward", [1])
                    time.sleep(random.randrange(1, 2))
                    self.osc_client.send_message("/input/MoveForward", [0])
                
                # Random look left
                elif num == 40:
                    self.osc_client.send_message("/input/LookLeft", [1])
                    time.sleep(random.randrange(10, 75) / 100)
                    self.osc_client.send_message("/input/LookLeft", [0])
                
                # Random look right
                elif num == 20:
                    self.osc_client.send_message("/input/LookRight", [1])
                    time.sleep(random.randrange(10, 75) / 100)
                    self.osc_client.send_message("/input/LookRight", [0])
    
    def vrchat_notifications_loop(self):
        """Handle VRChat notifications (friend requests, etc.)"""
        try:
            configuration = vrchatapi.Configuration(
                username=self.credentials.VRCHAT_USER,
                password=self.credentials.VRCHAT_PASSWORD,
            )
            
            with vrchatapi.ApiClient(configuration) as api_client:
                api_client.user_agent = self.credentials.USER_AGENT
                auth_api = authentication_api.AuthenticationApi(api_client)
                
                try:
                    current_user = auth_api.get_current_user()
                except UnauthorizedException as e:
                    if e.status == 200:
                        if "Email 2 Factor Authentication" in e.reason:
                            auth_api.verify2_fa_email_code(
                                two_factor_email_code=TwoFactorEmailCode(input("Email 2FA Code: "))
                            )
                        elif "2 Factor Authentication" in e.reason:
                            auth_api.verify2_fa(
                                two_factor_auth_code=TwoFactorAuthCode(input("2FA Code: "))
                            )
                            current_user = auth_api.get_current_user()
                        else:
                            print("Exception when calling API: %s\n", e)
                except vrchatapi.ApiException as e:
                    print("Exception when calling API: %s\n", e)
                
                print("Logged in as:", current_user.display_name)
                
                while True:
                    try:
                        notifications = notifications_api.NotificationsApi(api_client).get_notifications()
                        for notification in notifications:
                            if notification.type == 'friendRequest':
                                notifications_api.NotificationsApi(api_client).accept_friend_request(notification.id)
                                print("Accepted friend request!")
                                
                                if not self.filter_content(notification.sender_username):
                                    self.speak_text(f"Thanks for friending me, {notification.sender_username}!")
                                
                                # Add to group (if configured)
                                if hasattr(self.credentials, 'GROUP_ID'):
                                    invite_req = CreateGroupInviteRequest(notification.sender_user_id, True)
                                    groups_api.GroupsApi(api_client).create_group_invite(
                                        self.credentials.GROUP_ID, invite_req
                                    )
                        
                        time.sleep(7)  # Check every 7 seconds
                        
                    except Exception as e:
                        print(f"Notification error: {e}")
                        
        except KeyboardInterrupt:
            pass
        except Exception as e:
            print(f"VRChat API error: {e}")
    
    async def main_async_loop(self):
        """Main async loop for chat processing"""
        while True:
            try:
                # This is where we'd handle any async chat processing
                # For now, just keep the loop alive
                await asyncio.sleep(1)
                
            except Exception as e:
                print(f"Error in async loop: {e}")
    
    def run(self):
        """Start the chat bot"""
        print("Starting Qwen Chat Bot...")
        
        # Check audio devices
        mixer.init()
        print("Available audio outputs:", devices.audio.get_audio_device_names(False))
        mixer.quit()
        
        # Start background threads
        speech_thread = threading.Thread(target=self.speech_recognition_loop, daemon=True)
        speech_thread.start()
        
        movement_thread = threading.Thread(target=self.random_movement_loop, daemon=True)
        movement_thread.start()
        
        vrchat_thread = threading.Thread(target=self.vrchat_notifications_loop, daemon=True)
        vrchat_thread.start()
        
        # Start async event loop
        try:
            asyncio.run(self.main_async_loop())
        except KeyboardInterrupt:
            print("Shutting down Qwen Chat Bot...")

# Credentials class for local configuration
class LocalCredentials:
    """Local credentials configuration"""
    VRCHAT_USER = 'your_vrchat_username'
    VRCHAT_PASSWORD = 'your_vrchat_password'  
    USER_AGENT = 'QwenChatBot/1.0 MidnightCore'
    # Optional: GROUP_ID = "grp_your_group_id_here"

def main():
    """Main entry point"""
    try:
        credentials = LocalCredentials()
        bot = QwenChatBot(credentials)
        bot.run()
    except KeyboardInterrupt:
        print("Bot stopped by user")
    except Exception as e:
        print(f"Bot crashed: {e}")

if __name__ == "__main__":
    main()