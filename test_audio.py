from gtts import gTTS
from pathlib import Path

try:
    text = "మీ పంట ఆరోగ్యంగా ఉందో లేదో పరిశీలించండి."
    audio = gTTS(text=text, lang="te")
    audio.save("test_telugu.mp3")
    print("Success! Telugu audio saved.")
except Exception as e:
    print("Audio error:", e)