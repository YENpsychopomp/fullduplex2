from dataclasses import dataclass, field
from datetime import datetime
import os
import uuid

@dataclass
class AudioFormat:
    format: str = field(default="pcm")
    sample_rate: int = field(default=16000)
    sample_width: int = field(default=2)
    channels: int = field(default=1)
    has_set: bool = field(default=False)

@dataclass
class SessionData:
    system_prompt: str = field(default="你是一個帶有幽默感的語言模型，請用中文回答問題。")
    audio_formate: AudioFormat = field(default_factory=AudioFormat)
    audio_buffer: bytes = field(default_factory=bytes)

class SessionManager:
    def __init__(self):
        self.sessions = {}

    def create_session(self) -> str:
        session_id = str(uuid.uuid4())
        session = SessionData()
        self.sessions[session_id] = session
        return session_id

    def get_session_info(self, session_id):
        return self.sessions.get(session_id)

    def save_session_audio(self, session_id, base_dir):
        session = self.sessions.get(session_id)
        if not session or not session.audio_buffer:
            return None

        audio_format = session.audio_formate
        extension = audio_format.format or "pcm"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{session_id}_{timestamp}.{extension}"
        os.makedirs(base_dir, exist_ok=True)
        file_path = os.path.join(base_dir, filename)

        with open(file_path, "wb") as f:
            f.write(session.audio_buffer)

        return file_path

    def close_session(self, session_id):
        if session_id in self.sessions:
            del self.sessions[session_id]