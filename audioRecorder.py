import sounddevice as sd
import wave
import numpy as np

class AudioRecorder:
    def __init__(self, filename: str = "record.wav", samplerate: int = 44100):
        self.filename = filename
        self.samplerate = samplerate

        # Mikrofonun desteklediği maksimum kanal sayısını öğren
        device_info = sd.query_devices(kind='input')
        self.channels = min(device_info['max_input_channels'], 1)  # Mono kayıt için 1 kanal

    def record(self, duration: int):
        print(f"🎤 Recording for {duration} seconds...")

        # Doğru formatı kullan (int16 veya float32 olabilir)
        audio_data = sd.rec(int(self.samplerate * duration), samplerate=self.samplerate, channels=self.channels, dtype=np.int16)
        sd.wait()
        
        with wave.open(self.filename, "wb") as wf:
            wf.setnchannels(self.channels)  # 1 kanal (mono)
            wf.setsampwidth(2)  # 16-bit için 2 byte
            wf.setframerate(self.samplerate)  # Doğru örnekleme oranı
            wf.writeframes(audio_data.tobytes())

        print(f"✅ Recording saved as {self.filename}")

# Kullanım
# recorder = AudioRecorder()
# recorder.record(duration=5)  # 5 saniyelik düzgün kayıt yapar
