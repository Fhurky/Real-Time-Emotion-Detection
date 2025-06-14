import pyaudio
import wave
import librosa
import numpy as np
import time

class AudioRecorder:
    def __init__(self, filename="recorded_audio.wav", duration=2, rate=44100, channels=1, chunk=1024):
        self.filename = filename
        self.duration = duration
        self.rate = rate
        self.channels = channels
        self.chunk = chunk
        self.format = pyaudio.paInt16

    def record_audio(self):
        """ Belirtilen süre boyunca ses kaydı yapar ve bir .wav dosyasına kaydeder. """
        audio = pyaudio.PyAudio()
        
        stream = audio.open(format=self.format, channels=self.channels,
                            rate=self.rate, input=True,
                            frames_per_buffer=self.chunk)

        print("Recording...")
        frames = []

        for _ in range(0, int(self.rate / self.chunk * self.duration)):
            data = stream.read(self.chunk)
            frames.append(data)

        print("Recording finished.")

        stream.stop_stream()
        stream.close()
        audio.terminate()

        # WAV dosyasına kaydetme
        with wave.open(self.filename, 'wb') as wf:
            wf.setnchannels(self.channels)
            wf.setsampwidth(audio.get_sample_size(self.format))
            wf.setframerate(self.rate)
            wf.writeframes(b''.join(frames))

    def extract_feature(self, mfcc=True):
        """ WAV dosyasından ses özelliklerini çıkarır. """
        X, sample_rate = librosa.load(self.filename, sr=None)
        result = np.array([])

        if mfcc:
            mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
            result = np.hstack((result, mfccs))

        return result

    def record_and_extract(self):
        """ Hem sesi kaydeder hem de özellikleri çıkarır. """
        self.record_audio()
        return self.extract_feature()
