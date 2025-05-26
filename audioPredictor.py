from modelPredictor import ModelPredictor
import librosa as lb
import soundfile as sf

class AudioPredictor:
    def __init__(self, filename: str, model_path: str, json_path: str):
        self.filename = filename
        self.model_predictor = ModelPredictor(model_path, json_path)

    def extract_first_2_seconds(self, output_filename="trimmed_audio.wav"):
        audio, sr = lb.load(self.filename, sr=None, duration=2.0)
        sf.write(output_filename, audio, sr)
        return output_filename

    def predict_emotion(self):
        trimmed_audio = self.extract_first_2_seconds()
        prediction = self.model_predictor.predict(trimmed_audio)
        return prediction

# Kullanım
# predictor = AudioPredictor("./record.wav", "./Models/", "./")
# emotion_probs = predictor.predict_emotion()
# print(emotion_probs)