import speech_recognition as sr
import numpy as np
from needed_classes.trBert import EmotionPredictor
from needed_classes.engBert import EmotionPredictorEN

class EmotionRecognizer:
    def __init__(self, 
                 wav_file_path=None,
                 lang="tr-TR", 
                 eng_rate=0.8,
                 tr_rate=0.9):
        # Initialize recognizer
        self.recognizer = sr.Recognizer()
        
        # Initialize emotion predictors
        self.predictorTR = EmotionPredictor(model_path="/home/fhurkhan/Masaüstü/Emotion_Detection/models/text/turkish_emotion_analysis.pt")
        self.predictorENG = EmotionPredictorEN()
        
        # Audio file path
        self.wav_file_path = wav_file_path
        
        # Language parameter
        self.lang = lang  # Default language
        
        # Store the rate parameters as instance variables
        self.eng_rate = eng_rate
        self.tr_rate = tr_rate
    
    def transcribe_audio(self, audio_file=None):
        """Transcribe audio file to text using Google Speech Recognition"""
        # Use provided audio_file or default to instance variable
        file_to_use = audio_file or self.wav_file_path
        
        if not file_to_use:
            print("No audio file specified.")
            return None
            
        with sr.AudioFile(file_to_use) as source:
            audio_data = self.recognizer.record(source)
            
            try:
                # Convert speech to text using Google Web Speech API
                text = self.recognizer.recognize_google(audio_data, language=self.lang)
                return text
            except sr.UnknownValueError:
                print("Speech was not understood.")
                return None
            except sr.RequestError:
                print("Could not access Google API.")
                return None
    
    def analyze_emotion(self, text):
        """Analyze emotion in the given text and return only combined results"""
        if text:
            # Get emotion predictions
            tr_result = self.predictorTR.predict_and_calculate(text)
            eng_result = self.predictorENG.predict_and_calculate(text, decision=0)
            
            # İki diziyi ağırlıklı şekilde birleştir
            combined_result = (tr_result * self.tr_rate) + (eng_result * self.eng_rate)

            # Toplamı 100 olacak şekilde normalize et
            percentage_result = (combined_result / np.sum(combined_result)) * 100
            
            # Just return the combined results array
            return percentage_result
        return None
    
    def single_analysis(self, wav_file_path=None):
        """Read, transcribe, and analyze a single audio file"""
        # Use provided path or instance variable
        file_to_analyze = wav_file_path or self.wav_file_path
        text = self.transcribe_audio(file_to_analyze)
        if text is None:
            return np.array([0, 0, 0, 0, 0, 0])
        return self.analyze_emotion(text)