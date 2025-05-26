import speech_recognition as sr

class AudioToText:
    def __init__(self, filename: str = "record.wav"):
        self.filename = filename
        self.recognizer = sr.Recognizer()

    def transcribe(self, language: str = "en-US"):
        with sr.AudioFile(self.filename) as source:
            print("Converting audio to text...")
            audio_data = self.recognizer.record(source)
            
            try:
                text = self.recognizer.recognize_google(audio_data, language=language)
                print("Transcription: ", text)
                return text
            except sr.UnknownValueError:
                print("Speech Recognition could not understand the audio.")
            except sr.RequestError:
                print("Could not request results from Google Speech Recognition service.")

# Kullanım
# converter = AudioToText(filename="./record.wav") 
# text = converter.transcribe(language="tr-TR") 
# print(text)