from transformers import pipeline
import numpy as np
from deep_translator import GoogleTranslator

class EmotionPredictorEN:
    def __init__(self):
        self.classifier = pipeline("text-classification", model='bhadresh-savani/bert-base-uncased-emotion', return_all_scores=True)

    def translate_to_english(self, text):
        return GoogleTranslator(source='tr', target='en').translate(text)

    def predict_emotion(self, text, decision):
        if decision == 1:  # If the text is already in English
            english_text = text
        else:  # If the text is in Turkish, translate it first
            english_text = self.translate_to_english(text)

        prediction = self.classifier(english_text)
        
        # Extract emotions and scores
        emotions = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']
        scores = [item['score'] for item in prediction[0]]  # prediction[0] because the output is a list
        
        # Convert scores to numpy array
        emotion_array = np.array(scores)

        return emotion_array

    def calculate_eng(self, emotions):
        sad = float(int(emotions.get("sadness", 0.0) * 100)) / 100
        happy = float(int(emotions.get("joy", 0.0) * 100)) / 100
        love = float(int(emotions.get("love", 0.0) * 100)) / 100
        anger = float(int(emotions.get("anger", 0.0) * 100)) / 100
        fear = float(int(emotions.get("fear", 0.0) * 100)) / 100
        surprised = float(int(emotions.get("surprise", 0.0) * 100)) / 100
        happy = happy + love

        if anger > 0.30 and sad > 0.30:
            natural = (5 * anger) / 6 + (3 * sad) / 4
            anger = anger / 6
            sad = sad / 4
        else:
            natural = round(((1 * sad) / 8 + (1 * happy) / 8 + (1 * anger) / 8 + (1 * fear) / 8 + (1 * surprised) / 8) * 100, 2)
            sad = round((7*sad / 8) * 100, 2)
            happy = round((7*happy / 8) * 100, 2)
            anger = round((7*anger / 8) * 100, 2)
            fear = round((7*fear / 8) * 100, 2)
            surprised = round((7*surprised / 8) * 100, 2)

        return np.array([anger, fear, happy, natural, sad, surprised])

    def predict_and_calculate(self, text, decision):
        emotion_array = self.predict_emotion(text, decision)
        emotions = dict(zip(['sadness', 'joy', 'love', 'anger', 'fear', 'surprise'], emotion_array))
        return self.calculate_eng(emotions)