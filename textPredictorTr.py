import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import warnings
from transformers import logging
import re
import logging as pylogging

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

class TextPredictorTR:
    def __init__(self, model_path='./turkish_emotion_analysis.pt', device='cpu', verbose=False):
        # Setup logging
        self.logger = pylogging.getLogger(__name__)
        self.logger.setLevel(pylogging.INFO if verbose else pylogging.ERROR)

        # Initialize device
        self.device = torch.device(device)
        
        try:
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained("maymuni/bert-base-turkish-cased-emotion-analysis")
            self.bert = AutoModel.from_pretrained("maymuni/bert-base-turkish-cased-emotion-analysis", return_dict=False)
            
            # Build and load model
            self.model = self._build_model(self.bert)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            self.logger.info("Model initialized successfully")
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
    
    class Arch(nn.Module):
        def __init__(self, bert):
            super(TextPredictorTR.Arch, self).__init__()
            self.bert = bert 
            self.dropout = nn.Dropout(0.1)
            self.relu = nn.ReLU()
            self.fc1 = nn.Linear(768, 512)
            self.fc3 = nn.Linear(512, 6)
            self.softmax = nn.LogSoftmax(dim=1)

        def forward(self, sent_id, mask):
            _, cls_hs = self.bert(sent_id, attention_mask=mask, return_dict=False)
            x = self.fc1(cls_hs)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.fc3(x)
            x = self.softmax(x)
            return x

    def _build_model(self, bert):
        return self.Arch(bert)
    
    def filter(self, text: str) -> str:
        """
        Enhanced text filtering method using regex for better cleaning
        """
        if not text or not isinstance(text, str):
            self.logger.warning("Invalid text input")
            return ""
        
        # Remove URLs, mentions, hashtags, special tokens
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        text = re.sub(r'\@\w+|\#|\&|\!', '', text)
        text = re.sub(r'RT\s', '', text)
        
        # Remove extra whitespaces
        text = ' '.join(text.split())
        
        return text
    
    def predict_emotion(self, text: str) -> dict:
        try:
            # Filter text
            text = self.filter(text)
            
            if not text:
                self.logger.warning("Empty text after filtering")
                return {emotion: 0.0 for emotion in ['anger', 'surprise', 'joy', 'sadness', 'fear', 'disgust']}
            
            # Tokenize
            tokenized = self.tokenizer.encode_plus(
                text,
                max_length=512,  # Limit sequence length
                padding='max_length',
                truncation=True,
                return_token_type_ids=False,
                return_tensors='pt'
            )

            input_ids = tokenized['input_ids'].to(self.device)
            attention_mask = tokenized['attention_mask'].to(self.device)

            # Predict
            with torch.no_grad():
                preds = self.model(input_ids, attention_mask)
                probabilities = torch.nn.functional.softmax(preds, dim=1)
                probabilities = probabilities.cpu().numpy()[0]

            # Map probabilities to emotions
            emotion_map = {
                'anger': float(probabilities[0]),
                'surprise': float(probabilities[1]),
                'joy': float(probabilities[2]),
                'sadness': float(probabilities[3]),
                'fear': float(probabilities[4]),
                'disgust': float(probabilities[5])
            }
            
            return emotion_map
        
        except Exception as e:
            self.logger.error(f"Emotion prediction error: {e}")
            return {emotion: 0.0 for emotion in ['anger', 'surprise', 'joy', 'sadness', 'fear', 'disgust']}
    
    def calculateTR(self, emotions: dict) -> np.ndarray:
        try:
            # Extract and process emotion scores
            sad = float(int(emotions.get("sadness", 0.0)*100))/100
            happy = float(int(emotions.get("joy", 0.0)*100))/100
            disgust = float(int(emotions.get("disgust", 0.0)*100))/100
            anger = float(int(emotions.get("anger", 0.0)*100))/100
            fear = float(int(emotions.get("fear", 0.0)*100))/100
            surprised = float(int(emotions.get("surprise", 0.0)*100))/100       

            # Complex emotion redistribution logic
            if happy > 0.25 and disgust > 0.20:
                natural = (4 * happy) / 5 + disgust
                happy = (happy / 5) + (sad / 20) + (anger / 20) + (fear / 20) + (surprised / 20)
                sad = (19 * sad) / 20
                anger = (19 * anger) / 20
                fear = (19 * fear) / 20
                surprised = (19 * surprised) / 20
            else:
                natural = (sad / 6) + (happy / 10) + (disgust) + (anger / 10) + (fear / 10) + (surprised / 10)
                sad = (5 * sad) / 6
                happy = (9 * happy) / 10
                anger = (9 * anger) / 10
                fear = (9 * fear) / 10
                surprised = (9 * surprised) / 10

            # Create result array with rounded percentages
            result_array = np.array([
                round(100 * anger, 2), 
                round(100 * fear, 2), 
                round(100 * happy, 2),
                round(100 * natural, 2), 
                round(100 * sad, 2), 
                round(100 * surprised, 2)
            ])

            return result_array
        
        except Exception as e:
            self.logger.error(f"Emotion calculation error: {e}")
            return np.zeros(6)

    def predict_and_calculate(self, text: str) -> np.ndarray:
        emotions = self.predict_emotion(text)
        return self.calculateTR(emotions)

