import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
import numpy as np
import warnings
from transformers import logging

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

class EmotionPredictor:
    def __init__(self, model_path="/home/fhurkhan/Masaüstü/main/TEXT/turkish_emotion_analysis.pt"):
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained("maymuni/bert-base-turkish-cased-emotion-analysis")
        self.bert = AutoModel.from_pretrained("maymuni/bert-base-turkish-cased-emotion-analysis", return_dict=False)
        self.model = self._build_model(self.bert)
        self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    
    class Arch(nn.Module):
        def __init__(self, bert):
            super(EmotionPredictor.Arch, self).__init__()
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
        # Return the defined model architecture
        return self.Arch(bert)
    
    def filter(self, text):
        final_text = ''
        for word in text.split():
            if word.startswith('@') or word == 'RT' or word[-3:] in ['com', 'org'] or \
               word.startswith('pic') or word.startswith('http') or word.startswith('www') or \
               word.startswith('!') or word.startswith('&') or word.startswith('-'):
                continue
            else:
                final_text += word + ' '
        return final_text
    
    def predict_emotion(self, text):
        text = self.filter(text)
        tokenized = self.tokenizer.encode_plus(
            text,
            padding='max_length',
            truncation=True,
            return_token_type_ids=False
        )

        input_ids = tokenized['input_ids']
        attention_mask = tokenized['attention_mask']

        seq = torch.tensor(input_ids).unsqueeze(0)
        mask = torch.tensor(attention_mask).unsqueeze(0)
        preds = self.model(seq, mask)
        preds = preds.detach().cpu().numpy()
        result = np.argmax(preds, axis=1)
        preds = torch.tensor(preds)
        probabilities = nn.functional.softmax(preds)

        return {'anger': float(probabilities[0][0]),
                'surprise': float(probabilities[0][1]),
                'joy': float(probabilities[0][2]),
                'sadness': float(probabilities[0][3]),
                'fear': float(probabilities[0][4]),
                'disgust': float(probabilities[0][5])
                }

    
    def calculateTR(self, emotions):
        sad = float(int(emotions.get("sadness", 0.0)*100))/100
        happy = float(int(emotions.get("joy", 0.0)*100))/100
        disgust = float(int(emotions.get("disgust", 0.0)*100))/100
        anger = float(int(emotions.get("anger", 0.0)*100))/100
        fear = float(int(emotions.get("fear", 0.0)*100))/100
        surprised = float(int(emotions.get("surprise", 0.0)*100))/100       

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

        result_array = np.array([round(100 * anger, 2), round(100 * fear, 2), round(100 * happy, 2),
                                 round(100 * natural, 2), round(100 * sad, 2), round(100 * surprised, 2)])

        return result_array

    def predict_and_calculate(self, text):
        emotions = self.predict_emotion(text)
        return self.calculateTR(emotions)
    

