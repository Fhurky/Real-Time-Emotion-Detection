import numpy as np
from audioRecorder import AudioRecorder
from audioToText import AudioToText
from audioPredictor import AudioPredictor
from textPredictorEn import TextPredictorEN
from textPredictorTr import TextPredictorTR
from imagePrediction import ImagePredictor


print("Test 1:")
# 5sn kayıt yap ve test.wav şeklinde kaydet
recorder = AudioRecorder(filename="test.wav")
result1 = recorder.record(duration=5)
print("Test 1 result: ", result1)

print("Test 2:")
# test.wav dosyasını metne dönüştür
converter = AudioToText(filename="test.wav")
result2 = converter.transcribe(language="tr-TR") # Language="en-US"
print("Test 2 result: ", result2)

print("Test 3:")
audioPredicttion = AudioPredictor(filename="./test.wav", model_path="./Models", json_path="./")
result3 = audioPredicttion.predict_emotion()
print("Test 3 result: ", result3)

print("Test 4:")
textPredictTr = TextPredictorTR(model_path="./turkish_emotion_analysis.pt", device='cpu') # device='cuda'
result4 = textPredictTr.predict_and_calculate(text="Bugün hava çok güzel")
print("Test 4 result: ", result4)

print("Test 5:")
textPredictEn = TextPredictorEN(device='cpu') # device='cuda'
result5 = textPredictEn.predict_and_calculate(text=result2, decision=0) # decision 1 for english
print("Test 5 result: ", result5)

print("Test 6: ")
sample_image = np.random.rand(48, 48).astype(np.float32)
predictor = ImagePredictor(model_path="./Image_model.keras")
result6 = predictor.emotion_from_preprocessed_image(sample_image)
print("Test 6 result: ", result6)