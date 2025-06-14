from needed_classes.model_loading import ModelPredictor
from needed_classes.recorder import AudioRecorder
import time

def predict_emotion_from_audio(model_path, JSON_path, duration=2, sleep_time=1):
    """
    Gerçek zamanlı ses kaydı alır, MFCC özelliklerini çıkarır ve modelle tahmin yapar.
    
    Args:
    - model_path (str): Model dosyasının yolu
    - JSON_path (str): JSON yapılandırma dosyasının yolu
    - duration (int): Her kaydın süresi (saniye cinsinden)
    - sleep_time (int): Tahmin sonrası bekleme süresi (saniye cinsinden)
    
    Returns:
    - son_tahmin (any): Son yapılan tahmin
    """
    # Model yükleme
    predictor = ModelPredictor(model_path, JSON_path)
    
    # Ses kaydedici başlat
    recorder = AudioRecorder(duration=duration)
    
    # Sonsuz döngü: Her 10 saniyede bir ses kaydedecek ve tahmin yapacak
    while True:
        print("Yeni kayıt alınıyor...")
        
        # Ses kaydet ve özellikleri çıkar
        features = recorder.record_and_extract()
        
        # Özellikleri modele uygun hale getir (reshape)
        features = features.reshape(1, -1)  # Tek bir örnek olarak model için uygun hale getiriyoruz
        
        # Tahmin yap
        son_tahmin = predictor.predict(features)
        
        # Son tahmin sonucunu yazdır
        print("Son Tahmin:", son_tahmin)
        
        # Belirtilen süre kadar bekle
        time.sleep(sleep_time)
        
    return son_tahmin
