import tensorflow as tf
import numpy as np

class ImagePredictor:
    def __init__(self, model_path : str = "./Image_model.keras", input_shape=(48, 48, 1), num_classes=6):
        """
        Duygu tanıma modelini yükler ve sınıfı başlatır.
        
        Args:
            model_path (str): Model dosyasının yolu
            input_shape (tuple): Modelin beklediği giriş şekli
            num_classes (int): Duygu sınıfı sayısı
        """
        self.model_path = model_path
        # Modeli yükle
        self.model = tf.keras.models.load_model(model_path)
        
        # Duygu etiketlerini tanımla
        self.emotion_labels = {
            0: "Angry",
            1: "Fear",
            2: "Happy", 
            3: "Neutral",
            4: "Sad",
            5: "Surprise"
        }
        
        self.input_shape = input_shape
        self.num_classes = num_classes
    
    def preprocess_image(self, image):
        """
        Görüntüyü modele uygun hale getirmek için işleme yapılır.
        
        Args:
            image (np.array): 48x48 boyutunda işlenmiş gri tonlama görüntü
        
        Returns:
            np.array: Modelin kabul edeceği formatta giriş verisi
        """
        # Görüntüyü modele uygun hale getirme: 1 boyutlu batch ekleniyor ve normalize ediliyor
        image = np.expand_dims(image, axis=-1)  # (48, 48) -> (48, 48, 1)
        image = np.expand_dims(image, axis=0)  # (48, 48, 1) -> (1, 48, 48, 1)
        image = image.astype('float32') / 255.0  # Normalizasyon işlemi
        return image
    
    def predict_emotion(self, image):
        """
        Verilen 48x48 boyutlarındaki işlenmiş görüntü üzerinden duygu tahminini yapar.
        
        Args:
            image (np.array): 48x48 boyutlarında işlenmiş gri tonlama görüntü verisi (NumPy array)
        
        Returns:
            np.array: Duygu olasılıklarının array çıktısı
        """
        try:
            # Ön işleme işlemi
            processed_image = self.preprocess_image(image)
            
            # Tahmin yap
            predictions = self.model.predict(processed_image)
            
            # Olasılıkları döndür
            return predictions[0]
        
        except Exception as e:
            print(f"Hata oluştu: {str(e)}")
            return None

# Kameradan veya başka bir kaynaktan 48x48 işlenmiş görüntü üzerinden duygu tahmini yapan fonksiyon
def emotion_from_preprocessed_image(image):
    """
    Bu fonksiyon, önceden işlenmiş 48x48 boyutunda bir resim ile duygu tahmini yapar.
    
    Args:
        image (np.array): 48x48 boyutlarında işlenmiş gri tonlama görüntü.
    
    Returns:
        np.array: Duygu olasılıkları.
    """
    # Model dosyasının yolunu belirtin
    model_path = './emotion_model.keras'  # Buraya model dosyasının yolunu girin
    detector = ImagePredictor(model_path)

    # Duygu tahminini yap
    predictions = detector.predict_emotion(image)

    # Sonuçları döndür
    return predictions


# Örnek kullanım


# 48x48 boyutunda işlenmiş bir görüntü (örnek olarak rastgele oluşturulmuş bir görüntü)
# Gerçek bir resim geldiğinde bu yer değiştirilebilir

# sample_image = np.random.rand(48, 48).astype(np.float32)

# predictor = ImagePredictor(model_path="./Image_model.keras")

# # 48x48 boyutunda işlenmiş bir resim ile duygu tahmini yap
# predictions = predictor.emotion_from_preprocessed_image(sample_image)

# print(f"Duygu olasılıkları: {predictions}")

