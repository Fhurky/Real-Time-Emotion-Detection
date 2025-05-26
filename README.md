
# 🎭 Real-Time Multimodal Emotion Detection

Bu proje, metin, ses ve görüntü verileri üzerinden gerçek zamanlı olarak duygu analizi yapabilen çok modlu (multimodal) bir yapay zeka sistemidir. Python tabanlı bu sistemde hem makine öğrenmesi hem de derin öğrenme yöntemleri kullanılmıştır.

## 🔍 Özellikler

- 📷 **Görüntü Tabanlı Duygu Tespiti**  
  Kullanıcının yüz ifadesine göre duygular tespit edilir (`image_model.keras` ile CNN tabanlı model kullanılmıştır).

- 🔉 **Ses Tabanlı Duygu Tespiti**  
  Mikrofon üzerinden alınan ses verisi işlenip duygular sınıflandırılır (`turkish_emotion_analysis.pt` ile PyTorch modeli).

- 📝 **Metin Tabanlı Duygu Tespiti**  
  Girilen metnin duygu tonu analiz edilerek sınıflandırma yapılır (farklı modeller `Models/` klasöründe yer alır).

- 🌍 Türkçe ve İngilizce metin desteği  
  `textPredictorTr.py` ve `textPredictorEn.py` dosyaları ile iki dilde de duygu tahmini yapılabilir.

## 📁 Proje Yapısı

```
├── main.py                       # Uygulamanın ana çalışma dosyası
├── audioRecorder.py              # Mikrofon üzerinden ses kaydı
├── audioToText.py                # Ses verisini metne çevirme
├── audioPredictor.py             # Sesten duygu tespiti
├── imagePrediction.py            # Görüntüden duygu tespiti
├── modelPredictor.py             # Yüklenen modellerle tahmin işlemleri
├── textPredictorTr.py            # Türkçe metin analizi
├── textPredictorEn.py            # İngilizce metin analizi
├── Configs/                      # Ayar dosyaları
├── Models/                       # Kayıtlı modeller (.pkl, .keras vs.)
├── trimmed_audio/                # Kırpılmış ses kayıtları
├── record/, test/                # Ses testi/kayıt örnekleri
├── turkish_emotion_analysis.pt   # Türkçe veri kümesiyle eğitilmiş model
├── Image_model.keras             # Görüntü CNN modeli
└── README.md                     # Bu dosya
```

## 🧠 Kullanılan Modeller

`Models/` klasöründe çeşitli algoritmalarla eğitilmiş modeller bulunmaktadır:

- `svm_model.pkl` – Support Vector Machine
- `rf_model.pkl` – Random Forest
- `mlp_model(pipeline).pkl` – Multi-layer Perceptron (Scikit-learn pipeline)
- `nn_model.keras` – Derin öğrenme modeli (Keras)
- `aboost_model.pkl` – AdaBoost
- `gnb_model.pkl` – Gaussian Naive Bayes
- `lr_model.pkl` – Logistic Regression
- `knn_model.pkl` – K-Nearest Neighbors
- `nn_scaler.pkl` – Model için kullanılan ölçekleyici (Scaler)

## 🚀 Başlatmak İçin

Proje klasörüne gidin ve gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

Ardından uygulamayı çalıştırın:

```bash
python main.py
```

## 🧪 Bağımlılıklar

- Python 3.8+
- OpenCV
- PyTorch
- Keras / TensorFlow
- Scikit-learn
- SpeechRecognition
- Transformers (eğer kullanılmaktaysa)

> Tüm bağımlılıkları `requirements.txt` içerisine dahil etmeniz önerilir.

## 📌 Notlar

- Türkçe ses analizi için özel eğitilmiş bir PyTorch modeli (`turkish_emotion_analysis.pt`) kullanılmıştır.
- Her bir modal için ayrı modüler dosyalar yazılmıştır, bu sayede sadece ses, sadece metin veya sadece görüntü bazlı sistemler bağımsız olarak da çalıştırılabilir.

## ✍️ Geliştirici

Furkan Koçal – [LinkedIn](https://www.linkedin.com/in/furkankocal)
