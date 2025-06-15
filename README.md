# 🎭 Emotion Detection GUI

Bu proje, **metin**, **görüntü** ve **ses** girdilerinden Türkçe duygu analizi yapan çok modelli bir masaüstü uygulamasıdır. Arayüzü sayesinde kullanıcılar kolayca veri girişi yapabilir ve duygu tahminlerini anlık olarak gözlemleyebilir.
---

## 📂 Klasör Yapısı

```
./Emotion_Detection/
├── config_files
│   └── config.json
├── GUI.py
├── models
│   ├── image
│   │   └── IMAGE_MODEL_V1.keras
│   ├── text
│   │   └── turkish_emotion_analysis.pt
│   └── voice
│       ├── aboost_model.pkl
│       ├── gnb_model.pkl
│       ├── knn_model_deneme.pkl
│       ├── knn_model.pkl
│       ├── lr_model.pkl
│       ├── mlp_model(pipeline).pkl
│       ├── nn_model.keras
│       ├── nn_scaler.pkl
│       ├── rf_model.pkl
│       └── svm_model.pkl
├── needed_classes
│   ├── engBert.py
│   ├── model_loading.py
│   ├── predictor.py
│   ├── __pycache__
│   │   ├── engBert.cpython-311.pyc
│   │   ├── model_loading.cpython-311.pyc
│   │   ├── predictor.cpython-311.pyc
│   │   ├── recorder.cpython-311.pyc
│   │   ├── speech_to_text.cpython-311.pyc
│   │   └── trBert.cpython-311.pyc
│   ├── recorder.py
│   ├── speech_to_text.py
│   └── trBert.py
├── README.MD
├── requirements.txt
└── temp_files
    └── recorded_audio.wav
```

---

## ⚙️ Kurulum



1. Python 3.11 ortamı oluştur ve etkinleştir:
   ```bash
   conda create --name emotion_detection_gui python=3.11
   conda activate emotion_detection_gui
   ```

2. Gerekli kütüphaneleri yükle:
   ```bash
   pip install -r requirements.txt
   ```

3. (🔧 **Linux Kullanıcıları İçin**):
   Bazı modellerin TensorFlow/Torch ile çalışması için aşağıdaki satırı çalıştırman gerekebilir:
   ```bash
   export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
   ```

4. GUI'yi başlat:
   ```bash
   python GUI.py
   ```

---

## 🧠 Kullanılan Modeller

- **Metin (Text)**: `turkish_emotion_analysis.pt` (Türkçe BERT tabanlı duygu analiz modeli)
- **Görüntü (Image)**: `IMAGE_MODEL_V1.keras` (Yüz ifadelerinden duygu çıkarımı)
- **Ses (Voice)**: `aboost`, `svm`, `mlp`, `nn_model.keras` gibi çeşitli ML ve DL modelleri

---

## 🧪 Özellikler

- 📸 Görüntüden duygu tahmini
- 🎤 Sesten duygu analizi (5 saniyelik kayıtlarla)
- 📝 Metinden duygu tahmini
- 🖼️ GUI arayüzü üzerinden kolay kullanım
- ⚡ Anlık tahmin ve sonuç görselleştirmeleri

---

## 🛠 Geliştirici Notları

- Model yolları ve konfigürasyonlar `config_files/config.json` dosyasından yönetilmektedir.
- `./needed_classes` dizininde ses kaydı, metin analizörü ve model yükleyici gibi bileşenler modüler halde bulunur.
- `./temp_files` klasörü çalışma sırasında oluşan geçici dosyaları barındırır.
- Bu proje linux ortamında ve CUDA desteğiyle geliştirilmiş sonrasında windows için adapte edilmiştir.
- Proje çalıştırılmadan önce işletim sistemine göre config.json dosyasındaki path değişkenleri doğru bir şekilde ayarlanmalıdır.
- CUDA kullanımı için CuDNN ve gerekli CUDA kütüphaneleri yüklenmelidir. Versiyon kontrolleri requirements_linux.txt üzerinden yapılabilir. (trBert.py ve enBert.py dosyaları üzerinden device = "cuda" olarak değiştirilmelidir)

---

## 📌 Gereksinimler

- Python 3.11
- PyTorch
- TensorFlow
- Scikit-learn
- Transformers
- OpenCV
- torchaudio / librosa

Detaylı gereksinim listesi için `requirements.txt` dosyasına bakınız.

---

## 📬 İletişim

> Geliştirici: **Furkan Koçal**  
> GitHub: [github.com/fhurkhan](https://github.com/Fhurky)  
> Mail: furkocal@gmail.com

