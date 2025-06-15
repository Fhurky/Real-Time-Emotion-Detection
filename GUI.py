import customtkinter as ctk
import cv2
import numpy as np
import threading
import time
from PIL import Image, ImageTk
from datetime import datetime
import os
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from needed_classes.speech_to_text import EmotionRecognizer
from needed_classes.model_loading import ModelPredictor
from needed_classes.predictor import predict_emotion_from_audio
from needed_classes.recorder import AudioRecorder

class EmotionAnalysisGUI:
    def __init__(self, keras_model=None):
        if keras_model is not None:
            self.keras_model = keras_model
        else:
            config_path = os.path.join("config_files", "config.json")
            with open(config_path, "r", encoding="utf-8") as file:
                config = json.load(file)
            self.keras_model = config.get("keras")

        # CustomTkinter tema ayarları
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Ana pencere
        self.root = ctk.CTk()
        self.root.title("🎭 REAL TIME EMOTION DETECTOR V2.0")
        self.root.geometry("1800x1000")
        self.root.resizable(True, True)
        
        # Modern gradient background renkleri
        self.root.configure(fg_color=("#f0f0f0", "#0a0a0a"))
        
        # Değişkenler
        self.cap = None
        self.is_running = False
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Duygu kategorileri
        self.emotions = ["Anger", "Fear", "Happy", "Neutral", "Sad", "Surprised"]
        
        # Renk temaları - Her analiz türü için benzersiz renkler
        self.color_themes = {
            "image": {
                "primary": "#FF6B6B",    # Kırmızı tonları
                "secondary": "#FF8E8E",
                "accent": "#FFB3B3"
            },
            "voice": {
                "primary": "#4ECDC4",    # Turkuaz tonları
                "secondary": "#7ED7D1",
                "accent": "#A8E0DC"
            },
            "text": {
                "primary": "#45B7D1",    # Mavi tonları
                "secondary": "#6BC5DB",
                "accent": "#8FD3E5"
            },
            "combined": {
                "primary": "#96CEB4",    # Yeşil tonları
                "secondary": "#B5D6C3",
                "accent": "#D4E6D2"
            }
        }
        
        # Sonuç değişkenleri - Thread-safe erişim için lock
        self.results_lock = threading.Lock()
        self.image_results = np.zeros(6)
        self.voice_results = np.zeros(6)
        self.text_results = np.zeros(6)
        self.combined_results = np.zeros(6)
        
        # Timer değişkenleri
        self.last_image_time = 0
        self.last_voice_time = 0
        
        # Ağırlıklar - Başlangıç değerleri
        self.w_text = ctk.DoubleVar(value=0.8)
        self.w_image = ctk.DoubleVar(value=0.9)
        self.w_voice = ctk.DoubleVar(value=0.4)
        
        # Current face için lock
        self.face_lock = threading.Lock()
        self.current_face = None
        
        # Progress bar referanslarını saklamak için dict
        self.progress_bars = {}
        self.value_labels = {}
        
        self.setup_ui()
        
    def setup_ui(self):
        # Ana grid yapılandırması - 3 sütunlu layout
        self.root.grid_columnconfigure(0, weight=2)  # Sol panel (kamera) daha geniş
        self.root.grid_columnconfigure(1, weight=1)  # Orta panel (kontroller)
        self.root.grid_columnconfigure(2, weight=2)  # Sağ panel (sonuçlar) daha geniş
        self.root.grid_rowconfigure(0, weight=1)
        
        # Sol panel - Kamera
        self.setup_camera_panel()
        
        # Orta panel - Kontroller ve Ağırlıklar
        self.setup_control_panel()
        
        # Sağ panel - Sonuçlar
        self.setup_results_panel()
        
    def setup_camera_panel(self):
        # Sol frame - Kamera
        self.left_frame = ctk.CTkFrame(self.root, corner_radius=20, fg_color=("#ffffff", "#1a1a1a"))
        self.left_frame.grid(row=0, column=0, padx=(20, 10), pady=20, sticky="nsew")
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_rowconfigure(1, weight=1)
        
        # Gradient header
        self.camera_header = ctk.CTkFrame(self.left_frame, height=60, corner_radius=15, 
                                         fg_color=("#e8f4f8", "#1e3a5f"))
        self.camera_header.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="ew")
        self.camera_header.grid_columnconfigure(0, weight=1)
        self.camera_header.grid_propagate(False)
        
        # Kamera başlığı - Emoji ve modern font
        self.camera_title = ctk.CTkLabel(
            self.camera_header, 
            text="📹 CAMERA", 
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#2c5282", "#ffffff")
        )
        self.camera_title.grid(row=0, column=0, pady=15)
        
        # Kamera görüntüsü - Modern çerçeve
        self.camera_frame = ctk.CTkFrame(self.left_frame, corner_radius=15, fg_color=("#f7fafc", "#2d3748"))
        self.camera_frame.grid(row=1, column=0, padx=15, pady=(0, 15), sticky="nsew")
        self.camera_frame.grid_columnconfigure(0, weight=1)
        self.camera_frame.grid_rowconfigure(0, weight=1)
        
        self.camera_label = ctk.CTkLabel(
            self.camera_frame, 
            text="🎥 Waiting for camera...", 
            font=ctk.CTkFont(size=16),
            text_color=("#718096", "#a0aec0")
        )
        self.camera_label.grid(row=0, column=0, padx=20, pady=20, sticky="nsew")
        
    def setup_control_panel(self):
        # Orta frame - Kontroller ve Ağırlıklar
        self.middle_frame = ctk.CTkFrame(self.root, corner_radius=20, fg_color=("#ffffff", "#1a1a1a"))
        self.middle_frame.grid(row=0, column=1, padx=10, pady=20, sticky="nsew")
        self.middle_frame.grid_columnconfigure(0, weight=1)
        self.middle_frame.grid_rowconfigure(2, weight=1)
        
        # Kontrol Başlığı
        self.control_header = ctk.CTkFrame(self.middle_frame, height=60, corner_radius=15,
                                          fg_color=("#f0fff0", "#1a4d1a"))
        self.control_header.grid(row=0, column=0, padx=15, pady=(15, 10), sticky="ew")
        self.control_header.grid_columnconfigure(0, weight=1)
        self.control_header.grid_propagate(False)
        
        self.control_title = ctk.CTkLabel(
            self.control_header,
            text="⚙️ CONTROL PANEL",
            font=ctk.CTkFont(size=18, weight="bold"),
            text_color=("#2d7d32", "#ffffff")
        )
        self.control_title.grid(row=0, column=0, pady=15)
        
        # Kontrol butonları - Modern tasarım
        self.button_frame = ctk.CTkFrame(self.middle_frame, fg_color="transparent")
        self.button_frame.grid(row=1, column=0, padx=15, pady=10, sticky="ew")
        self.button_frame.grid_columnconfigure(0, weight=1)
        
        self.start_button = ctk.CTkButton(
            self.button_frame,
            text="🚀 START",
            command=self.start_analysis,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            corner_radius=25,
            fg_color=("#10b981", "#059669"),
            hover_color=("#047857", "#065f46")
        )
        self.start_button.grid(row=0, column=0, pady=(0, 10), sticky="ew")
        
        self.stop_button = ctk.CTkButton(
            self.button_frame,
            text="⏹️ STOP",
            command=self.stop_analysis,
            font=ctk.CTkFont(size=16, weight="bold"),
            height=50,
            corner_radius=25,
            state="disabled",
            fg_color=("#ef4444", "#dc2626"),
            hover_color=("#b91c1c", "#991b1b")
        )
        self.stop_button.grid(row=1, column=0, sticky="ew")
        
        # Ağırlık Kontrolü
        self.setup_weight_controls()
        
    def setup_weight_controls(self):
        # Ağırlık kontrol frame
        self.weight_frame = ctk.CTkFrame(self.middle_frame, corner_radius=15, fg_color=("#fef5e7", "#2d1b13"))
        self.weight_frame.grid(row=2, column=0, padx=15, pady=15, sticky="nsew")
        self.weight_frame.grid_columnconfigure(0, weight=1)
        
        # Ağırlık başlığı
        self.weight_title = ctk.CTkLabel(
            self.weight_frame,
            text="⚖️ WEIGHT SETTINGS",
            font=ctk.CTkFont(size=16, weight="bold"),
            text_color=("#d97706", "#fbbf24")
        )
        self.weight_title.grid(row=0, column=0, pady=(15, 10))
        
        # Ağırlık kontrolleri
        weight_configs = [
            ("🖼️ Image", self.w_image, self.color_themes["image"]["primary"]),
            ("🎙️ Voice", self.w_voice, self.color_themes["voice"]["primary"]),
            ("📝 Text", self.w_text, self.color_themes["text"]["primary"])
        ]
        
        for i, (label_text, var, color) in enumerate(weight_configs):
            # Ağırlık frame
            weight_item_frame = ctk.CTkFrame(self.weight_frame, fg_color="transparent")
            weight_item_frame.grid(row=i+1, column=0, padx=10, pady=5, sticky="ew")
            weight_item_frame.grid_columnconfigure(1, weight=1)
            
            # Label
            weight_label = ctk.CTkLabel(
                weight_item_frame,
                text=label_text,
                font=ctk.CTkFont(size=12, weight="bold"),
                width=80
            )
            weight_label.grid(row=0, column=0, padx=(0, 10), sticky="w")
            
            # Slider
            weight_slider = ctk.CTkSlider(
                weight_item_frame,
                from_=0.0,
                to=1.0,
                variable=var,
                command=self.on_weight_change,
                progress_color=color,
                button_color=color,
                button_hover_color=color
            )
            weight_slider.grid(row=0, column=1, padx=(0, 10), sticky="ew")
            
            # Value label
            value_label = ctk.CTkLabel(
                weight_item_frame,
                text=f"{var.get():.1f}",
                font=ctk.CTkFont(size=12),
                width=30
            )
            value_label.grid(row=0, column=2, sticky="e")
            
            # Store reference for updates
            setattr(self, f"weight_label_{i}", value_label)
            
    def on_weight_change(self, value=None):
        """Ağırlık değiştiğinde value labelları güncelle"""
        try:
            for i in range(3):
                label = getattr(self, f"weight_label_{i}", None)
                if label:
                    values = [self.w_image.get(), self.w_voice.get(), self.w_text.get()]
                    label.configure(text=f"{values[i]:.1f}")
        except Exception as e:
            print(f"Weight label update error: {e}")
            
    def setup_results_panel(self):
        # Sağ frame - Sonuçlar
        self.right_frame = ctk.CTkFrame(self.root, corner_radius=20, fg_color=("#ffffff", "#1a1a1a"))
        self.right_frame.grid(row=0, column=2, padx=(10, 20), pady=20, sticky="nsew")
        self.right_frame.grid_columnconfigure((0, 1), weight=1)
        self.right_frame.grid_rowconfigure((1, 2), weight=1)
        
        # Sonuçlar başlık header
        self.results_header = ctk.CTkFrame(self.right_frame, height=60, corner_radius=15,
                                          fg_color=("#f3e8ff", "#4c1d95"))
        self.results_header.grid(row=0, column=0, columnspan=2, padx=15, pady=(15, 10), sticky="ew")
        self.results_header.grid_columnconfigure(0, weight=1)
        self.results_header.grid_propagate(False)
        
        self.results_title = ctk.CTkLabel(
            self.results_header,
            text="📊 RESULTS OF EMOTION ANALYSIS",
            font=ctk.CTkFont(size=20, weight="bold"),
            text_color=("#5b21b6", "#ffffff")
        )
        self.results_title.grid(row=0, column=0, pady=15)
        
        # Sonuç framelerini oluştur
        result_configs = [
            ("image", "🖼️ Image Analysis", 1, 0),
            ("voice", "🎙️ Voice Analysis", 1, 1),
            ("text", "📝 Textual Analysis", 2, 0),
            ("combined", "🎯 FINAL", 2, 1, True)
        ]
        
        for config in result_configs:
            if len(config) == 5:
                self.setup_result_frame(config[0], config[1], config[2], config[3], config[4])
            else:
                self.setup_result_frame(config[0], config[1], config[2], config[3])
                
    def setup_result_frame(self, key, title, row, col, is_final=False):
        # Ana frame - Tema rengine göre
        colors = self.color_themes[key]
        
        frame = ctk.CTkFrame(
            self.right_frame, 
            corner_radius=15,
            fg_color=(colors["accent"], "#2a2a2a"),
            border_width=2,
            border_color=(colors["primary"], colors["secondary"])
        )
        frame.grid(row=row, column=col, padx=10, pady=8, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        
        # Başlık - Gradient efekti
        title_frame = ctk.CTkFrame(
            frame, 
            height=40,
            corner_radius=12,
            fg_color=(colors["secondary"], colors["primary"])
        )
        title_frame.grid(row=0, column=0, padx=8, pady=(8, 5), sticky="ew")
        title_frame.grid_columnconfigure(0, weight=1)
        title_frame.grid_propagate(False)
        
        title_label = ctk.CTkLabel(
            title_frame, 
            text=title, 
            font=ctk.CTkFont(size=13, weight="bold"),
            text_color=("#ffffff", "#ffffff")
        )
        title_label.grid(row=0, column=0, pady=8)
        
        # Progress bar'lar için ana frame
        progress_main_frame = ctk.CTkFrame(frame, fg_color="transparent")
        progress_main_frame.grid(row=1, column=0, padx=8, pady=(3, 8), sticky="nsew")
        progress_main_frame.grid_columnconfigure(1, weight=1)
        
        # Her duygu için progress bar ve etiket
        progress_bars = []
        value_labels = []
        
        for i, emotion in enumerate(self.emotions):
            # Duygu emoji'leri
            emotion_emojis = ["😠", "😨", "😊", "😐", "😢", "😲"]
            
            # Duygu etiketi
            emotion_label = ctk.CTkLabel(
                progress_main_frame, 
                text=f"{emotion_emojis[i]} {emotion[:4]}", 
                font=ctk.CTkFont(size=10, weight="bold"),
                width=60
            )
            emotion_label.grid(row=i, column=0, padx=(3, 8), pady=2, sticky="w")
            
            # Progress bar - Tema renginde
            progress_bar = ctk.CTkProgressBar(
                progress_main_frame, 
                height=16,
                corner_radius=8,
                progress_color=colors["primary"],
                fg_color=(colors["accent"], "#404040")
            )
            progress_bar.set(0)
            progress_bar.grid(row=i, column=1, padx=(0, 8), pady=2, sticky="ew")
            progress_bars.append(progress_bar)
            
            # Değer etiketi - Rengarenk
            value_label = ctk.CTkLabel(
                progress_main_frame, 
                text="0%", 
                font=ctk.CTkFont(size=10, weight="bold"),
                text_color=(colors["primary"], colors["secondary"]),
                width=35
            )
            value_label.grid(row=i, column=2, padx=(0, 3), pady=2)
            value_labels.append(value_label)
        
        # Dict'te sakla
        self.progress_bars[key] = progress_bars
        self.value_labels[key] = value_labels
        
        print(f"Setup result frame: {key} - Progress bars: {len(progress_bars)}, Value labels: {len(value_labels)}")
        
    def start_analysis(self):
        # Kamerayı başlat
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.show_error("Camera could not be opened!")
            return
            
        self.is_running = True
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        
        # Timer'ları sıfırla
        self.last_image_time = time.time()
        self.last_voice_time = time.time()
        
        # Analiz thread'lerini başlat
        self.camera_thread = threading.Thread(target=self.camera_loop, daemon=True)
        self.analysis_thread = threading.Thread(target=self.analysis_loop, daemon=True)
        
        self.camera_thread.start()
        self.analysis_thread.start()
        
    def stop_analysis(self):
        self.is_running = False
        if self.cap:
            self.cap.release()
        
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.camera_label.configure(image=None, text="🛑 Camera  has been stopped")
        
    def camera_loop(self):
        while self.is_running:
            if self.cap is None:
                break
                
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            # Yüz tespiti
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            
            # Yüzleri çerçevele - Modern stil
            for (x, y, w, h) in faces:
                # Padding ekle
                pad_w = int(w * 0.15)
                pad_h = int(h * 0.15)
                
                # Koordinatları ayarla
                x1 = max(0, x - pad_w)
                y1 = max(0, y - pad_h)
                x2 = min(frame.shape[1], x + w + pad_w)
                y2 = min(frame.shape[0], y + h + pad_h)
                
                # Modern çerçeve - Gradient efekti
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 150), 3)
                cv2.rectangle(frame, (x1-2, y1-2), (x2+2, y2+2), (100, 255, 100), 1)
                
                # Modern metin
                cv2.putText(frame, "FACE DETECTED", (x1, y1-15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 150), 2)
                cv2.putText(frame, f"AI ANALYZING...", (x1, y2+25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 255), 2)
                
                # Yüz crop'unu kaydet (analiz için) - Thread-safe
                with self.face_lock:
                    self.current_face = frame[y1:y2, x1:x2].copy()
            
            # Kamera çerçevesi ekle - Dekoratif çerçeve
            frame = self.add_decorative_frame(frame)
            
            # Görüntüyü GUI'ye göster
            self.update_camera_display(frame)
            time.sleep(0.033)  # ~30 FPS

    def add_decorative_frame(self, frame):
        """Kamera görüntüsüne dekoratif çerçeve ekler"""
        h, w = frame.shape[:2]
        
        # Çerçeve kalınlığı
        border_size = 20
        corner_size = 60
        
        # Ana çerçeve - Gradient efekti için birden fazla katman
        # Dış çerçeve - Koyu
        cv2.rectangle(frame, (0, 0), (w-1, h-1), (30, 30, 30), border_size)
        
        # Orta çerçeve - Parlak
        cv2.rectangle(frame, (border_size//2, border_size//2), 
                    (w-border_size//2-1, h-border_size//2-1), (100, 150, 255), 8)
        
        # İç çerçeve - Beyaz highlight
        cv2.rectangle(frame, (border_size-5, border_size-5), 
                    (w-border_size+4, h-border_size+4), (255, 255, 255), 2)
        
        # Köşe süslemeleri - Sol üst
        cv2.line(frame, (border_size-5, border_size+corner_size), 
                (border_size-5, border_size-5), (0, 255, 200), 4)
        cv2.line(frame, (border_size-5, border_size-5), 
                (border_size+corner_size, border_size-5), (0, 255, 200), 4)
        
        # Köşe süslemeleri - Sağ üst
        cv2.line(frame, (w-border_size+4, border_size-5), 
                (w-border_size-corner_size, border_size-5), (0, 255, 200), 4)
        cv2.line(frame, (w-border_size+4, border_size-5), 
                (w-border_size+4, border_size+corner_size), (0, 255, 200), 4)
        
        # Köşe süslemeleri - Sol alt
        cv2.line(frame, (border_size-5, h-border_size+4), 
                (border_size-5, h-border_size-corner_size), (0, 255, 200), 4)
        cv2.line(frame, (border_size-5, h-border_size+4), 
                (border_size+corner_size, h-border_size+4), (0, 255, 200), 4)
        
        # Köşe süslemeleri - Sağ alt
        cv2.line(frame, (w-border_size+4, h-border_size+4), 
                (w-border_size-corner_size, h-border_size+4), (0, 255, 200), 4)
        cv2.line(frame, (w-border_size+4, h-border_size+4), 
                (w-border_size+4, h-border_size-corner_size), (0, 255, 200), 4)
        
        # Dekoratif noktalar - Köşelerde
        cv2.circle(frame, (border_size+15, border_size+15), 3, (255, 255, 0), -1)
        cv2.circle(frame, (w-border_size-15, border_size+15), 3, (255, 255, 0), -1)
        cv2.circle(frame, (border_size+15, h-border_size-15), 3, (255, 255, 0), -1)
        cv2.circle(frame, (w-border_size-15, h-border_size-15), 3, (255, 255, 0), -1)
        
        # Kamera durumu göstergesi - Sağ üst köşe
        status_color = (0, 255, 0) if len(self.face_cascade.detectMultiScale(
            cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 1.1, 4)) > 0 else (255, 100, 100)
        cv2.circle(frame, (w-30, 30), 8, status_color, -1)
        cv2.circle(frame, (w-30, 30), 12, (255, 255, 255), 2)
        
        # Başlık metni - Üst orta
        title_text = "AI CAMERA SYSTEM"
        text_size = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
        text_x = (w - text_size[0]) // 2
        
        # Metin arka planı
        cv2.rectangle(frame, (text_x-10, 5), (text_x + text_size[0] + 10, 35), 
                    (0, 0, 0), -1)
        cv2.rectangle(frame, (text_x-10, 5), (text_x + text_size[0] + 10, 35), 
                    (100, 150, 255), 2)
        
        # Başlık metni
        cv2.putText(frame, title_text, (text_x, 28), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame

    def update_camera_display(self, frame):
        # OpenCV BGR'den RGB'ye çevir
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # PIL Image'e çevir
        pil_image = Image.fromarray(frame_rgb)
        
        # Boyutlandır - Daha büyük görüntü
        display_size = (520, 390)
        pil_image = pil_image.resize(display_size, Image.Resampling.LANCZOS)
        
        # PhotoImage'e çevir
        photo = ImageTk.PhotoImage(pil_image)
        
        # GUI'de göster
        self.root.after(0, lambda: self.camera_label.configure(image=photo, text=""))
        self.root.after(0, lambda: setattr(self.camera_label, 'image', photo))  # Referansı tut
        
    def analysis_loop(self):
        while self.is_running:
            current_time = time.time()
            
            # Görüntü analizi (her 2 saniye)
            if current_time - self.last_image_time >= 2.0:
                with self.face_lock:
                    current_face_copy = self.current_face.copy() if self.current_face is not None else None
                
                if current_face_copy is not None:
                    threading.Thread(target=self.analyze_image, args=(current_face_copy,), daemon=True).start()
                    self.last_image_time = current_time
            
            # Ses ve metin analizi (her 6 saniye)
            if current_time - self.last_voice_time >= 6.0:
                threading.Thread(target=self.analyze_voice_text, daemon=True).start()
                self.last_voice_time = current_time
            
            time.sleep(0.1)
    
    def analyze_image(self, face_image):
        try:
            # Geçici dosya için tam yol belirle
            temp_dir = "/tmp_files" if os.path.exists("/tmp_files") else os.getcwd()
            temp_path = os.path.join(temp_dir, f"temp_face_{threading.current_thread().ident}.jpg")
            
            # Geçici dosyaya kaydet
            success = cv2.imwrite(temp_path, face_image)
            if not success:
                print(f"Failed to write temporary file: {temp_path}")
                return
            
            # print(f"Geçici dosya oluşturuldu: {temp_path}")
            
            # Gerçek analyze_emotion fonksiyonunu çağır
            result = self.analyze_emotion(temp_path)
            
            if result is not None:
                # String'den float'a çevir
                result_array = np.array(result, dtype=float)
                
                # Softmax çıktısı zaten normalize edilmiş (toplam = 1), direkt yüzdeye çevir
                result_normalized = result_array * 100
                
                # Thread-safe güncelleme
                with self.results_lock:
                    self.image_results = result_normalized
                
                # print(f"Görüntü analizi sonucu: {result_normalized}")
                # print(f"Görüntü sonuç toplamı: {np.sum(result_normalized)}")
                
                # GUI'yi güncelle - Thread-safe şekilde
                self.root.after(0, self.update_results_display)
            
            # Geçici dosyayı sil
            if os.path.exists(temp_path):
                os.remove(temp_path)
                # print(f"Geçici dosya silindi: {temp_path}")
                
        except Exception as e:
            print(f"Image analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    def analyze_voice_text(self):
        try:
            # Gerçek fonksiyonları çağır
            voice_result = self.voice_models_predictor()
            text_result = self.text_emotion()
            
            # Thread-safe güncelleme
            with self.results_lock:
                if voice_result is not None:
                    voice_array = np.array(voice_result, dtype=float)
                    # Eğer zaten yüzdelik değerlerse direkt kullan
                    if np.sum(voice_array) > 50:  # Muhtemelen yüzdelik değer
                        self.voice_results = voice_array
                    else:
                        # Normalize et
                        if np.sum(voice_array) > 0:
                            self.voice_results = (voice_array / np.sum(voice_array)) * 100
                        else:
                            self.voice_results = np.zeros(6)
                
                if text_result is not None:
                    text_array = np.array(text_result, dtype=float)
                    # Eğer zaten yüzdelik değerlerse direkt kullan
                    if np.sum(text_array) > 50:  # Muhtemelen yüzdelik değer
                        self.text_results = text_array
                    else:
                        # Normalize et
                        if np.sum(text_array) > 0:
                            self.text_results = (text_array / np.sum(text_array)) * 100
                        else:
                            self.text_results = np.zeros(6)
            
            # print(f"Ses analizi sonucu: {self.voice_results}")
            # print(f"Metin analizi sonucu: {self.text_results}")
            
            # GUI'yi güncelle - Thread-safe şekilde
            self.root.after(0, self.update_results_display)
            
        except Exception as e:
            print(f"Voice/text analysis error: {e}")
            import traceback
            traceback.print_exc()
    
    # Orijinal fonksiyonlarınızı buraya ekleyin
    def analyze_emotion(self, image_path, model_path=None):
        if model_path is None:
            model_path = self.keras_model
        try:
            # Modeli yükle (sadece ilk seferde)
            if not hasattr(self, 'image_model'):
                self.image_model = load_model(model_path)
            
            # Görüntüyü numpy array olarak oku
            image_array = cv2.imread(image_path)
            if image_array is None:
                print(f"Image could not be read: {image_path}")
                return None
            
            # Çalışan kodunuzdaki gibi preprocessing
            gray_image = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)
            resized_image = cv2.resize(gray_image, (48, 48))
            normalized_image = resized_image.astype('float32') / 255.0
            input_image = np.expand_dims(normalized_image, axis=0)
            
            # Tahmini yap
            predictions = self.image_model.predict(input_image, verbose=0)
            result = np.array(predictions[0])
            
            # String'e çevir (orijinal kodunuzdaki gibi)
            string_predictions = [str(prob) for prob in result]
            return string_predictions
        
        except Exception as e:
            print(f"Image analysis error: {e}")
            return None
    
    def voice_models_predictor(self):
        try:
            # ModelPredictor sınıfını başlat
            model_path = "./models/voice/"
            json_path = "./config_files/"
            predictor = ModelPredictor(model_path, json_path)

            # AudioRecorder sınıfını kullanarak ses kaydı ve özellik çıkarımı
            recorder = AudioRecorder(filename="./temp_files/recorded_audio.wav", duration=5)  # 5 saniyelik kayıt
            features = recorder.record_and_extract()

            # Eğer özellik çıkarılamadıysa sıfır dön
            if features is None or len(features) == 0:
                print("Audio features could not be extracted.")
                return np.zeros(6)

            # Tahmin için reshape
            features = np.array([features])  # (1, n_features) şekline getir

            # Model tahminini al
            result = predictor.predict(features)
            return result

        except Exception as e:
            print(f"Error in audio-based emotion prediction: {e}")
            return np.zeros(6)

    def text_emotion(self):
        try:
            recognizer = EmotionRecognizer()
            # For a single recording and analysis
            result = recognizer.single_analysis(wav_file_path="./temp_files/recorded_audio.wav") # ['anger','fear','happy','natural','sad','surprised'] yüzdesel sonuçlar
            result = np.array(result, dtype=float)
            return result
        except Exception as e:
            print(f"Text analysis error: {e}")
            return np.zeros(6)
    
    def update_results_display(self):
        """Main thread'de çalışacak GUI güncelleme fonksiyonu"""
        try:
            # Thread-safe sonuç okuma
            with self.results_lock:
                image_results_copy = self.image_results.copy()
                voice_results_copy = self.voice_results.copy()
                text_results_copy = self.text_results.copy()
            
            # Birleşik sonucu hesapla - Dinamik ağırlıklarla
            # Tüm sonuçları 0-1 aralığına normalize et
            image_norm = image_results_copy / 100.0
            voice_norm = voice_results_copy / 100.0
            text_norm = text_results_copy / 100.0
            
            # Ağırlıklı ortalama - GUI'den güncel değerleri al
            w_text = self.w_text.get()
            w_image = self.w_image.get()
            w_voice = self.w_voice.get()
            
            combined = (w_text * text_norm + 
                       w_image * image_norm + 
                       w_voice * voice_norm)
            
            # Normalize et ve yüzdeye çevir
            if np.sum(combined) > 0:
                combined_normalized = (combined / np.sum(combined)) * 100
            else:
                combined_normalized = np.zeros(6)
            
            # Thread-safe güncelleme
            with self.results_lock:
                self.combined_results = combined_normalized
            
            # print(f"Birleşik sonuç (w_text:{w_text:.1f}, w_image:{w_image:.1f}, w_voice:{w_voice:.1f}): {combined_normalized}")
            
            # GUI'yi güncelle
            self._update_ui_results(image_results_copy, voice_results_copy, text_results_copy, combined_normalized)
            
        except Exception as e:
            print(f"GUI update error: {e}")
            import traceback
            traceback.print_exc()
    
    def _update_ui_results(self, image_results, voice_results, text_results, combined_results):
        """Gerçek GUI güncelleme fonksiyonu - Animasyonlu güncelleme"""
        try:
            # Sonuç verileri
            results_data = {
                "image": image_results,
                "voice": voice_results,
                "text": text_results,
                "combined": combined_results
            }
            
            # Her bir sonuç tipini güncelle
            for key, results in results_data.items():
                if key in self.progress_bars and key in self.value_labels:
                    progress_bars = self.progress_bars[key]
                    value_labels = self.value_labels[key]
                    
                    # print(f"GUI güncelleme - {key}: {results}")
                    
                    for i, (bar, label) in enumerate(zip(progress_bars, value_labels)):
                        if i < len(results):
                            value = float(results[i])
                            value = max(0, min(100, value))  # 0-100 aralığında sınırla
                            
                            print(f"{key} - {self.emotions[i]}: {value:.1f}%")
                            
                            try:
                                # Animasyonlu güncelleme
                                current_value = bar.get()
                                target_value = value / 100.0
                                
                                # Yumuşak geçiş efekti
                                if abs(current_value - target_value) > 0.01:
                                    steps = 5
                                    step_size = (target_value - current_value) / steps
                                    self.animate_progress_bar(bar, current_value, target_value, step_size, 0, steps)
                                else:
                                    bar.set(target_value)
                                
                                # Renkli yüzde gösterimi
                                if value > 70:
                                    color = "#10b981"  # Yeşil - Yüksek güven
                                elif value > 40:
                                    color = "#f59e0b"  # Turuncu - Orta güven
                                else:
                                    color = "#6b7280"  # Gri - Düşük güven
                                
                                label.configure(text=f"{value:.0f}%", text_color=color)
                                
                            except Exception as e:
                                print(f"GUI element update error {key}-{i}: {e}")
                        else:
                            # Veri yoksa sıfırla
                            bar.set(0)
                            label.configure(text="0%", text_color="#6b7280")
                else:
                    print(f"Progress bar not found: {key}")
                    print(f"Current keys: {list(self.progress_bars.keys())}")
                    
        except Exception as e:
            print(f"_update_ui_results error: {e}")
            import traceback
            traceback.print_exc()
    
    def animate_progress_bar(self, bar, current, target, step_size, step, max_steps):
        """Progress bar'ı yumuşak geçişle güncelle"""
        if step < max_steps:
            new_value = current + step_size * (step + 1)
            bar.set(new_value)
            # Bir sonraki adımı planla
            self.root.after(50, lambda: self.animate_progress_bar(bar, current, target, step_size, step + 1, max_steps))
        else:
            bar.set(target)  # Son değeri kesin olarak ayarla
    
    def show_error(self, message):
        # Modern hata dialog'u
        error_window = ctk.CTkToplevel(self.root)
        error_window.title("⚠️ Error!")
        error_window.geometry("400x200")
        error_window.resizable(False, False)
        error_window.configure(fg_color=("#ffffff", "#1a1a1a"))
        
        # Hata ikonu ve mesaj frame
        error_frame = ctk.CTkFrame(error_window, corner_radius=15, fg_color=("#fef2f2", "#451a1a"))
        error_frame.pack(fill="both", expand=True, padx=20, pady=20)
        error_frame.grid_columnconfigure(0, weight=1)
        
        # Hata ikonu
        error_icon = ctk.CTkLabel(error_frame, text="🚫", font=ctk.CTkFont(size=48))
        error_icon.grid(row=0, column=0, pady=(20, 10))
        
        # Hata mesajı
        error_label = ctk.CTkLabel(
            error_frame, 
            text=message, 
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=("#dc2626", "#ef4444"),
            wraplength=300
        )
        error_label.grid(row=1, column=0, pady=10)
        
        # Tamam butonu
        ok_button = ctk.CTkButton(
            error_frame, 
            text="✅ OK", 
            command=error_window.destroy,
            font=ctk.CTkFont(size=12, weight="bold"),
            fg_color=("#dc2626", "#ef4444"),
            hover_color=("#b91c1c", "#dc2626"),
            corner_radius=20
        )
        ok_button.grid(row=2, column=0, pady=(10, 20))
        
        # Pencereyi merkeze al
        error_window.transient(self.root)
        error_window.grab_set()
    
    def run(self):
        self.root.mainloop()
        # Temizlik
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = EmotionAnalysisGUI()
    app.run()