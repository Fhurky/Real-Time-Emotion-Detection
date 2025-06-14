import pickle
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import json
import os

class ModelPredictor:
    def __init__(self, model_path, JSON_path):
        """
        model_path: Tüm modellerin bulunduğu klasörün yolu (örn: "../Models/")
        """
        self.model_path = model_path.rstrip('/') + '/'  # Sonuna '/' ekleyerek düzenliyoruz
        self.JSON_path = JSON_path.rstrip('/') + '/'  # Sonuna '/' ekleyerek düzenliyoruz
        # Modelleri yükle
        self.knn_model = self.load_pickle_model("knn_model.pkl")
        self.lr_model = self.load_pickle_model("lr_model.pkl")
        self.mlp_model = joblib.load(self.model_path + "mlp_model(pipeline).pkl")
        self.gnb_model = self.load_pickle_model("gnb_model.pkl")
        self.rf_model = self.load_pickle_model("rf_model.pkl")
        self.svm_model = self.load_pickle_model("svm_model.pkl")
        self.aboost_model = self.load_pickle_model("aboost_model.pkl")
        self.nn_model = load_model(self.model_path + "nn_model.keras")
        self.scaler = joblib.load(self.model_path + "nn_scaler.pkl")

        # Konfigürasyon dosyasını yükle
        config_path = os.path.join(self.JSON_path, "config.json")
        with open(config_path, "r") as file:
            config = json.load(file)
        self.weights = config["weights"]
        self.class_names = ['angry', 'fear', 'happy', 'neutral', 'sad', 'surprised']

    def load_pickle_model(self, model_filename):
        """
        Pickle formatındaki bir modeli yükler.
        """
        model_full_path = os.path.join(self.model_path, model_filename)
        with open(model_full_path, "rb") as model_file:
            return pickle.load(model_file)

    def predict(self, X):
        """
        Verilen girdiye göre modellerin tahminlerini hesaplar ve birleştirir.
        """
        models_results = [
            self.KNN_MODEL(X),
            self.LR_MODEL(X),
            self.MLP_MODEL(X),
            self.GNB_MODEL(X),
            self.RF_MODEL(X),
            self.SVM_MODEL(X),
            self.ABOOST_MODEL(X),
            self.NN_MODEL(X)
        ]
        return self.combine_models_predict(models_results)

    def KNN_MODEL(self, X):
        return self.knn_model.predict_proba(X) if hasattr(self.knn_model, "predict_proba") else None

    def LR_MODEL(self, X):
        return self.lr_model.predict_proba(X) if hasattr(self.lr_model, "predict_proba") else None

    def MLP_MODEL(self, X):
        return self.mlp_model.predict_proba(X) if hasattr(self.mlp_model, "predict_proba") else None

    def GNB_MODEL(self, X):
        return self.gnb_model.predict_proba(X) if hasattr(self.gnb_model, "predict_proba") else None

    def RF_MODEL(self, X):
        return self.rf_model.predict_proba(X) if hasattr(self.rf_model, "predict_proba") else None

    def SVM_MODEL(self, X):
        return self.svm_model.predict_proba(X) if hasattr(self.svm_model, "predict_proba") else None

    def ABOOST_MODEL(self, X):
        return self.aboost_model.predict_proba(X) if hasattr(self.aboost_model, "predict_proba") else None

    def NN_MODEL(self, X):
        X_scaled = self.scaler.transform(X)
        return self.nn_model.predict(X_scaled)

    def combine_models_predict(self, models_results):
        """
        Model tahminlerini ağırlıklarla birleştirerek nihai tahmini üretir.
        """
        weighted_predictions = np.zeros(len(self.class_names))

        for model_result, weight in zip(models_results, self.weights):
            if model_result is not None:
                weighted_predictions += model_result[0] * weight

        total_weight = sum(self.weights)
        combined_probabilities = weighted_predictions / total_weight
        combined_probabilities = np.maximum(combined_probabilities, 0)

        if np.sum(combined_probabilities) > 0:
            combined_probabilities /= np.sum(combined_probabilities)

        return combined_probabilities * 100
