# =========================================================
# FINAL HYBRID CNN-KNN PREDICTION SERVICE
# BASELINE CNN vs MobileNetV2 + KNN
# TANPA PCA (LOKAL VERSION - WINDOWS)
# =========================================================

# =========================================================
# IMPORT LIBRARY
# =========================================================

import os
import numpy as np
import tensorflow as tf
import joblib

# pyrefly: ignore [missing-import]
from tensorflow.keras.preprocessing import image
# pyrefly: ignore [missing-import]
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================================================
# DATASET & MODEL PATHS (LOKAL)
# =========================================================

script_dir = os.path.dirname(os.path.abspath(__file__))
train_path = os.path.join(script_dir, "Chilli Plant Diseases Dataset", "train")
valid_path = os.path.join(script_dir, "Chilli Plant Diseases Dataset", "valid")
test_path = os.path.join(script_dir, "Chilli Plant Diseases Dataset", "test")

# Tentukan folder model di root proyek
root_dir = os.path.dirname(script_dir)
model_dir = os.path.join(root_dir, "model")
if not os.path.exists(model_dir):
    os.makedirs(model_dir, exist_ok=True)

save_dir = model_dir

# Definisikan path model hybrid & baseline
FEATURE_EXTRACTOR_PATH = os.path.join(model_dir, "mobilenetv2_feature_extractor.h5")
KNN_MODEL_PATH = os.path.join(model_dir, "hybrid_cnn_knn.pkl")
MOBILENET_BEST_PATH = os.path.join(model_dir, "mobilenetv2_best.h5")

# =========================================================
# LAZY LOAD GLOBAL MODELS FOR PREDICTION
# =========================================================

_feature_extractor = None
_knn_model = None
_cnn_best_model = None

# Class names mapping (obtained from directory list)
RAW_CLASSES = [
    'Chilli Anthracnose',
    'Chilli Healthy',
    'Chilli Leaf Curl Virus',
    'Chilli Leaf Spot',
    'Chilli Veinal Mottle Virus',
    'Chilli Whitefly',
    'Chilli Yellowish',
    'Non-Chilli'
]

# Mapping to match Flask app INFO_PENYAKIT dictionary keys
CLASS_MAPPING = {
    'Chilli Anthracnose': 'anthracnose ( Patek )',
    'Chilli Healthy': 'healthy ( Sehat )',
    'Chilli Leaf Curl Virus': 'leaf curl',
    'Chilli Leaf Spot': 'leaf spot',
    'Chilli Veinal Mottle Virus': 'veinal mottle virus',
    'Chilli Whitefly': 'whitefly',
    'Chilli Yellowish': 'yellowish',
    'Non-Chilli': 'non-chilli'
}

def predict(image_path):
    """
    Fungsi prediksi untuk mendeteksi penyakit tanaman cabai.
    Digunakan oleh app.py untuk melakukan diagnosa secara real-time.
    """
    global _feature_extractor, _knn_model
    
    # Load model jika belum terload di memory (Lazy Loading)
    if _feature_extractor is None:
        local_extractor_path = os.path.join(model_dir, "mobilenetv2_feature_extractor.h5")
        if not os.path.exists(local_extractor_path):
            # Fallback ke folder script jika folder model tidak punya file tersebut
            local_extractor_path = os.path.join(script_dir, "mobilenetv2_feature_extractor.h5")
            if not os.path.exists(local_extractor_path):
                print(f"[ERROR] File feature extractor tidak ditemukan di: {local_extractor_path}")
                return None, 0.0
        print(f"[INFO] Memuat Model Feature Extractor dari {local_extractor_path}...")
        _feature_extractor = tf.keras.models.load_model(local_extractor_path, compile=False)
        
    if _knn_model is None:
        local_knn_path = os.path.join(model_dir, "hybrid_cnn_knn.pkl")
        if not os.path.exists(local_knn_path):
            # Fallback ke folder script jika folder model tidak punya file tersebut
            local_knn_path = os.path.join(script_dir, "hybrid_cnn_knn.pkl")
            if not os.path.exists(local_knn_path):
                print(f"[ERROR] File KNN Classifier tidak ditemukan di: {local_knn_path}")
                return None, 0.0
        print(f"[INFO] Memuat Model KNN Classifier dari {local_knn_path}...")
        _knn_model = joblib.load(local_knn_path)
        
    try:
        # Preprocessing Gambar
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        # Ekstraksi Fitur menggunakan CNN
        features = _feature_extractor.predict(img_array, verbose=0)
        
        # Prediksi menggunakan KNN Pipeline (Scaler + KNN)
        pred_idx = _knn_model.predict(features)[0]
        
        # Hitung tingkat kepercayaan (confidence) dari probabilitas KNN
        pred_proba = _knn_model.predict_proba(features)[0]
        confidence = float(pred_proba[pred_idx] * 100.0)
        
        raw_pred_class = RAW_CLASSES[pred_idx]
        clean_pred_class = CLASS_MAPPING.get(raw_pred_class, raw_pred_class)
        
        print(f"[DIAGNOSA] Hasil: {raw_pred_class} -> {clean_pred_class} ({confidence:.2f}%)")
        return clean_pred_class, confidence
        
    except Exception as e:
        print(f"[ERROR] Gagal melakukan diagnosa: {e}")
        return None, 0.0
