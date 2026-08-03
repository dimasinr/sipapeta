# =========================================================
# HYBRID CNN-KNN PREDICTION SERVICE
# =========================================================

import os
import numpy as np
import tensorflow as tf
import joblib

from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================================================
# MODEL PATHS
# =========================================================
script_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(script_dir)
model_dir = os.path.join(root_dir, "model")

_feature_extractor = None
_knn_model = None

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

CLASS_MAPPING = {
    'Chilli Anthracnose': 'anthracnose',
    'Chilli Healthy': 'healthy',
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
    
    if _feature_extractor is None:
        local_extractor_path = os.path.join(model_dir, "mobilenetv2_feature_extractor.h5")
        if not os.path.exists(local_extractor_path):
            local_extractor_path = os.path.join(script_dir, "mobilenetv2_feature_extractor.h5")
            if not os.path.exists(local_extractor_path):
                print(f"[ERROR] File feature extractor tidak ditemukan di: {local_extractor_path}")
                return None, 0.0
        print(f"[INFO] Memuat Model Feature Extractor dari {local_extractor_path}...")
        _feature_extractor = tf.keras.models.load_model(local_extractor_path, compile=False)
        
    if _knn_model is None:
        local_knn_path = os.path.join(model_dir, "hybrid_cnn_knn.pkl")
        if not os.path.exists(local_knn_path):
            local_knn_path = os.path.join(script_dir, "hybrid_cnn_knn.pkl")
            if not os.path.exists(local_knn_path):
                print(f"[ERROR] File KNN Classifier tidak ditemukan di: {local_knn_path}")
                return None, 0.0
        print(f"[INFO] Memuat Model KNN Classifier dari {local_knn_path}...")
        _knn_model = joblib.load(local_knn_path)
        
    try:
        img = image.load_img(image_path, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)
        
        features = _feature_extractor.predict(img_array, verbose=0)
        pred_idx = _knn_model.predict(features)[0]
        
        pred_proba = _knn_model.predict_proba(features)[0]
        confidence = float(pred_proba[pred_idx] * 100.0)
        
        raw_pred_class = RAW_CLASSES[pred_idx]
        clean_pred_class = CLASS_MAPPING.get(raw_pred_class, raw_pred_class)
        
        all_predictions = []
        for idx, proba in enumerate(pred_proba):
            raw_cls = RAW_CLASSES[idx]
            clean_cls = CLASS_MAPPING.get(raw_cls, raw_cls)
            all_predictions.append({
                'class_name': clean_cls,
                'probability': float(proba * 100.0)
            })
        all_predictions = sorted(all_predictions, key=lambda x: x['probability'], reverse=True)
        
        print(f"[DIAGNOSA] Hasil: {raw_pred_class} -> {clean_pred_class} ({confidence:.2f}%)")
        return clean_pred_class, confidence, all_predictions
        
    except Exception as e:
        print(f"[ERROR] Gagal melakukan diagnosa: {e}")
        return None, 0.0, []

