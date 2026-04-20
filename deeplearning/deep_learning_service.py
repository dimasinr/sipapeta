import os
import numpy as np
import joblib
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# ======================
# CONFIG
# ======================
IMG_SIZE = 224
BATCH_SIZE = 32
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset/train/")
MODEL_DIR = os.path.join(BASE_DIR, "model/")

# Load model secara global agar tidak di-load ulang setiap kali prediksi
_base_model = None
_model = None
_pca = None
_knn = None
_classes = None

def get_feature_extractor():
    global _base_model, _model
    if _model is None:
        _base_model = MobileNetV2(weights='imagenet', include_top=False, pooling='avg', input_shape=(224, 224, 3))
        _model = Model(inputs=_base_model.input, outputs=_base_model.output)
    return _model

def get_best_k(features_pca, labels):
    # ======================
    # GRID SEARCH KNN
    # ======================
    print("Melakukan Grid Search untuk KNN...")

    param_grid = {
        'n_neighbors': [3, 5],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan']
    }

    grid = GridSearchCV(
        KNeighborsClassifier(),
        param_grid,
        cv=3,              
        verbose=1,
        n_jobs=-1
    )

    grid.fit(features_pca, labels)

    print("Best Parameters:", grid.best_params_)

    knn = grid.best_estimator_
    return knn

def train_model():
    """Melatih PCA dan KNN menggunakan fitur yang diekstrak dari MobileNetV2."""
    print("Memulai proses training...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # datagen = ImageDataGenerator(rescale=1./255)
    datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    generator = datagen.flow_from_directory(
        DATASET_PATH,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    model = get_feature_extractor()
    
    print("Mengekstraksi fitur...")
    features = model.predict(generator, verbose=1)
    labels = generator.classes
    
    print("Feature shape:", features.shape)
    
    print("Melakukan PCA...")
    # Menghindari error jika jumlah sampel kurang dari 100
    n_components = min(100, features.shape[1])
    pca = PCA(n_components=n_components, random_state=42)
    features_pca = pca.fit_transform(features)
    
    print("\n[1] Melatih Base KNN (Default n_neighbors=5)...")
    base_knn = KNeighborsClassifier(n_neighbors=5)
    base_knn.fit(features_pca, labels)
    base_predictions = base_knn.predict(features_pca)
    base_acc = accuracy_score(labels, base_predictions)
    
    print("\n[2] Memulai Evaluasi dengan Grid Search KNN...")
    best_knn = get_best_k(features_pca, labels) # memanggil fungsi yang sudah Anda buat
    best_predictions = best_knn.predict(features_pca)
    best_acc = accuracy_score(labels, best_predictions)
    
    class_names = list(generator.class_indices.keys())
    
    print("\n==============================================")
    print("HASIL PERBANDINGAN MODEL (Pada Data Training)")
    print("==============================================")
    
    print("\n--- 1. BASE KNN ---")
    print(f"Akurasi  : {base_acc*100:.2f}%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, base_predictions))
    print("\nClassification Report:")
    print(classification_report(labels, base_predictions, target_names=class_names))
    
    print("\n--- 2. GRID SEARCH KNN ---")
    print(f"Akurasi  : {best_acc*100:.2f}%")
    print("Parameter: ", best_knn.get_params())
    print("\nConfusion Matrix:")
    print(confusion_matrix(labels, best_predictions))
    print("\nClassification Report:")
    print(classification_report(labels, best_predictions, target_names=class_names))
    
    print("\nMenyimpan model terbaik hasil Grid Search...")
    joblib.dump(pca, os.path.join(MODEL_DIR, "pca.pkl"))
    joblib.dump(best_knn, os.path.join(MODEL_DIR, "knn.pkl"))
    
    # Simpan mapping label (id -> nama kelas)
    class_mapping = {v: k for k, v in generator.class_indices.items()}
    joblib.dump(class_mapping, os.path.join(MODEL_DIR, "classes.pkl"))
    
    print("Model berhasil disimpan di folder 'model/'!")

def load_models():
    """Memuat PCA, KNN, dan label kelas."""
    global _pca, _knn, _classes
    if _pca is None or _knn is None or _classes is None:
        try:
            _pca = joblib.load(os.path.join(MODEL_DIR, "pca.pkl"))
            _knn = joblib.load(os.path.join(MODEL_DIR, "knn.pkl"))
            _classes = joblib.load(os.path.join(MODEL_DIR, "classes.pkl"))
        except FileNotFoundError:
            print("Error: Model belum dilatih. Jalankan train_model() terlebih dahulu.")
            return False
    return True

def predict(image_path):
    """
    Memprediksi kelas dari gambar yang diberikan menggunakan model Hybrid CNN-KNN.
    """
    if not load_models():
        return None, 0.0
        
    model = get_feature_extractor()
    
    # Load and preprocess image

    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))
    img_array = img_to_array(img)
    # img_array = img_array / 255.0  # Rescale ke 0-1
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan dimensi batch
    
    # Extract features
    features = model.predict(img_array, verbose=0)
    
    # PCA transform
    features_pca = _pca.transform(features)
    
    # Predict with KNN
    class_idx = _knn.predict(features_pca)[0]
    
    # Get probabilities
    probas = _knn.predict_proba(features_pca)[0]
    confidence = float(np.max(probas) * 100) # Persentase confidence
    
    class_name = _classes[class_idx]
    
    return class_name, confidence

if __name__ == '__main__':
    train_model()