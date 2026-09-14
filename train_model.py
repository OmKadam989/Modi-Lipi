import os
import json
import random
import numpy as np
import cv2
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from tensorflow.keras import layers, models

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
EVALUATION_DIR = BASE_DIR / "evaluation"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
EVALUATION_DIR.mkdir(parents=True, exist_ok=True)

# Custom external dataset path specified by user
USER_DATASET_DIR = Path(r"C:\ModiLipi\MODI_Lipi_Dataset\archive\MODI MATRA DATASET\MODI MATRA DATASET")
LOCAL_DATASET_DIR = BASE_DIR / "datasets" / "real_modi"

def load_images_from_split_directory(dir_path: Path, label_map: dict = None, img_size: int = 32):
    """Loads images from class subfolders in a train or test split directory."""
    X_data = []
    y_data = []
    
    if not dir_path.exists():
        return np.array([]), np.array([]), {}

    class_names = sorted([d.name for d in dir_path.iterdir() if d.is_dir()])
    if label_map is None:
        label_map = {name: idx for idx, name in enumerate(class_names)}
        
    valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}

    for class_name in class_names:
        if class_name not in label_map:
            continue
        class_dir = dir_path / class_name
        class_idx = label_map[class_name]
        
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in valid_exts:
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    # Square pad and resize
                    h, w = img.shape
                    max_dim = max(h, w)
                    pad_top = (max_dim - h) // 2
                    pad_bottom = max_dim - h - pad_top
                    pad_left = (max_dim - w) // 2
                    pad_right = max_dim - w - pad_left
                    
                    padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
                    resized = cv2.resize(padded, (img_size, img_size), interpolation=cv2.INTER_AREA)
                    
                    X_data.append(resized)
                    y_data.append(class_idx)

    if not X_data:
        return np.array([]), np.array([]), label_map

    X = np.array(X_data, dtype=np.float32) / 255.0
    X = np.expand_dims(X, axis=-1)
    y = np.array(y_data, dtype=np.int32)
    return X, y, label_map

def build_cnn_model(input_shape: tuple, num_classes: int) -> models.Sequential:
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), padding='same', activation='relu', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Conv2D(128, (3, 3), padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def main():
    print("=== Modi Lipi CNN Character Model Training ===")
    img_size = 32
    
    # 1. Check user dataset path first
    target_dataset_dir = None
    if USER_DATASET_DIR.exists():
        target_dataset_dir = USER_DATASET_DIR
        print(f"Found User Dataset at: {USER_DATASET_DIR}")
    elif LOCAL_DATASET_DIR.exists():
        target_dataset_dir = LOCAL_DATASET_DIR
        print(f"Found Local Dataset at: {LOCAL_DATASET_DIR}")

    if target_dataset_dir:
        train_dir = target_dataset_dir / "train"
        test_dir = target_dataset_dir / "test"

        if train_dir.exists() and test_dir.exists():
            print(f"Loading split train/test datasets from {target_dataset_dir}...")
            class_names = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
            label_map = {name: idx for idx, name in enumerate(class_names)}
            
            X_train, y_train, _ = load_images_from_split_directory(train_dir, label_map, img_size)
            X_test, y_test, _ = load_images_from_split_directory(test_dir, label_map, img_size)
        else:
            print(f"Loading single dataset directory from {target_dataset_dir}...")
            class_names = sorted([d.name for d in target_dataset_dir.iterdir() if d.is_dir()])
            label_map = {name: idx for idx, name in enumerate(class_names)}
            X_all, y_all, _ = load_images_from_split_directory(target_dataset_dir, label_map, img_size)
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=0.2, random_state=42, stratify=y_all)

        num_classes = len(class_names)
        print(f"Successfully loaded dataset! Classes ({num_classes}): {class_names}")
        print(f"Train samples: {X_train.shape[0]} | Test samples: {X_test.shape[0]}")
    else:
        print("No dataset found on disk. Please check the dataset path.")
        return

    # Save label encoder JSON
    with open(MODELS_DIR / "label_encoder.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    model = build_cnn_model((img_size, img_size, 1), num_classes)
    model.summary()
    
    epochs = 15
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=32,
        verbose=1
    )
    
    model_save_path = MODELS_DIR / "trained_modi_cnn.keras"
    model.save(str(model_save_path))
    print(f"Model saved successfully to {model_save_path}")
    
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    acc = float(accuracy_score(y_test, y_pred))
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted', zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    metrics = {
        "status": "available",
        "model_loaded": True,
        "accuracy": round(acc, 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "total_classes": num_classes,
        "class_names": class_names,
        "confusion_matrix": cm,
        "training_accuracy_history": [round(float(a), 4) for a in history.history['accuracy']],
        "training_loss_history": [round(float(l), 4) for l in history.history['loss']]
    }
    
    metrics_path = EVALUATION_DIR / "evaluation_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
        
    print("=== Training & Evaluation Completed Successfully ===")
    print(f"Final Test Accuracy: {acc * 100:.2f}% | F1-Score: {f1:.4f}")

if __name__ == "__main__":
    main()
