# 🧠 AI‑vs‑Real Image Classifier (CNN)

**A compact Convolutional Neural Network (CNN) that accurately performs binary classification to distinguish between AI‑generated and real images.**

---

## 🚀 Project Overview

This Colab notebook implements a streamlined pipeline to train a CNN on a labeled dataset of images—some real, others AI‑generated. Using convolutional, pooling, and fully connected layers, the model learns visual patterns to output whether each image is **“Real”** or **“AI‑generated.”**

---
## LINK TO THE DATASET
[Dataset](kaggle competitions download -c iiti-ml-starters-hackathon)

## 🗂️ Dataset Preparation

- Load your images and labels (e.g., via Google Drive).  
- Organize into train / validation (and optionally test) folders with two subfolders:  
  - `real/`  
  - `ai_generated/`  
- Example structure:
  ```
  dataset/
  ├── train/
  │   ├── real/
  │   └── ai_generated/
  └── val/
      ├── real/
      └── ai_generated/
  ```

---

## 🔄 Data Augmentation & Preprocessing

- Resize all images to a uniform shape (e.g., 224×224×3).  
- Apply augmentations (flip, rotate, zoom, brightness, etc.) using `ImageDataGenerator` from Keras.  
- Normalize pixel values to `[0, 1]`.

---

## 🧠 Model Architecture

Build a compact CNN using Keras’ `Sequential` API:

```python
model = Sequential([
  Conv2D(filters=32, kernel_size=3, activation='relu', input_shape=(224,224,3)),
  MaxPooling2D(2),
  Conv2D(filters=64, kernel_size=3, activation='relu'),
  MaxPooling2D(2),
  Conv2D(filters=128, kernel_size=3, activation='relu'),
  MaxPooling2D(2),
  Flatten(),
  Dense(128, activation='relu'),
  Dropout(0.5),
  Dense(1, activation='sigmoid')  # Outputs probability [0,1]
])
```

- Loss Function: `binary_crossentropy`  
- Optimizer: `Adam`  
- Metric: `accuracy`

---

## 📈 Training & Validation

Compile and train the model:

```python
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

history = model.fit(train_generator,
                    validation_data=val_generator,
                    epochs=20)
```

---

## 🎯 Evaluation & Metrics

Evaluate model performance on validation data:

```python
results = model.evaluate(val_generator)
```

Generate a classification report and confusion matrix:

```python
from sklearn.metrics import classification_report, confusion_matrix

Y_pred = model.predict(val_generator)
y_pred = (Y_pred > 0.5).astype(int)

print(classification_report(val_generator.classes, y_pred))
print(confusion_matrix(val_generator.classes, y_pred))
```

---

## 🧾 Inference & Usage

Classify a new image:

```python
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

img = load_img('path/to/image.jpg', target_size=(224,224))
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

prob = model.predict(x)[0][0]
label = 'AI‑generated' if prob > 0.5 else 'Real'
print(f"{label} ({prob:.2f})")
```

---

## 📊 Results & Performance

The model typically achieves around **90‑95% accuracy**, depending on:

- Quality of labeled images  
- Diversity in augmentation  
- Model complexity & training data volume

---

## 🛠️ How to Use

1. Open the Colab notebook  
2. Upload or mount your dataset  
3. Adjust paths and model config as needed  
4. Run the cells to train, evaluate, and infer  
5. Use the inference section to classify new images

---

## ✅ Requirements

- Python 3.x  
- TensorFlow 2.x  
- NumPy  
- scikit-learn  
- Matplotlib (for plotting)

Install dependencies (if needed):

```bash
pip install tensorflow numpy scikit-learn matplotlib
```

---


## 📌 Future Improvements

- Use pre-trained models (e.g., ResNet, EfficientNet)  
- Add Grad-CAM for visual explainability  
- Improve dataset size and balance  
- Deploy using Flask, FastAPI, or Streamlit
