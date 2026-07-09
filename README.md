# 🍽️ Food Item Classifier

A CNN-based food image classifier built with TensorFlow/Keras and served via a Streamlit web app.

---

## ✅ Features

- Upload any food image (JPG / PNG / WEBP)
- Instant food name prediction
- Confidence score with colour-coded bar
- Category label (Fruit / Fast Food / Snack)
- Top-5 alternative predictions

---

## 📁 Folder Structure

```
Food_Classifier/
│
├── dataset/
│      ├── Apple/
│      ├── Banana/
│      ├── Pizza/
│      ├── Burger/
│      └── Sandwich/
│
├── model/
│      └── food_model.keras
│
├── train.py          ← Train the CNN model
├── app.py            ← Streamlit web app
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare dataset

Place images inside the matching class folder:

```
dataset/
  Apple/    → apple_1.jpg, apple_2.jpg …
  Banana/   → banana_1.jpg …
  Pizza/    → pizza_1.jpg …
  Burger/   → burger_1.jpg …
  Sandwich/ → sandwich_1.jpg …
```

Aim for **at least 100 images per class** for good accuracy.  
Free sources: [Kaggle](https://www.kaggle.com/), [Google Images](https://images.google.com/), [Food-101](https://huggingface.co/datasets/food101).

### 3. Train the model

```bash
python train.py
```

Training uses **MobileNetV2** (ImageNet pre-trained) with two phases:
- Phase 1 — top layers only (fast)
- Phase 2 — fine-tune last 30 base layers

The best checkpoint is automatically saved to `model/food_model.keras`.

### 4. Launch the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## 🛠️ Technologies

| Tool | Role |
|---|---|
| Python 3.10+ | Core language |
| TensorFlow / Keras | Model training & inference |
| MobileNetV2 (CNN) | Transfer learning backbone |
| Streamlit | Web UI |
| OpenCV | Image utilities |
| NumPy | Array processing |
| Pillow | Image I/O |

---

## 📊 Model Details

| Parameter | Value |
|---|---|
| Architecture | MobileNetV2 + custom head |
| Input size | 224 × 224 × 3 |
| Classes | 5 (Apple, Banana, Pizza, Burger, Sandwich) |
| Optimizer | Adam |
| Loss | Categorical cross-entropy |
| Augmentation | Rotation, flip, zoom, brightness |

---

## 💡 Tips

- Add more classes by creating new folders in `dataset/` and re-running `train.py`.
- Lower confidence (< 50 %) shown in red — try a clearer photo.
- Model auto-reloads in the app via `@st.cache_resource`.
