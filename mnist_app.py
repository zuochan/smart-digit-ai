import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
import random
import os
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import uuid
from pathlib import Path

st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(to right, #667eea, #764ba2);
        background-attachment: fixed;
        background-size: cover;
        color: white;
    }
    
    div[class*="CanvasToolbar"] {
        background-color: #000000 !important;
    }

    canvas {
        border: none !important;
    }

    h1, h2, h3, h4, h5, h6, p, span, label {
        color: white !important;
    }
    .stTextInput > div > div > input {
        color: white;
        background-color: #333333;
    }
    .main {
        background-color: #f4f4f4;
        padding: 20px;
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        font-size: 16px;
        border-radius: 5px;
    }
    </style>
""", unsafe_allow_html=True)

def load_feedback_data(folder="feedback_data"):
    images = []
    labels = []
    for filename in os.listdir(folder):
        if filename.endswith(".png") and filename.startswith("feedback_"):
            label = int(filename.split("_")[1])  # 例: "feedback_3_abcd1234.png"
            path = os.path.join(folder, filename)
            img = Image.open(path).resize((28, 28)).convert("L")
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(label)
    return np.array(images), np.array(labels)


st.markdown("""
<div style='
    padding: 25px 15px;
    border-radius: 12px;
    text-align: center;
    color: white;
'>
    <h1 style='margin-bottom: 10px;'>🔢 Smart Digit AI</h1>
    <p style='font-size: 17px;'>Sketch a digit. AI will try to recognize it. Let's see who's smarter.</p>
</div>
""", unsafe_allow_html=True)

# データ読み込み・前処理
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_test = x_test / 255.0
y_test = to_categorical(y_test, 10)

# 保存済みモデルがあれば読み込む、なければ学習して保存
if os.path.exists("mnist_model.keras"):
    model = load_model("mnist_model.keras")
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
else:
    model = Sequential([
        Flatten(input_shape=(28, 28)),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(x_train / 255.0, to_categorical(y_train, 10), epochs=3, batch_size=32, validation_split=0.1, verbose=0)
    model.save("mnist_model.keras")
    
# ランダムなテスト画像の選択
index = random.randint(0, 9999)
image = x_test[index]
label = np.argmax(y_test[index])

# 予測
prediction = model.predict(image.reshape(1, 28, 28))
predicted_label = np.argmax(prediction)

# 表示
with st.container():
    st.header("Model Prediction (from Test Set)")
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=f"Correct Label: {label}", width=150)
    with col2:
        st.subheader(f"Prediction: {predicted_label}")  

with st.container():
    st.header("Draw a Digit")
    st.markdown("Use your mouse to draw a number (0-9) in the box below:")

    col1, col2 = st.columns(2)
    with col1:
        canvas_result = st_canvas(
            fill_color="#000000",
            stroke_width=15,
            stroke_color="#FFFFFF",
            background_color="#333333",
            height=300,
            width=344,
            drawing_mode="freedraw",
            key="canvas"
        )
    with col2:
        if canvas_result.image_data is not None:
            # 画像の赤チャンネルを取得（手描き線は白＝255）
            red_channel = canvas_result.image_data[:, :, 0]

            # 白いピクセルの数をカウント
            white_pixels = np.sum(red_channel > 100)

            # ある程度以上描かれていたら処理実行（例: 100ピクセル以上）
            if white_pixels >= 100:
                image = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
                image = image.resize((28, 28)).convert('L')
                img_array = np.array(image) / 255.0

                prediction = model.predict(img_array.reshape(1, 28, 28))
                predicted_label = np.argmax(prediction)

                st.subheader(f"Prediction: {predicted_label}")    

                correct_label = st.text_input("If the prediction is incorrect, enter the correct digit (0–9):")

                if st.button("Save Correction & Retrain") and correct_label.isdigit() and 0 <= int(correct_label) <= 9:
                    label = int(correct_label)

                    # --------- 1. 画像の保存（ログ用） ---------
                    feedback_dir = Path("feedback_data")
                    feedback_dir.mkdir(exist_ok=True)
                    unique_id = uuid.uuid4().hex[:8]
                    filename = feedback_dir / f"feedback_{label}_{unique_id}.png"
                    image.save(filename)

                    # --------- 2. モデルの再学習 ---------
                    x = img_array.reshape(1, 28, 28)
                    y = to_categorical([label], 10)
                    model.fit(x, y, epochs=3, batch_size=1, verbose=0)
                    model.save("mnist_model.keras")
                    model.compile() 

                    # --------- 3. 通知 ---------
                    st.success(f"Correction saved to: {filename}")
                    st.success("Model retrained on new sample and saved!")
            else:
                st.subheader("Prediction: -")
