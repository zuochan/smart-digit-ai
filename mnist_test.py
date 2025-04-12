import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.utils import to_categorical
import random

# データを読み込む
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# データの形状を表示
print("訓練画像の形状:", x_train.shape)
print("訓練ラベルの形状:", y_train.shape)

# 最初の画像を表示
plt.imshow(x_train[0], cmap='gray')
plt.title(f"Label: {np.argmax(to_categorical(y_train)[0])}")
# plt.show()

# 正規化（0〜1に）
x_train = x_train / 255.0
x_test = x_test / 255.0

# one-hotエンコード
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# モデル構築
model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(128, activation='relu'),
    Dense(10, activation='softmax')
])

# モデルコンパイル
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


# モデル学習
model.fit(x_train, y_train, epochs=5, batch_size=32, validation_split=0.1, verbose=1)
model.save("mnist_model.keras")

# テストデータで評価
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"\nテストデータでの精度: {test_acc:.4f}")

# index = random.randint(0, 9999)
# image = x_test[index]

# # 予測（1枚だけ予測するには shape を調整）
# prediction = model.predict(image.reshape(1, 28, 28))
# predicted_label = np.argmax(prediction)

# # 結果表示
# plt.imshow(image, cmap='gray')
# plt.title(f"Prediction: {predicted_label}")
# plt.show()