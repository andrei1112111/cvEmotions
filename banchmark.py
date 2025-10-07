import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.datasets import fetch_lfw_people
import cv2
from detect_emotion import classify_emotion
import numpy as np
import os
import ssl

# Игнорируем проверку SSL при загрузке LFW
ssl._create_default_https_context = ssl._create_unverified_context

def detect_emotion(img_array):
    """Обертка для работы с numpy массивом из LFW"""
    img_bgr = cv2.cvtColor((img_array * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    tmp_path = "tmp_face.jpg"
    cv2.imwrite(tmp_path, img_bgr)
    emotion = classify_emotion(tmp_path)

    try: 
        os.remove(tmp_path)
    except OSError: 
        pass

    return img_array, emotion


print("Загрузка датасета...")
faces = fetch_lfw_people(color=False, resize=1)
print(f"Загружено {len(faces.images)} лиц")

# Обрабатываем и собираем только лица с эмоциями
faces_with_emotions = []
for img in tqdm(faces.images):
    _, emotion = detect_emotion(img)
    if emotion != "нейтрально":  # только лица с эмоцией
        faces_with_emotions.append((img, emotion))

print(f"Найдено {len(faces_with_emotions)} лиц с эмоциями")

# Отображаем в сетке 4x4
rows, cols = 4, 4
plt.figure(figsize=(12, 12))
for i, (img, emotion) in enumerate(faces_with_emotions[:16]):
    plt.subplot(rows, cols, i + 1)
    plt.imshow(img, cmap='gray')
    plt.axis("off")
    plt.title(emotion)
plt.tight_layout()
plt.show()
