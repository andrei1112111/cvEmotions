import cv2
import numpy as np
import os


def load_image(path: str):
    """
    Загружает изображение по указанному пути.

    Args:
        path (str): путь к изображению

    Returns:
        np.ndarray: изображение в формате BGR

    Raises:
        FileNotFoundError: если файл не найден или не читается
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл '{path}' не найден.")
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Не удалось прочитать изображение: {path}")
    return image


def detect_faces(image: np.ndarray, scale_factor: float = 1.1, min_neighbors: int = 3):
    """
    Находит лица на изображении с помощью каскада Хаара.

    Args:
        image (np.ndarray): изображение BGR
        scale_factor (float): коэффициент масштабирования
        min_neighbors (int): минимальное количество соседей

    Returns:
        list[tuple[int, int, int, int]]: список координат (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    return faces


def detect_eyes(image: np.ndarray, scale_factor: float = 1.05, min_neighbors: int = 4):
    """
    Находит глаза на изображении с помощью каскада Хаара.
    Обычно используется внутри области лица.

    Args:
        image (np.ndarray): изображение лица (BGR)
        scale_factor (float): коэффициент масштабирования
        min_neighbors (int): минимальное количество соседей

    Returns:
        list[tuple[int, int, int, int]]: список координат глаз (x, y, w, h)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
    eyes = eye_cascade.detectMultiScale(gray, scale_factor, min_neighbors)
    return eyes


def detect_eye_whites(image: np.ndarray, eyes: list[tuple[int, int, int, int]], step: int = 5):
    """
    Находит и подсвечивает белки глаз на изображении.

    Args:
        image (np.ndarray): изображение (BGR)
        eyes (list): список координат найденных глаз
        step (int): шаг увеличения порога яркости

    Returns:
        tuple[np.ndarray, int]: (обновлённое изображение, количество найденных белков)
    """
    total_whites = 0
    b_channel = cv2.split(image)[0]  # используем синий канал

    for (x, y, w, h) in eyes:
        # Центральная часть глаза
        x1, y1 = x + int(w * 0.2), y + int(h * 0.2)
        x2, y2 = x + int(w * 0.8), y + int(h * 0.8)

        roi = b_channel[y1:y2, x1:x2]
        roi_color = image[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        mid = np.median(roi) * 0.8
        valid_contours = []

        # Подбор порога по количеству найденных областей
        while mid < 255:
            _, mask = cv2.threshold(roi, float(mid), 255, cv2.THRESH_BINARY)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            valid_contours = [c for c in contours if 20 < cv2.contourArea(c) < 500]

            if len(valid_contours) <= 1:
                break
            mid += step

        cv2.drawContours(roi_color, valid_contours, -1, (0, 255, 255), 2)
        total_whites += len(valid_contours)

    return image, total_whites
