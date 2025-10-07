"""
Скрипт для распознавания лиц с использованием каскадов Хаара (OpenCV).
Включает предобработку изображения (улучшение контраста, сглаживание шума) и
дополнительные фильтры (наличие глаз). Опционально детекция лиц в профиль и NMS.
"""

import os
from typing import Tuple
import cv2
import numpy as np


def load_image_from_path(path: str) -> Tuple[np.ndarray, str]:
    """
    Загружает изображение по указанному пути и конвертирует его из BGR (формат OpenCV)
    в RGB (формат matplotlib).

    cv2.imread читает изображение в BGR, даже если исходник png/jpg в RGB,
    поэтому необходимо конвертировать для корректного отображения в matplotlib.

    Args:
        path (str): Путь к изображению.

    Returns:
        Tuple[np.ndarray, str]: Изображение в формате RGB и имя исходного файла.

    Raises:
        FileNotFoundError: Если изображение не может быть прочитано.
    """
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(f"Cannot read image at: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb, os.path.basename(path)


def improve_gray_contrast(gray: np.ndarray) -> np.ndarray:
    """
    Улучшает контрастность градаций серого с использованием адаптивного гистограммного
    выравнивания (CLAHE). Этот метод устойчив к неравномерному освещению.

    Args:
        gray (np.ndarray): Одноканальное изображение (градации серого).

    Returns:
        np.ndarray: Обработанное изображение с улучшенным контрастом.
    """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)


def det_faces(path_image, scale=1.1, neighbors=5, minsize=60, 
              with_profiles=True, check_eyes=True):
    """
    Основная функция:
    - Разбирает аргументы командной строки.
    - Загружает изображение и преобразует его в оттенки серого.
    - Выполняет предобработку (контраст, сглаживание).
    - Детектирует лица в анфас и (опционально) в профиль.
    - (Опционально) фильтрует лица по наличию глаз.
    - Применяет Non-Maximum Suppression (NMS) для устранения дубликатов.
    - Отображает и сохраняет результат.
    """

    rgb, src_name = load_image_from_path(path_image)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    # Предобработка: улучшение контраста и сглаживание шума.
    gray = improve_gray_contrast(gray)
    gray = cv2.GaussianBlur(gray, (3,3), 0)

    face_xml = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(face_xml)

    # Детекция лиц в анфас.
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale,
        minNeighbors=neighbors,
        minSize=(minsize, minsize)
    )
    faces = list(faces)

    # Дополнительно: детекция лиц в профиль.
    if with_profiles:
        prof_xml = cv2.data.haarcascades + "haarcascade_profileface.xml"
        prof = cv2.CascadeClassifier(prof_xml)
        pf = prof.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=neighbors,
            minSize=(minsize, minsize)
        )
        faces.extend(pf)

    # Фильтрация по наличию глаз для уменьшения ложных срабатываний.
    if check_eyes:
        eye_xml = cv2.data.haarcascades + "haarcascade_eye.xml"
        eye_cascade = cv2.CascadeClassifier(eye_xml)
        kept = []
        for (x,y,w,h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(
                roi_gray,
                scaleFactor=1.05,  # шаг меньше стандартного для поиска мелких объектов
                minNeighbors=6,
                minSize=(int(w*0.15), int(h*0.15))  # минимум ~15% ширины/высоты лица
            )
            # Если найден хотя бы один глаз — лицо считается действительным.
            if len(eyes) >= 1:
                kept.append((x,y,w,h))
        faces = kept

    def nms(boxes, iou_thr=0.3):
        """
        Выполняет Non-Maximum Suppression (NMS) для объединения близких
        боксов и устранения дубликатов.

        Args:
            boxes (list): Список ограничивающих прямоугольников [x, y, w, h].
            iou_thr (float): Порог IoU для слияния.

        Returns:
            list: Отфильтрованные боксы.
        """
        if not boxes: return []
        boxes = np.array(boxes)
        x1,y1,w,h = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
        x2, y2 = x1+w, y1+h
        areas = w*h
        order = np.argsort(areas)[::-1]  # индексы по убыванию площади
        keep=[]
        while order.size>0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
            iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
            inds = np.where(iou <= iou_thr)[0]
            order = order[inds+1]
        return boxes[keep].tolist()

    faces = nms(faces)

    return faces
