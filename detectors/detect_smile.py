"""
Скрипт для обнаружения улыбок на изображениях с использованием каскадов Хаара.
Использует detect_faces для обнаружения лиц, затем ищет улыбки в области рта.
"""

import cv2
from detectors.detect_faces import load_image_from_path, improve_gray_contrast, det_faces


def detect_smiles(image_path, scale=1.1, neighbors=5, minsize=60, 
                 with_profiles=True, check_eyes=False, smile_neighbors=3):
    """
    Обнаруживает лица на изображении и находит улыбки в области рта.
    
    Args:
        image_path (str): Путь к изображению
        scale (float): Параметр масштабирования для детектора лиц
        neighbors (int): Минимальное количество соседей для детектора лиц
        minsize (int): Минимальный размер лица
        with_profiles (bool): Использовать детекцию профильных лиц
        check_eyes (bool): Проверять наличие глаз
        smile_neighbors (int): Параметр для детектора улыбок
        
    Returns:
        tuple: Изображение с выделенными лицами и улыбками, количество найденных улыбок
    """
    
    # Загружаем изображение
    rgb, src_name = load_image_from_path(image_path)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    
    # Предобработка
    gray = improve_gray_contrast(gray)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    
    # Обнаруживаем лица
    faces = det_faces(image_path, scale, neighbors, minsize, with_profiles, check_eyes)
    
    # Загружаем каскад для обнаружения улыбок
    smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    smile_count = 0
    
    # Для каждого обнаруженного лица ищем улыбку
    for (x, y, w, h) in faces:
        # Определяем область рта (нижняя часть лица)
        roi_gray = gray[y + int(h * 0.6):y + h, x:x + w]
        roi_color = rgb[y + int(h * 0.6):y + h, x:x + w]
        
        # Обнаруживаем улыбки
        smiles = smile_cascade.detectMultiScale(
            roi_gray,
            scaleFactor=1.5,
            minNeighbors=smile_neighbors,
            minSize=(int(w * 0.15), int(h * 0.08))
        )
        
        # Рисуем прямоугольник вокруг лица
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (0, 255, 0), 2)
        
        # Рисуем прямоугольники вокруг улыбок
        for (sx, sy, sw, sh) in smiles:
            cv2.rectangle(rgb, (x + sx, y + int(h * 0.6) + sy), 
                         (x + sx + sw, y + int(h * 0.6) + sy + sh), 
                         (255, 0, 0), 2)
            smile_count += 1
            
            # Добавляем текст "Smile"
            cv2.putText(rgb, 'Smile', (x + sx, y + int(h * 0.6) + sy - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    return rgb, smile_count
