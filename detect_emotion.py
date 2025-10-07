from detectors.detect_smile import detect_smiles
from detectors.detect_eye_whites import load_image, detect_faces, detect_eyes, detect_eye_whites


def classify_emotion(image_path: str) -> str:
    """
    Классифицирует эмоцию на изображении по признакам:
    - улыбка → радость
    - глаза закрыты → усталость
    - иначе → нейтральное состояние
    """
    # обработка ошибок включена в функции load_image, detect_faces, detect_smiles, detect_eyes, detect_eye_whites
    
    img = load_image(image_path)
    faces = detect_faces(img)
    emotion = "нейтрально"

    for (x, y, w, h) in faces:
        face_roi = img[y:y+h, x:x+w]  # бокс с лицом
        _, smiles = detect_smiles(image_path)
        eyes = detect_eyes(face_roi)
        _, whites = detect_eye_whites(face_roi, eyes)

        if smiles > 0:
            emotion = "радость"
        elif whites == 0:
            emotion = "усталость"

    return emotion


def parse_args():
    """
    парсер аргументов командной строки
    """
    
    import argparse
    parser = argparse.ArgumentParser(description="наивный детектор эмоций по картинке")

    parser.add_argument(
        "--image", "-i", type=str, required=True,
        help="картинка с лицом .jpg"
    )

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    argv = parse_args()
    print(classify_emotion(argv.image))


# python emotion_integration.py --image path/to/image.jpg
