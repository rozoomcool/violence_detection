import cv2
import numpy as np
import tensorflow as tf
from keras.src.saving import load_model

# from tensorflow.keras.models import load_model

# Загрузка обученной модели
model = load_model('action_recognition_model.keras')  # Укажите путь к вашей модели

# Список классов (например, действия)
actions = ["Нормальное", "Драка", "Драка с оружием"]

# Параметры видеопотока
cap = cv2.VideoCapture(0)  # Используем камеру (0 - это встроенная камера)
sequence = []
SEQ_LENGTH = 16  # Длина последовательности кадров

def preprocess_frame(frame):
    # Изменение размера кадра до 224x224 и нормализация
    frame = cv2.resize(frame, (224, 224))
    frame = frame.astype('float32') / 255.0
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Предобработка кадра
    preprocessed_frame = preprocess_frame(frame)

    # Добавляем кадр в последовательность
    sequence.append(preprocessed_frame)
    if len(sequence) == SEQ_LENGTH:
        # Преобразуем последовательность в нужную форму (batch_size, SEQ_LENGTH, 224, 224, 3)
        input_data = np.expand_dims(sequence, axis=0)

        # Выполняем предсказание
        preds = model.predict(input_data)[0]

        # Получаем индекс класса с максимальной вероятностью
        action = actions[np.argmax(preds)]

        # Выводим распознанное действие
        cv2.putText(frame, f'Action: {action}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Сбрасываем последовательность
        sequence = []

    # Отображаем кадр
    cv2.imshow('Action Recognition', frame)

    # Выход из программы при нажатии клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()

