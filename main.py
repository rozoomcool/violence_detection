import cv2
import numpy as np
import tensorflow as tf
from keras import Model
from keras.src.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.layers import TimeDistributed, Dense, Dropout, GlobalAveragePooling2D, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Input
import threading

# Функция для сборки модели
def build_model():
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model.trainable = False  # Замораживаем веса базы модели

    # Входная последовательность из 16 кадров
    inputs = Input(shape=(16, 128, 128, 3))

    # Применяем TimeDistributed для обработки каждого кадра через базовую модель
    x = TimeDistributed(base_model)(inputs)
    x = TimeDistributed(GlobalAveragePooling2D())(x)

    # Добавляем LSTM для обработки временной информации
    x = LSTM(64, return_sequences=False)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(3, activation='softmax')(x)  # 3 класса

    model = Model(inputs, outputs)
    model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Загрузка модели
model = build_model()

# Классы действий
actions = ["NORMAL", "Violence", "Weaponized"]
action = actions[0]

# Видеопоток с камеры
cap = cv2.VideoCapture(0)
sequence = []
SEQ_LENGTH = 16  # Последовательность из 16 кадров
PROCESS_FRAME_RATE = 5  # Обрабатываем только каждый 5-ый кадр

# Флаг и блокировка для многопоточности
lock = threading.Lock()
frame_processed = False

# Функция предобработки кадров
def preprocess_frame(frame):
    frame = cv2.resize(frame, (128, 128))  # Уменьшение размера для ускорения
    frame = frame.astype('float32') / 255.0
    return frame

# Функция для распознавания действий в отдельном потоке
def run_inference(sequence):
    global action, frame_processed
    input_data = np.expand_dims(sequence, axis=0)  # (1, 16, 128, 128, 3)

    # Выполняем предсказание
    preds = model.predict(input_data)[0]

    # Определяем действие с максимальной вероятностью
    with lock:
        action = actions[np.argmax(preds)]
        frame_processed = True

frame_count = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Предобработка текущего кадра
    preprocessed_frame = preprocess_frame(frame)

    frame_count += 1
    # Добавляем кадр в последовательность
    if frame_count % PROCESS_FRAME_RATE == 0:
        sequence.append(preprocessed_frame)

    # Если последовательность заполнена, запускаем инференс в отдельном потоке
    if len(sequence) == SEQ_LENGTH:
        threading.Thread(target=run_inference, args=(sequence,)).start()
        sequence = []  # Очищаем последовательность для следующего анализа

    # Отображаем результат
    with lock:
        cv2.putText(frame, f'Action: {action}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow('Action Recognition', frame)

    # Выход при нажатии 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Освобождаем ресурсы
cap.release()
cv2.destroyAllWindows()
