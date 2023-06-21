import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import cv2
import json
import main
from tensorflow.keras.applications import VGG16


class NeuralNetwork:
    def __init__(self):
        self.__network_path = main.NETWORK_NAME
        self.__model = None
        self.__batch_size = main.BATCH_SIZE  # Размер батча
        self.__img_shape = main.IMG_SHAPE  # Разрешение
        self.__datebase_name = main.DATEBASE_NAME  # Относительный путь до базы данных
        self.__q_train = main.Q_TRAIN  # Количество обучающих примеров
        self.__q_test = main.Q_TEST  # Количество тестовых примеров
        self.__q_general = main.Q_GENERAL  # Общее количество примеров
        self.__epochs = main.EPOCHS  # Кол-во эпох

    def convert_img(self, path):
        """Преобразование изображений из базы данных аод вход в нейросеть"""
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        # x, y = 80, 60
        # height, width = img.shape[:2]
        # img = img[y: height - y, x:width - x]
        img = img / 255.0
        img = cv2.resize(img, (self.__img_shape, self.__img_shape))
        img = np.array(img).reshape(128, 128, 3)
        return img

    def __call__(self, *args, **kwargs):
        return self.__model(*args, **kwargs)

    def create(self):
        """Создание модели"""
        # Используется готовая модель VGG16 из библиотеки keras
        base_model = VGG16(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
        x = base_model.output
        x = tf.keras.layers.GlobalAveragePooling2D()(x)
        x = tf.keras.layers.Dense(128, activation='relu')(x)
        predictions = tf.keras.layers.Dense(8, activation='softmax')(x)
        model = tf.keras.models.Model(inputs=base_model.input, outputs=predictions)
        for layer in base_model.layers:
            layer.trainable = False

        # Компилирование модели
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        # Представление модели
        model.summary()
        self.__model = model

    def save(self):
        """Сохранение модели"""
        self.__model.save(self.__network_path)
        print("Модель сохранена")

    def load(self):
        """Загрузка модели"""
        try:
            self.__model = tf.keras.models.load_model(self.__network_path)
            print(f"Moдель {self.__network_path} загружена")
        except:
            print(f"Moдель {self.__network_path} не найдена")

    def train(self):
        """Обучение модели"""
        # Подготовка обучающих и тестовых данных
        x_train = np.array([self.convert_img(self.__datebase_name + "/train/" + str(i) + ".png")
                            for i in range(self.__q_train)])
        with open(self.__datebase_name + "/annotations_train.json", "r") as json_file:
            data = json.load(json_file)  # передаем файловый объект
            y_train = np.array([value for value in data.values()])
            y_train_cat = tf.keras.utils.to_categorical(y_train, 8)

        x_test = np.array([self.convert_img(self.__datebase_name + "/test/" + str(i) + ".png")
                           for i in range(self.__q_train, self.__q_general)])
        with open(self.__datebase_name + "/annotations_test.json", "r") as json_file:
            data = json.load(json_file)  # передаем файловый объект
            y_test = np.array([value for value in data.values()])
            y_test_cat = tf.keras.utils.to_categorical(y_test, 8)

        # Обучение модели
        history = self.__model.fit(x_train, y_train_cat, batch_size=self.__batch_size,
                                   epochs=self.__epochs,
                                   validation_split=0.2)
        # Тестирование модели
        self.__model.evaluate(x_test, y_test_cat)
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        # Построение графика точности
        epochs_range = range(self.__epochs)
        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Точность на обучении')
        plt.plot(epochs_range, val_acc, label='Точность на валидации')
        plt.legend(loc='lower right')
        plt.title('Точность на обучающих и валидационных данных')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Потери на обучении')
        plt.plot(epochs_range, val_loss, label='Потери на валидации')
        plt.legend(loc='upper right')
        plt.title('Потери на обучающих и валидационных данных')
        plt.savefig('./foo.png')
        plt.show()
        print("Модель обучена")

        self.save()

    def predict(self, img_way):
        """Предсказание изображения"""
        img = self.convert_img(img_way)
        x = np.expand_dims(img, axis=0)  # представляем изображение в виде трёхмерного тензора
        res = self.__model.predict(x)
        print(res)
        print(f"Ожидаемое значение: {np.argmax(res)}")  # Аргумент с наибольшей вероятностью
        plt.imshow(img)
        plt.show()

    def check(self, img):
        """Проверка тренировочных данных"""
        x_train = np.array([self.convert_img(self.__datebase_name + "/train/" + str(i) + ".png")
                            for i in range(self.__q_train)])
        with open(self.__datebase_name + "/annotations_train.json", "r") as json_file:
            data = json.load(json_file)  # передаем файловый объект
            y_train = np.array([value for value in data.values()])
            y_train_cat = tf.keras.utils.to_categorical(y_train, 8)
        print("y_train = ", y_train[img])
        print("y_train_cat = ", y_train_cat[img])
        plt.imshow(x_train[img])
        plt.show()


