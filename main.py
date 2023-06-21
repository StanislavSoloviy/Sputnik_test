import numpy as np
import building_images
from CNN import NeuralNetwork


"""Глобальные константы"""
Q_TRAIN = 3000                    # Количество обучающих примеров
Q_TEST = 200                          # Количество тестовых примеров
Q_GENERAL = Q_TRAIN + Q_TEST        # Общее количество примеров
DATEBASE_NAME = 'datebase'          # Имя базы данных для обучения
X = np.arange(-10, 10.01, 0.1)      # Диапазон входящих значений Х для функций
BATCH_SIZE = 32                     # количество тренировочных изображений для обработки перед обновлением параметров модели
IMG_SHAPE = 128                     # размерность 128x128 к которой будет приведено входное изображение
NETWORK_NAME = "CNN.h5"             # Имя нейросети для загрузки и сохранения
EPOCHS = 5                        # Количество эпох




if __name__ == '__main__':
    """Основной раздел программы"""
    model = NeuralNetwork()

    """Создание базы данных. Закоментировать, если база уже создана"""
    #database = building_images.DataBase()
    #database.create()

    """Создание и обучение нейросети. Закомментировать, если модель обучена"""
    # model.create()
    # model.train()

    """Загрузка обученной модели"""
    model.load()
    """Проверка"""
    model.predict("datebase/test/3000.png")
    model.predict("datebase/test/3001.png")
    model.predict("datebase/test/3002.png")
