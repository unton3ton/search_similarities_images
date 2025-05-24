# pip install opencv-python numpy matplotlib
# pip install -U scikit-learn

# https://telegra.ph/Sistema-poiska-pohozhih-izobrazhenij-04-28
# Импортируем нужные библиотеки

import cv2
import numpy as np
import os
# import matplotlib
# matplotlib.use('TkAgg')  # Установите бэкенд на TkAgg
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity


# ========== Настройка ==========

# Папка с изображениями базы
# IMAGE_FOLDER = "/media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat"
IMAGE_FOLDER = "/media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog"


# Путь к изображению-запросу
QUERY_IMAGE_PATH = "11.jpg"


# Сколько похожих картинок показать
TOP_K = 5


# ========== Функция: Вытаскиваем вектор признаков изображения ==========

def extract_features(image_path):
    """
    Извлекает вектор признаков из изображения:
    уменьшаем его и усредняем цвета
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Не удалось загрузить изображение: {image_path}")
        return None  # Возвращаем None, если изображение не загружено

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Уменьшаем размер изображения до 32x32 для упрощения
    # image = cv2.resize(image, (32, 32))

    image = cv2.resize(image, (64, 64))

    # Преобразуем изображение в один длинный вектор
    features = image.flatten()

    # Нормализуем вектор признаков
    features = features / 255.0

    return features


# ========== Загружаем базу изображений ==========

database_features = []
database_filenames = []

for filename in os.listdir(IMAGE_FOLDER):
    if filename.endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(IMAGE_FOLDER, filename)
        features = extract_features(path)
        
        if features is not None:  # Проверяем, что features не None
            database_features.append(features)
            database_filenames.append(filename)
        else:
            print(f"Пропускаем файл из-за ошибки: {path}")

# Преобразуем в массив NumPy только если есть валидные признаки
if database_features:
    database_features = np.array(database_features)
else:
    print("Нет доступных признаков для создания массива.")



database_features = np.array(database_features)

# ========== Обрабатываем изображение-запрос ==========
query_features = extract_features(QUERY_IMAGE_PATH)

# ========== Вычисляем похожесть ==========
# Используем косинусную меру близости
similarities = cosine_similarity([query_features], database_features)[0]

# Сортируем индексы по убыванию схожести
sorted_indices = np.argsort(similarities)[::-1]

# ========== Показываем результаты ==========
# Загружаем изображение-запрос
query_image = cv2.imread(QUERY_IMAGE_PATH)
query_image = cv2.cvtColor(query_image, cv2.COLOR_BGR2RGB)

# Ри# Рисуем
plt.figure(figsize=(15, 5))

# Показываем запрос
plt.subplot(1, TOP_K + 1, 1)
plt.imshow(query_image)
plt.title("Запрос")
plt.axis('off')

# Показываем топ-K похожих
for i in range(TOP_K):
    filename = database_filenames[sorted_indices[i]]
    image_path = os.path.join(IMAGE_FOLDER, filename)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    plt.subplot(1, TOP_K + 1, i + 2)
    plt.imshow(img)
    plt.title(f"Похожее {i+1}")
    plt.axis('off')

plt.tight_layout()

# Сохраняем график в файл
plt.savefig(f"results-{QUERY_IMAGE_PATH[:-4]}-64x64.png")  # Укажите нужное имя файла
plt.close()  # Закрываем фигуру, чтобы освободить память
# plt.show()

'''
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/10125.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/10125.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/10404.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/10404.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/10501.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/10501.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/10820.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/10820.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/11210.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/11210.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/11565.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/11565.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/11874.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/11874.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/11935.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/11935.jpg
Corrupt JPEG data: 239 extraneous bytes before marker 0xd9
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/140.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/140.jpg
Corrupt JPEG data: 214 extraneous bytes before marker 0xd9
Corrupt JPEG data: 128 extraneous bytes before marker 0xd9
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/2663.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/2663.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/3300.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/3300.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/3491.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/3491.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/4833.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/4833.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/5553.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/5553.jpg
Corrupt JPEG data: 99 extraneous bytes before marker 0xd9
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/660.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/660.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/666.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/666.jpg
Corrupt JPEG data: 1153 extraneous bytes before marker 0xd9
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/7968.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/7968.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/7978.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/7978.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/8470.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/8470.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/850.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/850.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/9171.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/9171.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/936.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/936.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/9565.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/9565.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/9778.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Cat/9778.jpg
'''

'''
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/10158.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/10158.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/10401.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/10401.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/10747.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/10747.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/10797.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/10797.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/11410.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/11410.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/11675.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/11675.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/11702.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/11702.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/11849.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/11849.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/11853.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/11853.jpg
Corrupt JPEG data: 399 extraneous bytes before marker 0xd9
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/1308.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/1308.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/1866.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/1866.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/2384.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/2384.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/2688.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/2688.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/2877.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/2877.jpg
Corrupt JPEG data: 226 extraneous bytes before marker 0xd9
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/3136.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/3136.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/3288.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/3288.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/3588.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/3588.jpg
Corrupt JPEG data: 162 extraneous bytes before marker 0xd9
Warning: unknown JFIF revision number 0.00
Corrupt JPEG data: 1403 extraneous bytes before marker 0xd9
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/4367.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/4367.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/5604.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/5604.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/5736.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/5736.jpg
Corrupt JPEG data: 254 extraneous bytes before marker 0xd9
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/6059.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/6059.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/6238.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/6238.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/6718.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/6718.jpg
Corrupt JPEG data: 2230 extraneous bytes before marker 0xd9
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/7112.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/7112.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/7133.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/7133.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/7369.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/7369.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/7459.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/7459.jpg
Corrupt JPEG data: 65 extraneous bytes before marker 0xd9
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/7969.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/7969.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/8730.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/8730.jpg
Не удалось загрузить изображение: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/9188.jpg
Пропускаем файл из-за ошибки: /media/chollet/Новый том/PLAATJES/PetImagesOriginalSize/Dog/9188.jpg
'''