from kaggle.api.kaggle_api_extended import KaggleApi
import pandas as pd
import os.path
from sklearn.model_selection import train_test_split

# Инициализация API Kaggle
api = KaggleApi()
api.authenticate()

# Загрузка датасета
try:
    if os.path.exists(r".\data\car_data.csv") is True:
        pass
    else:
        api.dataset_download_files("volkanastasia/dataset-of-used-cars",
                                   path="./data", unzip=True)
except Exception as e:
    print(e)

# Чтение датасета
df = pd.read_csv(r".\data\car_data.csv")

# Разбиение на обучающую и тестовую выборки
train, test = train_test_split(df, test_size=0.2, random_state=42,
                               shuffle=True)


# Сохранение обучающей и тестовой выборок
try:
    if os.path.exists(r".\data\train.csv") is True:
        pass
    else:
        train.to_csv(r".\data\train.csv")
except Exception as e:
    print(e)

try:
    if os.path.exists(r".\data\test.csv") is True:
        pass
    else:
        test.to_csv(r".\data\test.csv")
except Exception as e:
    print(e)
