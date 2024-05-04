from catboost.datasets import titanic
import pandas as pd

# загружаем датасет
titanic_train, titanic_test = titanic()

# сохраняем датасет в csv файл
titanic_test.to_csv(r"C:\Users\Santerr80\OneDrive\Документы\GitHub\mlops_practice\lab4\datasets\titanic.csv", index=False)

# выведем первые 3 строки датасета
print(titanic_train.info())
