from catboost.datasets import titanic
import pandas as pd
import os

# загружаем датасет
titanic_train, titanic_test = titanic()

pd.set_option('display.max_columns', None)

# сохраняем датасет в csv файл
try:
    if os.path.exists(r"C:\Users\Santerr80\OneDrive\Документы\GitHub\
                     mlops_practice\lab4\datasets\titanic.csv") is True:
        pass
    else:
        titanic_test.to_csv(r"C:\Users\Santerr80\OneDrive\Документы\GitHub\
                            mlops_practice\lab4\datasets\titanic.csv",
                            index=False)
except Exception as e:
    print(e)

# выведем строки датасета
print(titanic_test)

# внесем изменения в датасет
titanic_test.iloc[0, 3] = 'female'
titanic_test.iloc[0, 4] = 52.0

# выведем первые строки датасета
print(titanic_test.head())

# сохраняем датасет в csv файл
titanic_test.to_csv(r"C:\Users\Santerr80\OneDrive\Документы\GitHub\mlops_practice\lab4\datasets\titanic.csv", index=False)

# заполним пропуски в данных столбца Age средним значением
titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].mean())

# выведем первые строки датасета
print(titanic_test)

# сохраняем датасет в csv файл
titanic_test.to_csv(r"C:\Users\Santerr80\OneDrive\Документы\GitHub\mlops_practice\lab4\datasets\titanic.csv", index=False)




