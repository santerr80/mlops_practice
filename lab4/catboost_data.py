from catboost.datasets import titanic
import pandas as pd
import os

from sklearn.preprocessing import OneHotEncoder


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


# внесем изменения в первую строку датасета
titanic_test.iloc[0, 3] = 'female'
titanic_test.iloc[0, 4] = 52.0

# сохраняем датасет в csv файл
titanic_test.to_csv(r"C:\Users\Santerr80\OneDrive\Документы\GitHub\mlops_practice\lab4\datasets\titanic.csv", index=False)

# заполним пропуски в данных столбца Age средним значением
titanic_test['Age'] = titanic_test['Age'].fillna(titanic_test['Age'].mean())


# сохраняем датасет в csv файл
titanic_test.to_csv(r"C:\Users\Santerr80\OneDrive\Документы\GitHub\mlops_practice\lab4\datasets\titanic.csv", index=False)

# создаем кодировщик
enc = OneHotEncoder(handle_unknown='ignore')

# подготовка данных для кодирования
X = titanic_test['Sex']

# кодирование
enc.fit(X.values.reshape(-1, 1))

# преобразование
y = enc.transform(X.values.reshape(-1, 1)).toarray()

# создание датафрейма преобразования
y = pd.DataFrame(y, columns=['female', 'male']) 

# объединение датафреймов
titanic_test = pd.concat([titanic_test, y], axis=1)

# сохраняем обновленный датасет в csv файл
titanic_test.to_csv(r"C:\Users\Santerr80\OneDrive\Документы\GitHub\mlops_practice\lab4\datasets\titanic.csv", index=False)

