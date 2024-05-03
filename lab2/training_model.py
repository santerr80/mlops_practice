from preparation_data import preparation  # функция для подготовки данных
from sklearn.neighbors import KNeighborsClassifier  # модель нейронной сети
from sklearn.model_selection import train_test_split  # для разбиения данных
import pickle  # для сохранения модели


# функция для обучения модели
def main():
    # Путь для тренировочных данных
    path = r".\data\train.csv"

    # Подготовка данных
    df = preparation(path)

    # Разбиение на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
        df.drop('car_price', axis=1),
        df['car_price'], test_size=0.2,
        random_state=42)

    # Обучение модели
    clf = KNeighborsClassifier(n_neighbors=50, p=2, metric='minkowski',
                               weights='distance', algorithm='brute')
    clf.fit(X_train, y_train)

    # Вывод оценки модели
    print(clf.score(X_test, y_test))

    # Сохранение модели
    with open('model.pkl', 'wb') as f:
        pickle.dump(clf, f)


# Запуск функции обучения модели

if __name__ == "__main__":
    main()
