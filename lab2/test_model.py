import pickle  # для загрузки модели
from preparation_data import preparation  # функция для подготовки данных


# функция для загрузки модели
def load_model(model_path: str):
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


# путь к модели
model_path = r'.\model.pkl'

# путь к тестовым данным
path = r'.\data\test.csv'

# загружаем модель
clf = load_model(model_path)

# подготовка данных
df = preparation(path)

# разделяем данные на X и y
X_test, y_test = df.drop('car_price', axis=1), df['car_price']

# делаем предсказания
y_pred = clf.predict(X_test)


print("Теститорвание прошло успешно:")

with open(r'.\predictions.txt', 'w') as f:
    f.write(str(y_pred))
print("Результаты сохранены в файл predictions.txt")

print(f"Точность модели: {clf.score(X_test, y_test)}")
