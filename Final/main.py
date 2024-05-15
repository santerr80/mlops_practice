# Импорт модулей
from transformers import TapexTokenizer, BartForConditionalGeneration
import pandas as pd
import warnings

# Отключение предупреждений
warnings.filterwarnings("ignore")


# Загрузка модели
def load_model():
    tokenizer = TapexTokenizer.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    model = BartForConditionalGeneration.from_pretrained("microsoft/tapex-large-finetuned-wtq")
    return tokenizer, model


# Запрос к модели
def tapex_tokenizer(data, query):
    encoding = tokenizer(table=data, query=query, return_tensors="pt")
    outputs = model.generate(**encoding, max_new_tokens=50)
    answer = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return answer


# Сохранение в файл
def save_to_file(content, file=r"D:\GitHub\mlops_practice\Final\data\output.txt"):
    try:
        with open(file, "a") as myfile:
            myfile.write(str(content) + "\n")
    except IOError:
        print("I/O error")


# Загрузка модели
tokenizer, model = load_model()


# Цикл запросов к модели
def main():
    # Чтение данных
    data = pd.read_csv(r"D:\GitHub\mlops_practice\Final\data\table.csv", sep=",")

    # Цикл запросов к модели
    while True:
        # Ввод запроса
        query = input("Enter your query: ")
        # Запрос к модели
        answer = tapex_tokenizer(data, query)
        # Вывод ответа
        print(answer)
        # Сохранение в файл
        content = [query, answer]
        save_to_file(content)


# Запуск
if __name__ == "__main__":
    main()
