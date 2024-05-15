import pytest
from main import load_model, tapex_tokenizer, save_to_file
import pandas as pd
import warnings

warnings.filterwarnings("ignore")


# Тестирование загрузки модели
def test_load_model():
    tokenizer, model = load_model()
    assert tokenizer is not None, "Токенизатор не был загружен"
    assert model is not None, "Модель не была загружена"

# Тестирование функции токенизации
@pytest.mark.parametrize("data, query", [
    (pd.DataFrame({'A': ['1', '2'], 'B': ['3', '4']}), "Get max number in column A?"),
    # ....
])
def test_tapex_tokenizer(data, query):
    answer = tapex_tokenizer(data, query)
    assert isinstance(answer, list), "Ответ должен быть списком"
    assert len(answer) > 0, "Список ответов не может быть пустым"


# Тестирование функции сохранения в файл
def test_save_to_file(tmpdir):
    content = "Тестовое содержимое"
    file = tmpdir.join("output.txt")
    save_to_file(content, file)
    assert file.read() == content + "\n", "Содержимое файла не соответствует ожидаемому"


# Настройка для тестирования функции сохранения в файл
@pytest.fixture
def mock_open(mocker):
    return mocker.patch("builtins.open", mocker.mock_open())


def test_save_to_file_with_mock_open(mock_open):
    content = "Тестовое содержимое"
    save_to_file(content)
    mock_open.assert_called_once_with(r"D:\GitHub\mlops_practice\Final\data\output.txt", "a")
    mock_open().write.assert_called_once_with(content + "\n")