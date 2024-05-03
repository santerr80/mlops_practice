# Импорт библиотек
import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

# Заголовок
st.header('Roberta-base for QA', divider='rainbow')
'This is the roberta-base model, fine-tuned using the SQuAD2.0 dataset.' 
'It\'s been trained on question-answer pairs, including unanswerable questions, for the task of Question Answering.'

# Поле ввода для контекста
context = st.text_input('Context', 'Today I came back to home, when rain is start.')
st.write('Context:', context)

# Поле ввода для вопроса
question = st.text_input('Question Answering', 'Why I come back to home?')
st.write('Question:', question)

# Модель
model_name = "deepset/roberta-base-squad2"

# Обработчик
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)

# Данные для ввода
QA_input = {
    'question': question,
    'context': context
}

# Предсказание
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Обработка
res = nlp(QA_input)

# Вывод
st.write('**Answer:**', res['answer'])
st.write('**Score:**', res['score'])

