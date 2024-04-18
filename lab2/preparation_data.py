import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler,  OrdinalEncoder


def preparation(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df_data = df.drop('car_price', axis=1)
    df_target = df['car_price']

    # Обработка числовых признаков
    num_data_columns_name = ['car_mileage',
                             'car_engine_capacity',
                             'car_engine_hp', 'car_age']
    df_data_num = df_data[num_data_columns_name]

    # выполним стандартизацию данных
    standart = StandardScaler()
    standart.fit(df_data_num)

    # Применяем трансформер
    standarted = standart.transform(df_data_num)
    df_standard = pd.DataFrame(standarted)

    # выполним масштабирование данных
    scaler = MinMaxScaler()
    scaler.fit(df_standard)

    # Применяем трансформер
    scaled = scaler.transform(df_standard)
    df_scaled = pd.DataFrame(scaled, columns=num_data_columns_name)

    # Обработка категориальных признаков
    cat_data_columns_name = ['car_brand', 'car_model', 'car_city', 'car_fuel',
                             'car_transmission', 'car_drive', 'car_country']
    df_data_cat = df_data[cat_data_columns_name]

    # выполним кодирование категориальных признаков
    enc = OrdinalEncoder()
    enc.fit(df_data_cat)

    # Применяем трансформер
    encoded = enc.transform(df_data_cat)

    # Преобразование в датафрейм
    df_encoded = pd.DataFrame(encoded, columns=cat_data_columns_name)

    # Объединим все признаки в один датафрейм
    df_prepared = pd.concat([df_encoded, df_scaled, df_target], axis=1)

    # Возвращаем датафрейм
    return df_prepared
