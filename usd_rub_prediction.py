# Импорт необходимых библиотек
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# Загрузка данных
# Предполагается, что у вас есть CSV файл с данными о курсе доллара к рублю
data = pd.read_csv('usd_rub_data.csv')

# Предобработка данных
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data)

# Разделение данных на обучающую и тестовую выборки
X = data_scaled[:-1]
y = data_scaled[1:]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Создание модели нейронной сети
model = Sequential()
model.add(Dense(100, input_dim=1, activation='relu'))
model.add(Dense(1))

# Компиляция модели
model.compile(loss='mean_squared_error', optimizer='adam')

# Обучение модели
model.fit(X_train, y_train, epochs=50, batch_size=1, verbose=2)

# Предсказание цены на завтра
prediction = model.predict(X_test)