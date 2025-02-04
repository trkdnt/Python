import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime



# Időintervallum beállítása: az elmúlt 5 év
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.datetime.today() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')

ticker = 'nvda'
df = yf.download(ticker, start=start_date, end=end_date)
print("Letöltött adatok:")
print(df.tail())  # az utolsó néhány sor megtekintése

# Csak a záróárakat használjuk
data = df[['Close']].copy()

# Normalizálás 0 és 1 közé
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# Sliding window mérete: hány nap adataiból jósolunk
window_size = 30

def create_dataset(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i+window_size])
        y.append(dataset[i+window_size])
    return np.array(X), np.array(y)

X, y = create_dataset(scaled_data, window_size)

# Az adatok 80%-át használjuk tanuláshoz
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)

model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# Modell betanítása
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))


# Utolsó 30 napos ablak az ismert adatokból
last_window = scaled_data[-window_size:]
forecast_input = np.array(last_window)
forecast_input = forecast_input.reshape(1, window_size, 1)

forecast = []

# 30 napos előrejelzés
n_future = 30
for _ in range(n_future):
    pred = model.predict(forecast_input)
    forecast.append(pred[0,0])
    
    # Az új értéket hozzáfűzzük a windowhoz, majd eltávolítjuk a legrégebbit
    forecast_input = np.append(forecast_input[:,1:,:], [[pred[0]]], axis=1)

# Visszakonvertálás az eredeti skálára
forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# Készítünk egy időindexet az előrejelzéshez
future_dates = pd.date_range(start=df.index[-1] + pd.Timedelta(days=1), periods=n_future)
forecast_df = pd.DataFrame(forecast_actual, index=future_dates, columns=['Forecast'])

print("Előrejelzett árak a következő 30 napra:")
print(forecast_df.head())

