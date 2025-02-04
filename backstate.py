import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime

# 1. Adatok letöltése az elmúlt 5 évre (például NVDA)
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.datetime.today() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')

ticker = 'NVDA'
df = yf.download(ticker, start=start_date, end=end_date)
print("Letöltött adatok:")
print(df.tail())  # Az utolsó néhány sor megtekintése

# Csak a záróárakat használjuk
data = df[['Close']].copy()

# 2. Backtesting beállítása: válassz egy múltbeli dátumot, mintha azon a napon indult volna az előrejelzés
backtest_date = '2025-01-03'  # Ezt módosíthatod, hogy melyik dátummal teszteled a modellt

# A training adatok: azokat az adatokat, melyek a backtest kezdete előttiek
train_data = data[data.index < backtest_date]

# A test (valós) adatok: a backtest kezdő dátumától számított 30 napos időszak
test_data = data[(data.index >= backtest_date) & (data.index < pd.to_datetime(backtest_date) + pd.Timedelta(days=30))]

print("Training data period:", train_data.index.min(), "to", train_data.index.max())
print("Test data period:", test_data.index.min(), "to", test_data.index.max())

# 3. Normalizálás: A scaler-t a training adatokra fiteljük, majd mindkét részhalmazra alkalmazzuk
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train = scaler.fit_transform(train_data)

# Sliding window beállítása: hány nap adataiból jósolunk
window_size = 30

def create_dataset(dataset, window_size):
    X, y = [], []
    for i in range(len(dataset) - window_size):
        X.append(dataset[i:i+window_size])
        y.append(dataset[i+window_size])
    return np.array(X), np.array(y)

X_train, y_train = create_dataset(scaled_train, window_size)
print("X_train shape:", X_train.shape)

# 4. LSTM modell építése
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dense(25),
    Dense(1)
])

model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

# 5. Modell betanítása
history = model.fit(X_train, y_train, epochs=50, batch_size=32)

# 6. Előrejelzés: A training adatok utolsó 30 napos ablakából előrejelzés 30 napra
last_window = scaled_train[-window_size:]
forecast_input = np.array(last_window).reshape(1, window_size, 1)
forecast = []
n_future = 30

for _ in range(n_future):
    pred = model.predict(forecast_input)
    forecast.append(pred[0, 0])
    # Az új predikció hozzáfűzése az ablakhoz, a legrégebbi eltávolítása
    forecast_input = np.append(forecast_input[:, 1:, :], [[pred[0]]], axis=1)

# Visszakonvertálás az eredeti skálára
forecast_actual = scaler.inverse_transform(np.array(forecast).reshape(-1, 1))

# 7. Időindex létrehozása az előrejelzéshez: a backtest_date-tól induló 30 kereskedési nap
forecast_dates = pd.date_range(start=pd.to_datetime(backtest_date), periods=n_future, freq='B')
forecast_df = pd.DataFrame(forecast_actual, index=forecast_dates, columns=['Forecast'])

print("Előrejelzett árak a következő 30 napra:")
print(forecast_df.head())

# 8. Diagram: megjelenítjük az elmúlt fél év (6 hónap) történeti adatát és a 30 napos előrejelzést
# Csak az elmúlt 6 hónap adatait tartalmazza a historikus rész
historical_plot = data[data.index >= (data.index.max() - pd.DateOffset(months=6))]

plt.figure(figsize=(14,7))
plt.plot(historical_plot.index, historical_plot['Close'], label='Történeti záróár (elmúlt 6 hónap)')
plt.plot(forecast_df.index, forecast_df['Forecast'], label='30 napos előrejelzés', linestyle='--', color='orange')
if len(test_data) > 0:
    # Ha a backtest időszak beleesik az elmúlt 6 hónapba, megjelenítjük a valós adatokat is.
    plt.plot(test_data.index, test_data['Close'], label='Valós árak (backtest period)', color='green')
plt.title(f'{ticker} záróár (elmúlt 6 hónap) és 30 napos előrejelzés backtesting: kezdő dátum {backtest_date}')
plt.xlabel('Dátum')
plt.ylabel('Ár (USD)')
plt.legend()
plt.style.use("dark_background")
plt.show()
