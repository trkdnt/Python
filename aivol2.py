# =============================================================================
# 1. IMPORTÁLÁS, PARAMÉTEREK, KONFIGURÁCIÓ
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import datetime
import talib  # TA-Lib telepítve kell legyen

# Alap konfiguráció:
main_ticker = 'AAPL'
# Technológiai szektor további részvényei – bővíthető a kívánt tickerekre
additional_tickers = ['MSFT', 'GOOGL', 'NVDA', 'AMD','AMZN','META','INTC','AAPL']
all_tickers = [main_ticker] + additional_tickers

# Backtesting mód:
# Ha megadsz egy dátumot (YYYY-MM-DD), akkor a modell úgy működik, mintha azon a napon indult volna.
# A forecast_horizon kereskedési napokra jósol (pl. 30 nap).
# Ha ez az érték üres vagy None, akkor az "élő" üzemmódot használjuk.
backtest_date_str = '2025-01-15.'  # Ha backtest, ez legyen egy múltbéli dátum!
# backtest_date_str = None  # Élő mód

forecast_horizon = 30  # előrejelzés 30 kereskedési napra

# Sliding window mérete: hány napos adatsor alapján jósol
window_size = 30

# Modell finomhangolási paraméterek
lstm_units = 100
dropout_rate = 0.3
epochs = 100
batch_size = 64

# Vizsgálati periódus: letöltjük az elmúlt 5 év adatait
end_date = datetime.datetime.today().strftime('%Y-%m-%d')
start_date = (datetime.datetime.today() - datetime.timedelta(days=5*365)).strftime('%Y-%m-%d')

plt.style.use("dark_background")

# =============================================================================
# 2. TECHNIKAI INDICÁTOROK SZÁMÍTÁSA (TA-Lib segítségével)
# =============================================================================
def add_technical_indicators(df):
    """
    Az adott DataFrame-hez (egy részvényre vonatkozóan) kiszámolja:
      - SMA20: 20 napos egyszerű mozgóátlag
      - EMA20: 20 napos exponenciális mozgóátlag
      - RSI14: 14 napos relatív erősség index
      - MACD: MACD vonal és signal vonal
      - Bollinger-szalagok: felső, középső és alsó sáv
    """
    df['SMA20'] = df['Close'].rolling(window=20).mean()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['RSI14'] = talib.RSI(df['Close'].values, timeperiod=14)
    macd, macdsignal, macdhist = talib.MACD(df['Close'].values, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD'] = macd
    df['MACD_signal'] = macdsignal
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(df['Close'].values, timeperiod=20)
    return df

# =============================================================================
# 3. ADATOK LETÖLTÉSE ÉS ÖSSZEGYŰJTÉSE
# =============================================================================
print("Adatok letöltése...")

# Letöltjük az összes részvény adatát egyszerre (Close, Volume)
data_all = yf.download(all_tickers, start=start_date, end=end_date, group_by='ticker', auto_adjust=True)

dfs = {}
for ticker in all_tickers:
    try:
        df_ticker = data_all[ticker].copy()  # group_by='ticker' esetén
    except Exception:
        df_ticker = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True)
    df_ticker = df_ticker[['Close', 'Volume']]
    df_ticker = add_technical_indicators(df_ticker)
    if isinstance(df_ticker.columns, pd.MultiIndex):
        df_ticker.columns = df_ticker.columns.droplevel(0)
    df_ticker.dropna(inplace=True)
    dfs[ticker] = df_ticker

# Az időbeli igazítás: a fő részvény indexét vesszük, majd a többi részvény Close adatait csatoljuk
df_main = dfs[main_ticker].copy()
for ticker in additional_tickers:
    df_ticker = dfs[ticker][['Close']].rename(columns={'Close': f'{ticker}_Close'})
    df_main = df_main.join(df_ticker, how='inner')

# =============================================================================
# XLK ETF (technológiai szektor index) hozzáadása
# =============================================================================
df_xlk = yf.download('XLK', start=start_date, end=end_date, auto_adjust=True)
if isinstance(df_xlk.columns, pd.MultiIndex):
    df_xlk.columns = df_xlk.columns.droplevel(0)
# Néha az oszlopok nevei mind "XLK" lehetnek, ezért egyedi neveket generálunk:
if all(col == 'XLK' for col in df_xlk.columns):
    df_xlk.columns = [f'XLK_{i}' for i in range(len(df_xlk.columns))]
# Válasszuk az első oszlopot, ami általában a záróárnak felel meg:
df_xlk = df_xlk.iloc[:, 0].to_frame()
df_xlk.columns = ['XLK_Close']
df_xlk = df_xlk.reindex(df_main.index)
print("XLK ETF adatstruktúra a módosítás után:")
print(df_xlk.head())

# Végső illesztés: csatoljuk az XLK ETF-et a fő DataFrame-hez
df_main = df_main.join(df_xlk, how='inner')
df_main.dropna(inplace=True)
print("Összeillesztett adatok (df_main) első néhány sora:")
print(df_main.head())

# =============================================================================
# 4. ADATOK ELŐFELDOLGOZÁSA: FEATUREK ÉS TARGET
# =============================================================================
target_col = 'Close'  # a fő részvény záróára
feature_cols = df_main.columns.tolist()
print("Használt featurek:")
print(feature_cols)

# =============================================================================
# 5. ADATOK SKÁLÁZÁSA
# =============================================================================
scaler_features = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler_features.fit_transform(df_main[feature_cols])

scaler_target = MinMaxScaler(feature_range=(0, 1))
scaled_target = scaler_target.fit_transform(df_main[[target_col]])

# =============================================================================
# 6. SLIDING WINDOW TECHNIKÁVAL ADATKÉPZÉS
# =============================================================================
def create_multivariate_dataset(scaled_feats, scaled_tgt, window_size):
    X, y = [], []
    for i in range(len(scaled_feats) - window_size):
        X.append(scaled_feats[i:i+window_size, :])
        y.append(scaled_tgt[i+window_size, 0])
    return np.array(X), np.array(y)

X_all, y_all = create_multivariate_dataset(scaled_features, scaled_target, window_size)
print("Teljes dataset: X_all shape =", X_all.shape, ", y_all shape =", y_all.shape)

# =============================================================================
# 7. TRAINING / BACKTEST SPLIT
# =============================================================================
if backtest_date_str and backtest_date_str.strip():
    backtest_date = pd.to_datetime(backtest_date_str)
    train_mask = df_main.index < backtest_date
    df_train = df_main.loc[train_mask].iloc[window_size:]  # az első window_size sor nem használható
    train_length = len(df_train)
    X_train = X_all[:train_length]
    y_train = y_all[:train_length]
    
    # A teszt adatok: a backtest_date-tól kezdődő időszak, legalább forecast_horizon kereskedési napra
    test_end_date = backtest_date + pd.Timedelta(days=forecast_horizon*1.5)
    df_test = df_main[(df_main.index >= backtest_date) & (df_main.index <= test_end_date)]
else:
    X_train = X_all
    y_train = y_all
    backtest_date = df_main.index[-1]
    df_test = pd.DataFrame()  # nincs valódi teszt adat

print("Training adatok száma:", X_train.shape[0])

# =============================================================================
# 8. LSTM MODELL ÉPÍTÉSE ÉS TANÍTÁSA
# =============================================================================
model = Sequential([
    LSTM(lstm_units, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
    Dropout(dropout_rate),
    LSTM(lstm_units, return_sequences=True),
    Dropout(dropout_rate),
    LSTM(lstm_units, return_sequences=False),
    Dense(50, activation='relu'),
    Dense(25, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')
model.summary()

history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.1, verbose=1)

# =============================================================================
# 9. ELŐREJELZÉS (FORECAST) - ITERATÍV MÓD
# =============================================================================
last_window = X_train[-1]  # utolsó training ablak
forecast_input = last_window.reshape(1, window_size, X_train.shape[2])
forecast = []

for _ in range(forecast_horizon):
    pred = model.predict(forecast_input, verbose=0)
    forecast.append(pred[0, 0])
    # Új időlépés: a legutolsó sor alapján, csak a target értéke frissül
    new_step = forecast_input[:, -1, :].copy()
    target_idx = feature_cols.index(target_col)
    new_step[0, target_idx] = pred[0, 0]
    forecast_input = np.append(forecast_input[:, 1:, :], new_step.reshape(1, 1, -1), axis=1)

forecast_actual = scaler_target.inverse_transform(np.array(forecast).reshape(-1, 1))

# =============================================================================
# 10. IDŐINDEX, DIAGRAM, ÉRTÉKELSÉG
# =============================================================================
if backtest_date_str and backtest_date_str.strip():
    # Backtest módban: a forecast kezdete legyen a megadott backtest dátum
    forecast_start = pd.to_datetime(backtest_date_str)
    # Generáljuk a forecast dátumokat üzleti napokból
    forecast_dates = pd.bdate_range(start=forecast_start, periods=forecast_horizon)
    # Historikus rész: csak az utolsó 6 hónap adatai a forecast_start előtt
    historical_plot = df_main.loc[df_main.index >= (forecast_start - pd.DateOffset(months=6)), [target_col]]
else:
    # Élő mód: a forecast kezdete az utolsó historikus nap utáni első üzleti nap
    last_date = df_main.index[-1]
    forecast_dates = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=forecast_horizon)
    # Historikus rész: az utolsó 6 hónap adatai
    historical_plot = df_main.loc[df_main.index >= (last_date - pd.DateOffset(months=6)), [target_col]]

# Készítünk egy DataFrame-et az előrejelzett értékekkel
forecast_df = pd.DataFrame(forecast_actual, index=forecast_dates, columns=['Forecast'])

print("Előrejelzett target árak a következő 30 napra:")
print(forecast_df.head())

# Diagram rajzolása
plt.figure(figsize=(16,8))
# Historikus adatok (kék vonal)
plt.plot(historical_plot.index, historical_plot[target_col], label='Historikus ' + target_col, linewidth=2)
# Valós (teszt) adatok backtest esetén (zöld vonal)
if not df_test.empty:
    # Csak a forecast időszakhoz tartozó napokat vesszük a df_test-ből
    df_test_period = df_test[df_test.index.isin(forecast_dates)]
    if not df_test_period.empty:
        plt.plot(df_test_period.index, df_test_period[target_col], label='Valós ' + target_col, color='lime', linewidth=2)
# Forecast (előrejelzés) – sárga/orange szaggatott vonal
plt.plot(forecast_df.index, forecast_df['Forecast'], label='30 napos előrejelzés', linestyle='--', color='orange', linewidth=2)

# Cím, tengelyek, legend és rács
if backtest_date_str and backtest_date_str.strip():
    title_str = f'{main_ticker} historikus adatok (utolsó 6 hónap) és 30 napos előrejelzés\nBacktest kezdete: {backtest_date_str}'
else:
    title_str = f'{main_ticker} historikus adatok (utolsó 6 hónap) és 30 napos előrejelzés (Élő mód)'
plt.title(title_str, fontsize=16)
plt.xlabel('Dátum', fontsize=14)
plt.ylabel('Ár (USD)', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.3)
plt.show()


# =============================================================================
# 11. EREDMÉNYEK KIÉRTÉKELÉSE (például RMSE számítás)
# =============================================================================
from sklearn.metrics import mean_squared_error
import math

if not df_test.empty:
    test_targets = df_test.loc[df_test.index.isin(forecast_dates), target_col].values
    if len(test_targets) == len(forecast_actual):
        rmse = math.sqrt(mean_squared_error(test_targets, forecast_actual))
        print("RMSE a forecast_horizon időszakra:", rmse)
    else:
        print("A forecast_horizon időszakra nem áll rendelkezésre elegendő valódi adat a kiértékeléshez.")
else:
    print("Élő mód: nincs elérhető teszt adat a kiértékeléshez.")

# =============================================================================
# 12. FINOMHANGOLÁSI TIPPEK
# =============================================================================
print("A fejlett multivariáns LSTM modell elkészült. A kód igyekszik a lehető legtöbb adatot és technikai indikátort felhasználni a pontosság növelése érdekében.")
