import os
import sys
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Fix Windows encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# TensorFlow/Keras
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import MinMaxScaler
import pickle

def fetch_market_data(ticker, period="10y"):
    """Fetch long-term historical data for training"""
    print(f"📥 กำลังดึงข้อมูล {ticker} ย้อนหลัง {period}...")
    data = yf.download(ticker, period=period, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    print(f"✅ ดึงข้อมูลได้ {len(data)} วัน")
    return data

def create_advanced_features(data):
    """Create technical indicators"""
    df = data.copy()
    
    # Moving Averages
    df['MA5'] = df['Close'].rolling(window=5).mean()
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df['MA200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA12'] - df['EMA26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    bb_std = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (bb_std * 2)
    df['BB_Lower'] = df['BB_Middle'] - (bb_std * 2)
    df['BB_Width'] = ((df['BB_Upper'] - df['BB_Lower']) / df['BB_Middle']).fillna(0)
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
    
    # ATR
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # OBV
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # ROC
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    return df.dropna()

def prepare_training_data(data, lookback=60):
    """Prepare data for LSTM training - predict percentage return instead of price"""
    df = create_advanced_features(data)
    
    # Calculate percentage return (target variable)
    df['Return'] = df['Close'].pct_change().shift(-1)  # Next day return
    df = df.dropna()
    
    # Select features
    feature_cols = ['Close', 'Volume', 'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 
                    'BB_Width', 'Stoch_K', 'ATR', 'OBV', 'ROC', 'Volatility', 'Volume_Ratio']
    
    df_features = df[feature_cols].values
    returns = df['Return'].values
    
    # Split: 80% train, 20% validation
    split_idx = int(len(df_features) * 0.8)
    train_features = df_features[:split_idx]
    
    # Fit scaler ONLY on training data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_features)
    
    # Transform all data
    scaled_data = scaler.transform(df_features)
    
    # Create sequences
    X, y = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i-lookback:i])
        y.append(returns[i])  # Predict return, not price
    
    X = np.array(X)
    y = np.array(y)
    
    # Split train/validation
    X_train = X[:split_idx-lookback]
    y_train = y[:split_idx-lookback]
    X_val = X[split_idx-lookback:]
    y_val = y[split_idx-lookback:]
    
    return X_train, y_train, X_val, y_val, scaler

def build_lstm_model(input_shape):
    """Build advanced Bidirectional LSTM model"""
    model = Sequential([
        Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
        Dropout(0.3),
        Bidirectional(LSTM(64, return_sequences=True)),
        Dropout(0.3),
        LSTM(32, return_sequences=False),
        Dropout(0.2),
        Dense(16, activation='relu'),
        Dense(1)  # Predict return (single value)
    ])
    
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    return model

def train_and_save_model(ticker, name, lookback=60):
    """Train model and save it with scaler"""
    print(f"\n{'='*60}")
    print(f"🚀 เริ่มเทรนโมเดลสำหรับ {name} ({ticker})")
    print(f"{'='*60}")
    
    try:
        # Fetch data
        data = fetch_market_data(ticker, period="10y")
        
        # Prepare data
        print("🔧 กำลังเตรียมข้อมูลและสร้าง features...")
        X_train, y_train, X_val, y_val, scaler = prepare_training_data(data, lookback)
        
        print(f"📊 ข้อมูล Training: {len(X_train)} samples")
        print(f"📊 ข้อมูล Validation: {len(X_val)} samples")
        
        # Build model
        print("🏗️ กำลังสร้างโมเดล LSTM...")
        model = build_lstm_model((lookback, X_train.shape[2]))
        
        # Callbacks
        model_filename = f"models/{ticker.replace('^', '').replace('.', '_')}_model.keras"
        os.makedirs("models", exist_ok=True)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(model_filename, monitor='val_loss', save_best_only=True)
        ]
        
        # Train
        print("🎓 กำลังเทรนโมเดล (อาจใช้เวลา 5-10 นาที)...")
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=100,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save scaler
        scaler_filename = f"models/{ticker.replace('^', '').replace('.', '_')}_scaler.pkl"
        with open(scaler_filename, 'wb') as f:
            pickle.dump(scaler, f)
        
        # Evaluate
        train_loss = history.history['loss'][-1]
        val_loss = history.history['val_loss'][-1]
        
        print(f"\n✅ เทรนเสร็จสิ้น!")
        print(f"📈 Training Loss: {train_loss:.6f}")
        print(f"📉 Validation Loss: {val_loss:.6f}")
        print(f"💾 บันทึกโมเดล: {model_filename}")
        print(f"💾 บันทึก Scaler: {scaler_filename}")
        
        return True
        
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาด: {str(e)}")
        return False

def main():
    print("🤖 โปรแกรมเทรนโมเดล AI สำหรับทำนายตลาดหุ้น")
    print("=" * 60)
    
    # Markets to train
    markets = {
        "^SET.BK": "ดัชนี SET (ไทย)",
        "^GSPC": "S&P 500 (สหรัฐ)",
        "^DJI": "Dow Jones (สหรัฐ)"
    }
    
    results = {}
    
    for ticker, name in markets.items():
        success = train_and_save_model(ticker, name)
        results[name] = "✅ สำเร็จ" if success else "❌ ล้มเหลว"
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 สรุปผลการเทรน")
    print(f"{'='*60}")
    for name, status in results.items():
        print(f"{status} {name}")
    
    print(f"\n✅ เสร็จสิ้นทั้งหมด! โมเดลพร้อมใช้งาน")
    print(f"💡 รันคำสั่ง: python predict_markets.py เพื่อทำนาย")

if __name__ == "__main__":
    main()
