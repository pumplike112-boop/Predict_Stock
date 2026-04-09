import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import requests
import json
import warnings
import sys
import pickle
warnings.filterwarnings('ignore')

# Fix Windows encoding for emoji support
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Load environment variables
load_dotenv()

# TensorFlow/Keras for LSTM
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

def fetch_market_data(ticker, period="2y"):
    """Fetch historical market data with error handling"""
    try:
        data = yf.download(ticker, period=period, progress=False)
        # Flatten MultiIndex columns if present
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        
        if data.empty:
            raise ValueError(f"ไม่สามารถดึงข้อมูลสำหรับ {ticker}")
        
        return data
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการดึงข้อมูล {ticker}: {str(e)}")
        raise

def create_advanced_features(data):
    """Create advanced technical indicators"""
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
    
    # ATR (Average True Range)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # OBV (On-Balance Volume)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Price Rate of Change
    df['ROC'] = df['Close'].pct_change(periods=10) * 100
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Volume indicators
    df['Volume_MA'] = df['Volume'].rolling(window=20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']
    
    return df.dropna()

def load_model_and_scaler(ticker):
    """Load pre-trained model and scaler"""
    model_filename = f"models/{ticker.replace('^', '').replace('.', '_')}_model.keras"
    scaler_filename = f"models/{ticker.replace('^', '').replace('.', '_')}_scaler.pkl"
    
    if not os.path.exists(model_filename) or not os.path.exists(scaler_filename):
        raise FileNotFoundError(f"โมเดลสำหรับ {ticker} ไม่พบ กรุณารัน train_models.py ก่อน")
    
    model = keras.models.load_model(model_filename)
    with open(scaler_filename, 'rb') as f:
        scaler = pickle.load(f)
    
    return model, scaler

def predict_one_month(data, ticker, days_ahead=20):
    """Predict 1 month ahead (20 trading days) using iterative prediction"""
    lookback = 60
    
    try:
        # Load pre-trained model
        model, scaler = load_model_and_scaler(ticker)
        
        # Prepare features
        df = create_advanced_features(data)
        feature_cols = ['Close', 'Volume', 'MA5', 'MA20', 'MA50', 'RSI', 'MACD', 
                        'BB_Width', 'Stoch_K', 'ATR', 'OBV', 'ROC', 'Volatility', 'Volume_Ratio']
        
        df_features = df[feature_cols].values
        
        # Scale using pre-fitted scaler
        scaled_data = scaler.transform(df_features)
        
        # Get last sequence
        if len(scaled_data) < lookback:
            raise ValueError("ข้อมูลไม่เพียงพอ")
        
        current_price = data['Close'].iloc[-1]
        
        # Iterative prediction for 20 days
        sequence = scaled_data[-lookback:].copy()
        cumulative_return = 0
        
        for day in range(days_ahead):
            # Predict next day return
            input_seq = sequence.reshape(1, lookback, len(feature_cols))
            daily_return = model.predict(input_seq, verbose=0)[0][0]
            cumulative_return += daily_return
            
            # Update sequence: shift and add new predicted values
            new_row = sequence[-1].copy()
            new_row[0] = new_row[0] * (1 + daily_return * 0.5)  # Dampen effect to avoid explosion
            sequence = np.vstack([sequence[1:], new_row])
        
        # Calculate final predicted price
        predicted_price = current_price * (1 + cumulative_return)
        change_pct = cumulative_return * 100
        
        # Calculate confidence (decreases with prediction horizon)
        base_confidence = 100 - abs(cumulative_return * 500)
        time_decay = 0.7  # 70% confidence for 1-month prediction
        confidence = min(95, max(40, base_confidence * time_decay))
        
        return predicted_price, change_pct, confidence
        
    except FileNotFoundError as e:
        print(f"⚠️ {str(e)}")
        current_price = data['Close'].iloc[-1]
        return current_price, 0.0, 30.0
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการทำนาย: {str(e)}")
        current_price = data['Close'].iloc[-1]
        return current_price, 0.0, 30.0

def get_trend_emoji(change_pct):
    """Return trend emoji based on percentage change"""
    return "📈" if change_pct > 0 else "📉"

def send_discord_webhook(webhook_url, embed_data):
    """Send formatted embed to Discord with error handling"""
    try:
        payload = {"embeds": [embed_data]}
        response = requests.post(webhook_url, json=payload, timeout=10)
        
        if response.status_code == 204:
            return 204
        elif response.status_code == 429:
            print("⚠️ Discord Rate Limit - รอสักครู่แล้วลองใหม่")
            return 429
        else:
            print(f"⚠️ Discord ตอบกลับด้วย status code: {response.status_code}")
            return response.status_code
            
    except requests.exceptions.Timeout:
        print("❌ Discord Webhook Timeout - ใช้เวลานานเกินไป")
        return 0
    except Exception as e:
        print(f"❌ เกิดข้อผิดพลาดในการส่ง Discord: {str(e)}")
        return 0

def main():
    # Market tickers
    markets = {
        "thai": {"^SET.BK": "ดัชนี SET"},
        "us": {
            "^GSPC": "S&P 500",
            "^DJI": "Dow Jones"
        }
    }
    
    # Collect predictions
    predictions = {"thai": {}, "us": {}}
    
    # Process Thai market
    for ticker, name in markets["thai"].items():
        try:
            print(f"🔄 กำลังประมวลผล {name} (ทำนาย 1 เดือนข้างหน้า)...")
            data = fetch_market_data(ticker)
            current_price = data['Close'].iloc[-1]
            
            predicted_price, change_pct, confidence = predict_one_month(data, ticker, days_ahead=20)
            
            predictions["thai"][name] = {
                "current": current_price,
                "predicted": predicted_price,
                "change_pct": change_pct,
                "trend": get_trend_emoji(change_pct),
                "confidence": confidence
            }
            print(f"✅ {name}: {current_price:.2f} → {predicted_price:.2f} ({change_pct:+.2f}%) [ความมั่นใจ: {confidence:.0f}%]")
        except Exception as e:
            print(f"❌ ไม่สามารถประมวลผล {name}: {str(e)}")
            continue
    
    # Process US markets
    us_temp = {}
    for ticker, name in markets["us"].items():
        try:
            print(f"🔄 กำลังประมวลผล {name} (ทำนาย 1 เดือนข้างหน้า)...")
            data = fetch_market_data(ticker)
            current_price = data['Close'].iloc[-1]
            
            predicted_price, change_pct, confidence = predict_one_month(data, ticker, days_ahead=20)
            
            us_temp[name] = {
                "current": current_price,
                "predicted": predicted_price,
                "change_pct": change_pct,
                "trend": get_trend_emoji(change_pct),
                "confidence": confidence
            }
            print(f"✅ {name}: {current_price:.2f} → {predicted_price:.2f} ({change_pct:+.2f}%) [ความมั่นใจ: {confidence:.0f}%]")
        except Exception as e:
            print(f"❌ ไม่สามารถประมวลผล {name}: {str(e)}")
            continue
    
    # Check if we have US data
    if len(us_temp) < 2:
        print("⚠️ ข้อมูลตลาดสหรัฐไม่ครบ")
        return
    
    # Store US predictions
    predictions["us"] = us_temp
    
    # Calculate US combined average
    sp500_data = us_temp["S&P 500"]
    dji_data = us_temp["Dow Jones"]
    avg_change_pct = (sp500_data['change_pct'] + dji_data['change_pct']) / 2
    avg_confidence = (sp500_data['confidence'] + dji_data['confidence']) / 2
    
    # Determine overall market sentiment
    all_changes = [predictions["thai"][k]['change_pct'] for k in predictions["thai"]]
    all_changes.append(avg_change_pct)
    avg_change = np.mean(all_changes)
    
    # Smooth gradient color based on average change (-5% to +5% range)
    # Clamp value between -5 and +5 for color calculation
    clamped_change = max(-5, min(5, avg_change))
    
    # Normalize to 0-1 range (0 = -5%, 0.5 = 0%, 1 = +5%)
    normalized = (clamped_change + 5) / 10
    
    # Create smooth gradient: Red -> Orange -> Yellow -> Light Green -> Green
    if normalized < 0.25:  # -5% to -2.5%: Deep Red to Red
        r = 255
        g = int(100 * (normalized / 0.25))
        b = 100
    elif normalized < 0.4:  # -2.5% to -1%: Red to Orange
        r = 255
        g = int(100 + 100 * ((normalized - 0.25) / 0.15))
        b = int(100 - 100 * ((normalized - 0.25) / 0.15))
    elif normalized < 0.5:  # -1% to 0%: Orange to Yellow
        r = 255
        g = int(200 + 55 * ((normalized - 0.4) / 0.1))
        b = 0
    elif normalized < 0.6:  # 0% to +1%: Yellow to Light Green
        r = int(255 - 155 * ((normalized - 0.5) / 0.1))
        g = 255
        b = int(100 * ((normalized - 0.5) / 0.1))
    elif normalized < 0.75:  # +1% to +2.5%: Light Green to Green
        r = int(100 - 100 * ((normalized - 0.6) / 0.15))
        g = 255
        b = int(100 + 100 * ((normalized - 0.6) / 0.15))
    else:  # +2.5% to +5%: Green to Deep Green
        r = 0
        g = int(255 - 55 * ((normalized - 0.75) / 0.25))
        b = int(200 - 50 * ((normalized - 0.75) / 0.25))
    
    # Convert RGB to hex
    embed_color = (r << 16) + (g << 8) + b
    
    # Sentiment description in Thai
    if avg_change > 3:
        sentiment = "แนวโน้มขาขึ้นแรง 🚀🐂"
    elif avg_change > 1:
        sentiment = "แนวโน้มขาขึ้น 📈🐂"
    elif avg_change > -1:
        sentiment = "แนวโน้มปานกลาง ⚖️"
    elif avg_change > -3:
        sentiment = "แนวโน้มขาลง 📉🐻"
    else:
        sentiment = "แนวโน้มขาลงแรง 💥🐻"
    
    # Build Discord Embed
    embed = {
        "title": "🤖 การทำนายตลาดหุ้นด้วย AI - 1 เดือนข้างหน้า (20 วันทำการ)",
        "description": (f"**โมเดล Deep Learning LSTM** | Bidirectional Neural Network\n"
                       f"**แนวโน้มตลาดโดยรวม:** {sentiment}\n"
                       f"**การเปลี่ยนแปลงเฉลี่ย:** `{avg_change:+.2f}%`"),
        "color": embed_color,
        "timestamp": datetime.utcnow().isoformat(),
        "fields": [],
        "footer": {
            "text": "⚡ ขับเคลื่อนด้วย LSTM + 14 ตัวชี้วัดทางเทคนิค | ⚠️ ไม่ใช่คำแนะนำทางการเงิน"
        },
        "thumbnail": {
            "url": "https://cdn-icons-png.flaticon.com/512/2936/2936719.png"
        }
    }
    
    # Thai Market Section
    embed["fields"].append({
        "name": "🇹🇭 **ตลาดหุ้นไทย**",
        "value": "```diff\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n```",
        "inline": False
    })
    
    for name, data in predictions["thai"].items():
        change_indicator = "🟢" if data['change_pct'] > 0 else "🔴"
        
        # Simple trend description
        if data['change_pct'] > 3:
            trend_text = "แนวโน้มขาขึ้นแรง"
        elif data['change_pct'] > 0:
            trend_text = "แนวโน้มขาขึ้น"
        elif data['change_pct'] > -3:
            trend_text = "แนวโน้มขาลง"
        else:
            trend_text = "แนวโน้มขาลงแรง"
        
        value = (f"{change_indicator} **ราคาปัจจุบัน:** `{data['current']:,.2f} บาท`\n"
                f"🎯 **ราคาที่คาดการณ์ (1 เดือน):** `{data['predicted']:,.2f} บาท`\n"
                f"💹 **การเปลี่ยนแปลง:** `{data['change_pct']:+.2f}%` {data['trend']}\n"
                f"🎲 **ความมั่นใจ:** `{data['confidence']:.0f}%`\n"
                f"📊 **{trend_text}**")
        embed["fields"].append({
            "name": f"📌 {name}",
            "value": value,
            "inline": False
        })
    
    # US Markets Section (Combined)
    embed["fields"].append({
        "name": "\n🇺🇸 **ตลาดหุ้นสหรัฐอเมริกา**",
        "value": "```diff\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n```",
        "inline": False
    })
    
    # Combined US market summary
    change_indicator = "🟢" if avg_change_pct > 0 else "🔴"
    
    if avg_change_pct > 3:
        trend_text = "แนวโน้มขาขึ้นแรง"
    elif avg_change_pct > 0:
        trend_text = "แนวโน้มขาขึ้น"
    elif avg_change_pct > -3:
        trend_text = "แนวโน้มขาลง"
    else:
        trend_text = "แนวโน้มขาลงแรง"
    
    us_value = (f"{change_indicator} **{trend_text}** `{avg_change_pct:+.2f}%` {get_trend_emoji(avg_change_pct)}\n"
                f"🎲 **ความมั่นใจ:** `{avg_confidence:.0f}%`\n\n"
                f"� **S&P 500:** `{sp500_data['change_pct']:+.2f}%`\n"
                f"   ปัจจุบัน: `${sp500_data['current']:,.2f}` → คาดการณ์: `${sp500_data['predicted']:,.2f}`\n\n"
                f"📊 **Dow Jones:** `{dji_data['change_pct']:+.2f}%`\n"
                f"   ปัจจุบัน: `${dji_data['current']:,.2f}` → คาดการณ์: `${dji_data['predicted']:,.2f}`")
    
    embed["fields"].append({
        "name": f"� ตลาดหุ้นสหรัฐ (S&P 500 + Dow Jones)",
        "value": us_value,
        "inline": False
    })
    
    # Send to Discord
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL")
    if not webhook_url:
        print("❌ ข้อผิดพลาด: ไม่พบ DISCORD_WEBHOOK_URL ใน environment variables")
        return
    
    status = send_discord_webhook(webhook_url, embed)
    
    if status == 204:
        print(f"✅ ส่ง Discord webhook สำเร็จ (status code: {status})")
    elif status == 429:
        print("⚠️ Discord Rate Limit - ลองอีกครั้งในภายหลัง")
    elif status == 0:
        print("❌ ไม่สามารถเชื่อมต่อ Discord ได้")
    else:
        print(f"⚠️ Discord ตอบกลับด้วย status code: {status}")

if __name__ == "__main__":
    main()
