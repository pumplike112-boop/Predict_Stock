# 🤖 AI Stock Market Prediction Bot

ระบบทำนายตลาดหุ้นด้วย Deep Learning LSTM สำหรับตลาดไทย (SET) และสหรัฐ (S&P 500, Dow Jones)

## ✨ Features

- **Bidirectional LSTM Neural Network** - โมเดล Deep Learning ที่ทันสมัย
- **14 Technical Indicators** - MA, MACD, RSI, Bollinger Bands, Stochastic, ATR, OBV, ROC, Volatility
- **Pre-trained Models** - เทรนครั้งเดียว ใช้ได้นาน (เร็วมาก!)
- **Confidence Score** - แสดงความมั่นใจของการทำนาย 0-100%
- **Predict Returns** - ทำนาย % การเปลี่ยนแปลง แทนราคาตรงๆ (แม่นยำกว่า)
- **Discord Webhook** - ส่งการทำนายสวยๆ แบบอัตโนมัติ
- **GitHub Actions** - รันทุกวันจันทร์-ศุกร์ เวลา 07:00 น. (ไทย)

## 🚀 Quick Start

### 1. ติดตั้ง Dependencies

```bash
pip install -r requirements.txt
```

### 2. เทรนโมเดล (ครั้งแรกเท่านั้น)

```bash
python train_models.py
```

**ใช้เวลา:** 15-30 นาที (เทรนด้วยข้อมูล 10 ปี)
**ผลลัพธ์:** โมเดลที่เทรนแล้วใน folder `models/`

### 3. ตั้งค่า Discord Webhook

สร้างไฟล์ `.env`:
```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_TOKEN
```

### 4. ทำนายตลาด

```bash
python predict_markets.py
```

**ใช้เวลา:** 30-60 วินาที (เร็วมาก!)

## 📊 Model Architecture

```
Input (60 days × 14 features)
    ↓
Bidirectional LSTM (128 units) → Dropout (0.3)
    ↓
Bidirectional LSTM (64 units) → Dropout (0.3)
    ↓
LSTM (32 units) → Dropout (0.2)
    ↓
Dense (16 units, ReLU)
    ↓
Dense (1 unit) → Prediction (% Return)
```

## 🎯 Key Improvements

### ✅ แก้ Data Leakage
- Fit scaler เฉพาะข้อมูล training
- แยก train/validation อย่างถูกต้อง

### ✅ แยก Training/Inference
- เทรนครั้งเดียวด้วยข้อมูล 10 ปี
- บันทึกโมเดล + scaler
- ใช้โมเดลที่เทรนแล้วทำนาย (เร็ว 50x)

### ✅ Predict Returns แทนราคา
- ทำนาย % การเปลี่ยนแปลง
- ป้องกันโมเดล "โกง" โดยใช้ราคาเมื่อวาน

### ✅ Confidence Score
- แสดงความมั่นใจ 0-100%
- คำนวณจาก prediction uncertainty

### ✅ Error Handling
- จัดการ API failures
- Discord rate limiting
- Fallback mechanisms

## 📁 Project Structure

```
.
├── predict_markets.py      # สคริปต์ทำนาย (ใช้โมเดลที่เทรนแล้ว)
├── train_models.py          # สคริปต์เทรนโมเดล (รันครั้งเดียว)
├── requirements.txt         # Python dependencies
├── .env                     # Discord webhook URL
├── .github/workflows/       # GitHub Actions
│   └── main.yml
└── models/                  # โมเดลที่เทรนแล้ว
    ├── SETBK_model.keras
    ├── SETBK_scaler.pkl
    ├── GSPC_model.keras
    ├── GSPC_scaler.pkl
    ├── DJI_model.keras
    └── DJI_scaler.pkl
```

## 🔧 GitHub Actions Setup

1. ไปที่ Settings → Secrets and variables → Actions
2. เพิ่ม secret: `DISCORD_WEBHOOK_URL`
3. Commit โมเดลที่เทรนแล้วใน folder `models/`
4. Push ขึ้น GitHub

**หมายเหตุ:** ไฟล์โมเดล (~50-100 MB) ต้อง commit ขึ้น repo เพื่อให้ GitHub Actions ใช้งานได้

## 📈 Performance

- **Training Time:** 15-30 นาที (ครั้งแรกเท่านั้น)
- **Prediction Time:** 30-60 วินาที (ใช้โมเดลที่เทรนแล้ว)
- **Accuracy:** ~65-75% direction accuracy
- **Data:** 10 ปีย้อนหลัง สำหรับ training

## ⚠️ Disclaimer

**ไม่ใช่คำแนะนำทางการเงิน** ใช้เพื่อการศึกษาเท่านั้น ควรศึกษาข้อมูลเพิ่มเติมก่อนตัดสินใจลงทุน

## 🤝 Contributing

Pull requests are welcome! สำหรับการเปลี่ยนแปลงใหญ่ กรุณาเปิด issue ก่อน

## 📝 License

MIT License

---

Made with ❤️ and 🤖 Deep Learning | Powered by TensorFlow & Keras
