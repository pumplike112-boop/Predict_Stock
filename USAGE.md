# 📖 คู่มือการใช้งาน

## 🎯 ขั้นตอนการใช้งาน

### ขั้นตอนที่ 1: เทรนโมเดล (ครั้งแรกเท่านั้น)

```bash
python train_models.py
```

**สิ่งที่จะเกิดขึ้น:**
1. ดึงข้อมูลหุ้นย้อนหลัง 10 ปี จาก Yahoo Finance
2. สร้าง 14 technical indicators
3. เทรนโมเดล LSTM 3 ตัว (SET, S&P 500, Dow Jones)
4. บันทึกโมเดล + scaler ใน folder `models/`

**ใช้เวลา:** 15-30 นาที

**ผลลัพธ์:**
```
models/
├── SETBK_model.keras      (~30 MB)
├── SETBK_scaler.pkl       (~1 KB)
├── GSPC_model.keras       (~30 MB)
├── GSPC_scaler.pkl        (~1 KB)
├── DJI_model.keras        (~30 MB)
└── DJI_scaler.pkl         (~1 KB)
```

---

### ขั้นตอนที่ 2: ทำนายตลาด

```bash
python predict_markets.py
```

**สิ่งที่จะเกิดขึ้น:**
1. โหลดโมเดลที่เทรนไว้แล้ว
2. ดึงข้อมูลล่าสุดจาก Yahoo Finance
3. ทำนายราคาวันถัดไป
4. ส่งผลลัพธ์ไปที่ Discord

**ใช้เวลา:** 30-60 วินาที

**ตัวอย่างผลลัพธ์:**
```
🔄 กำลังประมวลผล ดัชนี SET...
✅ ดัชนี SET: 1489.66 → 1465.23 (-1.64%) [ความมั่นใจ: 72%]

🔄 กำลังประมวลผล S&P 500...
✅ S&P 500: 6768.32 → 6851.03 (+1.22%) [ความมั่นใจ: 68%]

🔄 กำลังประมวลผล Dow Jones...
✅ Dow Jones: 47785.07 → 48069.05 (+0.59%) [ความมั่นใจ: 65%]

✅ ส่ง Discord webhook สำเร็จ (status code: 204)
```

---

## 🔄 เมื่อไหร่ต้องเทรนโมเดลใหม่?

**แนะนำให้เทรนใหม่:**
- ทุก 1-3 เดือน (เพื่อให้โมเดลเรียนรู้ข้อมูลใหม่)
- เมื่อตลาดมีการเปลี่ยนแปลงครั้งใหญ่
- เมื่อความแม่นยำลดลงอย่างเห็นได้ชัด

**วิธีเทรนใหม่:**
```bash
# ลบโมเดลเก่า
rm -rf models/

# เทรนใหม่
python train_models.py
```

---

## 🤖 GitHub Actions (รันอัตโนมัติ)

### ตั้งค่า:

1. **เพิ่ม Discord Webhook Secret:**
   - ไปที่ Settings → Secrets and variables → Actions
   - คลิก "New repository secret"
   - Name: `DISCORD_WEBHOOK_URL`
   - Value: `https://discord.com/api/webhooks/...`

2. **Commit โมเดล:**
   ```bash
   git add models/
   git commit -m "Add trained models"
   git push
   ```

3. **เช็คว่ารันสำเร็จ:**
   - ไปที่ Actions tab
   - ดู workflow "Daily Market Prediction"

### กำหนดการรัน:

- **อัตโนมัติ:** จันทร์-ศุกร์ เวลา 00:00 UTC (07:00 น. เวลาไทย)
- **Manual:** คลิก "Run workflow" ใน Actions tab

---

## 🐛 Troubleshooting

### ปัญหา: โมเดลไม่พบ

```
⚠️ โมเดลสำหรับ ^SET.BK ไม่พบ กรุณารัน train_models.py ก่อน
💡 กำลังใช้โมเดลแบบ real-time แทน...
```

**วิธีแก้:** รัน `python train_models.py`

---

### ปัญหา: Discord Rate Limit

```
⚠️ Discord Rate Limit - รอสักครู่แล้วลองใหม่
```

**วิธีแก้:** รอ 1-2 นาที แล้วรันใหม่

---

### ปัญหา: Yahoo Finance ล้ม

```
❌ เกิดข้อผิดพลาดในการดึงข้อมูล ^SET.BK: ...
```

**วิธีแก้:** 
- เช็คอินเทอร์เน็ต
- ลองรันใหม่อีกครั้ง
- Yahoo Finance อาจมีปัญหาชั่วคราว

---

## 💡 Tips

### เพิ่มตลาดอื่นๆ:

แก้ไขใน `train_models.py` และ `predict_markets.py`:

```python
markets = {
    "^SET.BK": "ดัชนี SET",
    "^GSPC": "S&P 500",
    "^DJI": "Dow Jones",
    "^IXIC": "NASDAQ",  # เพิ่มใหม่
    "^N225": "Nikkei 225"  # เพิ่มใหม่
}
```

จากนั้นรัน `python train_models.py` ใหม่

---

### ปรับแต่ง Confidence Score:

แก้ไขใน `predict_markets.py`:

```python
# ปัจจุบัน
confidence = min(100, max(0, 100 - abs(predicted_return * 10000)))

# ปรับให้เข้มงวดขึ้น
confidence = min(100, max(0, 100 - abs(predicted_return * 20000)))
```

---

## 📊 ทำความเข้าใจผลลัพธ์

### ความมั่นใจ (Confidence):

- **80-100%:** มั่นใจมาก (แนวโน้มชัดเจน)
- **60-79%:** มั่นใจปานกลาง
- **40-59%:** มั่นใจน้อย
- **0-39%:** ไม่มั่นใจ (ตลาดผันผวน)

### การเปลี่ยนแปลง (%):

- **+2% ขึ้นไป:** แนวโน้มขาขึ้นแรง 🚀
- **0% ถึง +2%:** แนวโน้มขาขึ้น 📈
- **0% ถึง -2%:** แนวโน้มขาลง 📉
- **-2% ลงไป:** แนวโน้มขาลงแรง 💥

---

## 🎓 Advanced Usage

### Backtesting (ทดสอบย้อนหลัง):

```python
# TODO: เพิ่มฟีเจอร์นี้ในอนาคต
# จะทดสอบว่าถ้าเทรดตามสัญญาณจะได้กำไรเท่าไร
```

### Ensemble Models:

```python
# TODO: รวมหลายโมเดลเข้าด้วยกัน
# เช่น LSTM + GRU + Transformer
```

---

**หากมีคำถาม:** เปิด Issue ใน GitHub
