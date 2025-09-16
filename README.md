# Price Prediction Model (SET Index + Machine Learning)

โปรเจคนี้เป็นการพัฒนา **โมเดลคาดการณ์ราคาหุ้นในตลาดหลักทรัพย์แห่งประเทศไทย (SET Index)**  
โดยใช้ทั้ง **Technical Indicators** (SMA, EMA, Support, Resistance, Breakout)  
และการประยุกต์ใช้ **Machine Learning** เพื่อเพิ่มความแม่นยำในการทำนายแนวโน้มราคา

---

## เป้าหมายของโปรเจค
- ทดลองสร้าง **โมเดลทำนายราคาหุ้นไทย (SET Index)**  
- เปรียบเทียบผลลัพธ์ระหว่าง **Rule-based (Indicator)** กับ **Machine Learning Models**  
- ทำให้ผู้เริ่มต้นเข้าใจการนำ Data Science มาประยุกต์ใช้กับการวิเคราะห์หุ้น  

---

## เทคโนโลยีและเครื่องมือที่ใช้
- **ภาษา**: Python
- **Library หลัก**: pandas, numpy, matplotlib, scikit-learn, tensorflow/keras, yfinance  
- **เครื่องมือ**: Jupyter Notebook  

---

## วิธีการทำงานของโปรเจค

### 1. การเตรียมข้อมูล (Data Preparation)
- ดึงข้อมูลราคาหุ้นจาก Yahoo Finance (`SET.BK`, หรือหุ้นรายตัวในตลาด)  
- ทำความสะอาดและจัดรูปแบบข้อมูลให้อยู่ใน Time Series  
- สร้างฟีเจอร์จาก Technical Indicators เช่น:
  - SMA (Simple Moving Average): ค่าเฉลี่ยราคาย้อนหลังแบบง่ายๆ ช่วยดูแนวโน้ม
    *SMA (Simple Moving Average)**  
    - SMA 20 วัน → แนวโน้มระยะสั้น  
    - SMA 50 วัน → แนวโน้มกลาง  
    - SMA 200 วัน → แนวโน้มยาว  
  - EMA (Exponential Moving Average) : คล้าย SMA แต่ให้น้ำหนักกับราคาล่าสุดมากกว่า
    **EMA (Exponential Moving Average)**  
    - EMA 12 วัน (สั้น, ไวต่อราคา)  
    - EMA 26 วัน (ยาวกว่า, ใช้คู่กับ EMA 12)
  - RSI (Relative Strength Index)
    **RSI (Relative Strength Index)**  
    - RSI 14 วัน (มาตรฐานทั่วไป)  
    - RSI > 70 → Overbought  
    - RSI < 30 → Oversold  
  - MACD
    **MACD (Moving Average Convergence Divergence)**  
    - MACD line = EMA(12) – EMA(26)  
    - Signal line = EMA(9) ของ MACD  
    - ใช้ตรวจจับสัญญาณตัดขึ้น (Bullish) / ตัดลง (Bearish)  
  - Volume-based signals
    **Volume-based Signals**  
    - ราคาขึ้นพร้อม Volume เพิ่มขึ้น → แนวโน้มแข็งแรง  
    - ราคาลงพร้อม Volume เพิ่มขึ้น → แนวโน้มลงจริง  
    - ใช้เป็นตัวเสริมยืนยัน (Confirmation)  

---

### 2. การสร้างโมเดล Machine Learning
โปรเจคนี้จะทดลองโมเดลหลายแบบเพื่อเปรียบเทียบผลลัพธ์:

- **Regression Models**
  ใช้ในการคาดการณ์ราคาหุ้นล่วงหน้า (Price Prediction)
  - Linear Regression
  - Random Forest Regressor
  - XGBoost

- **Classification Models**
  ใช้จำแนกสัญญาณ เช่น “ซื้อ/ขาย/ถือ” (Buy / Sell / Hold)
  - Logistic Regression (ทำนาย Buy / Sell / Hold)
  - Support Vector Machine (SVM)
  - Neural Network (MLP)

- **Time Series Models**
  เหมาะกับข้อมูลราคาหุ้นที่เป็นลำดับเวลา (Time Series) โดยเฉพาะ LSTM ซึ่งใช้ได้ดีกับแนวโน้มระยะยาว
  - ARIMA (ฐานเปรียบเทียบ)
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)

---

### 3. การประเมินผล (Evaluation)
- ใช้ Train/Test Split และ Cross-validation  
- Metric ที่ใช้:
  - RMSE (Root Mean Square Error) สำหรับการทำนายราคา
  - Accuracy / F1-score สำหรับการจำแนกสัญญาณซื้อขาย
- เปรียบเทียบผลลัพธ์ระหว่าง Indicator-only กับ Machine Learning  

---

### 4. การแสดงผล (Visualization)
- กราฟราคาจริง (Actual) เทียบกับราคาที่โมเดลทำนาย (Predicted)  
- จุดสัญญาณ Breakout, Buy, Sell  
- Visualization ของ Feature Importance (เช่นจาก Random Forest/XGBoost)  

---

## โครงสร้างไฟล์ในโปรเจค
```bash
price-prediction-model/
│
├── data/                  # เก็บข้อมูลราคาหุ้น (CSV)
├── notebooks/             # Jupyter Notebooks
│   ├── EDA.ipynb          # สำรวจและทำความสะอาดข้อมูล
│   ├── Indicators.ipynb   # ทดลองคำนวณ Technical Indicators
│   ├── ML_models.ipynb    # โมเดล Machine Learning
│   └── LSTM_model.ipynb   # โมเดล LSTM สำหรับ Time Series
├── src/                   # โค้ดหลัก
│   ├── data_loader.py     # โหลดและเตรียมข้อมูล
│   ├── indicators.py      # ฟังก์ชันคำนวณ Indicator
│   ├── models.py          # โมเดล ML ที่ใช้
│   ├── evaluate.py        # ฟังก์ชันประเมินผล
│   └── visualize.py       # วาดกราฟผลลัพธ์
├── requirements.txt       # Library ที่ต้องใช้
└── README.md              # คู่มือโปรเจค
⚙️ วิธีการติดตั้งและใช้งาน
Clone โปรเจค

bash
Copy code
git clone https://github.com/username/price-prediction-model.git
cd price-prediction-model
ติดตั้ง Library

bash
Copy code
pip install -r requirements.txt
รัน Jupyter Notebook

bash
Copy code
jupyter notebook notebooks/EDA.ipynb
✅ สิ่งที่ได้จากโปรเจคนี้
เรียนรู้การสร้าง Indicator เบื้องต้น (SMA, EMA, MACD, RSI)

ทดลองใช้ Machine Learning และ Deep Learning (LSTM) กับข้อมูลหุ้นจริง

เปรียบเทียบผลลัพธ์ระหว่าง Rule-based vs ML-based models
