# ⚡ Energy Consumption Forecasting – Time Series ML Project

Forecasting hourly energy consumption using historical time series data, lag features, rolling statistics, and calendar signals. This project demonstrates a full ML workflow from data cleaning to deployment.

---

## 🧠 Problem Statement

Electric utilities must anticipate consumption to prevent overloads and optimize grid performance. This project predicts future energy demand using historical values and temporal patterns.

---

## 🗂️ Dataset

- Source: [Kaggle – Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)
- Granularity: Hourly time series
- Target: Energy in megawatts (MW)
- Features: Lag values, rolling stats, time of day, day of week, etc.

---

## 🔁 Project Workflow

1. **Data Parsing & Cleaning**:
   - Parsed and indexed `Datetime`
   - Handled missing timestamps with interpolation
   - Reindexed to consistent hourly frequency

2. **Feature Engineering**:
   - Lag features: `t-1`, `t-24`, `t-168`
   - Rolling mean & std over 24-hour windows
   - Calendar decomposition (hour, weekday, weekend)

3. **Modeling**:
   - Chronological train-test split (no leakage)
   - Trained baseline Linear Regression
   - Trained and tuned XGBoost Regressor
   - Evaluation using RMSE, MAE, R²

4. **Deployment**:
   - Built a Streamlit app for interactive forecasting
   - Accepts past 168 hourly values and predicts the next hour
   - Shows model input + forecasted output

---

## 🎯 Sample Model Performance

| Model            | RMSE   | R²     |
|------------------|--------|--------|
| LinearRegression | 71.62  | 0.96   |
| XGBoost          | 25.79  | 0.99   |

*Note: Performance will vary based on dataset slice*

---

## 🌐 Streamlit App (Local Demo)

### 🔧 How to Run

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Test case Input: 
1423, 1430, 1437, 1442, 1439, 1441, 1435, 1444, 1447, 1452, 1450, 1456, 1461, 1459, 1463, 1466, 1462, 1464, 1460, 1455, 1453, 1456, 1458, 1462,
1470, 1475, 1473, 1471, 1476, 1478, 1480, 1485, 1487, 1489, 1490, 1492, 1491, 1488, 1484, 1482, 1486, 1489, 1493, 1497, 1499, 1500, 1502, 1505,
1510, 1508, 1506, 1503, 1501, 1502, 1500, 1497, 1495, 1493, 1491, 1492, 1490, 1488, 1487, 1489, 1492, 1494, 1496, 1498, 1499, 1502, 1505, 1508,
1512, 1516, 1513, 1510, 1508, 1506, 1505, 1504, 1503, 1502, 1504, 1505, 1506, 1507, 1508, 1510, 1512, 1514, 1515, 1516, 1517, 1518, 1519, 1520,
1522, 1524, 1525, 1526, 1527, 1526, 1525, 1523, 1521, 1519, 1518, 1517, 1516, 1515, 1514, 1513, 1512, 1510, 1509, 1508, 1507, 1506, 1505, 1503,
1500, 1497, 1494, 1492, 1490, 1488, 1487, 1486, 1485, 1484, 1482, 1480, 1479, 1478, 1477, 1476, 1475, 1474, 1472, 1470, 1468, 1467, 1465, 1462,
1460, 1458, 1455, 1453, 1450, 1448, 1446, 1444, 1442, 1440, 1438, 1436, 1435, 1433, 1431, 1430, 1428, 1426, 1424, 1422, 1420, 1419, 1417, 1415
