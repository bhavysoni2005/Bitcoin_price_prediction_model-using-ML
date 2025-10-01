# 🪙 Bitcoin Price Prediction – Desktop App (Python + Tkinter)

A machine learning project to forecast **Bitcoin prices** using historical market data.  
It provides an **interactive desktop application** built with Tkinter, where users can fetch live data, train models and view predicted prices instantly.

---

## 📌 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation](#installation)
- [Usage](#usage)
- [Project Workflow](#project-workflow)
- [Results](#results)
- [Future Enhancements](#future-enhancements)
- [Author](#author)

---

## 📖 Project Overview
This project implements a **machine learning-based system** to predict the next-day closing price of Bitcoin.  
Using historical data fetched from Yahoo Finance, the app allows users to train models (Linear Regression / Random Forest) and get forecasts with metrics like **MAE, RMSE and R²**.  
It is designed as a **desktop GUI** for ease of use, requiring no command-line interaction.

---

## ✨ Features
- Fetch **live Bitcoin data** from Yahoo Finance API.
- Load your own CSV datasets.
- **Data Preprocessing & Feature Engineering** built-in.
- Choose between **Linear Regression** and **Random Forest** models.
- View **MAE, RMSE and R² metrics** instantly.
- Predict **next-day closing price** and see trend direction (📈/📉).
- Visualize actual vs predicted prices in a chart.

---

## 🛠️ Tech Stack
- **Frontend (GUI):** Tkinter
- **Backend (ML):** Python, Scikit-learn
- **Data Handling:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Data Source:** Yahoo Finance (via `yfinance`)

---

🚀 Usage

- Click Load CSV to upload your own Bitcoin historical data.
- Or click Fetch Latest Data to get live data from Yahoo Finance.
- Select the desired model from the dropdown.
- Click Train Model to train and view predictions & metrics.

🔄 Project Workflow

- Data Collection: Load CSV or fetch via Yahoo Finance.
- Data Preprocessing: Handle missing values, feature engineering.
- Model Training: Train Linear Regression / Random Forest.
- Prediction: Predict next-day closing price.
- Visualization: Plot Actual vs Predicted price chart.

📊 Results

- Achieved metrics (example): MAE ~300 USD, RMSE ~500 USD, R² ~0.95.
- Intuitive desktop GUI for easy operation.
- Trend indicator to show if price may go up or down next day.

📝 Future Enhancements

- Add advanced time-series models (LSTM, GRU).
- Integrate sentiment analysis from news/social media.
- Deploy as a web or mobile app for wider use.
