import tkinter as tk 
from tkinter import filedialog, messagebox
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# ---------------- FUNCTIONS ---------------- #

df = None

def load_csv():
    """Load data from CSV file."""
    global df
    file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    if file_path:
        df = pd.read_csv(file_path)
        messagebox.showinfo("File Loaded", f"Loaded file: {file_path}")
        print(df.head())

def fetch_data():
    """Fetch latest BTC-USD data from Yahoo Finance."""
    global df
    try:
        end = datetime.now()
        start = end - timedelta(days=5*365)  # last 5 years
        df = yf.download("BTC-USD", start=start, end=end)
        df.reset_index(inplace=True)
        messagebox.showinfo("Data Fetched", "Latest Bitcoin data fetched successfully!")
        print(df.tail())
    except Exception as e:
        messagebox.showerror("Error", str(e))

def train_model():
    """Train model, evaluate, and predict next day's price."""
    global df
    if df is None:
        messagebox.showerror("Error", "Please load CSV or fetch data first.")
        return
    
    # Features & target
    X = df[['Open', 'High', 'Low', 'Volume']]
    y = df['Close']
    
    # Train/Test Split
    split_idx = int(len(df) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Choose model
    if model_choice.get() == "Linear Regression":
        model = LinearRegression()
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
    
    # Train model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    metrics_label.config(text=f"MAE: {mae:.2f} | RMSE: {rmse:.2f} | R²: {r2:.4f}")
    
    # Predict Next Day Price
    last_row = df[['Open', 'High', 'Low', 'Volume']].iloc[-1].values.reshape(1, -1)
    next_day_price_value = float(model.predict(last_row)[0]) if last_row is not None else None
    prev_close_value = float(df['Close'].iloc[-1]) if not df['Close'].empty else None

    # Trend comparison logic
    if next_day_price_value is not None and prev_close_value is not None:
        if next_day_price_value > prev_close_value:
            result_label.config(text="Prediction: Price will go UP 📈", fg="green")
        elif next_day_price_value < prev_close_value:
            result_label.config(text="Prediction: Price will go DOWN 📉", fg="red")
        else:
            result_label.config(text="Prediction: Price will remain the SAME ➡", fg="blue")
    else:
        result_label.config(text="Error: Could not compute prediction.", fg="orange")
    
    # Show predicted price
    next_price_label.config(
        text=f"Predicted Next Day Price: ${next_day_price_value:,.2f}",
        fg="black"
    )
    
    # Plot results
    plt.figure(figsize=(8, 4))
    plt.plot(y_test.values, label="Actual")
    plt.plot(y_pred, label="Predicted")
    plt.title("Bitcoin Price Prediction")
    plt.legend()
    plt.show()


# ---------------- UI ---------------- #

root = tk.Tk()
root.title("Bitcoin Price Prediction")
root.geometry("850x650")
root.configure(bg="#f5f5f5")

# Buttons
tk.Button(root, text="Load CSV", command=load_csv, bg="#2196F3", fg="white", width=20).pack(pady=10)
tk.Button(root, text="Fetch Latest Data", command=fetch_data, bg="#4CAF50", fg="white", width=20).pack(pady=5)

# Model Choice
model_choice = tk.StringVar(value="Linear Regression")
tk.Label(root, text="Select Model:", font=("Arial", 12), bg="#f5f5f5").pack(pady=5)
tk.OptionMenu(root, model_choice, "Linear Regression", "Random Forest").pack(pady=5)

# Train Button
tk.Button(root, text="Train Model", command=train_model, bg="#FF9800", fg="white", width=20).pack(pady=10)

# Metrics Label
metrics_label = tk.Label(root, text="", font=("Arial", 12), bg="#f5f5f5")
metrics_label.pack(pady=10)

# Next Price Label
next_price_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#f5f5f5")
next_price_label.pack(pady=10)

# Result Label (trend direction)
result_label = tk.Label(root, text="", font=("Arial", 16, "bold"), bg="#f5f5f5")
result_label.pack(pady=10)

root.mainloop()
