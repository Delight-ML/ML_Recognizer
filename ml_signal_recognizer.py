import pandas as pd
import time
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

import os

# === Telegram Notification Setup ===
import requests

TELEGRAM_BOT_TOKEN = "8309841481:AAEK933Vbeenkk9FiK4cDWVue5OLcboQvBo"
TELEGRAM_CHAT_ID = "7741874426"

def send_telegram_message(message):
    """Send message to Telegram chat using your bot."""
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
    try:
        requests.post(url, data=payload)
        print("✅ Telegram notification sent!")
    except Exception as e:
        print("⚠️ Failed to send Telegram message:", e)

# === Ensure signal log folder exists ===
log_folder = "signal_logs"
os.makedirs(log_folder, exist_ok=True)
log_file = os.path.join(log_folder, "signal_log.csv")

def send_telegram_message(message):
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        requests.post(url, json=payload)
        print(f"📩 Telegram notification sent: {message}")
    except Exception as e:
        print("Error sending Telegram message:", e)

# === Load & prepare data (robust cleaner for messy MT5 CSVs) ===
import pandas as pd

# Read file as raw text
with open("market_data.csv", "r", encoding="utf-8") as f:
    text = f.read()

# Force-split into multiple lines (this handles weird CSV exports from MT5)
if "\n" not in text and ";" in text:
    text = text.replace(";", "\n")

# Save fixed version
with open("market_data_fixed.csv", "w", encoding="utf-8") as f:
    f.write(text)

# Load again with pandas
# === Load and Prepare Data ===
data = pd.read_csv(
    'market_data_clean.csv',
    sep=',', 
    engine='python',
    encoding='utf-8',
    on_bad_lines='skip'
)
# Auto-fix weird spacing or missing commas
if len(data) <= 1:
    print("Detected malformed CSV — attempting to auto-fix...")
    with open('market_data_clean.csv', 'r', encoding='utf-8', errors='ignore') as f:
        text = f.read()
        text = text.replace(' ', ' ')  # remove non-breaking spaces
        text = text.replace(' ,', ',').replace(', ', ',')  # tighten commas
        text = text.replace(',,', ',')  # clean double commas

    with open('market_data_fixed.csv', 'w', encoding='utf-8') as f:
        f.write(text)

    data = pd.read_csv('market_data_fixed.csv', sep=',', engine='python', on_bad_lines='skip')

# Convert numeric columns to proper float or int
numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
for col in numeric_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Drop rows that failed to convert (if any)
data.dropna(subset=numeric_cols, inplace=True)

print("Rows after numeric cleaning:", len(data))
print(data.head())

# === Feature Engineering ===
import re
import pandas as pd
from io import StringIO

clean_fname = "market_data_clean.csv"
orig_fname = "market_data.csv"

# 1) read raw bytes and decode safely (try utf-8, fallback latin1)
with open(orig_fname, "rb") as f:
    raw = f.read()

try:
    text = raw.decode("utf-8")
except Exception:
    text = raw.decode("latin1", errors="replace")

# 2) remove non-breaking spaces and weird Unicode spaces
text = text.replace("\u00A0", " ").replace("\r\n", "\n")

# 3) remove spaces that are splitting digits (e.g. "3 6 7 8 5 3 . 2 7 0 0 0" -> "367853.27000")
#    This removes spaces between digits or between digit and dot.
text = re.sub(r'(?<=\d)\s+(?=\d)', '', text)          # spaces between digits
text = re.sub(r'(?<=\d)\s+(?=\.)', '', text)          # spaces before dot
text = re.sub(r'(?<=\.)\s+(?=\d)', '', text)          # spaces after dot

# 4) normalize separators: collapse spaces around commas to single comma
text = re.sub(r'\s*,\s*', ',', text)

# 5) remove any double commas that may have appeared
text = re.sub(r',+', ',', text)

# 6) write cleaned temporary CSV (utf-8)
with open(clean_fname, "w", encoding="utf-8") as f:
    f.write(text)

# 7) Now try to load the cleaned CSV with pandas
#    Detect if header line exists (contains 'time' or 'open'); otherwise set header=None and assign names
first_line = text.splitlines()[0].lower() if len(text) > 0 else ""
has_header = ("time" in first_line) or ("open" in first_line)

if has_header:
    data = pd.read_csv(clean_fname, engine='python', on_bad_lines='skip')
else:
    # if no header, assume standard columns and force names
    data = pd.read_csv(clean_fname, engine='python', header=None, names=["Time","Open","High","Low","Close","Volume"], on_bad_lines='skip')
print("Rows loaded from CSV:", len(data))
print(data.head(10))

# 8) Final cleanup: strip column names and remove empty entirely-empty columns
data.columns = data.columns.astype(str).str.strip()
data = data.loc[:, ~data.columns.str.fullmatch('Unnamed:.*')]

# 9) Show how many rows loaded and the top rows for verification
# Fix case where data is all on one line
if len(data) <= 1:
    print("⚠️ Detected single-line CSV, trying to auto-split rows...")
    with open('market_data.csv', 'r', encoding='utf-8') as f:
        raw = f.read().replace('  ', ',').replace(' ,', ',')
    with open('market_data_fixed.csv', 'w', encoding='utf-8') as f:
        f.write(raw.replace(' ,', '\n'))
    data = pd.read_csv('market_data_fixed.csv')
print("Rows loaded:", len(data))
print(data.head())

# Clean column names
data.columns = data.columns.str.strip().str.capitalize()

# Ensure the required columns exist
expected_cols = ['Open', 'High', 'Low', 'Close']
for col in expected_cols:
    if col not in data.columns:
        raise ValueError(f"Missing column: {col}")

# Create target variable (example logic — adjust if needed)
# Ensure numeric columns are clean
for col in ['Open', 'High', 'Low', 'Close']:
    data[col] = pd.to_numeric(data[col], errors='coerce')
data['Signal'] = (data['Close'].shift(-1) > data['Close']).astype(int)

# Features and labels
X = data[expected_cols]
y = data['Signal']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
preds = model.predict(X_test)
accuracy = accuracy_score(y_test, preds)
print(f"Model training complete - Ready for signal generation! ✅")
print(f"Accuracy: {accuracy * 100:.2f}%")

# === Signal Generation ===
latest_features = X.tail(1)
signal = model.predict(latest_features)[0]
print(f"Latest Signal: {'BUY 🟢' if signal == 1 else 'SELL 🔴'}")

# === Function to log signals ===
def log_signal(timestamp, signal, price):
    """Append each new signal to a CSV file with timestamp and price."""
    with open(log_file, "a") as f:
        f.write(f"{timestamp},{signal},{price}\n")
    print(f"✅ Signal logged: {signal} at {price} ({timestamp})")

# === Live CSV Monitoring ===
def monitor_csv(model, csv_path, refresh_rate=30):
    print("\nMonitoring CSV for new signals...\n")
    last_signal = None
csv_path = 'market_data_fixed.csv'  # Path to your CSV file
refresh_rate = 5  # seconds (you can adjust to how often you want it to check for updates)

import time
import os

print("\n🔄 Monitoring CSV for new signals...\n")

last_row_count = 0
last_signal = None

while True:
    try:
        # Check if file exists and has grown
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path, encoding='ISO-8859-1', engine='python')
            data.columns = data.columns.str.strip().str.capitalize()

            # Ensure numeric columns are cleaned
            for col in ['Open', 'High', 'Low', 'Close']:
                data[col] = data[col].astype(str).str.replace(' ', '').astype(float)

            # Only process if new rows are added
            if len(data) > last_row_count:
                print(f"\n📈 New data detected! Total rows: {len(data)}")
                data['Signal'] = (data['Close'].shift(-1) > data['Close']).astype(int)

            # --- Handle latest signal safely (replace old print/log/send code with this) ---
            # convert numeric 0/1 signal to string label
                latest_signal = 'Buy' if int(data['Signal'].iloc[-1]) == 1 else 'Sell'

            # get latest price and timestamp (ensure price is numeric)
                latest_price = float(data['Close'].iloc[-1])
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # print to console
                print(f"{timestamp} - New signal: {latest_signal}")

            # log to csv (uses your existing log_signal function)
                log_signal(timestamp, latest_signal, latest_price)

            # send Telegram (use your existing send_telegram_message function)
                send_telegram_message(f"📊 New Signal Detected: {latest_signal.upper()} at {latest_price}")

            # update row counter after handling the new row
                last_row_count = len(data)

        else:
            print("⚠️ CSV file not found! Waiting...")

    except Exception as e:
        print("❌ Error during monitoring:", e)

    time.sleep(refresh_rate)

# === Start Monitoring ===
monitor_csv(model, "market_data.csv", refresh_rate=30)