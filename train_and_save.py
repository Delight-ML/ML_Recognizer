import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Load the data
file_path = "market_data.csv"   # make sure this is your actual CSV name
df = pd.read_csv(file_path, encoding='utf-16', on_bad_lines='skip')

# Step 2: Rename columns if needed
df.columns = ["Time", "Open", "High", "Low", "Close", "Volume", "Extra"]

# Step 3: Create a simple target signal
df["Signal"] = 0
df.loc[df["Close"] > df["Open"], "Signal"] = 1  # 1 = Buy
df.loc[df["Close"] < df["Open"], "Signal"] = -1 # -1 = Sell

# Step 4: Select features and target
X = df[["Open", "High", "Low", "Close"]]
y = df["Signal"]

# Step 5: Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 7: Save the trained model
joblib.dump(model, "signal_model.pkl")

print("✅ Model trained and saved successfully as 'signal_model.pkl'!")