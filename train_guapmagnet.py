import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
import joblib

df = pd.read_csv("guapmagnet_dataset.csv")
df.dropna(inplace=True)
df = df[["bid", "ask", "signal"]]
df["signal"] = df["signal"].map({"BUY": 1, "SELL": 0, "HOLD": 2})
X = df[["bid", "ask"]]
y = df["signal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train, y_train)
joblib.dump(model, "guapmagnet_model.joblib")
print("✅ Model saved as guapmagnet_model.joblib")

