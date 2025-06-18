import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# 🔸 Step 1: Create dummy data based on your logic
data = {
    "pages_visited": [10, 2, 5, 12, 3, 15, 8, 1, 9, 4],
    "time_spent": [200, 30, 60, 220, 25, 300, 120, 15, 180, 50],
    "days_since_signup": [9, 1, 2, 15, 1, 20, 7, 1, 10, 3],
    "is_logged_in": [1, 0, 1, 1, 0, 1, 1, 0, 1, 0],
    "churn": [0, 1, 1, 0, 1, 0, 0, 1, 0, 1]  # 0 = Happy user, 1 = Churned user
}

# 🔸 Step 2: Convert to DataFrame
df = pd.DataFrame(data)

# 🔸 Step 3: Split features and target
X = df.drop("churn", axis=1)
y = df["churn"]

# 🔸 Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🔸 Step 5: Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 🔸 Step 6: Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy:.2f}")

# 🔸 Step 7: Save model
joblib.dump(model, "models/behavior_model.pkl")
print("📦 Model saved to models/behavior_model.pkl")
