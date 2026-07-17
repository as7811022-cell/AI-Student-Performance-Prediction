import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Dataset load
data = pd.read_csv("student_data.csv")

# Features
X = data[['StudyHours','Attendance','PreviousMarks','Assignments']]

# Target
y = data['Result']

# Model
model = RandomForestClassifier(random_state=42)

# Training
model.fit(X, y)

# Save Model
joblib.dump(model, "model.pkl")

print("Model Trained Successfully!")