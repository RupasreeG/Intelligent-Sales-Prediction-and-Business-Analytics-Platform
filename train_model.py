import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load dataset
data = pd.read_csv("Superstore.csv")

# Select useful columns
data = data[["Category", "Region", "Segment", "Sales"]]

# Encode categorical columns
le_category = LabelEncoder()
le_region = LabelEncoder()
le_segment = LabelEncoder()

data["Category"] = le_category.fit_transform(data["Category"])
data["Region"] = le_region.fit_transform(data["Region"])
data["Segment"] = le_segment.fit_transform(data["Segment"])

X = data[["Category", "Region", "Segment"]]
y = data["Sales"]

# Train model
model = RandomForestRegressor()
model.fit(X, y)

# Save everything
os.makedirs("model", exist_ok=True)

with open("model/model.pkl", "wb") as f:
    pickle.dump((model, le_category, le_region, le_segment), f)

print("Model trained successfully")








