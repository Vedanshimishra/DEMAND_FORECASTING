import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np
import pickle

# Step 1: Load the Dataset
df = pd.read_csv("train.csv")  # Replace "data.csv" with your actual file path

# Step 2: Define Features (X) and Target (y)
X = df[['id', 'week', 'center_id', 'meal_id', 'checkout_price', 'base_price', 
        'emailer_for_promotion', 'homepage_featured']]
y = df['num_orders']

# Step 3: Split Data into Training and Testing Sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Model
xgb_model = xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
xgb_model.fit(X_train, y_train)

# Step 5: Evaluate the Model
y_pred = xgb_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Root Mean Squared Error:", np.sqrt(mse))

# Step 6: Save the Model using Pickle
with open("xgb_model.pkl", "wb") as file:
    pickle.dump(xgb_model, file)
