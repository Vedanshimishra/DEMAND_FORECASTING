import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import pickle

# Load your dataset
df_ = pd.read_csv('train.csv')

# Preprocess the data to use relevant features
X = df_.drop(['id', 'num_orders'], axis=1).values
y = df_['num_orders'].values

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Train the model
lr = LinearRegression()
lr.fit(X_train, y_train)

# Make predictions
y_pred = lr.predict(X_test)

# Evaluate the model
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error for LinearRegression:', rmse)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(lr, f)
