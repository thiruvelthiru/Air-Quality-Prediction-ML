import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data
data = pd.read_csv("Air Quality Index- Delhi.csv")

# Check for NaN values in the entire DataFrame
print(data.isnull().any())

# Drop rows with NaN values
data = data.dropna()

# Define features (X) and target variable (Y)
X = data.drop("PM 2.5", axis=1)
Y = data["PM 2.5"]

# Calculate correlation with the target variable
correlation_matrix = X.corrwith(Y)
selected_features = correlation_matrix[correlation_matrix.abs() > 0.2].index.tolist()

# Select relevant features
X_selected = X[selected_features]

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_selected, Y, test_size=0.25, random_state=123)

# Model building & training using Random Forest
model_rf = RandomForestRegressor(random_state=6878)

model_rf.fit(X_train, Y_train)

# Model validation
Y_pred_rf = model_rf.predict(X_test)
mse= mean_squared_error(Y_pred_rf, Y_test)
accuracy_cal_rf = np.sqrt(mse)
print("accuracy:", accuracy_cal_rf)
