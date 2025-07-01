import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import numpy as np

# Load the dataset
df = pd.read_csv("BSE SENSEX DATASET.csv")

# Convert 'Date' to datetime format
df['Date'] = pd.to_datetime(df['Date'], format='%d-%B-%Y')

# Sort by date
df.sort_values('Date', inplace=True)

# Convert dates to ordinal format (for regression)
df['Date_ordinal'] = df['Date'].map(pd.Timestamp.toordinal)

# Prepare X and y for regression
X = df['Date_ordinal'].values.reshape(-1, 1)
y = df['Close'].values.reshape(-1, 1)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X, y)
y_lr_pred = lr_model.predict(X)

# Support Vector Regression (SVM)
svr_model = SVR(kernel='rbf', C=1000, gamma=0.00001)
svr_model.fit(X, y.ravel())
y_svr_pred = svr_model.predict(X)

# Random Forest Regression
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X, y.ravel())
y_rf_pred = rf_model.predict(X)

# R² Scores
lr_r2 = r2_score(y, y_lr_pred)
svr_r2 = r2_score(y, y_svr_pred)
rf_r2 = r2_score(y, y_rf_pred)

print("Linear Regression R² Score (Accuracy):", lr_r2)
print("SVM Regression R² Score (Accuracy):", svr_r2)
print("Random Forest Regression R² Score (Accuracy):", rf_r2)

# Plotting actual vs predicted close prices
plt.figure(figsize=(14, 6))
plt.plot(df['Date'], df['Close'], label='Actual Close Price', color='blue')
plt.plot(df['Date'], y_lr_pred, label='Linear Regression', color='red', linestyle='--')
plt.plot(df['Date'], y_svr_pred, label='SVM Regression (RBF Kernel)', color='green', linestyle='-.')
plt.plot(df['Date'], y_rf_pred, label='Random Forest Regression', color='purple', linestyle=':')
plt.title('BSE SENSEX Close Price with Linear, SVM & Random Forest Regression')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Bar plot of model accuracies
models = ['Linear Regression', 'SVM Regression', 'Random Forest']
accuracies = [lr_r2 * 100, svr_r2 * 100, rf_r2 * 100]

plt.figure(figsize=(8, 4))
bars = plt.bar(models, accuracies, color=['red', 'green', 'purple'] ,width = 0.4)

# Add data labels on top of the bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2, yval + 1, f'{yval:.2f}%', ha='center', va='bottom')

plt.ylim(0, 100)
plt.ylabel('Accuracy (%)')
plt.title('Model Accuracy Comparison')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
