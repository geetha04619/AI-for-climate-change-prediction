
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

# Step 1: Load the dataset
data = {
    'year': np.arange(1980, 2021),
    'co2': [338.7, 340.1, 341.3, 343.1, 344.5, 346.0, 347.2, 348.9, 351.5, 353.2,
            354.4, 355.7, 357.1, 358.8, 360.2, 362.1, 364.0, 366.5, 368.0, 369.5,
            371.1, 373.2, 375.7, 377.4, 379.8, 381.9, 384.0, 386.2, 388.5, 390.7,
            392.9, 395.0, 397.1, 399.3, 401.5, 403.7, 405.9, 408.1, 410.3, 412.5, 414.7],
    'temperature': [0.27, 0.31, 0.34, 0.33, 0.35, 0.38, 0.40, 0.41, 0.43, 0.45,
                    0.47, 0.50, 0.52, 0.54, 0.56, 0.57, 0.59, 0.60, 0.62, 0.64,
                    0.66, 0.68, 0.70, 0.72, 0.74, 0.75, 0.77, 0.79, 0.80, 0.82,
                    0.84, 0.86, 0.88, 0.89, 0.91, 0.93, 0.94, 0.96, 0.98, 1.00, 1.02]
}

df = pd.DataFrame(data)

# Step 2: Train-test split
X = df[['year', 'co2']]
y = df['temperature']
X_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Step 3: Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Evaluate model
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {mae:.3f}")

# Step 5: Predict future temperature (e.g., 2035, 430 ppm CO2)
future_year = 2035
future_co2 = 430
future_temp = model.predict([[future_year, future_co2]])
print(f"Predicted global temperature in {future_year}: {future_temp[0]:.2f}°C")

# Step 6: Visualization
plt.scatter(df['year'], df['temperature'], color='blue', label='Actual')
plt.plot(df['year'], model.predict(df[['year', 'co2']]), color='red', label='Predicted')
plt.xlabel('Year')
plt.ylabel('Global Temperature (°C)')
plt.title('Climate Change Prediction')
plt.legend()
plt.grid(True)
plt.show()

