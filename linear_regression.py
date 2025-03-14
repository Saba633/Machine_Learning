import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample Data: House Size (in square feet) and Price (in thousands)
X = np.array([500, 700, 800, 1000, 1200, 1500]).reshape(-1, 1)  # Features (House Size)
y = np.array([150, 200, 220, 260, 300, 360])  # Target (House Price)

# Create and Train the Model
model = LinearRegression()
model.fit(X, y)

# Make a Prediction
new_size = np.array([[1100]])  # Predict price for 1100 sq ft house
predicted_price = model.predict(new_size)
print(f"Predicted Price for 1100 sq ft: {predicted_price[0]:.2f}K")

# Plot the Data and Regression Line
plt.scatter(X, y, color="blue", label="Actual Data")
plt.plot(X, model.predict(X), color="red", label="Regression Line")
plt.xlabel("House Size (sq ft)")
plt.ylabel("Price (K)")
plt.legend()
plt.show()
