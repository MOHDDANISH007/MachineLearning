import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler



# Load data
df = pd.read_csv("placement.csv")
print("Data shape:", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nData info:")
print(df.info())


# Split into X (features) and y (target)
X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



print(f"\nTraining set size: {len(X_train)}")
print(f"Test set size: {len(X_test)}")


# Optional: Scale features for better gradient descent performance
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ==========================
# 1. Scikit-Learn Model
# ==========================
print("\n" + "="*50)
print("SCIKIT-LEARN LINEAR REGRESSION")
print("="*50)


lr = LinearRegression()
lr.fit(X_train, y_train)

# Predict on test data (not training data!)
y_pred_lr = lr.predict(X_test)

print(f"\nMean Squared Error: {mean_squared_error(y_test, y_pred_lr):.4f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_:.4f}")


# ==========================
# 2. Custom Batch Gradient Descent
# ==========================
print("\n" + "="*50)
print("CUSTOM BATCH GRADIENT DESCENT")
print("="*50)



class BatchGradientDescent:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None
        self.cost_history = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0
        
        for i in range(self.epochs):
            # Forward pass
            y_pred = np.dot(X, self.coef_) + self.intercept_

            # Calculate gradients
            error = y_pred - y
            d_coef = (2 / n_samples) * np.dot(X.T, error)
            d_intercept = (2 / n_samples) * np.sum(error)
            
            # Update parameters
            self.coef_ -= self.learning_rate * d_coef
            self.intercept_ -= self.learning_rate * d_intercept
            
            # Calculate and store cost (MSE)
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # Optional: Print progress every 100 epochs
            if i % 100 == 0:
                print(f"Epoch {i:4d}: Cost = {cost:.6f}")
    
    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_



# Train custom model
print("\nTraining Custom Model...")
gg = BatchGradientDescent(learning_rate=0.01, epochs=1000)
gg.fit(X_train_scaled, y_train)  # Use scaled data for better convergence


# Predict on test data
y_pred_gg = gg.predict(X_test_scaled)

print(f"\nMean Squared Error: {mean_squared_error(y_test, y_pred_gg):.4f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_gg):.4f}")
print(f"Coefficients: {gg.coef_}")
print(f"Intercept: {gg.intercept_:.4f}")




# ==========================
# 2. Custom Stochastic Gradient Descent
# ==========================
print("\n" + "="*50)
print("CUSTOM Stochastic GRADIENT DESCENT")
print("="*50)



class StochasticGradientDescent:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.coef_ = None
        self.intercept_ = None
        self.cost_history = []

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # Initialize parameters
        self.coef_ = np.ones(n_features)
        self.intercept_ = 0

        for i in range(self.epochs):

            for j in range(n_samples):

                # Randomly select one training example
                random_idx = np.random.randint(0, n_samples)

                # Forward pass
                y_pred = np.dot(X[random_idx], self.coef_) + self.intercept_

                # Error
                error = y[random_idx] -  y_pred 

                # Gradients
                d_coef = -2 * X[random_idx] * error
                d_intercept = -2 * error

                # Update parameters
                self.coef_ -= self.learning_rate * d_coef
                self.intercept_ -= self.learning_rate * d_intercept

            # Calculate cost after one epoch
            y_pred_all = np.dot(X, self.coef_) + self.intercept_
            cost = np.mean((y_pred_all - y) ** 2)
            self.cost_history.append(cost)

            if i % 100 == 0:
                print(f"Epoch {i:4d}: Cost = {cost:.6f}")

    def predict(self, X):
        return np.dot(X, self.coef_) + self.intercept_


sgd = StochasticGradientDescent(
    learning_rate=0.01,
    epochs=20
)

sgd.fit(X_train_scaled, y_train)

y_pred_sgd = sgd.predict(X_test_scaled)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_sgd):.4f}")
print(f"R² Score: {r2_score(y_test, y_pred_sgd):.4f}")
print(f"Coefficients: {sgd.coef_}")
print(f"Intercept: {sgd.intercept_:.4f}")