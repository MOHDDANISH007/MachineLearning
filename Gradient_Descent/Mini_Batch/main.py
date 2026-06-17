from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
# random module
import random



# Load diabetes dataset
diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target


# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)



# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)



# Train linear regression model
lr = LinearRegression()
lr.fit(X_train_scaled, y_train)

y_pred_lr = lr.predict(X_test_scaled)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_lr):.4f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_lr):.4f}")
print(f"Coefficients: {lr.coef_}")
print(f"Intercept: {lr.intercept_:.4f}")


import numpy as np
import random

class Mini_Batch:

    def __init__(self, learning_rate=0.01, epochs=1000, batch_size=32):
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size

        self.coef_ = None
        self.intercept_ = None
        self.cost_history = []

    def fit(self, X, y):

        n_samples, n_features = X.shape

        # Initialize parameters
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0

        for i in range(self.epochs):

            # Number of batches per epoch
            for j in range(int(n_samples / self.batch_size)):

                # Select random batch
                random_idx = random.sample(
                    range(n_samples),
                    self.batch_size
                )

                X_batch = X[random_idx]
                y_batch = y[random_idx]

                # Predictions
                y_pred = np.dot(X_batch, self.coef_) + self.intercept_

                # Error
                error = y_pred - y_batch

                # Gradients
                d_coef = (2 / self.batch_size) * np.dot(
                    X_batch.T,
                    error
                )

                d_intercept = (2 / self.batch_size) * np.sum(error)

                # Update parameters
                self.coef_ -= self.learning_rate * d_coef
                self.intercept_ -= self.learning_rate * d_intercept

            # Cost after each epoch
            y_pred_all = np.dot(X, self.coef_) + self.intercept_

            cost = np.mean((y - y_pred_all) ** 2)

            self.cost_history.append(cost)

            if i % 100 == 0:
                print(f"Epoch {i:4d}: Cost = {cost:.6f}")

    def predict(self, X):

        return np.dot(X, self.coef_) + self.intercept_


# Train mini-batch gradient descent model
mb = Mini_Batch(learning_rate=0.01, epochs=100, batch_size=40)
mb.fit(X_train_scaled, y_train)

y_pred_mb = mb.predict(X_test_scaled)

print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred_mb):.4f}")
print(f"R^2 Score: {r2_score(y_test, y_pred_mb):.4f}")
print(f"Coefficients: {mb.coef_}")
print(f"Intercept: {mb.intercept_:.4f}")