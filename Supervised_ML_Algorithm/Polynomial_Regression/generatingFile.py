import numpy as np
import pandas as pd

np.random.seed(42)

X = 6 * np.random.rand(200, 1) - 3
y = X**2 + 2 + np.random.randn(200, 1)

df = pd.DataFrame({
    "X": X.flatten(),
    "Y": y.flatten()
})

df.to_csv("polynomial_dataset.csv", index=False)

print(df.head())