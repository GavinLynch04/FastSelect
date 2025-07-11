import pandas as pd
from sklearn.datasets import make_classification

X, y = make_classification(
        n_samples=4000,
        n_features=4000,
        n_informative=100,
        n_redundant=85,
        random_state=42
    )
pd.DataFrame(X).to_csv("benchmark_X.csv", index=False)
pd.DataFrame(y, columns=['target']).to_csv("benchmark_y.csv", index=False)