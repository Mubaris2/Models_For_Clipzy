import pandas as pd
import numpy as np
from joblib import dump
try:
    df = pd.read_csv('output.csv')
except:
    print("Error: output.csv not found")
    exit(1)

class ClipRecommender:
  def __init__(self):
    self.C = None
    self.features = None
    self.lastdf = None
    self.trending = []

  def fit(self, df: pd.DataFrame, drop_cols=("Uname",)):
    self.features = [c for c in df.columns if c not in set(drop_cols)]
    if self.lastdf is not None:
      trending_scores = {f: 0.0 for f in self.features}
      for row in range(len(self.lastdf)):
        for f in self.features:
          trending_scores[f] += df.iloc[row][f] - self.lastdf.iloc[row][f]
      self.trending = sorted(trending_scores.items(), key=lambda x:x[1], reverse=True)

    X = df[self.features].to_numpy(dtype=float)
    norm = np.linalg.norm(X, axis=0, keepdims=True)
    norm[norm == 0] = 1
    Xn = X / norm

    self.C = Xn.T @ Xn
    self.lastdf = df.copy()
    return self

  def predict(self, x: dict, squash=True):
    vec = np.array([float(x.get(f, 0.0)) for f in self.features])
    adjusted = self.C @ vec

    if squash:
      adjusted = np.tanh(adjusted)
      adjusted = 0.8 * adjusted + 0.2 * vec  

    return {f: float(f"{a:.6f}") for f, a in zip(self.features, adjusted)}
    
model = ClipRecommender()
model.fit(df)
dump(model, 'clipRecmodel.pkl')