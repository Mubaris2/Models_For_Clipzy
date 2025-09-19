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
    self.norm = None 

  def fit(self, df: pd.DataFrame, drop_cols=("Uname",)):
    self.features = [c for c in df.columns if c not in set(drop_cols)]
    if self.lastdf is not None:
      common = df.merge(self.lastdf, on="Uname", suffixes=("_new", "_old"))
      trending_scores = {}
      for f in self.features:
        trending_scores[f] = (common[f + "_new"] - common[f + "_old"]).sum()
      self.trending = sorted(trending_scores.items(), key=lambda x:x[1], reverse=True)

      X_new = df[self.features].iloc[len(self.lastdf):].to_numpy(dtype=float)
      Xn_new = X_new / self.norm
      self.C += Xn_new.T @ Xn_new
    else:
      X = df[self.features].to_numpy(dtype=float)
      self.norm = np.linalg.norm(X, axis=0, keepdims=True)
      self.norm[self.norm == 0] = 1
      Xn = X / self.norm
      self.C = Xn.T @ Xn

    self.lastdf = df.copy()
    return self

  def predict(self, x: dict, squash=True):
    vec = np.array([float(x.get(f, 0.0)) for f in self.features])
    vec = vec / self.norm.flatten()
    adjusted = self.C @ vec

    if squash:
      adjusted = np.tanh(adjusted)
      adjusted = 0.8 * adjusted + 0.2 * vec  

    return {f: float(f"{a:.6f}") for f, a in zip(self.features, adjusted)}
    
model = ClipRecommender()
model.fit(df)
dump(model, 'clipRecmodel.pkl')