import pandas as pd
import numpy as np
from joblib import dump
try:
    df = pd.read_csv('output.csv')
except:
    print("Error: output.csv not found")
    print("Try running 'csvMaker.py' first")
    exit(1)

class ClipRecommender:
  def __init__(self):
    self.C = None
    self.features = None

  def fit(self, df: pd.DataFrame, drop_cols=("Uname",)):
    feats = [c for c in df.columns if c not in set(drop_cols)]
    self.features = feats
    X = df[self.features].to_numpy(dtype=float)
    norm = np.linalg.norm(X, axis=0, keepdims=True)
    norm[norm == 0] = 1
    Xn = X / norm
    self.C = Xn.T @ Xn
    return self

  def predict(self, x: dict):
    vec = np.array([float(x.get(f, 0.0)) for f in self.features])
    adjusted = vec @ self.C
    max_abs = np.max(np.abs(adjusted)) or 1
    adjusted = adjusted / max_abs
    return {f: float(a) for f, a in zip(self.features, adjusted)}
  
model = ClipRecommender()
model.fit(df)
dump(model, 'clipRecmodel.pkl')