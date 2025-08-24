import pandas as pd
import numpy as np
from joblib import dump
try:
    df = pd.read_csv('output.csv')
except:
    print("Error: output.csv not found")
    print("Try running 'csvMaker.py' first")
    exit(1)

class MetadataCorrelationModel:
    def __init__(self):
        self.corr_matrix = None
        self.features = None

    def fit(self, user_clip_matrix: pd.DataFrame):
        user_clip_matrix = user_clip_matrix.drop(columns=["user_id"])
        self.features = user_clip_matrix.columns
        self.corr_matrix = user_clip_matrix.corr().fillna(0)

    def predict(self, clip_metadata: dict):
        vec = np.array([clip_metadata[f] for f in self.features])
        adjusted = vec @ self.corr_matrix.values
        return dict(zip(self.features, adjusted))

model = MetadataCorrelationModel()
model.fit(df)
dump(model, 'clipRecmodel.pkl')