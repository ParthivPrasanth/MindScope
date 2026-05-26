import os
import joblib
import pandas as pd
import xgboost as xgb

from src.config import MODEL_PATH, ENCODER_PATH, FEATURE_COLUMNS_PATH
from src.data_utils import prepare_features


class DepressionPredictor:
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

        if not os.path.exists(ENCODER_PATH):
            raise FileNotFoundError(f"Missing encoder file: {ENCODER_PATH}")

        if not os.path.exists(FEATURE_COLUMNS_PATH):
            raise FileNotFoundError(f"Missing feature columns file: {FEATURE_COLUMNS_PATH}")

        self.encoder = joblib.load(ENCODER_PATH)
        self.feature_columns = joblib.load(FEATURE_COLUMNS_PATH)

        self.model = xgb.Booster()
        self.model.load_model(MODEL_PATH)

    def predict(self, input_dict):
        input_df = pd.DataFrame([input_dict])

        input_df = prepare_features(input_df)

        input_encoded = self.encoder.transform(input_df)

        for col in self.feature_columns:
            if col not in input_encoded.columns:
                input_encoded[col] = 0

        input_encoded = input_encoded[self.feature_columns]


        dmatrix = xgb.DMatrix(input_encoded)
        raw_prob = float(self.model.predict(dmatrix)[0])
        prob = raw_prob
        pred = int(raw_prob > 0.5)



        return pred, prob