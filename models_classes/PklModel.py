import joblib
from BaseModel import BaseModel

from utils import format_prediction


class PklModel(BaseModel):
    def __init__(self, model_path: str, vectorizer_path: str, model_name: str) -> None:
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.model_name = model_name

    def predict(self, text: str) -> str:
        X_new = self.vectorizer.transform([text])
        predictions = self.model.predict(X_new)

        return format_prediction(predictions[0], text, self.model_name)
