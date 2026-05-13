import joblib

class IntentModel:

    def __init__(
        self,
        path="model/artifacts/intent_model.pkl"
    ):
        self.pipeline = joblib.load(path)

    def predict(self, text):

        prediction = self.pipeline.predict([text])[0]

        probabilities = self.pipeline.predict_proba([text])[0]

        classes = self.pipeline.classes_

        probability_dict = {
            classes[i]: float(probabilities[i])
            for i in range(len(classes))
        }

        return {
            "intent": prediction,
            "probabilities": probability_dict
        }