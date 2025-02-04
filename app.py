from flask import Flask, render_template, request
import toml

# from predict import make_prediction

from models_classes.PklModel import PklModel

# get the config
with open('config.toml', 'r', encoding="utf8") as f:
    config = toml.load(f)


app = Flask(__name__)

# Load models
models = {
    "random_forest": PklModel(
        config['random_forest']['model_path'], config['random_forest']['vectorizer_path'], "Random Forest"
    ),
    # "lstm": PklModel(
    #     config['lstm']['model_path'], config['lstm']['vectorizer_path']
    # ),
    "logreg": PklModel(
        config['logreg']['model_path'], config['logreg']['vectorizer_path'], "Logistic Regression"
    ),
    "naibay": PklModel(
        config['naibay']['model_path'], config['naibay']['vectorizer_path'], "Naive Bayers"
    ),
}


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html", probability=None)


@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form["user_text"]

    selected_model = request.form["model"]
    print("selected model:", selected_model)

    if selected_model in models:
        prediction = models[selected_model].predict(user_text)
    else:
        prediction = "Invalid model selection"

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
