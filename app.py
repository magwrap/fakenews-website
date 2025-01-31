from flask import Flask, render_template, request

from predict import make_prediction

app = Flask(__name__)

# # Load LSTM model

#
# # Load tokenizer (used during training)
# with open("model/tokenizer.pkl", "rb") as handle:
#     tokenizer = pickle.load(handle)
#
# # Define max sequence length (same as during training)
# MAX_SEQUENCE_LENGTH = 100


@app.route('/', methods=['GET'])
def index():
    return render_template("index.html", probability=None)


@app.route("/predict", methods=["POST"])
def predict():
    user_text = request.form["user_text"]
    prediction = make_prediction(user_text)

    # Tokenize and pad input
    # sequence = tokenizer.texts_to_sequences([user_text])
    # padded_sequence = pad_sequences(sequence, maxlen=MAX_SEQUENCE_LENGTH)

    # Make prediction
    # prediction = model.predict(padded_sequence)
    # Assuming a binary classification model
    # probability = float(prediction[0][0])

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)
