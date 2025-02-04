import toml
import csv
import itertools
from models_classes.PklModel import PklModel

with open('config.toml', 'r', encoding="utf8") as f:
    config = toml.load(f)

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


if __name__ == "__main__":
    fake_news_path = config["config"]["data_dir"] + \
        config["config"]["fake_news_csv"]
    with open(fake_news_path, "r", encoding="utf-8") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header (if applicable)

        for row in itertools.islice(reader, 100):
            text = row[0]  # Assuming the text is in the first column
            print(text)
            for model in models.values():
                model.predict(text)

            print("\n")
