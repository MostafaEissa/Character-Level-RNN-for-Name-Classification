import fire
import os
import sys
import torch
from data.load_dataset import load_dataset, n_letters
from features.build_features import build_features
from models.train_model import train, RNN
from models.predict_model import predict

sys.path.append(os.path.join(os.path.dirname(__file__)))
_categories = ['Arabic', 'Chinese', 'Czech', 'Dutch', 'English', 'French', 'German', 
'Greek', 'Irish', 'Italian', 'Japanese', 'Korean', 'Polish', 'Portuguese',
'Russian', 'Scottish', 'Spanish', 'Vietnamese']

def train_model(data_path, n_hidden=128, num_epochs=8):
    categories, category_lines = load_dataset(data_path)
    print("Data set loaded from disk.")

    X, y = build_features(category_lines)
    print("Features built successfully.")

    print("About to start training...")
    print()
    model = RNN(n_letters, n_hidden, len(categories))
    train(model, X, y, categories, num_epochs)

    print()
    torch.save(model, "../models/model.pt")
    print("Model saved to disk")

def predict_model(model_path, name, categories=_categories):
    predict(model_path, name, categories)


def main():
    fire.Fire({
    'train-model': train_model,
    'predict-model': predict_model,
    })

if __name__ == "__main__":
    main()