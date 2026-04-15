import json
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


################# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])
test_data_path = os.path.join(config['test_data_path'])


def save_prob_distribution(y_probs, y_test, threshold):
    plt.figure(figsize=(8, 5))
    plt.hist(y_probs[y_test == 0], bins=50, alpha=0.5, label='No Exit')
    plt.hist(y_probs[y_test == 1], bins=50, alpha=0.5, label='Exit')
    plt.axvline(threshold, color='red', linestyle='--', label='Best threshold')
    plt.legend()
    plt.title(f'Predicted probability distribution - {config['model']}')
    plt.xlabel('Predicted probability')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(os.path.join(model_path, f"probability_distributions_{config['model']}.png"))
    plt.close()

################# Function for model scoring
def score_model():
    test_data = pd.read_csv(os.path.join(test_data_path, 'testdata.csv'))
    X_test = test_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y_test = test_data['exited']

    print(test_data.describe())
    print(y_test.value_counts(normalize=True))

    with open(os.path.join(model_path, f'trainedmodel_{config['model']}.pkl'), 'rb') as f:
        model = pickle.load(f)

    y_probs = model.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.1, 0.9, 50)
    f1s = [f1_score(y_test, y_probs > t) for t in thresholds]

    best_threshold = thresholds[np.argmax(f1s)]
    print(f"best_threshold: {best_threshold}")

    predictions = (y_probs > best_threshold).astype(int)
    print(f"y_probs: \t{np.round(y_probs, 2)}")
    print(f"predictions: \t{predictions}")
    print(f"y_test: \t{y_test.values}")

    save_prob_distribution(y_probs, y_test, best_threshold)

    f1 = f1_score(y_test, predictions)
    print(f"F1: {f1}")

    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, f'latestscore_{config['model']}.txt'), 'w') as f:
        f.write(str(f1))

    return f1


if __name__ == '__main__':
    score_model()
