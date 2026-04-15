import json
import os
import pickle

import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV


################### Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])
model_path = os.path.join(config['output_model_path'])

param_grid_lr = {
    'model__C': [0.01, 0.1, 1, 10, 100]
}

param_grid_rf = {
    'model__n_estimators': [200, 400],
    'model__max_depth': [None, 5, 10],
    'model__min_samples_split': [2, 5, 10],
    'model__min_samples_leaf': [1, 2, 5],
    'model__max_features': ['sqrt', 'log2']
}


################# Function for training the model
def train_model():
    training_data = pd.read_csv(os.path.join(dataset_csv_path, 'finaldata.csv'))
    X = training_data[['lastmonth_activity', 'lastyear_activity', 'number_of_employees']]
    y = training_data['exited']

    print(training_data.describe())

    lr = LogisticRegression(
            solver='liblinear',
            class_weight='balanced',
            max_iter=500,
            random_state=0
        )

    rf = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=0
        )
    
    m = {'lr': { 
            'pipeline': Pipeline([
                ('scaler', StandardScaler()),
                ('model', lr)]), 
            'params': param_grid_lr }, 
        'rf': { 
            'pipeline': Pipeline([
                ('model', rf)]), 
            'params': param_grid_rf }
    }

    model = m[config['model']]

    search = GridSearchCV(
        model['pipeline'],
        model['params'],
        scoring='f1',
        cv=5
    )

    search.fit(X, y)
    
    os.makedirs(model_path, exist_ok=True)
    with open(os.path.join(model_path, f'trainedmodel_{config['model']}.pkl'), 'wb') as f:
        pickle.dump(search.best_estimator_, f)

    return model['pipeline']


if __name__ == '__main__':
    train_model()
