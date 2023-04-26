#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   training.py
@Time    :   2023/04/25 22:10:02
'''

import pandas as pd
import os
import json
import logging
import joblib
from utils import preprocess_data
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# define logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get path variables
logger.info("Loading config...")
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = config['output_folder_path']
model_path = config['output_model_path']

if not os.path.exists(dataset_csv_path):
    logger.info("Not exists: %s" % dataset_csv_path)
    os.makedirs(dataset_csv_path)
    logger.info("Created: %s" % dataset_csv_path)

if not os.path.exists(model_path):
    logger.info("Not exists: %s" % model_path)
    os.makedirs(model_path)
    logger.info("Created: %s" % model_path)

# Function for training the model


def train_model():
    """Train model
    """
    df = pd.read_csv(os.path.join(dataset_csv_path, "finaldata.csv"))
    df_x, df_y, encoder = preprocess_data(df, None)

    X_train, X_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.20)

    # use this logistic regression for training
    logger.info("Initializing model...")
    model = LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
                               intercept_scaling=1, l1_ratio=None, max_iter=100,
                               multi_class='ovr', n_jobs=None, penalty='l2',
                               random_state=0, solver='liblinear', tol=0.0001, verbose=0,
                               warm_start=False)

    # fit the logistic regression to your data
    logger.info("Starting to train the logistic regression model...")
    model.fit(X_train, y_train)

    logger.info("Train score: %.2f", model.score(X_train, y_train))
    logger.info("Test score: %.2f", model.score(X_test, y_test))

    # write the trained model to your workspace in a file called trainedmodel.pkl
    saved_model_path = os.path.join(model_path, "trainedmodel.pkl")
    saved_encoder_path = os.path.join(model_path, "encoder.pkl")

    logger.info("Writing trained model to %s", saved_model_path)
    logger.info("Writing encoder to %s", saved_encoder_path)
    joblib.dump(model, saved_model_path)
    joblib.dump(encoder, saved_encoder_path)

    logger.info("Training completely!")


if __name__ == "__main__":
    train_model()
