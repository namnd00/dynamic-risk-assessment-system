#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   scoring.py
@Time    :   2023/04/26 22:32:47
'''

import argparse
import pandas as pd
import os
from sklearn import metrics
from utils import preprocess_data
import json
import joblib
import logging

# define logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)

output_folder_path = config['output_folder_path']
test_data_path = config['test_data_path']
model_path = config['output_model_path']
prod_model_path = config['prod_deployment_path']


def create_dir(dirname):
    """Create a directory"""
    if not os.path.exists(dirname):
        logger.info("Not exists: %s" % dirname)
        os.makedirs(dirname)
        logger.info("Created: %s" % dirname)


create_dir(output_folder_path)
create_dir(test_data_path)
create_dir(model_path)


# Function for model scoring
def score_model(production=False):
    # this function should take a trained model, load test data,
    # and calculate an F1 score for the model relative to the test data
    # it should write the result to the latestscore.txt file

    logger.info("Loading model...")
    trained_model_name = "trainedmodel.pkl"
    encoder_name = "encoder.pkl"

    if not production:
        trained_model_path = os.path.join(model_path, trained_model_name)
        encoder_path = os.path.join(model_path, encoder_name)
    else:
        trained_model_path = os.path.join(prod_model_path, trained_model_name)
        encoder_path = os.path.join(prod_model_path, encoder_name)

    logger.info("Loading model from %s" % trained_model_path)
    logger.info("Loading encoder from %s" % encoder_path)
    model = joblib.load(trained_model_path)
    encoder = joblib.load(encoder_path)

    if production:
        df = pd.read_csv(os.path.join(output_folder_path, "finaldata.csv"))
    else:
        df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    logger.info("Preprocessing data...")
    df_x, df_y, _ = preprocess_data(df, encoder)

    logger.info("Predict test data...")
    y_pred = model.predict(df_x)

    logger.info("Compute f1 score...")
    f1 = metrics.f1_score(df_y, y_pred)
    if not production:
        score_file_path = os.path.join(model_path, "latestscore.txt")
    else:
        score_file_path = os.path.join(prod_model_path, "latestscore.txt")
    logger.info("F1 score: %.2f", f1)
    logger.info("Write prediction results to %s", score_file_path)
    with open(score_file_path, "w") as score_file:
        score_file.write(str(f1) + "\n")

    return f1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scoring model")
    parser.add_argument("--production", type=bool, default=False, help="Whether to produce production")
    args = parser.parse_args()
    _production = args.production
    _ = score_model(_production)
