#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   app.py
@Time    :   2023/04/26 21:51:47
'''

from flask import Flask, request
from diagnostics import (
    model_predictions,
    dataframe_summary,
    check_missing_data,
    outdated_packages_list,
    execution_time
)
from scoring import score_model
import json
import os
import logging


# define logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Set up variables for use in our script
app = Flask(__name__)
app.secret_key = '1652d576-484a-49fd-913a-6879acfa6ba4'

with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = os.path.join(config['output_folder_path'])

prediction_model = None


# Prediction Endpoint
@app.route("/prediction", methods=['POST', 'OPTIONS'])
def predict():
    # call the prediction function you created in Step 3
    dataset_path = request.json.get('dataset_path')
    logger.info("Received: %s", dataset_path)
    y_pred, _ = model_predictions(dataset_path)
    return str(y_pred)

# Scoring Endpoint


@app.route("/scoring", methods=['GET', 'OPTIONS'])
def scoring():
    # check the score of the deployed model
    score = score_model()
    logger.info("Score: %s" % str(score))
    return str(score)

# Summary Statistics Endpoint


@app.route("/summarystats", methods=['GET', 'OPTIONS'])
def stats():
    # check means, medians, and modes for each column
    summary = dataframe_summary()
    logger.info("Summary:\n%s" % str(summary))
    return str(summary)

# Diagnostics Endpoint


@app.route("/diagnostics", methods=['GET', 'OPTIONS'])
def diagnostics():
    # check timing and percent NA values
    et = execution_time()
    md = check_missing_data()
    op = outdated_packages_list()
    return str("Execution_time:" + et + "\nMissing_data;" + md + "\nOutdated_packages:" + op)


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, debug=True, threaded=True)
