#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   diagnostics.py
@Time    :   2023/04/26 21:43:12
'''

import pandas as pd
import timeit
import os
import json
import joblib
import logging
from utils import preprocess_data
import subprocess
import sys

# define logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Load config.json and get environment variables
with open('config.json', 'r') as f:
    config = json.load(f)

dataset_csv_path = config['output_folder_path']
test_data_path = config['test_data_path']
model_path = config['prod_deployment_path']


# Function to get model predictions
def model_predictions(dataset_path=None):
    # read the deployed model and a test dataset, calculate predictions
    logger.info(">>>Model predictions...")
    logger.info("Loading model...")
    model = joblib.load(os.path.join(model_path, "trainedmodel.pkl"))
    encoder = joblib.load(os.path.join(model_path, "encoder.pkl"))

    logger.info("Loading test dataset...")
    if dataset_path is None:
        dataset_path = "testdata.csv"

    df = pd.read_csv(os.path.join(test_data_path, dataset_path))

    logger.info("Preprocessing dataset...")
    df_x, df_y, _ = preprocess_data(df, encoder)

    logger.info("Start to predict...")
    y_pred = model.predict(df_x)

    return y_pred, df_y


# Function to get summary statistics
def dataframe_summary():
    # calculate summary statistics here
    logger.info(">>>Summary statistics...")
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))
    numeric_columns = [
        "lastmonth_activity",
        "lastyear_activity",
        "number_of_employees"
    ]

    result = []
    for column in numeric_columns:
        logger.info("Column %s" % column)
        result.append([column, "mean", df[column].mean()])
        result.append([column, "median", df[column].median()])
        result.append([column, "standard deviation", df[column].std()])

    return result


# check missing data
def check_missing_data():
    # calculate summary statistics here
    logger.info(">>>Check missing data...")
    df = pd.read_csv(os.path.join(test_data_path, "testdata.csv"))

    result = []
    for column in df.columns:
        count_na = df[column].isna().sum()
        count_not_na = df[column].count()
        count_total = count_not_na + count_na
        percentage_na = int(count_na / count_total * 100)

        logger.info("Column {} has {} N/A values, {} not N/A values, total {} values => {}%"
                    .format(column, count_na, count_not_na, count_total, percentage_na))

        result.append([column, str(percentage_na) + "%"])

    return str(result)


# Function to get timings
def execution_time():
    # calculate timing of training.py and ingestion.py
    logger.info(">>>Execution time...")
    result = []
    for procedure in ["training.py", "ingestion.py"]:
        logger.info("Check on %s..." % procedure)
        starttime = timeit.default_timer()
        os.system('python3 %s' % procedure)
        timing = timeit.default_timer() - starttime
        logger.info("Timing on %s..." % timing)
        result.append([procedure, timing])

    return str(result)


# Function to check dependencies
def outdated_packages_list():
    # check outupdated
    logger.info(">>>Checking outupdated...")
    outdated_packages = subprocess.check_output(['pip', 'list', '--outdated'])\
        .decode(sys.stdout.encoding)
    logger.info(outdated_packages)
    return str(outdated_packages)


if __name__ == '__main__':
    model_predictions()
    dataframe_summary()
    check_missing_data()
    execution_time()
    outdated_packages_list()
