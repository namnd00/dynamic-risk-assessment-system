#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   fullprocess.py
@Time    :   2023/04/26 22:19:21
'''

import ingestion
import training
import scoring
import deployment
import diagnostics
import reporting
import json
import os

import logging

# define logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


with open("config.json", "r") as f:
    config = json.load(f)

input_folder_path = config["input_folder_path"]
prod_deployment_path = os.path.join(config['prod_deployment_path'])
model_path = os.path.join(config['output_model_path'])

# Check and read new data
# first, read ingestedfiles.txt\
ingested_file_path = os.path.join(prod_deployment_path, "ingestedfiles.txt")
logger.info("Reading data from %s..." % ingested_file_path)
ingested_files = []
with open(ingested_file_path, "r") as report_file:
    for line in report_file:
        ingested_files.append(line.rstrip())

logger.info("Checking new data...")
# second, determine whether the source data folder has files that aren't listed in ingestedfiles.txt
new_files = False
for filename in os.listdir(input_folder_path):
    if input_folder_path + "/" + filename not in ingested_files:
        logger.info("Found new file: %s" % filename)
        new_files = True

# Deciding whether to proceed, part 1
# if you found new data, you should proceed. otherwise, do end the process here
if not new_files:
    logger.info("No new ingested data, exiting")
    exit(0)


# Checking for model drift
# check whether the score from the deployed model is different from the
# score from the model that uses the newest ingested data
ingestion.merge_multiple_dataframe()
scoring.score_model(production=True)

prod_latest_score_file_path = os.path.join(prod_deployment_path, "latestscore.txt")
with open(prod_latest_score_file_path, "r") as report_file:
    old_f1 = float(report_file.read())

latest_score_file_path = os.path.join(model_path, "latestscore.txt")
with open(latest_score_file_path, "r") as report_file:
    new_f1 = float(report_file.read())

# Deciding whether to proceed, part 2
# if you found model drift, you should proceed. otherwise, do end the process here

if new_f1 >= old_f1:
    logger.info("Actual F1 (%s) is better/equal than old F1 (%s), \
                no drift detected -> exiting" % (new_f1, old_f1))
    exit(0)

logger.info("Actual F1 (%s) is WORSE than old F1 (%s), drift detected -> training model" % (new_f1, old_f1))

logger.info("Start to re-training...")
training.train_model()

# Re-deployment
# if you found evidence for model drift, re-run the deployment.py script
logger.info("Start to re-deploy...")
deployment.store_model_into_pickle()

# Diagnostics and reporting
# run diagnostics.py and reporting.py for the re-deployed model
diagnostics.model_predictions(None)
diagnostics.execution_time()
diagnostics.dataframe_summary()
diagnostics.check_missing_data()
diagnostics.outdated_packages_list()
reporting.score_model()

# check api calls
# os.system("python3 apicalls.py")
