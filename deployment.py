#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   deployment.py
@Time    :   2023/04/26 21:20:02
'''

import os
import json
import shutil
import logging

# define logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Load config.json and correct path variable
with open('config.json', 'r') as f:
    config = json.load(f)


model_path = os.path.join(config['output_model_path'])
dataset_csv_path = os.path.join(config['output_folder_path'])
prod_deployment_path = os.path.join(config['prod_deployment_path'])


# function for deployment
def store_model_into_pickle():
    # copy the latest pickle file, the latestscore.txt value, and the ingestfiles.txt file into the deployment directory
    for f in [
            "ingestedfiles.txt", "trainedmodel.pkl", "encoder.pkl", "latestscore.txt", ]:
        if f in ["ingestedfiles.txt"]:
            source_filepath = os.path.join(dataset_csv_path, f)
        else:
            source_filepath = os.path.join(model_path, f)
        new_filepath = os.path.join(prod_deployment_path, f)
        logger.info('Copying %s to %s' % (source_filepath, new_filepath))
        shutil.copy2(source_filepath, new_filepath)

    logger.info("Completely!")


if __name__ == '__main__':
    store_model_into_pickle()
