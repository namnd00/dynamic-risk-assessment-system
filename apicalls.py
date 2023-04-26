#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   apicalls.py
@Time    :   2023/04/26 21:58:17
'''

import requests
import os
import json
import logging

# define logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Specify a URL that resolves to your workspace
URL = "http://127.0.0.1:8000"
logger.info("Listen to %s" % URL)

# Call each API endpoint and store the responses
headers = {'Content-type': 'application/json', 'Accept': 'text/plain'}

response1 = requests.post("%s/prediction" % URL, json={"dataset_path": "testdata.csv"},
                          headers=headers).text
logger.info("Respone 1: %s", response1)

response2 = requests.get("%s/scoring" % URL, headers=headers).text
logger.info("Respone 2: %s", response2)

response3 = requests.get("%s/summarystats" % URL, headers=headers).text
logger.info("Respone 3: %s", response3)

response4 = requests.get("%s/diagnostics" % URL, headers=headers).text
logger.info("Respone 4: %s", response4)

# combine all API responses
responses = response1 + "\n" + response2 + "\n" + response3 + "\n" + response4

# write the responses to your workspace
with open('config.json', 'r') as f:
    config = json.load(f)

model_path = os.path.join(config['output_model_path'])

saved_path = os.path.join(model_path, "apireturns.txt")
logger.info("Writing responses to %s", saved_path)
with open(saved_path, "w") as returns_file:
    returns_file.write(responses)

logger.info("Completed.")
