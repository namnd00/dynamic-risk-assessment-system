#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   reporting.py
@Time    :   2023/04/26 21:46:26
'''

from sklearn import metrics
import matplotlib.pyplot as plt
import json
import os
import logging
from diagnostics import model_predictions

# define logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


# Load config.json and get path variables
with open('config.json', 'r') as f:
    config = json.load(f)


dataset_csv_path = config['test_data_path']
model_path = config['output_model_path']

# Function for reporting
def score_model():
    # calculate a confusion matrix using the test data and the deployed model
    logger.info(">>>Score model")
    y_pred, df_y = model_predictions()
    df_cm = metrics.confusion_matrix(df_y, y_pred)
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(df_cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(df_cm.shape[0]):
        for j in range(df_cm.shape[1]):
            ax.text(x=j, y=i, s=df_cm[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    saved_path = os.path.join(model_path, "confusionmatrix.png")
    plt.savefig(saved_path)
    logger.info("Save confusion matrix to: %s" % saved_path)


if __name__ == "__main__":
    score_model()
