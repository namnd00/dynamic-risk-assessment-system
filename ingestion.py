#!/usr/bin/env python
# -*-coding:utf-8 -*-
'''
@File    :   ingestion.py
@Time    :   2023/04/25 21:49:34
'''

import pandas as pd
import os
import json
import glob
import logging

# define logging
logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()

# Load config.json and get input and output paths
logger.info("Loading config...")
with open('config.json', 'r') as f:
    config = json.load(f)

input_folder_path = config['input_folder_path']
output_folder_path = config['output_folder_path']

if not os.path.exists(input_folder_path):
    logger.info("Not exists: %s" % input_folder_path)
    os.makedirs(input_folder_path)
    logger.info("Created: %s" % input_folder_path)

if not os.path.exists(output_folder_path):
    logger.info("Not exists: %s" % output_folder_path)
    os.makedirs(output_folder_path)
    logger.info("Created: %s" % output_folder_path)

# Function for data ingestion


def merge_multiple_dataframe():
    """Merge multiple dataframes and process data.
    """
    # Get all CSV files in the input folder
    logger.info("Getting all CSV files...")
    csv_files = glob.glob("%s/*.csv" % input_folder_path)
    logger.info("Found %d CSV files" % len(csv_files))

    # Concatenate all dataframes from the CSV files into a single dataframe
    df = pd.concat(map(pd.read_csv, csv_files), ignore_index=True)

    # Drop duplicates from the final dataframe
    logger.info("Dropping duplicates...")
    df.drop_duplicates(inplace=True)

    # Write the final dataframe to a CSV file in the output folder
    logger.info("Writing final dataframe...")
    df.to_csv("%s/finaldata.csv" % output_folder_path, index=False)

    # Write the list of ingested CSV files to a text file in the output folder
    logger.info("Writing list of ingested CSV files...")
    ingested_csv_path = os.path.join(output_folder_path, "ingestedfiles.txt")
    with open(ingested_csv_path, "w") as report_file:
        for line in csv_files:
            report_file.write(line + "\n")
    logger.info("List of ingested CSV files written to %s", ingested_csv_path)


if __name__ == '__main__':
    merge_multiple_dataframe()
