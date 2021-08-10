#!/usr/bin/python

import sys, os, io
import shutil
import glob
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, Model, PipelineModel
import requests, re
import dsx_core_utils, jaydebeapi
from sqlalchemy import *
from sqlalchemy.types import String, Boolean


# setup dsxr environmental vars from command line input
from dsx_ml.ml import dsxr_setup_environment
dsxr_setup_environment()

# define variables
args = {u'remoteHostImage': u'', u'target': u'/datasets/churn_predict_out.csv', u'remote_input_data': [{u'datasource': u'datasource', u'dataset': u'CUST_CHURN_1M'}], u'output_datasource_type': u'', u'livyVersion': u'livyspark2', u'execution_type': u'DSX', u'source': u'', u'output_type': u'Localfile', u'remoteHost': u'', u'sysparm': u''}
model_path = os.getenv("DSX_PROJECT_DIR") + os.path.join("/models", os.getenv("DSX_MODEL_NAME","ChurnPredict"), os.getenv("DSX_MODEL_VERSION","1"),"model")
input_data = args.get("remote_input_data")[0]
output_data = os.getenv("DEF_DSX_DATASOURCE_OUTPUT_FILE", (os.getenv("DSX_PROJECT_DIR") + args.get("target")))

# create spark context
spark = SparkSession.builder.getOrCreate()
sc = spark.sparkContext
datasource_name = 'datasource'

# Add asset from remote connection
dataframe = None
dataset_name = input_data.get('dataset')
dataSet = dsx_core_utils.get_remote_data_set_info(dataset_name)
dataSource = dsx_core_utils.get_data_source_info(dataSet.get('datasource'))
spark = spark.builder.getOrCreate()

# Load JDBC data to Spark dataframe
dbTableOrQuery = (dataSet['schema'] + '.' if(len(dataSet['schema'].strip()) != 0) else '') + dataSet['table']
if (dataSet['query']):
    dbTableOrQuery = "(" + dataSet['query'] + ") TBL"

# Optionally override table or query based on environmental vars
dbTable = os.getenv("DEF_DSX_DATASOURCE_SCHEMA_TABLE", None)
dbQuery = os.getenv("DEF_DSX_DATASOURCE_QUERY", None)
if dbQuery:
    dbTableOrQuery = "(" + dbQuery + ")"
elif dbTable:
    dbTableOrQuery = dbTable

# Read dataframe
dataframe = spark.read.format("jdbc").option("url", dataSource['URL']).option("dbtable",dbTableOrQuery).option("user",dataSource['user']).option("password",dataSource['password']).load()
dataframe.show(5)

# read test dataframe (inputJson = "input.json")
testDF = dataframe

# load model
model_rf = PipelineModel.load(model_path)

# prediction
outputDF = model_rf.transform(testDF) 

# save scoring result to given target
scoring_df = outputDF.toPandas()

# save output to csv
scoring_df.to_csv(output_data, encoding='utf-8')