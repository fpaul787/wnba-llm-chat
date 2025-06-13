# Databricks notebook source
# Read in data
catalog = "frantzpaul_tech"
schema = "wnba_rag"
df = spark.read.table(f"{catalog}.{schema}.team_box_2025")
display(df)

# COMMAND ----------


