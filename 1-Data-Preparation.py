# Databricks notebook source
# Read in data
catalog = "frantzpaul_tech"
schema = "wnba_rag"
wnba_df = spark.read.table(f"{catalog}.{schema}.team_box_2025")
display(wnba_df)

# COMMAND ----------

wnba_clean_df = wnba_df.drop("game_id", "season_type", "team_id", "team_uid", "team_slug", "team_abbreviation", "team_short_display_name",
"team_color", "team_alternate_color", "team_logo", "opponent_team_id",
"opponent_team_uid", "opponent_team_slug", "opponent_team_abbreviation",
"opponent_team_short_display_name", "opponent_team_color", "opponent_team_alternate_color", "opponent_team_logo")
display(wnba_clean_df)

# COMMAND ----------

wnba_clean_df.columns

# COMMAND ----------

test_df = wnba_clean_df.limit(2)


# COMMAND ----------

from pyspark.sql.functions import udf, struct
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def create_game_summary(row):
    return (f"On {row['game_date']}, during the {row['season']}, the {row['team_display_name']} played a {row['team_home_away']} game against the {row['opponent_team_display_name']}. "
    f"The final score was {row['team_score']}-{row['opponent_team_score']}, with the {row['team_name']} {'winning' if row['team_winner'] == True else 'losing'} the game."        
    )

df = test_df.withColumn("game_summary", create_game_summary(struct(*test_df.columns)))
display(df)

# COMMAND ----------

test_df = wnba_clean_df.limit(1)
display(test_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert Each Row Into a Natural Language Chunk

# COMMAND ----------


