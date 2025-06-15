# Databricks notebook source
# Read in data
catalog = "frantzpaul_tech"
schema = "wnba_rag"
wnba_df = spark.read.table(f"{catalog}.{schema}.team_box_2025")
display(wnba_df)

# COMMAND ----------

wnba_clean_df = wnba_df.drop("season_type", "team_id", "team_uid", "team_slug", "team_abbreviation", "team_short_display_name",
"team_color", "team_alternate_color", "team_logo", "opponent_team_id",
"opponent_team_uid", "opponent_team_slug", "opponent_team_abbreviation",
"opponent_team_short_display_name", "opponent_team_color", "opponent_team_alternate_color", "opponent_team_logo")
display(wnba_clean_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert Each Row Into a Natural Language Chunk

# COMMAND ----------

from pyspark.sql.functions import udf, struct
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def create_game_summary(row):
    return (f"On {row['game_date']}, during the {row['season']} WNBA season, the {row['team_display_name']} played an {row['team_home_away']} game against the {row['opponent_team_display_name']}. "
    f"The final score was {row['team_score']}-{row['opponent_team_score']}, with the {row['team_name']} {'winning' if row['team_winner'] == True else 'losing'} the game. "
    f"The team recorded {row['points_in_paint']} points in the paint, {row['fast_break_points']} fast break points, "
    f"{row['assists']} assists, {row['steals']} steals, and {row['blocks']} {'blocks' if row['blocks'] > 1 else 'block'}. "
    f"They had {row['defensive_rebounds']} defensive rebounds and {row['offensive_rebounds']} offensive rebounds, totaling {row['total_rebounds']} rebounds. "
    f"The team shot {row['field_goals_made']}/{row['field_goals_attempted']} from the field  for a {row['field_goal_pct']}% FG, "
    f"{row['free_throws_made']}/{row['free_throws_attempted']} from the free throw line for a  {row['free_throw_pct']}% FT, and "
    f"{row['three_point_field_goals_made']}/{row['three_point_field_goals_attempted']} from three point range for a {row['three_point_field_goal_pct']}% 3PT. "
    f"They committed {row['turnovers']} turnovers, leading to {row['turnover_points']} opponent points, {row['team_turnovers']} team turnovers, and had {row['fouls']} personal fouls, "
    f"{row['technical_fouls']} technical fouls (total {row['total_technical_fouls']}) and {row['flagrant_fouls']} flagrant fouls. "
    f"Their largest lead was {row['largest_lead']} points."
    )

wnba_summaries_df = wnba_clean_df.withColumn("game_summary", create_game_summary(struct(*wnba_clean_df.columns)))
display(wnba_summaries_df)

# COMMAND ----------

df = wnba_summaries_df.select("game_id", "game_summary")

# save to table
table_name = "wnba_game_summaries"
df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.{table_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Chunking

# COMMAND ----------

# MAGIC %md
# MAGIC Since our game summaries are short, there is no need to chunk.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
import pandas as pd
# Create a UDF to chunk our summaries
@pandas_udf("array<string>")
def read_as_chunk(texts: pd.Series) -> pd.Series:
    return texts.apply(lambda x: [x] if pd.notnull(x) and x.strip() else [])

# COMMAND ----------

@pandas_udf("array<float>")
def get_embedding(contents: pd.Series) -> pd.Series:
    import mlflow.deployments
    deploy_client = mlflow.deployments.get_deploy_client("databricks")
    def get_embeddings(batch):
        #Note: this will fail if an exception is thrown during embedding creation (add try/except if needed) 
        response = deploy_client.predict(endpoint="databricks-gte-large-en", inputs={"input": batch})
        return [e['embedding'] for e in response.data]

    # Splitting the contents into batches of 150 items each, since the embedding model takes at most 150 inputs per request.
    max_batch_size = 150
    batches = [contents.iloc[i:i + max_batch_size] for i in range(0, len(contents), max_batch_size)]

    # Process each batch and collect the results
    all_embeddings = []
    for batch in batches:
        all_embeddings += get_embeddings(batch.tolist())

    return pd.Series(all_embeddings)

# COMMAND ----------

# MAGIC %md
# MAGIC This part really isn't necessary since the content will end up being exactly the same as the game_summary column, but I'm following through with it as part of procedure. 

# COMMAND ----------

from pyspark.sql.functions import explode
table_name = "wnba_game_summaries"
df = spark.read.table(f"{catalog}.{schema}.{table_name}")

chunks_df = df \
    .withColumn("content", explode(read_as_chunk("game_summary"))) \
    .withColumn("embedding", get_embedding("content")) \
    .drop("game_summary")

# COMMAND ----------

display(chunks_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save to Delta Table

# COMMAND ----------

# MAGIC %sql
# MAGIC --Note that we need to enable Change Data Feed on the table to create the index
# MAGIC CREATE TABLE IF NOT EXISTS frantzpaul_tech.wnba_rag.wnba_summary_embeddings (
# MAGIC   id BIGINT GENERATED BY DEFAULT AS IDENTITY,
# MAGIC   game_id STRING,
# MAGIC   content STRING,
# MAGIC   embedding ARRAY <FLOAT>
# MAGIC ) TBLPROPERTIES (delta.enableChangeDataFeed = true); 

# COMMAND ----------

chunks_df.write.mode("append").saveAsTable(f"{catalog}.{schema}.wnba_summary_embeddings")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM frantzpaul_tech.wnba_rag.wnba_summary_embeddings limit 1
