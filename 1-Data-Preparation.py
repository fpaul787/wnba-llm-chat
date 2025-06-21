# Databricks notebook source
# MAGIC %md
# MAGIC # Preparing Game Data for LLM and Self Managed Vector Search Embeddings

# COMMAND ----------

# MAGIC %pip install --quiet llama-index==0.10.43 transformers==4.49.0 langchain-text-splitters==0.2.0 torch==2.6.0
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

catalog = "frantzpaul_tech"
schema = "wnba_rag"
team_box_combined_table_name = f"{catalog}.{schema}.team_box_combined"
player_box_combined_table_name = f"{catalog}.{schema}.player_box_combined"

# COMMAND ----------

combined_team_box_df = spark.read.table(team_box_combined_table_name)
combined_player_box_df = spark.read.table(player_box_combined_table_name)

# COMMAND ----------

display(combined_team_box_df)

# COMMAND ----------

wnba_team_box_clean_df = combined_team_box_df.drop("season_type", "team_id", "team_uid", "team_slug", "team_abbreviation", "team_short_display_name",
"team_color", "team_alternate_color", "team_logo", "opponent_team_id",
"opponent_team_uid", "opponent_team_slug", "opponent_team_abbreviation",
"opponent_team_short_display_name", "opponent_team_color", "opponent_team_alternate_color", "opponent_team_logo")
display(wnba_team_box_clean_df)

# COMMAND ----------

display(combined_player_box_df)

# COMMAND ----------

wnba_player_box_clean_df = combined_player_box_df.drop("season_type", "athlete_id", "team_id", "team_short_display_name", "athlete_short_name", "athlete_position_abbreviation", "team_display_name", "team_uid", "team_slug", "team_logo", "team_abbreviation", "team_color", "team_alternate_color",  "opponent_team_id", "opponent_team_abbreviation", "opponent_team_logo", "opponent_team_color", "opponent_team_alternate_color", "athlete_headshot_href")
display(wnba_player_box_clean_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Convert Each Row Into a Natural Language Chunk

# COMMAND ----------

from pyspark.sql.functions import udf, struct
from pyspark.sql.types import StringType

@udf(returnType=StringType())
def create_game_summary(row):
    largest_lead_part = ""
    if row.largest_lead is not None:
        largest_lead_part = f"Their largest lead was {row.largest_lead} points."
    else:
        largest_lead_part = ""

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
    f"{largest_lead_part}"
    )

@udf(returnType=StringType())
def create_player_game_summary(row):
    """Create player summary"""

    if row is None:
        return None
    
    win_loss_part = ""
    if row["team_winner"] is None:
        win_loss_part = ""
    else:
        win_loss_part = "win" if row["team_winner"] == True else "loss"

    ejected_part = ""
    if row.ejected is not None and row.ejected == True:
        ejected_part = "She was ejected."

    did_not_play_part = ""
    if row.did_not_play is not None and row.did_not_play == True:
        if row.reason is not None:
            did_not_play_part = "Did not play due to " + row.reason
        else:
            did_not_play_part = "Did not play."

    jersey_part = ""
    if row.athlete_jersey is not None:
        jersey_part = f"#{row.athlete_jersey}."
    else:
        jersey_part = ""

    position_part = ""
    if row.athlete_position_name is not None and row.athlete_position_name != "Not Available":
        position_part = f"{row.athlete_position_name},"
    else:
        position_part = ""

    jersery_position_part = f", {jersey_part} {position_part}"
    shooting_part = ""
    if row.field_goals_made is not None and row.field_goals_attempted is not None:
        shooting_part = (
            f"shooting {row.field_goals_made}/{row.field_goals_attempted} from the field, "
            f"{row.three_point_field_goals_made}/{row.three_point_field_goals_attempted} from three, "
            f"and {row.free_throws_made}/{row.free_throws_attempted} from the line. "
        )
    else:
        shooting_part = " "

    plus_minus_part = ""
    if row.plus_minus is not None:
        plus_minus_part = f", finishing with a plus/minus was {row.plus_minus} "
    else:
        plus_minus_part = ""

    if row.did_not_play is not None and row.did_not_play == True:
        reason = ""
        if row.reason is not None:
            reason = f"Did not play due to {row.reason}"
        else:
            reason = "Did not play"
        
        return (
             f"On {row.game_date} ({row.season}), {row.athlete_display_name} "
             f"(#{jersey_part}, {position_part}) {reason}"
        )

    
    return (
        f"On {row.game_date} ({row.season}), {row.athlete_display_name} "
        f"{jersery_position_part} played for the {row.team_location} {row.team_name} "
        f"in a {row.home_away} game against the {row.opponent_team_location} {row.opponent_team_name} ({row.opponent_team_display_name}). "
        f"The final score was {row.team_score}-{row.opponent_team_score}, and the {win_loss_part} impacted {row.athlete_display_name}. "
        f"She played {row.minutes} minutes, recording {row.points} points, {shooting_part}"
        f"She added {row.rebounds} rebounds ({row.offensive_rebounds} offensive, {row.defensive_rebounds} defensive), "
        f"{row.assists} assists, {row.steals} steals, and {row.blocks} blocks, while committing {row.fouls} fouls and {row.turnovers} turnovers"
        f"{plus_minus_part}{ejected_part}{did_not_play_part}."
    )



# COMMAND ----------

wnba_team_summaries_df = wnba_team_box_clean_df.withColumn("game_summary", create_game_summary(struct(*wnba_team_box_clean_df.columns)))
display(wnba_team_summaries_df)

# COMMAND ----------

wnba_player_summaries_df = wnba_player_box_clean_df.withColumn("player_summary", create_player_game_summary(struct(*wnba_player_box_clean_df.columns)))
display(wnba_player_summaries_df)


# COMMAND ----------

# MAGIC %md
# MAGIC ### Append the rows together based on `game_id` and concat summary

# COMMAND ----------

from pyspark.sql import functions as F

merged_team_box_df = wnba_team_summaries_df.groupBy("game_id").agg(
  F.concat_ws("\n\n", F.collect_list("game_summary")).alias("summary")
)

merged_player_box_df = wnba_player_summaries_df.groupBy("game_id").agg(
  F.concat_ws("\n\n", F.collect_list("player_summary")).alias("summary")
)

games_box_df = merged_team_box_df.withColumnRenamed("summary", "game_summary")
players_box_df = merged_player_box_df.withColumnRenamed("summary", "player_summary")




# COMMAND ----------

combined_box_df = (
    games_box_df.join(players_box_df, on="game_id", how="inner")
    .withColumn(
        "summary",
        F.concat_ws("\n\n",
                    F.coalesce(F.col("game_summary"), F.lit("")),
                    F.coalesce(F.col("player_summary"), F.lit(""))
                   )
    )
)

display(combined_box_df)

# COMMAND ----------

df = combined_box_df.select("game_id", "summary")

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

# MAGIC %md
# MAGIC Update: game summaries used to short, now they are long because of game summaries and player summaries. Now we should chunk.

# COMMAND ----------

from pyspark.sql.functions import pandas_udf
from llama_index.core.node_parser import SentenceSplitter
import pandas as pd
from llama_index.core import Document, set_global_tokenizer
from transformers import AutoTokenizer
from typing import Iterator

@pandas_udf("array<string>")
def read_as_chunk(batch_iter: Iterator[pd.Series]) -> Iterator[pd.Series]:
    tokenizer = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer", cache_dir="/tmp/hf_cache")
    set_global_tokenizer(tokenizer)

    splitter = SentenceSplitter(chunk_size=1000, chunk_overlap=10)
    for batch in batch_iter:
        # Convert each text to Document, split, return chunk texts
        yield batch.apply(
            lambda x: [node.text for node in splitter.get_nodes_from_documents([Document(text=x)])] if pd.notnull(x) else []
        )

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute Embeddings

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
    .withColumn("content", explode(read_as_chunk("summary"))) \
    .withColumn("embedding", get_embedding("content")) \
    .drop("summary")

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

chunks_df.write.mode("overwrite").saveAsTable(f"{catalog}.{schema}.wnba_summary_embeddings")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM frantzpaul_tech.wnba_rag.wnba_summary_embeddings limit 1
