# wnba-llm-chat

This project creates a RAG application and chatbot with message history for WNBA game data. This project was on done on Databricks. I modeled the project based on the tutorial that Databricks provides [here](https://notebooks.databricks.com/demos/llm-rag-chatbot/index.html). 

## Dataset

The dataset comes from the [sportsdataverse/wehoop-wnba-data repository](https://github.com/sportsdataverse/wehoop-wnba-data). This is a very neat repository that collects daily game and player data from WNBA games. For this project, I only used
team data from the 2025 season. Since the data was mostly numeric with some short text. I had to create a 'natural language chunk' and use that for the embeddings. Essentially, I took the most important stats from a game log and combined them in a neat game summary. Here is an example:


  _On 2025-06-11, during the 2025 WNBA season, the Los Angeles Sparks played an away game against the Las Vegas Aces. The final score was 97-89, with the Sparks winning the game. The team recorded 44 points in the paint, 8 fast break points, 24 assists, 5 steals, and 1 block. They had 34 defensive rebounds and 4 offensive rebounds, totaling 38 rebounds. The team shot 33/58 from the field  for a 56.9% FG, 22/27 from the free throw line for a  81.5% FT, and 9/20 from three point range for a 45.0% 3PT. They committed 15 turnovers, leading to 19 opponent points, 2 team turnovers, and had 26 personal fouls, 2 technical fouls (total 2) and 0 flagrant fouls. Their largest lead was 20 points._


### Embeddings

After curating the dataset, I created embeddings based on the game summaries. These embeddings were created using the `databricks-gte-large-en` embedding model provided by Databricks. Databricks refers to the embeddings that you created on your own as `self managed embeddings`.  I saved the embeddings, along with the content to a delta table in Databricks. These embeddings were used in the vector search index step.

## Vector Search Index
Databricks provides the infrastructure for creating the vector search index and providing an endpoint to access the index. Its relatively simple through the Python SDK and the Databricks UI. More information can be found [here](https://docs.databricks.com/aws/en/generative-ai/create-query-vector-search).

## RAG Chain
After preparing the dataset and vector search index, now is time for the fun part of performing RAG and building our chatbot. 

## Next Steps

For further development, I would like to incorporate data from previous seasons and include player data with that as well. This small project made me realize that for LLM applications, your dataset and how you're training matters a lot and sometimes takes the most time.