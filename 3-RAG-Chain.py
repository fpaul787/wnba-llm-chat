# Databricks notebook source
# MAGIC %md
# MAGIC ### Langchain model to perform RAG.

# COMMAND ----------

# MAGIC %pip install --quiet -U databricks-sdk==0.49.0 databricks-agents mlflow[databricks] "databricks-langchain>=0.4.0" langchain==0.3.25 langchain_core==0.3.59 databricks-vectorsearch==0.55 pydantic==2.10.1
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

VECTOR_SEARCH_ENDPOINT_NAME = "wnba_vs_endpoint"
catalog = "frantzpaul_tech"
schema = "wnba_rag"

#The table we'd like to index
source_table_fullname = f"{catalog}.{schema}.wnba_summary_embeddings"

# vector search index
vs_index_fullname = f"{catalog}.{schema}.wnba_summary_embeddings_self_managed_vs_index"

# COMMAND ----------

# MAGIC %md
# MAGIC ### Building our Chain

# COMMAND ----------

import mlflow
import yaml
rag_chain_config = {
    "databricks_resources": {
        "llm_endpoint_name": "databricks-meta-llama-3-3-70b-instruct",
        "vector_search_endpoint_name": VECTOR_SEARCH_ENDPOINT_NAME,
    },
    "input_example": {
        "messages": [
            {"role": "user", "content": "Did the Sparks beat the Aces?"},
            {"role": "assistant", "content": "The Los Angeles Sparks beat the Aces"},
            {"role": "user", "content": "What was the score?"}
        ]
    },
    "llm_config": {
        "llm_parameters": {"max_tokens": 1500, "temperature": 0.01},
        "llm_prompt_template": "You are a trusted assistant that helps answer questions based only on the provided information. If you do not know the answer to a question, you truthfully say you do not know.  Here is some context which might or might not help you answer: {context}.  Answer directly, do not repeat the question, do not start with something like: the answer to the question, do not add AI in front of your answer, do not say: here is the answer, do not mention the context or the question. Based on this context, answer this question: {question}",
        "llm_prompt_template_variables": ["context", "question"],
    },
    "retriever_config": {
        "embedding_model": "databricks-gte-large-en",
        "chunk_template": "Passage: {chunk_text}\n",
        "data_pipeline_tag": "poc",
        "parameters": {"k": 3, "query_type": "ann"},
        "schema": {"chunk_text": "content", "document_game_id": "game_id", "primary_key": "id"},
        "vector_search_index": f"{vs_index_fullname}",
    }
}

try:
    with open('rag_chain_config.yaml', 'w') as f:
        yaml.dump(rag_chain_config, f)
except:
    print('pass to work on build job')
model_config = mlflow.models.ModelConfig(development_config='rag_chain_config.yaml')

# COMMAND ----------

# MAGIC %%writefile chain.py
# MAGIC from databricks_langchain.embeddings import DatabricksEmbeddings
# MAGIC from operator import itemgetter
# MAGIC import mlflow
# MAGIC import os
# MAGIC
# MAGIC from databricks.vector_search.client import VectorSearchClient
# MAGIC
# MAGIC from databricks_langchain.chat_models import ChatDatabricks
# MAGIC from databricks_langchain.vectorstores import DatabricksVectorSearch
# MAGIC
# MAGIC from langchain_core.runnables import RunnableLambda
# MAGIC from langchain_core.output_parsers import StrOutputParser
# MAGIC from langchain_core.prompts import (
# MAGIC     PromptTemplate,
# MAGIC     ChatPromptTemplate,
# MAGIC     MessagesPlaceholder,
# MAGIC )
# MAGIC from langchain_core.runnables import RunnablePassthrough, RunnableBranch
# MAGIC from langchain_core.messages import HumanMessage, AIMessage
# MAGIC
# MAGIC ## Enable MLflow Tracing
# MAGIC mlflow.langchain.autolog()
# MAGIC
# MAGIC # Return the string contents of the most recent message from the user
# MAGIC def extract_user_query_string(chat_messages_array):
# MAGIC     return chat_messages_array[-1]["content"]
# MAGIC
# MAGIC # Return the chat history, which is is everything before the last question
# MAGIC def extract_chat_history(chat_messages_array):
# MAGIC     return chat_messages_array[:-1]
# MAGIC
# MAGIC # Load the chain's configuration
# MAGIC model_config = mlflow.models.ModelConfig(development_config="rag_chain_config.yaml")
# MAGIC
# MAGIC databricks_resources = model_config.get("databricks_resources")
# MAGIC retriever_config = model_config.get("retriever_config")
# MAGIC llm_config = model_config.get("llm_config")
# MAGIC
# MAGIC vector_search_schema = retriever_config.get("schema")
# MAGIC
# MAGIC embedding_model = DatabricksEmbeddings(endpoint=retriever_config.get("embedding_model"))
# MAGIC
# MAGIC # Turn the Vector Search index into a LangChain retriever
# MAGIC vector_search_as_retriever = DatabricksVectorSearch(
# MAGIC     endpoint=databricks_resources.get("vector_search_endpoint_name"),
# MAGIC     index_name=retriever_config.get("vector_search_index"),
# MAGIC     text_column=vector_search_schema.get("chunk_text"),
# MAGIC     embedding=embedding_model, 
# MAGIC     columns=[
# MAGIC         vector_search_schema.get("primary_key"),
# MAGIC         vector_search_schema.get("chunk_text"),
# MAGIC         vector_search_schema.get("document_game_id"),
# MAGIC     ],
# MAGIC ).as_retriever(search_kwargs=retriever_config.get("parameters"))
# MAGIC
# MAGIC # Enable the RAG Studio Review App to properly display retrieved chunks and evaluation suite to measure the retriever
# MAGIC mlflow.models.set_retriever_schema(
# MAGIC     primary_key=vector_search_schema.get("primary_key"),
# MAGIC     text_column=vector_search_schema.get("chunk_text"),
# MAGIC     doc_uri=vector_search_schema.get("document_game_id") 
# MAGIC )
# MAGIC
# MAGIC
# MAGIC # Method to format the docs returned by the retriever into the prompt
# MAGIC def format_context(docs):
# MAGIC     chunk_template = retriever_config.get("chunk_template")
# MAGIC     chunk_contents = [
# MAGIC         chunk_template.format(
# MAGIC             chunk_text=d.page_content,
# MAGIC             document_uri=d.metadata[vector_search_schema.get("document_game_id")],
# MAGIC         )
# MAGIC         for d in docs
# MAGIC     ]
# MAGIC     return "".join(chunk_contents)
# MAGIC
# MAGIC
# MAGIC # Prompt Template for generation
# MAGIC prompt = ChatPromptTemplate.from_messages(
# MAGIC     [
# MAGIC         ("system", llm_config.get("llm_prompt_template")),
# MAGIC         # Note: This chain does not compress the history, so very long converastions can overflow the context window.
# MAGIC         MessagesPlaceholder(variable_name="formatted_chat_history"),
# MAGIC         # User's most current question
# MAGIC         ("user", "{question}"),
# MAGIC     ]
# MAGIC )
# MAGIC
# MAGIC
# MAGIC # Format the converastion history to fit into the prompt template above.
# MAGIC def format_chat_history_for_prompt(chat_messages_array):
# MAGIC     history = extract_chat_history(chat_messages_array)
# MAGIC     formatted_chat_history = []
# MAGIC     if len(history) > 0:
# MAGIC         for chat_message in history:
# MAGIC             if chat_message["role"] == "user":
# MAGIC                 formatted_chat_history.append(HumanMessage(content=chat_message["content"]))
# MAGIC             elif chat_message["role"] == "assistant":
# MAGIC                 formatted_chat_history.append(AIMessage(content=chat_message["content"]))
# MAGIC     return formatted_chat_history
# MAGIC
# MAGIC
# MAGIC # Prompt Template for query rewriting to allow converastion history to work - this will translate a query such as "how does it work?" after a question such as "what is spark?" to "how does spark work?".
# MAGIC query_rewrite_template = """Based on the chat history below, we want you to generate a query for an external data source to retrieve relevant documents so that we can better answer the question. The query should be in natural language. The external data source uses similarity search to search for relevant documents in a vector space. So the query should be similar to the relevant documents semantically. Answer with only the query. Do not add explanation.
# MAGIC
# MAGIC Chat history: {chat_history}
# MAGIC
# MAGIC Question: {question}"""
# MAGIC
# MAGIC query_rewrite_prompt = PromptTemplate(
# MAGIC     template=query_rewrite_template,
# MAGIC     input_variables=["chat_history", "question"],
# MAGIC )
# MAGIC
# MAGIC
# MAGIC # FM for generation
# MAGIC model = ChatDatabricks(
# MAGIC     endpoint=databricks_resources.get("llm_endpoint_name"),
# MAGIC     extra_params=llm_config.get("llm_parameters"),
# MAGIC )
# MAGIC
# MAGIC # RAG Chain
# MAGIC chain = (
# MAGIC     {
# MAGIC         "question": itemgetter("messages") | RunnableLambda(extract_user_query_string),
# MAGIC         "chat_history": itemgetter("messages") | RunnableLambda(extract_chat_history),
# MAGIC         "formatted_chat_history": itemgetter("messages")
# MAGIC         | RunnableLambda(format_chat_history_for_prompt),
# MAGIC     }
# MAGIC     | RunnablePassthrough()
# MAGIC     | {
# MAGIC         "context": RunnableBranch(  # Only re-write the question if there is a chat history
# MAGIC             (
# MAGIC                 lambda x: len(x["chat_history"]) > 0,
# MAGIC                 query_rewrite_prompt | model | StrOutputParser(),
# MAGIC             ),
# MAGIC             itemgetter("question"),
# MAGIC         )
# MAGIC         | vector_search_as_retriever
# MAGIC         | RunnableLambda(format_context),
# MAGIC         "formatted_chat_history": itemgetter("formatted_chat_history"),
# MAGIC         "question": itemgetter("question"),
# MAGIC     }
# MAGIC     | prompt
# MAGIC     | model
# MAGIC     | StrOutputParser()
# MAGIC )
# MAGIC
# MAGIC ## Tell MLflow logging where to find your chain.
# MAGIC mlflow.models.set_model(model=chain)

# COMMAND ----------

from mlflow.models.resources import DatabricksVectorSearchIndex, DatabricksServingEndpoint
import mlflow
import os
# Log the model to MLflow
with mlflow.start_run(run_name=f"dbdemos_rag_advanced"):
    logged_chain_info = mlflow.langchain.log_model(
        lc_model=os.path.join(os.getcwd(), 'chain.py'),  # Chain code file e.g., /path/to/the/chain.py 
        model_config='rag_chain_config.yaml',  # Chain configuration 
        artifact_path="chain",  # Required by MLflow
        input_example=model_config.get("input_example"),  # Save the chain's input schema.  MLflow will execute the chain before logging & capture it's output schema.
        example_no_conversion=True,  # Required by MLflow to use the input_example as the chain's schema
        resources=[
            DatabricksVectorSearchIndex(index_name=model_config.get("retriever_config").get("vector_search_index")),
            DatabricksServingEndpoint(endpoint_name=model_config.get("retriever_config").get("embedding_model")),
            DatabricksServingEndpoint(endpoint_name=model_config.get("databricks_resources").get("llm_endpoint_name"))
        ],
        extra_pip_requirements=[
            "databricks-connect"
        ]
    )

# Test the chain locally
chain = mlflow.langchain.load_model(logged_chain_info.model_uri)
chain.invoke(model_config.get("input_example"))

# COMMAND ----------

MODEL_NAME = "wnba_rag"
MODEL_TABLE_NAME = f"{catalog}.{schema}.{MODEL_NAME}"

# COMMAND ----------

def wait_for_model_serving_endpoint_to_be_ready(ep_name):
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.serving import EndpointStateReady, EndpointStateConfigUpdate
    import time

    w = WorkspaceClient()
    state = ""
    for i in range(200):
        state = w.serving_endpoints.get(ep_name).state
        if state.config_update == EndpointStateConfigUpdate.IN_PROGRESS:
            if i % 40 == 0:
                print(f"Waiting for endpoint to deploy {ep_name}. Current state: {state}")
            time.sleep(10)
        elif state.ready == EndpointStateReady.READY:
          print('endpoint ready.')
          return
        else:
          break
    raise Exception(f"Couldn't start the endpoint, timeout, please check your endpoint for more details: {state}")

# COMMAND ----------

from databricks import agents

unity_catalog_model_info = mlflow.register_model(model_uri=logged_chain_info.model_uri, name=MODEL_TABLE_NAME)

deployment_info = agents.deploy(model_name=MODEL_TABLE_NAME, model_version=unity_catalog_model_info.version, scale_to_zero=True)

# COMMAND ----------

instructions_to_reviewer = f"""### Instructions for Testing the our Databricks WNBA Chatbot assistant

Your inputs are invaluable for the development team. By providing detailed feedback and corrections, you help us fix issues and improve the overall quality of the application. We rely on your expertise to identify any gaps or areas needing enhancement.

1. **Variety of Questions**:
   - Please try a wide range of questions that you anticipate the end users of the application will ask. This helps us ensure the application can handle the expected queries effectively.

2. **Feedback on Answers**:
   - After asking each question, use the feedback widgets provided to review the answer given by the application.
   - If you think the answer is incorrect or could be improved, please use "Edit Answer" to correct it. Your corrections will enable our team to refine the application's accuracy.

3. **Review of Returned Documents**:
   - Carefully review each document that the system returns in response to your question.
   - Use the thumbs up/down feature to indicate whether the document was relevant to the question asked. A thumbs up signifies relevance, while a thumbs down indicates the document was not useful.

Thank you for your time and effort in testing our assistant. Your contributions are essential to delivering a high-quality product to our end users."""

agents.set_review_instructions(MODEL_TABLE_NAME, instructions_to_reviewer)
wait_for_model_serving_endpoint_to_be_ready(deployment_info.endpoint_name)

# COMMAND ----------

user_list = ["frantz@frantzpaul.tech"]
# Set the permissions.
agents.set_permissions(model_name=MODEL_TABLE_NAME, users=user_list, permission_level=agents.PermissionLevel.CAN_QUERY)

print(f"Share this URL with your stakeholders: {deployment_info.review_app_url}")

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT * FROM frantzpaul_tech.wnba_rag.wnba_game_summaries
# MAGIC WHERE game_id = '401736152';
