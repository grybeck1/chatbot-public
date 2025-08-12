import os
import mlflow
from databricks_langchain import VectorSearchRetrieverTool, ChatDatabricks
    
# Try to initialize Vector Search tool using databricks_langchain
from databricks_langchain import VectorSearchRetrieverTool, ChatDatabricks
mlflow.langchain.autolog()

# Simple query function using databricks_langchain
def query_endpoint(endpoint_name, messages, max_tokens, use_tools=True) -> str:
    """Simple query using ChatDatabricks with optional tools, returns trace_id"""

    # Initialize the retriever tool
    vs_tool = VectorSearchRetrieverTool(
        index_name="main.dbdemos_rag_chatbot.databricks_documentation_vs_index",
        tool_name="databricks_docs_retriever",
        tool_description="Retrieves information about Databricks products from official Databricks documentation."
    )
    #print(vs_tool.invoke("Databricks Agent Framework?"))
    print("✅ Vector Search tool initialized successfully")

    # Initialize ChatDatabricks LLM
    llm = ChatDatabricks(endpoint=endpoint_name)
    
    # Bind tools if available and requested
    if use_tools:
        llm_with_tools = llm.bind_tools([vs_tool])
    else:
        llm_with_tools = llm
    
    # Convert messages to LangChain format (just take the last user message for simplicity)
    user_message = messages[-1]["content"] if messages else ""
    
    # Make the chat completion call
    response = llm_with_tools.invoke(user_message)
    
    # Return in the expected format
    return {"content": response.content,
            "trace_id": mlflow.get_active_trace_id(),
            "trace": response}
