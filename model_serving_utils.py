import os
import mlflow
from databricks_langchain import VectorSearchRetrieverTool, ChatDatabricks
from mlflow.deployments import get_deploy_client

mlflow.langchain.autolog()

@mlflow.trace
def query_endpoint(endpoint_name, messages, user="unknown", use_tools=True) -> dict:
    """Simple query using ChatDatabricks with tool execution capability"""

    mlflow.update_current_trace(tags={"user": user if user else "unknown"})
    # Initialize the retriever tool
    vs_tool = VectorSearchRetrieverTool(
        index_name="main.dbdemos_rag_chatbot.databricks_documentation_vs_index",
        tool_name="databricks_docs_retriever",
        tool_description="Retrieves information about Databricks products from official Databricks documentation."
    )
    # Initialize ChatDatabricks LLM
    llm = ChatDatabricks(endpoint=endpoint_name)
    
    # Convert OpenAI format messages to LangChain format
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    
    langchain_messages = []
    for msg in messages:
        if msg["role"] == "system":
            langchain_messages.append(SystemMessage(content=msg["content"]))
        elif msg["role"] == "user":
            langchain_messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            langchain_messages.append(AIMessage(content=msg["content"]))
    
    if use_tools:
        # Bind tools to the LLM
        llm_with_tools = llm.bind_tools([vs_tool])
        
        # First call - model may decide to call tools
        response = llm_with_tools.invoke(langchain_messages)
        
        # Check if model wants to call tools
        if hasattr(response, 'tool_calls') and response.tool_calls:
            print(f"🔧 Model is calling {len(response.tool_calls)} tool(s)")
            
            # Execute each tool call
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call.get('name', 'unknown')
                tool_args = tool_call.get('args', {})
                
                print(f"📞 Calling tool: {tool_name} with args: {tool_args}")
                
                if tool_name == "databricks_docs_retriever":
                    # Execute the vector search tool
                    try:
                        tool_result = vs_tool.invoke(tool_args)
                        tool_results.append(f"Tool Result: {tool_result}")
                        print(f"✅ Tool executed successfully")
                    except Exception as e:
                        tool_results.append(f"Tool Error: {str(e)}")
                        print(f"❌ Tool execution failed: {e}")
            
            # Create follow-up messages with tool results
            if tool_results:
                # Add tool results as a new system message
                enhanced_messages = langchain_messages + [
                    SystemMessage(content="Retrieved context:\n" + "\n".join(tool_results))
                ]
                # Final call with tool results
                final_response = llm.invoke(enhanced_messages)
                response_content = final_response.content
            else:
                response_content = response.content
        else:
            # No tool calls, use response as-is
            response_content = response.content
            print("💬 Model responded without calling tools")
    else:
        # No tools enabled - use full message history
        response = llm.invoke(langchain_messages)
        response_content = response.content
    
    return {
        "content": response_content,
        "trace_id": mlflow.get_active_trace_id(),
    }
