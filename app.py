import streamlit as st
import mlflow
from model_serving_utils import query_endpoint

# Configure Streamlit page
st.set_page_config(
    page_title="Databricks Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Initialize MLflow
mlflow.set_tracking_uri("databricks")
mlflow.set_experiment("/Users/gabe.rybeck@databricks.com/test")

# Enable MLflow autologging
mlflow.openai.autolog()
mlflow.langchain.autolog()
print("🔍 MLflow tracing initialized")

# Streamlit UI
st.title("🤖 Databricks Chatbot")

# Sidebar controls
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    selected_model = st.selectbox(
        "🤖 Choose Model:",
        ["databricks-claude-3-7-sonnet", "databricks-meta-llama-3-1-405b-instruct"],
        help="Select the model to use for responses"
    )
    
    # Vector search toggle
    use_tools = st.checkbox(
        "🔍 Enable Vector Search",
        value=True,
        help="Enable vector search for retrieving relevant context"
    )
    
    # Evaluation toggle
    run_evaluation = st.checkbox(
        "📊 Enable Evaluation", 
        value=True,
        help="Run safety and relevance evaluation on responses"
    )
    
    # Clear chat
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask me anything about Databricks..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Generate assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Prepare messages (simple format)
                messages = [
                    {"role": "system", "content": "You are a helpful assistant that answers questions about Databricks."},
                ] + st.session_state.messages
                
                # Get response from model
                response_dict = query_endpoint(
                    endpoint_name=selected_model,
                    messages=messages,
                    max_tokens=400,
                    use_tools=use_tools
                )
                
                assistant_response = response_dict["content"]
                trace = response_dict["trace"]
                trace_id = response_dict.get("trace_id")
                
                # Display response
                st.markdown(assistant_response)
                st.markdown(trace)

                # Run evaluation if enabled
                if run_evaluation:
                    with st.expander("📊 Evaluation Results", expanded=True):
                        try:
                            from mlflow.genai.scorers import Safety, RelevanceToQuery, RetrievalRelevance
                            
                            st.write(f"**Trace ID:** `{trace_id}`")
                            
                            with st.spinner("Running evaluation..."):
                                # Run LLM scorers directly
                                fb_safety = Safety()( 
                                            outputs=assistant_response,
                                        )
                                mlflow.log_assessment(trace_id=trace_id, assessment=fb_safety)

                                fb_rel = RelevanceToQuery()(
                                    inputs={"question": prompt},
                                    outputs=assistant_response,
                                )
                                mlflow.log_assessment(trace_id=trace_id, assessment=fb_rel)

                                # Add retrieval relevance if vector search is enabled
                                fb_retrieval = None
                                if use_tools:
                                    fb_retrieval = RetrievalRelevance()(
                                        trace={"data":trace},
                                    )
                                    mlflow.log_assessment(trace_id=trace_id, assessment=fb_retrieval)
                            
                            # Display results in Streamlit
                            col1, col2, col3 = st.columns(3)
        
                            
                            with col1:
                                st.subheader("🛡️ Safety Score")
                                st.metric("Safety", f"{fb_safety.value}")
                            
                            with col2:
                                st.subheader("🎯 Relevance Score")
                                st.metric("Relevance", f"{fb_rel.value}")
                            
                            with col3:
                                st.subheader("🔍 Retrieval Score")
                                st.metric("Retrieval", f"{fb_retrieval.value if fb_retrieval else 'N/A'}")
                                
                            
                            st.success("✅ Assessments logged to MLflow successfully!")
                            
                        except Exception as eval_error:
                            st.error(f"❌ Evaluation error: {eval_error}")
                            st.write("**Error details:**")
                            st.exception(eval_error)
                
                # Show status indicators
                if use_tools:
                    st.caption("🔍 Vector search enabled")
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
                assistant_response = f"I apologize, but I encountered an error: {str(e)}"
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": assistant_response})

# Show evaluation status
if run_evaluation:
    st.info("💡 **Evaluation enabled** - Safety and relevance scores are logged to MLflow traces")
else:
    st.info("💡 Enable evaluation in sidebar to get response quality metrics")