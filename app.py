import streamlit as st
import mlflow
from model_serving_utils import query_endpoint
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit app
if "visibility" not in st.session_state:
    st.session_state.visibility = "visible"
    st.session_state.disabled = False

# Configure Streamlit page
st.set_page_config(
    page_title="Databricks Chatbot",
    page_icon="🤖",
    layout="wide"
)

# Access headers passed by Databricks Apps
def get_user_info():
    headers = st.context.headers
    return dict(
        user_name=headers.get("X-Forwarded-Preferred-Username"),
        user_email=headers.get("X-Forwarded-Email"),
        user_id=headers.get("X-Forwarded-User"),
    )

user_name = get_user_info().get("user_name")

# Display the user's information
st.write(f"Logged in as: {user_name}")

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
        [os.getenv("SERVING_ENDPOINT"), os.getenv("SERVING_ENDPOINT_2")],
        help="Select the model to use for responses"
    )
    
    # Vector search toggle
    use_tools = st.checkbox(
        "🔍 Enable Vector Search",
        value=True,
        help="Enable vector search for retrieving relevant context"
    )
    
    # System prompt editor
    st.subheader("🎯 System Prompt")
    system_prompt = st.text_area(
        "Edit the system prompt:",
        value="You are a helpful assistant that answers questions about Databricks. Be concise, accurate, and helpful. When using retrieved context, incorporate it naturally into your response.",
        height=100,
        help="Customize how the AI assistant behaves and responds"
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
    
    # Load previous traces
    if st.button("📋 Load Previous 5 Traces"):
        try:
            # Search traces (this is a simplified version - MLflow's trace search API may vary)
            traces = mlflow.search_traces(
                experiment_ids=["2827072201875595"],
                filter_string=f"tag.user = '{user_name}'",
                max_results=5,
                order_by=["timestamp DESC"]
            )
            
            if traces:
                st.success(f"Found {len(traces)} previous traces!")
                
                # Display trace information
                for i, trace in enumerate(traces):
                    with st.expander(f"Trace {i+1}: {trace.info.trace_id[:8]}..."):
                        st.write(f"**Trace ID:** {trace.info.trace_id}")
                        st.write(f"**Timestamp:** {trace.info.timestamp}")
                        if hasattr(trace, 'inputs') and trace.inputs:
                            st.write(f"**Input:** {trace.inputs}")
                        if hasattr(trace, 'outputs') and trace.outputs:
                            st.write(f"**Output:** {trace.outputs}")
                        
            else:
                st.info("No previous traces found for this user")
                
        except Exception as e:
            st.error(f"Error loading traces: {e}")
            st.write("Note: This feature requires appropriate MLflow permissions and trace search capabilities")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display system prompt as first message
with st.chat_message("system"):
    st.markdown(f"**System:** {system_prompt}")

# Display chat history
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
    
    # Add feedback buttons after each assistant message
    if message["role"] == "assistant":
        st.markdown("---")
        feedback_col1, feedback_col2, feedback_col3 = st.columns([1, 2, 1])
        
        with feedback_col2:
            col1, col2, col3 = st.columns([1, 1, 1])
            with col1:
                if st.button("👍 Good", key=f"good_{i}"):
                    print(f"🔥 THUMBS UP BUTTON CLICKED for message {i}!")  # Debug
                    try:
                        current_trace_id = st.session_state.get("current_trace_id")
                        print(f"🔍 Current trace ID for feedback: {current_trace_id}")  # Debug
                        
                        if current_trace_id:
                            mlflow.log_feedback(
                                trace_id=current_trace_id,
                                name='user_feedback_thumbs',
                                value=1.0,
                                rationale="Thumbs up - helpful response",
                                source=mlflow.entities.AssessmentSource(
                                    source_type='HUMAN',
                                    source_id='streamlit_user',
                                ),
                            )
                            print("🔍 Positive feedback logged successfully!")  # Debug
                            st.success("✅ Positive feedback logged!")
                            st.balloons()  # Visual confirmation
                        else:
                            print("🔍 No trace ID available for feedback")  # Debug
                            st.warning("⚠️ No trace ID available")
                    except Exception as e:
                        print(f"🔍 Feedback error: {e}")  # Debug
                        st.error(f"Feedback error: {e}")
            
            with col2:
                if st.button("👎 Poor", key=f"poor_{i}"):
                    print(f"🔥 THUMBS DOWN BUTTON CLICKED for message {i}!")  # Debug
                    try:
                        current_trace_id = st.session_state.get("current_trace_id")
                        print(f"🔍 Current trace ID for feedback: {current_trace_id}")  # Debug
                        
                        if current_trace_id:
                            mlflow.log_feedback(
                                trace_id=current_trace_id,
                                name='user_feedback_thumbs',
                                value=0.0,
                                rationale="Thumbs down - poor response",
                                source=mlflow.entities.AssessmentSource(
                                    source_type='HUMAN',
                                    source_id='streamlit_user',
                                ),
                            )
                            print("🔍 Negative feedback logged successfully!")  # Debug
                            st.success("✅ Negative feedback logged!")
                            st.snow()  # Visual confirmation
                        else:
                            print("🔍 No trace ID available for feedback")  # Debug
                            st.warning("⚠️ No trace ID available")
                    except Exception as e:
                        print(f"🔍 Feedback error: {e}")  # Debug
                        st.error(f"Feedback error: {e}")
        
        st.markdown("")  # Add spacing

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
                # Prepare messages with custom system prompt
                messages = [
                    {"role": "system", "content": system_prompt},
                ] + st.session_state.messages
                
                # Get response from model
                response_dict = query_endpoint(
                    endpoint_name=selected_model,
                    messages=messages,
                    user=user_name,
                    use_tools=use_tools,
                )
                
                assistant_response = response_dict["content"]
                trace_id = response_dict.get("trace_id")
                
                # Debug: Show what trace_id we got
                print(f"🔍 Received trace_id: {trace_id}")
                
                # Store trace_id in session state for feedback
                st.session_state.current_trace_id = trace_id
                
                # Debug: Show what's stored in session state
                print(f"🔍 Stored in session state: {st.session_state.current_trace_id}")
                
                # Display response
                st.markdown(assistant_response)

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
                                trace = mlflow.get_trace(trace_id=trace_id)
                                if use_tools:
                                    fb_retrieval = RetrievalRelevance()(
                                        trace=trace
                                    )
                                    if fb_retrieval:
                                        mlflow.log_assessment(trace_id=trace_id, assessment=fb_retrieval[0])
                            
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
                                st.metric("Retrieval", f"{fb_retrieval[0].feedback.value if fb_retrieval else 'N/A'}")
                                
                            
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
                # Set empty trace_id if there was an error
                st.session_state.current_trace_id = None

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})
    
    # Show feedback buttons for the current response immediately
    st.markdown("---")
    
    # Trigger a rerun to show the buttons
    st.rerun()


# Show evaluation status
if run_evaluation:
    st.info("💡 **Evaluation enabled** - Safety and relevance scores are logged to MLflow traces")
else:
    st.info("💡 Enable evaluation in sidebar to get response quality metrics")