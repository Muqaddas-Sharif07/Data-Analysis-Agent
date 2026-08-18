import streamlit as st
import pandas as pd
from langchain_groq import ChatGroq
from langchain_experimental.agents import create_pandas_dataframe_agent

# Page setup
st.set_page_config(page_title="AI Data Analysis Agent", page_icon="📊", layout="wide")
st.title("📊 AI-Powered Data Analysis Agent")
st.write("Upload any CSV file and ask natural language questions about your data!")

# Sidebar for API Key & File Upload
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Groq API Key", type="password")
# Sidebar File Upload
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

# Code tabhi chalega jab File AUR API Key dono mojud hon
if uploaded_file is not None and api_key:
# 1. Read CSV
    df = pd.read_csv(uploaded_file)

# 2. Data Preview
    st.subheader("Data Preview")
    st.dataframe(df.head())
    import os
    
    # Active key determine karein
    active_key = api_key.strip() if (api_key and api_key.strip()) else st.secrets.get("GROQ_API_KEY", "")
        
    # Environment variable me force-set karein
    os.environ["GROQ_API_KEY"] = active_key
    
    # LLM Initialize karein
    llm = ChatGroq(
            model_name="llama-3.3-70b-versatile",
            groq_api_key=active_key
        )
    
    agent = create_pandas_dataframe_agent(
        llm,
        df,
        verbose=False,
        agent_type="zero-shot-react-description",
        allow_dangerous_code=True,
        handle_parsing_errors=True,
    )
    

# 4. Chat Interface
st.subheader("Ask Questions About Your Data")
user_query = st.text_input("Type your question here:")
if st.button("Analyze Data"):
        if user_query:
            with st.spinner("Analyzing..."):
                response = agent.invoke(user_query)
                st.write(response["output"])
        else:
            st.warning("Please enter a question first.")

elif not api_key:
    st.warning("Please enter your Groq API Key in the sidebar.")
else:
    st.info("Please upload a CSV file from the sidebar to get started.")
