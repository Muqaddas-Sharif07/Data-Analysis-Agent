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
uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file and api_key:
    # Read CSV
    df = pd.read_csv(uploaded_file)
    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Initialize LLM & Agent
llm = ChatGroq(model_name="llama-3.1-8b-instant", api_key=api_key)
    agent = create_pandas_dataframe_agent(
        llm, 
        df, 
        verbose=False, 
        agent_type="zero-shot-react-description", 
        allow_dangerous_code=True, 
        handle_parsing_errors=True
    )

    # Chat Interface
    st.subheader("Ask Questions About Your Data")
    user_query = st.text_input("Type your question here:")

    if st.button("Analyze"):
        if user_query:
            with st.spinner("Analyzing data..."):
                try:
                    response = agent.run(user_query)
                    st.success("Done!")
                    st.write(response)
                except Exception as e:
                    error_msg = str(e)
                    if "Could not parse LLM output:" in error_msg:
                        clean_res = error_msg.split("Could not parse LLM output:")[-1]
                        st.success("Done!")
                        st.write(clean_res)
                    else:
                        st.error(f"Error: {error_msg}")
        else:
            st.warning("Please enter a question first!")

elif not api_key:
    st.info("👈 Please enter your Groq API Key in the sidebar to get started.")

elif not uploaded_file:
    st.info("👈 Please upload a CSV file in the sidebar.")
   
           

     
