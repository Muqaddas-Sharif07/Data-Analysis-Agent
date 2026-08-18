import pandas as pd
import plotly.express as px
import streamlit as st
from langchain_groq import ChatGroq

# App Title & Layout Configuration
st.set_page_config(page_title="Data Analysis Agent", layout="wide")
st.title("📊 AI Data Analysis Agent")

# Sidebar - Settings & File Upload
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")
uploaded_file = st.sidebar.file_uploader("Upload CSV File", type=["csv"])

# Data Loading
df = None
if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.sidebar.success("CSV Uploaded Successfully!")
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # =========================================================
        # 📊 NEW VISUALIZATION SECTION (Aapka Naya Section)
        # =========================================================
        st.markdown("---")
        st.subheader("📊 Interactive Data Visualizations")

        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
        all_cols = df.columns.tolist()

        if numeric_cols:
            col1, col2 = st.columns(2)
            with col1:
                x_var = st.selectbox("Select X-axis:", all_cols)
            with col2:
                y_var = st.selectbox("Select Y-axis:", numeric_cols)

            fig = px.bar(
                df,
                x=x_var,
                y=y_var,
                title=f"{y_var} by {x_var}",
                template="plotly_white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No numeric columns found for visualization.")
        st.markdown("---")
        # =========================================================

    except Exception as e:
        st.sidebar.error(f"Error reading CSV: {e}")


# Function to generate Python Pandas code via Groq
def analyze_data(df, query, api_key):
    llm = ChatGroq(model_name="openai/gpt-oss-120b", api_key=api_key)

    prompt = f"""
    You are an expert Python data analyst.
    Here is the schema of the dataframe `df`:
    Columns and types: {df.dtypes.to_dict()}
    
    User Request: "{query}"
    
    Write ONLY a valid executable Python expression or statement using pandas to calculate/get the answer (the dataframe is named `df`).
    Do not print, just write the expression that returns the output (e.g., `df.describe()` or `df.groupby('col').mean()`).
    DO NOT generate any plotting code like .plot() or matplotlib code. Just return data.
    Return ONLY the code block enclosed in ```python ``` without any extra text or explanation.
    """

    response = llm.invoke(prompt)
    return response.content


# Chat Interface
st.subheader("Ask Questions About Your Data")
user_query = st.text_input("Type your question here:")

if st.button("Analyze"):
    if not api_key:
        st.info("👈 Please enter your Groq API Key in the sidebar.")
    elif uploaded_file is None or df is None:
        st.info("👈 Please upload a CSV file in the sidebar.")
    elif not user_query:
        st.warning("Please enter a question to ask!")
    else:
        with st.spinner("Analyzing data..."):
            try:
                # 1. Get Python code from LLM
                generated_code = analyze_data(df, user_query, api_key)

                # 2. Clean the code string
                clean_code = (
                    generated_code.replace("```python", "")
                    .replace("```", "")
                    .strip()
                )

                # 3. Safely execute code on local DataFrame
                local_vars = {"df": df, "pd": pd}
                exec(f"result = {clean_code}", {}, local_vars)

                # 4. Show Result
                st.success("Done!")
                st.write("**Answer:**")
                st.write(local_vars.get("result"))

            except Exception as e:
                st.error(f"Error analyzing data: {e}")
