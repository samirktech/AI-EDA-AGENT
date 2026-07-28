import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import langchain
import langchain_community
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq
from langchain.chat_models import init_chat_model
import streamlit as st

st.set_page_config(page_title="AI Powered Data Analyst Agent", layout="wide")
st.title("🤖 AI-Powered Data Analyst Agent")
st.write("Automatically analyze your data, generate univariate, bivariate, and multivariate charts, execute code, and chat with your dataset!")

# Initialize API keys and Models
GOOGLE_API_KEY = st.sidebar.text_input("Enter Google API Key", type="password")
GROQ_API_KEY = st.sidebar.text_input("Enter Groq API Key", type="password")

if GOOGLE_API_KEY and GROQ_API_KEY:
    gemini_llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=GOOGLE_API_KEY
    )

    groq_llm = ChatGroq(
        model="qwen-2.5-coder-32b-instruct",
        api_key=GROQ_API_KEY
    )

    from langchain.agents import create_agent
    
    def temp_tool():
        """This is just a dummy tool"""
        return "Hello world"

    agent = create_agent(
        model=gemini_llm,
        tools=[temp_tool]
    )

    # File Uploader
    uploaded_file = st.file_uploader("Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success("Dataset loaded successfully!")
            st.write("### Dataset Preview", df.head())

            # Tabs for Analysis and Chat
            tab1, tab2, tab3 = st.tabs(["📊 Basic & Advanced EDA", "📈 Visualizations", "💬 Chat with Data"])

            with tab1:
                st.subheader("Basic & Advanced EDA Generation")
                if st.button("Run AI EDA Analysis"):
                    with st.spinner("Agent is generating EDA code and analyzing data..."):
                        try:
                            # Generate basic and advanced EDA scripts dynamically
                            df_sample = df.sample(min(5, len(df)))
                            prompt = f"""You are a data analyst perform basic eda python single function perform_eda code and give all required analysis like missing values and columns. Data frame sample : {df_sample}"""
                            response = agent.invoke({'messages': [{'role': 'user', 'content': prompt}]})
                            ans = response["messages"][-1].content
                            
                            # Handle different response structures safely
                            if isinstance(ans, list):
                                ans = ans[-1].get('text', str(ans))
                            
                            if "```python" in ans:
                                code = ans.split("```python")[1].split("```")[0]
                            elif "```" in ans:
                                code = ans.split("```")[1]
                            else:
                                code = ans

                            with open('basic_eda.py', 'w') as f:
                                f.write(code)

                            # Advanced EDA prompt
                            advance_prompt = """Give Python advance_eda.py file with every code inside a single function eda_by_ai with parameter as dataframe and no need to load file, df is already loaded, starts with using df and include describe, corr, univariate numerical and object column analysis, multivariate analysis using seaborn/matplotlib plots with comments."""
                            response_adv = agent.invoke({'messages': [{'role': 'user', 'content': advance_prompt}]})
                            ans_adv = response_adv["messages"][-1].content
                            
                            if isinstance(ans_adv, list):
                                ans_adv = ans_adv[-1].get('text', str(ans_adv))
                                
                            if "```python" in ans_adv:
                                code_adv = ans_adv.split("```python")[1].split("```")[0]
                            elif "```" in ans_adv:
                                code_adv = ans_adv.split("```")[1]
                            else:
                                code_adv = ans_adv

                            with open('advance_eda.py', 'w') as f:
                                f.write(code_adv)

                            st.success("EDA scripts generated and executed successfully!")
                            
                            # Display basic stats directly
                            st.write("### Dataset Summary Statistics")
                            st.write(df.describe())
                            st.write("### Missing Values")
                            st.write(df.isnull().sum())

                        except Exception as e:
                            st.error(f"Error during EDA generation: {e}")

            with tab2:
                st.subheader("Automatic Charts & Multivariate Analysis")
                if st.button("Generate Charts"):
                    st.write("#### Numerical Correlation Heatmap")
                    fig, ax = plt.subplots(figsize=(8, 6))
                    numeric_df = df.select_dtypes(include=[np.number])
                    if not numeric_df.empty:
                        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                        st.pyplot(fig)
                    else:
                        st.info("No numerical columns found for correlation heatmap.")

                    st.write("#### Univariate & Bivariate Distributions")
                    for col in numeric_df.columns[:3]:
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.histplot(df[col], kde=True, ax=ax)
                        st.pyplot(fig)

            with tab3:
                st.subheader("Chat with your Dataset")
                user_query = st.text_input("Ask a question about your data:")
                if user_query:
                    with st.spinner("Analyzing query..."):
                        chat_prompt = f"Given the dataframe df with columns {list(df.columns)}, write and execute python code to answer this query: {user_query}. Return only executable python code that prints or stores the result in a variable named 'result'."
                        chat_response = agent.invoke({'messages': [{'role': 'user', 'content': chat_prompt}]})
                        chat_ans = chat_response["messages"][-1].content
                        
                        if isinstance(chat_ans, list):
                            chat_ans = chat_ans[-1].get('text', str(chat_ans))
                            
                        st.write("### Generated Code:")
                        st.code(chat_ans, language='python')
                        
                        # Extra thing in comments: You can safely execute sanitized code here using local scope context `{'df': df}`.

        except Exception as e:
            st.error(f"Error reading file: {e}")
else:
    st.warning("Please enter your Google API Key and Groq API Key in the sidebar to proceed.")
