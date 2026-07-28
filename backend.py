# Step 3: Load all modules
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
print("Modules Loaded Successfully!!")


# Step 4: Model creation
GOOGLE_API_KEY = "API-KEY"
GROQ_API_KEY = "API-KEY"

gemini_llm = ChatGoogleGenerativeAI(
    model = "gemini-3.5-flash-lite",
    google_api_key = GOOGLE_API_KEY
    )

groq_llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    api_key = GROQ_API_KEY
    )


# step 5: Agent Creation
def temp_tool():
  """This is just a dummy tool"""
  return "Hello world"

from langchain.agents import create_agent
agent = create_agent(
    model = gemini_llm,
    tools = [temp_tool] # we will give multiple tools here
    )


def load_dataset(path:str, agent = agent):
  """This Function help to take uploaded
  file path or url, and call agent
  here agent is responsible to return python code
  that helps to read uploaded dataset
  like CSV, XLSX, XLS, from Github url that ends
  with csv.
  System instructons: Don't give any reboot or harm code
  give correct executable code only
  example:
  df = pd.read_csv(file_path) or uploaded bytes string
  when file uploaded by streamlit or direct"""

  prompt = """return python code to read file
              in pandas using uploaded file extension"""


  try:
    import os
    if 'file_loader.py' not in os.getcwd():
      response = agent.invoke({'messages':[{'role':'user','content':prompt}]})
      ans = response["messages"][-1].content[-1]['text']
      code = ans.split("```")[1]
      with open('file_loader.py', 'w') as f:
        f.write(code[6:])
  except:
    pass

  return "Success"

  # # time.sleep(3)
  # from file_loader import read_uploaded_file
  # return read_uploaded_file(path)


url = 'https://raw.githubusercontent.com/axisgras-hash/DATASETS/refs/heads/main/Superstore.csv'
ans = load_dataset(url, agent)
ans


def read_file(path):
  """Read file using file_loader.py module
  read_uploaded_file function"""

  from file_loader import read_uploaded_file
  return read_uploaded_file(path)


  # tool 3
def perform_eda_func(data, agent):
  """
    Takes a pandas DataFrame as input and passes a sample to an AI agent.
    The agent is tasked with generating a custom Python function (`perform_eda`) 
    to perform basic Exploratory Data Analysis based on the dataset's schema.
    """
    
    # Use .head(5) for consistency, and convert to string or dict so the LLM reads it cleanly
    df_sample = data.head(5).to_string() 
    df_stats = data.describe(include='all').to_string()
    
    prompt = f"""You are an expert Python Data Analyst. 

Your task is to write a single, self-contained Python function named `perform_eda(df)` that performs basic Exploratory Data Analysis on a pandas DataFrame.

DATA CONTEXT:
Use the following data sample and statistics to understand the schema and tailor your code (e.g., handling specific column types if necessary).
- Data Sample:\n{df_sample}
- Data Stats:\n{df_stats}

REQUIREMENTS:
1. The function `perform_eda(df)` must calculate and return key metrics in a dictionary: shape, size, data types, column names, and missing values per column.
2. Include all necessary imports (e.g., `import pandas as pd`) inside or above the function.
3. The returned code must be generalized to run on the full version of this dataset.

CONSTRAINTS:
- Output ONLY valid, executable Python code. 
- DO NOT wrap the output in markdown code blocks (e.g., no ```python ... ```).
- DO NOT include any conversational filler, explanations, or print statements outside the function.
"""

  response = agent.invoke({'messages':[{'role':'user','content':prompt}]})
  ans = response["messages"][-1].content[-1]['text']
  code = ans.split("```")[1]

  with open('basic_eda.py', 'w') as f:
    f.write(code[6:])


  #====================AdvaNce EDA======================
 """
    Passes the dataset's schema to the agent to generate an advanced, 
    modular EDA script with specific univariate and multivariate requirements.
    """
    
    # Extracting columns and datatypes helps the LLM know exactly which columns 
    # to use for the multivariate bar plots (like Sales, Region, Segment).
    schema = data.dtypes.to_string()
    
    prompt = f"""You are an Expert Data Scientist. Your task is to write a comprehensive, modular Python script to perform Advanced Exploratory Data Analysis (EDA) on a dataset.

DATA SCHEMA:
Use the following column names and data types to dynamically generate your analysis code:
\n{schema}\n

REQUIREMENTS:
1. DEPENDENCIES: At the very top of the script, include standard Python code using `subprocess` and `sys` to `pip install` any required modules (e.g., seaborn, matplotlib, pandas) if they are not already installed.
2. MODULAR ARCHITECTURE: Write distinct functions for each of the following tasks:
   - `get_summary()`: Print `.describe()` for both numerical and object columns.
   - `get_correlation()`: Calculate and display the correlation matrix.
   - `univariate_analysis()`: Generate distributions (e.g., histograms/boxplots) for numerical columns and value counts for object columns.
   - `multivariate_analysis()`: Generate bar plots using `seaborn` with the `hue` parameter to show relationships across multiple columns (e.g., comparing a numeric metric like Sales across a categorical Region, segmented by a hue like Segment).
3. MAIN EXECUTION: Include a `main(df)` function that sequentially calls all the modular functions above.

CONSTRAINTS:
- Output ONLY valid, executable Python code. 
- DO NOT include markdown formatting (no ```python blocks).
- DO NOT include conversational text.
- Ensure the code handles potential errors (e.g., skipping correlation if no numeric columns exist).
"""
  response = agent.invoke({'messages':[{'role':'user','content':advance_prompt}]})
  system_prompt_model = response["messages"][-1].content[-1]['text']

  #========================The above will give detailed prompt only================

  new_prompt = """Give Python advance_eda.py file with
  every code inside a single function eda_by_ai with parameter as dataframe
  and no need to load file, df is already loaded,
  starts with using df and
  any notes with comment""" + system_prompt_model

  response = agent.invoke({'messages':[{'role':'user','content':new_prompt}]})
  ans = response["messages"][-1].content[-1]['text']
  code = ans.split("```")[1]

  with open('advance_eda.py', 'w') as f:
    f.write(code[6:])

  return "Success"

df = read_file(url)
df


from basic_eda import perform_eda
from advance_eda import eda_by_ai
