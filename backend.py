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
Your task is to write a single, self-contained Python function named `perform_eda(df)` that performs basic Exploratory Data Analysis on a pandas DataFrame and returns the results as a dictionary.

DATA CONTEXT:
Use the following data sample and statistics to understand the schema and tailor your code (e.g., handling specific column types, datetime parsing, or categorical vs numeric logic if necessary).
- Data Sample:\n{df_sample}
- Data Stats:\n{df_stats}

REQUIREMENTS:
The function `perform_eda(df)` must calculate and return the following as a single dictionary:
1. `shape` - tuple of (rows, columns)
2. `size` - total number of elements
3. `dtypes` - dictionary of column name to data type (as strings, not dtype objects)
4. `columns` - list of column names
5. `missing_values` - dictionary of column name to count of missing/null values
6. `missing_percentage` - dictionary of column name to percentage of missing values
7. `duplicate_rows` - count of fully duplicate rows
8. `numeric_summary` - summary statistics (mean, std, min, max, quartiles) for numeric columns only, as a dictionary (use `.to_dict()` on the describe() output)
9. `unique_counts` - dictionary of column name to number of unique values

REQUIREMENTS FOR CODE QUALITY:
- Include all necessary imports (e.g., `import pandas as pd`, `import numpy as np`) inside or above the function.
- Handle edge cases gracefully: empty DataFrames, columns with all-null values, and non-numeric columns being passed to numeric operations.
- Ensure all dictionary values are JSON-serializable (convert numpy types like `np.int64`/`np.float64` to native Python `int`/`float`, and convert dtype objects to strings).
- The function must not raise an exception on valid pandas DataFrames.
- The code must be generalized to run on the full version of this dataset, not just the sample shown above.

CONSTRAINTS:
- Output ONLY valid, executable Python code.
- The code must define exactly one function: `perform_eda(df)`.
- DO NOT wrap the output in markdown code blocks (e.g., no ```python ... ```).
- DO NOT include any conversational filler, explanations, comments about the task, or print statements outside the function.
- DO NOT include example usage or a call to `perform_eda()` at the end of the script.
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
    
    advance_prompt = f"""You are an expert Python Data Analyst and Data Visualization specialist.
Your task is to write a single, self-contained Python function named `perform_advanced_eda(df)` that performs advanced Exploratory Data Analysis on a pandas DataFrame and generates a comprehensive set of charts based on the actual structure of the data.

DATA CONTEXT:
Use the following data sample and statistics to understand the schema, column types, and relationships, and tailor your chart choices accordingly (e.g., only plot correlation heatmaps if there are 2+ numeric columns, only plot time series if a datetime column exists).
- Data Sample:\n{df_sample}
- Data Stats:\n{df_stats}

REQUIREMENTS:
1. The function `perform_advanced_eda(df)` must inspect the DataFrame's columns and dtypes at runtime, and dynamically decide which charts are relevant. Do not hardcode column names from the sample — detect numeric, categorical, and datetime columns programmatically.
2. Generate charts covering as many of the following as are applicable to the actual data:
   - Distribution plots (histograms/KDE) for numeric columns
   - Box plots for numeric columns to visualize outliers
   - Bar charts for categorical columns (top N categories if high cardinality)
   - Correlation heatmap for numeric columns (only if 2+ numeric columns exist)
   - Pairplot/scatter matrix for numeric columns (limit to a reasonable number of columns to avoid clutter, e.g., top 5 by variance)
   - Time series line plots (only if a datetime-like column is detected)
   - Missing value heatmap/bar chart showing null distribution across columns
   - Count plots for categorical vs a numeric target if an obvious target-like column exists
3. Use `matplotlib` and `seaborn` for all visualizations. Set a consistent, readable style (e.g., `sns.set_style`, reasonable figure sizes).
4. Save each generated chart as a PNG file to a folder named `eda_charts/` (create the folder if it doesn't exist), using descriptive filenames (e.g., `eda_charts/histogram_age.png`, `eda_charts/correlation_heatmap.png`).
5. The function should return a dictionary with:
   - `charts_generated`: list of file paths for all saved charts
   - `insights`: a dictionary of basic auto-derived observations (e.g., most correlated pair of numeric columns, column with most missing values, most skewed numeric column, categorical column with highest cardinality)
6. Include all necessary imports (`pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `os`) inside or above the function.
7. Close each matplotlib figure after saving (`plt.close()`) to avoid memory issues when generating many charts.

CONSTRAINTS:
- The function must not raise an exception on valid pandas DataFrames — wrap chart-generation steps in try/except blocks so one failed chart doesn't stop the rest from generating.
- Skip a chart type gracefully (do not error) if the required column type isn't present in the data.
- The code must be generalized to run on the full version of this dataset, not just the sample shown above.
- Ensure all values in the returned dictionary are JSON-serializable (convert numpy types to native Python types).
- Output ONLY valid, executable Python code.
- The code must define exactly one function: `perform_advanced_eda(df)`.
- DO NOT wrap the output in markdown code blocks (e.g., no ```python ... ```).
- DO NOT include any conversational filler, explanations, comments about the task, or print statements outside the function.
- DO NOT include example usage or a call to `perform_advanced_eda()` at the end of the script.
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
