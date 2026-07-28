import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
import glob
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

# ----------------------------------------------------------------------------
# PAGE CONFIG + GLOBAL STYLE
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Data Analyst Agent",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main .block-container {padding-top: 2rem; padding-bottom: 3rem;}

    .app-header {
        background: linear-gradient(90deg, #4F46E5 0%, #7C3AED 100%);
        padding: 1.6rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.5rem;
    }
    .app-header h1 {margin: 0; font-size: 1.9rem;}
    .app-header p {margin: 0.3rem 0 0 0; opacity: 0.9; font-size: 0.95rem;}

    div[data-testid="stMetric"] {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }

    .section-card {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        margin-bottom: 1rem;
    }

    .stTabs [data-baseweb="tab-list"] {gap: 6px;}
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 18px;
        background-color: #F1F5F9;
    }
    .stTabs [aria-selected="true"] {
        background-color: #EEF2FF !important;
        border-bottom: 3px solid #4F46E5 !important;
    }

    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
        padding: 0.5rem 1.2rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

st.markdown(
    """
    <div class="app-header">
        <h1>📊 AI-Powered Data Analyst Agent</h1>
        <p>Upload a dataset and let the agent write, run, and explain a full basic + advanced
        exploratory data analysis — then chat with your data directly.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# SIDEBAR — API KEYS + MODEL SELECTION
# ----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 🔑 API Configuration")
    GOOGLE_API_KEY = st.text_input("Google API Key", type="password", help="Used for Gemini")
    GROQ_API_KEY = st.text_input("Groq API Key", type="password", help="Used for Groq-hosted models")

    st.markdown("---")
    st.markdown("### ⚙️ Model")
    model_choice = st.radio("Choose the model that writes the analysis code", ["Gemini", "Groq"], horizontal=True)

    st.markdown("---")
    with st.expander("ℹ️ How this works"):
        st.write(
            "1. Upload a CSV/Excel file.\n"
            "2. The agent writes a self-contained Python function tailored to your "
            "data's actual columns and dtypes.\n"
            "3. That code runs locally in this app and the results are rendered below."
        )

if not (GOOGLE_API_KEY and GROQ_API_KEY):
    st.info("👈 Enter both API keys in the sidebar to get started.")
    st.stop()

llm = (
    ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite", google_api_key=GOOGLE_API_KEY)
    if model_choice == "Gemini"
    else ChatGroq(model="llama-3.3-70b-versatile", api_key=GROQ_API_KEY)
)

# ----------------------------------------------------------------------------
# HELPERS
# ----------------------------------------------------------------------------
def extract_code(raw_text: str) -> str:
    """Pull clean, executable Python out of an LLM response, stripping any
    markdown fences or stray prose the model may have added despite instructions."""
    if isinstance(raw_text, list):
        raw_text = raw_text[-1].get("text", str(raw_text)) if raw_text else ""

    match = re.search(r"```(?:python)?\s*(.*?)```", raw_text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return raw_text.strip()


def build_basic_eda_prompt(df_sample: str, df_stats: str) -> str:
    return f"""You are an expert Python Data Analyst.
Your task is to write a single, self-contained Python function named `perform_eda(df)` that performs basic Exploratory Data Analysis on a pandas DataFrame and returns the results as a dictionary. The function must rely ONLY on the `df` argument passed to it — do not reference any external variables, files, or a hardcoded schema.

DATA CONTEXT (for understanding schema only — do not hardcode these exact values):
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
8. `numeric_summary` - summary statistics (mean, std, min, max, quartiles) for numeric columns only, as a dictionary
9. `unique_counts` - dictionary of column name to number of unique values

CODE QUALITY:
- Include all necessary imports (`pandas`, `numpy`) inside or above the function.
- Handle edge cases gracefully: empty DataFrames, all-null columns, non-numeric columns.
- Ensure every dictionary value is JSON-serializable (convert numpy types to native Python `int`/`float`, dtypes to strings).
- The function must not raise an exception on any valid pandas DataFrame and must be fully generalized (works on the full dataset, not just the sample above).

CONSTRAINTS:
- Output ONLY valid, executable Python code.
- The code must define exactly one function: `perform_eda(df)`.
- DO NOT wrap the output in markdown code blocks.
- DO NOT include explanations, comments about the task, print statements, example usage, or a call to `perform_eda()` outside the function.
"""


def build_advanced_eda_prompt(df_sample: str, df_stats: str) -> str:
    return f"""You are an expert Python Data Analyst and Data Visualization specialist.
Your task is to write a single, self-contained Python function named `perform_advanced_eda(df)` that performs advanced Exploratory Data Analysis on a pandas DataFrame and generates a comprehensive set of charts. The function must rely ONLY on the `df` argument passed to it — do not reference any external variables, files, or a hardcoded schema.

DATA CONTEXT (for understanding schema only — do not hardcode these exact values):
- Data Sample:\n{df_sample}
- Data Stats:\n{df_stats}

REQUIREMENTS:
1. Inspect `df`'s columns and dtypes at runtime and dynamically decide which charts are relevant. Detect numeric, categorical, and datetime columns programmatically — never hardcode column names.
2. Generate charts covering as many of the following as are applicable to the actual data:
   - Distribution plots (histograms/KDE) for numeric columns
   - Box plots for numeric columns to visualize outliers
   - Bar charts for categorical columns (top N categories if high cardinality)
   - Correlation heatmap for numeric columns (only if 2+ numeric columns exist)
   - Scatter matrix for numeric columns (limit to top 5 by variance to avoid clutter)
   - Time series line plots (only if a datetime-like column is detected)
   - Missing value bar chart across columns
3. Use `matplotlib` and `seaborn`. Set a consistent, readable style and reasonable figure sizes.
4. Save each chart as a PNG to a folder named `eda_charts/` (create it if missing), with descriptive filenames (e.g. `eda_charts/histogram_age.png`).
5. Return a dictionary with:
   - `charts_generated`: list of file paths for all saved charts
   - `insights`: dictionary of auto-derived observations (e.g. most correlated pair of numeric columns, column with most missing values, most skewed numeric column, categorical column with highest cardinality)
6. Include all necessary imports (`pandas`, `numpy`, `matplotlib.pyplot`, `seaborn`, `os`) inside or above the function.
7. Close each matplotlib figure after saving (`plt.close()`).

CONSTRAINTS:
- Wrap each chart-generation step in its own try/except so one failure doesn't stop the rest.
- Skip a chart type gracefully (no error) if the required column type isn't present.
- The function must be fully generalized (works on the full dataset, not just the sample above) and must not raise an exception on any valid pandas DataFrame.
- Ensure every value in the returned dictionary is JSON-serializable.
- Output ONLY valid, executable Python code.
- The code must define exactly one function: `perform_advanced_eda(df)`.
- DO NOT wrap the output in markdown code blocks.
- DO NOT include explanations, comments about the task, print statements, example usage, or a call to `perform_advanced_eda()` outside the function.
"""


def get_df_context(df: pd.DataFrame):
    sample = df.sample(min(5, len(df)), random_state=42).to_string()
    stats = df.describe(include="all").to_string()
    return sample, stats


def run_generated_function(code: str, func_name: str, df: pd.DataFrame):
    """Executes LLM-generated code in an isolated namespace and calls func_name(df)."""
    namespace = {}
    exec(code, namespace)
    if func_name not in namespace:
        raise ValueError(f"Generated code did not define `{func_name}`.")
    return namespace[func_name](df)


# ----------------------------------------------------------------------------
# FILE UPLOAD
# ----------------------------------------------------------------------------
uploaded_file = st.file_uploader("📁 Upload your CSV or Excel file", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith(".csv") else pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    st.success(f"Dataset loaded — **{df.shape[0]:,} rows × {df.shape[1]} columns**")

    # Quick metrics row
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Rows", f"{df.shape[0]:,}")
    m2.metric("Columns", df.shape[1])
    m3.metric("Missing cells", f"{int(df.isnull().sum().sum()):,}")
    m4.metric("Duplicate rows", int(df.duplicated().sum()))

    with st.expander("🔍 Preview dataset"):
        st.dataframe(df.head(20), use_container_width=True)

    tab1, tab2, tab3, tab4 = st.tabs(
        ["🧪 Basic EDA", "🚀 Advanced EDA", "📈 Quick Visualizations", "💬 Chat with Data"]
    )

    # ------------------------------------------------------------------
    # TAB 1 — BASIC EDA
    # ------------------------------------------------------------------
    with tab1:
        st.markdown("#### Basic Exploratory Data Analysis")
        st.caption("The agent writes a self-contained `perform_eda(df)` function tailored to your data, then runs it here.")

        if st.button("Run Basic EDA", key="basic_eda_btn"):
            with st.spinner("Agent is writing and executing the basic EDA code..."):
                try:
                    df_sample, df_stats = get_df_context(df)
                    prompt = build_basic_eda_prompt(df_sample, df_stats)
                    response = llm.invoke(prompt)
                    code = extract_code(response.content)

                    with st.expander("View generated code"):
                        st.code(code, language="python")

                    result = run_generated_function(code, "perform_eda", df)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Shape", f"{result['shape'][0]} × {result['shape'][1]}")
                    c2.metric("Size", result.get("size", "—"))
                    c3.metric("Duplicate rows", result.get("duplicate_rows", "—"))

                    st.markdown("**Data types**")
                    st.dataframe(pd.Series(result["dtypes"], name="dtype"), use_container_width=True)

                    st.markdown("**Missing values**")
                    missing_df = pd.DataFrame({
                        "missing_count": result["missing_values"],
                        "missing_%": result.get("missing_percentage", {}),
                    })
                    st.dataframe(missing_df, use_container_width=True)

                    st.markdown("**Unique value counts**")
                    st.dataframe(pd.Series(result["unique_counts"], name="unique_count"), use_container_width=True)

                    if result.get("numeric_summary"):
                        st.markdown("**Numeric summary**")
                        st.dataframe(pd.DataFrame(result["numeric_summary"]), use_container_width=True)

                except Exception as e:
                    st.error(f"Error during basic EDA: {e}")

    # ------------------------------------------------------------------
    # TAB 2 — ADVANCED EDA
    # ------------------------------------------------------------------
    with tab2:
        st.markdown("#### Advanced Exploratory Data Analysis")
        st.caption("The agent writes a self-contained `perform_advanced_eda(df)` function that generates and saves charts, then displays them here.")

        if st.button("Run Advanced EDA", key="advanced_eda_btn"):
            with st.spinner("Agent is writing and executing the advanced EDA code..."):
                try:
                    df_sample, df_stats = get_df_context(df)
                    prompt = build_advanced_eda_prompt(df_sample, df_stats)
                    response = llm.invoke(prompt)
                    code = extract_code(response.content)

                    with st.expander("View generated code"):
                        st.code(code, language="python")

                    os.makedirs("eda_charts", exist_ok=True)
                    # clear stale charts from a previous run
                    for f in glob.glob("eda_charts/*.png"):
                        os.remove(f)

                    result = run_generated_function(code, "perform_advanced_eda", df)

                    if result.get("insights"):
                        st.markdown("**Auto-derived insights**")
                        st.json(result["insights"])

                    charts = result.get("charts_generated", [])
                    if charts:
                        st.markdown(f"**Generated {len(charts)} chart(s)**")
                        cols = st.columns(2)
                        for i, chart_path in enumerate(charts):
                            if os.path.exists(chart_path):
                                cols[i % 2].image(chart_path, use_container_width=True)
                    else:
                        st.info("No charts were generated for this dataset.")

                except Exception as e:
                    st.error(f"Error during advanced EDA: {e}")

    # ------------------------------------------------------------------
    # TAB 3 — QUICK VISUALIZATIONS (built-in, no LLM call)
    # ------------------------------------------------------------------
    with tab3:
        st.markdown("#### Quick Built-in Visualizations")
        numeric_df = df.select_dtypes(include=[np.number])

        if st.button("Generate Charts", key="quick_charts_btn"):
            if numeric_df.shape[1] >= 2:
                st.markdown("**Correlation heatmap**")
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
                st.pyplot(fig)
                plt.close(fig)
            else:
                st.info("Need at least 2 numeric columns for a correlation heatmap.")

            if not numeric_df.empty:
                st.markdown("**Distributions**")
                cols = st.columns(min(3, numeric_df.shape[1]))
                for i, col in enumerate(numeric_df.columns[:3]):
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.histplot(df[col].dropna(), kde=True, ax=ax)
                    ax.set_title(col)
                    cols[i].pyplot(fig)
                    plt.close(fig)
            else:
                st.info("No numeric columns found.")

    # ------------------------------------------------------------------
    # TAB 4 — CHAT WITH DATA
    # ------------------------------------------------------------------
    with tab4:
        st.markdown("#### Chat with your Dataset")
        user_query = st.text_input("Ask a question about your data:")

        if user_query:
            with st.spinner("Analyzing your question..."):
                try:
                    df_sample, df_stats = get_df_context(df)
                    chat_prompt = f"""You are a Python data analyst. Given a pandas DataFrame `df` with:
- Columns and dtypes:\n{df.dtypes.to_string()}
- Sample rows:\n{df_sample}

Write Python code that answers this question about `df`: "{user_query}"

CONSTRAINTS:
- Assume `df` is already loaded — do not read any file.
- Store the final answer in a variable named `result` (a number, string, DataFrame, or Series).
- Do not include print statements, explanations, or markdown fences.
- Output ONLY valid, executable Python code.
"""
                    response = llm.invoke(chat_prompt)
                    code = extract_code(response.content)

                    st.markdown("**Generated code**")
                    st.code(code, language="python")

                    namespace = {"df": df, "pd": pd, "np": np}
                    exec(code, namespace)
                    result = namespace.get("result", "No `result` variable was set by the generated code.")

                    st.markdown("**Result**")
                    if isinstance(result, (pd.DataFrame, pd.Series)):
                        st.dataframe(result, use_container_width=True)
                    else:
                        st.write(result)

                except Exception as e:
                    st.error(f"Error answering query: {e}")
else:
    st.info("Upload a dataset above to begin.")
    st.warning("Please enter your Google API Key and Groq API Key in the sidebar to proceed.")
