import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import re
import traceback
import os
from dotenv import load_dotenv
import io
import sys
import traceback

# LangChain imports for AI-powered data analysis
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()
GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")

prompt_1 = PromptTemplate(
                input_variables=["request", "columns", "sample_data"],
                template='''You are an intelligent Python data assistant.

You are given:
- request : {request}
- columns : {columns}
- Sample rows : {sample_data}

Your job is to classify the intent and generate appropriate Python code or skip code generation depending on the instruction.

---

### 🔍 1. Intent Classification

Categorize the user request into one of the following:
- **"question"**: e.g., "What is the average price?", "Which brand has the most products?"
- **"analysis"**: e.g., grouping, correlation, filtering, statistical summaries
- **"visualization"**: e.g., bar chart, histogram, scatter plot
- **"driver_analysis"**: e.g., "What drives sales?", "Top features influencing price", "Which variables affect churn?"
- **"casual"**: e.g., greetings, thanks, jokes, or inputs not related to the dataset

---

### 🧠 2. Code Generation Rules

- Use the existing pandas DataFrame named `df` — no need to import or redefine it
- Use standard pandas operations or `scikit-learn` for model-based tasks
- Handle missing values using `.dropna()`, `fillna()`, etc.
- If the request is for **summary**, **insights**, or **brief explanation**, include:
  - Proper `print()` statements for all outputs
  - A final line: `print("__NEEDS_INTERPRETATION__")` so the system can interpret the results

---

### 🧩 3. Driver Analysis Rules

If the user intent is **driver_analysis**:
- Use a tree-based model like `RandomForestRegressor` or `RandomForestClassifier` from `sklearn`
- Encode categorical features using `LabelEncoder`
- Identify the target variable based on user query (e.g., “price”, “churn”, etc.)
- Show **feature importances** in a sorted bar chart
- Use either:
  - `plt.figure(figsize=(10, 6))` with `plt.barh()` and `plt.show()`, or
  - `plotly.express.bar()` and assign chart to `fig`
- Do **not** include `__NEEDS_INTERPRETATION__` if a chart is generated

---

### 📊 4. Visualization Rules

- Use `matplotlib.pyplot` (`plt`) or `plotly.express` (`px`)
- For `matplotlib`:
  - Use `plt.figure(figsize=(10, 6))`, proper titles, axis labels, and `plt.show()`
- For `plotly`, assign the chart to a variable `fig`
- **Do not print** anything or include `__NEEDS_INTERPRETATION__` for visualizations
- Your response should include only the chart code

---

### 💬 5. Handling Casual Requests

If the user's message is casual or unrelated to the data:
- Respond like a helpful assistant
- Do **not** include any code
- Example:
  - User: Hey
    Response: Hello! How can I assist you with your data today?
  - User: Thank you
    Response: You're welcome! Let me know if you have more questions.

---

### 🔐 6. Code Format Requirement

Always respond with a single valid Python code block:
````python
<your generated code here>

'''
            )


prompt_2 = PromptTemplate(
                input_variables=["request", "code", "output"],
                template='''
                question : {request}
                code : {code}
                output : {output}

                Please provide a human-readable summary of this output keep it short. 
                Example 1:
                """
                question: What is the average of price?
                code: df['price'].mean
                output: 24.34
                
                llm output: The average price of 24.34
                """
                '''
            )

df = 0
def get_llm():
        return ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=GEMINI_API_KEY  # Replace with your API key
        )
llm = get_llm()
def callLLM(df, text, question):
    chain = LLMChain(llm=llm, prompt = text)
    try:
        result = chain.run(
                        request=question,
                        columns=", ".join(df.columns.tolist()),
                        sample_data=df.head(3).to_string()
                    )
        return result
    except:
        print("Some error happend while running command")
    pass

# ... keep your imports and setup the same until below st.set_page_config()

st.set_page_config(page_title="AI Data Analyst", page_icon="🤖", layout="wide")
st.title("💬 AI Data Analyst")

st.sidebar.header("📁 Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Choose a CSV file", type=["csv"])

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

df = pd.DataFrame()  # Define df globally

def get_llm():
    return ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=GEMINI_API_KEY)

llm = get_llm()

def callLLM(df, text, question):
    chain = LLMChain(llm=llm, prompt=text)
    try:
        result = chain.run(
            request=question,
            columns=", ".join(df.columns.tolist()),
            sample_data=df.head(3).to_string()
        )
        return result
    except:
        print("Some error happened while running command")
    return ""

def handle_send():
    user_input = st.session_state.user_input_key.strip()
    if not user_input:
        return

    result = callLLM(df, prompt_1, user_input)
    code_match = re.search(r'```python\n(.*?)```', result, re.DOTALL)

    bot_response = ""
    fig_object = None
    generated_code = ""

    exec_globals = {
        'df': df,
        'pd': pd,
        'plt': plt,
        'sns': sns,
        'px': px,
        'go': go,
        'st': st
    }

    if code_match:
        generated_code = code_match.group(1).strip()

        old_stdout = sys.stdout
        sys.stdout = buffer = io.StringIO()

        try:
            exec(generated_code, exec_globals)
        except Exception as e:
            sys.stdout = old_stdout
            error_trace = traceback.format_exc()
            bot_response = f"❌ Error executing code:\n```\n{error_trace}\n```"
        else:
            sys.stdout = old_stdout
            executed_output = buffer.getvalue()

            if 'plt.show()' in generated_code:
                fig_object = plt.gcf()
                plt.close()
                bot_response = "✅ Visualization rendered above."

            elif 'fig' in exec_globals:
                fig_object = exec_globals['fig']
                bot_response = "✅ Plot generated using Plotly."

            elif "__NEEDS_INTERPRETATION__" in executed_output:
                cleaned_output = executed_output.replace("__NEEDS_INTERPRETATION__", "").strip()
                chain = LLMChain(llm=llm, prompt=prompt_2)
                summary = chain.run(
                    request=user_input,
                    code=generated_code,
                    output=cleaned_output
                )
                bot_response = summary
            else:
                bot_response = executed_output.strip()
    else:
        bot_response = result

    st.session_state.chat_history.append({
        "user": user_input,
        "type": "plot" if fig_object else "text",
        "code": generated_code,
        "output": bot_response,
        "fig": fig_object
    })

    st.session_state.user_input_key = ""


# === MAIN INTERFACE ===
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    st.subheader("📊 Dataset Info")
    st.write(f"Shape: {df.shape}")
    st.write("Columns:", list(df.columns))

    st.markdown("---")
    st.subheader("🧠 Ask Anything About Your Data")

    chat_container = st.container()
    with chat_container:
        for msg in st.session_state.chat_history:
            with st.chat_message("user"):
                st.markdown(msg["user"])

            with st.chat_message("assistant"):
                if msg["type"] == "plot":
                    if msg["fig"]:
                        st.plotly_chart(msg["fig"], use_container_width=True)
                    elif 'plt.show()' in msg["code"]:
                        st.pyplot(plt.gcf())
                        plt.close()
                st.markdown(msg["output"])

    st.text_input("💬 Type your question here:", key="user_input_key")
    st.button("📨 Send", on_click=handle_send)

else:
    st.info("📌 Please upload a CSV file to begin.")
