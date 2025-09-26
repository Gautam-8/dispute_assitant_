from pathlib import Path
import re
import pandas as pd
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate

load_dotenv()  # loads OPENAI_API_KEY

# --- Load all CSVs into a dict of DataFrames ---
BASE_DIR = Path(__file__).resolve().parents[1]   # project root
CSV_DIR  = BASE_DIR / "csv"

tables = {}
schema_info = []
for csv_file in CSV_DIR.glob("*.csv"):
    df = pd.read_csv(csv_file)
    name = csv_file.stem
    tables[name] = df
    cols = ", ".join(df.columns)
    schema_info.append(f"Table {name}: columns = [{cols}]")

schema_text = "\n".join(schema_info)

# --- LLM setup ---
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# Prompt template for pure pandas code
prompt_template = ChatPromptTemplate.from_template(
    """
    You are an expert Python data analyst.
    I have the following pandas DataFrames already loaded in a dict named `tables`:
    {schema}

    Some transaction IDs may contain duplicates indicated by a suffix like '_DUP1'
    (e.g., T001_DUP1). Treat those as duplicate transactions if the user asks.

    Write ONLY the Python pandas code (no explanations) that answers the question below.
    Your code should:
    - Use the `tables` dict to access the DataFrames, e.g. tables["disputes"].
    - Print the final answer or DataFrame as the last line.
    - Avoid plotting; only textual pandas operations.

    Question: {question}
    """
)

def ask_llm_for_code(question: str) -> str:
    """Ask the LLM to generate pandas code for a natural-language question."""
    prompt = prompt_template.format(schema=schema_text, question=question)
    raw = llm.invoke(prompt).content.strip()

    # --- Clean markdown/code fences if present ---
    cleaned = re.sub(r"```(?:python)?", "", raw, flags=re.IGNORECASE)
    cleaned = cleaned.replace("```", "").strip()

    # Optional basic safety check
    if any(forbidden in cleaned.lower() for forbidden in ["import ", "os.", "sys.", "subprocess"]):
        raise ValueError("⚠️ Unsafe code detected.")
    return cleaned

def run_generated_code(code: str):
    """Execute generated pandas code with our tables dict in a restricted namespace."""
    local_env = {"tables": tables, "pd": pd}
    try:
        exec(code, {}, local_env)
    except Exception as e:
        print("⚠️ Error executing code:", e)
    return local_env.get("result")


def query_pandas(q : str):
    d = {}
    q = q.strip()
    code = ask_llm_for_code(q)
    d["code"] = code
    result = run_generated_code(code)
    d["result"] = result
    return d
