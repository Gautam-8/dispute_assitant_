from pathlib import Path
from fastapi import FastAPI
import pandas as pd
from database import engine
from models import Base
from tasks.task1 import hard_code_classifier, llm_classify
from tasks.task2 import suggest_action
from tasks.task3 import query_db, init_disputes_db
from tasks.task3_pandas import query_pandas

app = FastAPI()

# ---------- DB Models ----------
Base.metadata.create_all(bind=engine)

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent      # directory containing this file
CSV_DIR  = BASE_DIR / "csv"                     # csv folder at project root

# ---------- Load Input CSVs ----------
# Always load disputes + transactions from csv/ folder
disputes = pd.read_csv(CSV_DIR / "disputes.csv")
transactions = pd.read_csv(CSV_DIR / "transactions.csv")

# Merge for enriched data
data = disputes.merge(
    transactions[["txn_id", "status"]],
    on="txn_id",
    how="left",
    suffixes=("_disp", "_txn")
)

@app.get("/")
async def read_root():
    return {"Hello": "World"}

# ---------- Task 1 ----------
@app.get("/complete_task1")
def task1():
    data["predicted_category"] = data.apply(hard_code_classifier, axis=1)
    data["confidence"] = 0.9
    data["explanation"] = "Matched hard-coded rules"

    low_conf_mask = data["predicted_category"] == "OTHERS"
    data.loc[low_conf_mask, "predicted_category"] = data[low_conf_mask].apply(
        llm_classify, axis=1
    )
    data.loc[low_conf_mask, "confidence"] = 0.6
    data.loc[low_conf_mask, "explanation"] = "LLM classification"

    out_path = CSV_DIR / "classified_disputes.csv"
    data[["dispute_id", "predicted_category", "confidence", "explanation"]].to_csv(
        out_path, index=False
    )
    print(f"✅ classified_disputes.csv created at {out_path}")
    return "Completed Task1"

# ---------- Task 2 ----------
@app.get("/complete_task2")
def task2():
    data[["suggested_action", "justification"]] = data.apply(
        lambda r: pd.Series(suggest_action(r)), axis=1
    )

    out_path = CSV_DIR / "resolutions.csv"
    data[["dispute_id", "suggested_action", "justification"]].to_csv(
        out_path, index=False
    )
    print(f"✅ resolutions.csv created at {out_path}")
    return "Completed Task2"

# ---------- Task 3 : LangChain SQL Queries ----------
@app.get("/push_data")
def task3_push_data():
    result = init_disputes_db()
    return result

@app.post("/query/{query}")
def task3_query(query: str):
    response = query_db(query)
    return response

# ---------- Task 3 : Pandas LLM Queries ----------
@app.post("/query_pandas/{query}")
def task3_pandas_query(query: str):
    response = query_pandas(query)
    return response
