from dotenv import load_dotenv
import pandas as pd
load_dotenv()
from sqlalchemy import create_engine
from langchain_community.utilities import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_community.agent_toolkits import SQLDatabaseToolkit, create_sql_agent
from langchain.agents import AgentExecutor

db = SQLDatabase.from_uri("sqlite:///disputes.db")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

toolkit = SQLDatabaseToolkit(db=db, llm=llm)
agent: AgentExecutor = create_sql_agent(
    llm=llm,
    toolkit=toolkit,
    agent_type="openai-tools",   # uses tool-calling; robust and up-to-date
    verbose=True,
)

def query_db(question: str):
    """Natural-language to SQL -> executes -> returns final answer."""
    return agent.invoke({"input": question})["output"]


def init_disputes_db(db_path: str = "disputes.db"):
    """
    Create a SQLite database and load the 4 CSVs as tables.
    Overwrites tables if they already exist.
    """
    engine = create_engine(f"sqlite:///{db_path}")

    # Load each CSV into its corresponding table
    from pathlib import Path
    csv_dir = Path("csv")
    
    try:
        classified_disputes = pd.read_csv(csv_dir / "classified_disputes.csv")
        classified_disputes.to_sql("classified_disputes", engine, if_exists="replace", index=False)
    except FileNotFoundError:
        print("classified_disputes.csv not found - run Task 1 first")

    disputes = pd.read_csv(csv_dir / "disputes.csv")
    disputes.to_sql("disputes", engine, if_exists="replace", index=False)

    try:
        resolutions = pd.read_csv(csv_dir / "resolutions.csv")
        resolutions.to_sql("resolutions", engine, if_exists="replace", index=False)
    except FileNotFoundError:
        print("resolutions.csv not found - run Task 2 first")

    transactions = pd.read_csv(csv_dir / "transactions.csv")
    transactions.to_sql("transactions", engine, if_exists="replace", index=False)

    return "Completed"