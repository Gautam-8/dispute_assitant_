"""
Dynamic query processor that works with session state data instead of fixed CSV files
"""
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import re

load_dotenv()

class DynamicQueryProcessor:
    def __init__(self):
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
    def get_tables_from_session(self):
        """Get tables dictionary from Streamlit session state"""
        tables = {}
        schema_info = []
        
        # Get disputes data
        if 'disputes' in st.session_state and not st.session_state.disputes.empty:
            disputes_df = st.session_state.disputes
            tables['disputes'] = disputes_df
            cols = ", ".join(disputes_df.columns)
            schema_info.append(f"Table disputes: columns = [{cols}]")
        
        # Get transactions data
        if 'transactions' in st.session_state and not st.session_state.transactions.empty:
            transactions_df = st.session_state.transactions
            tables['transactions'] = transactions_df
            cols = ", ".join(transactions_df.columns)
            schema_info.append(f"Table transactions: columns = [{cols}]")
        
        # Get merged data
        if 'merged_data' in st.session_state and not st.session_state.merged_data.empty:
            merged_df = st.session_state.merged_data
            tables['merged_data'] = merged_df
            cols = ", ".join(merged_df.columns)
            schema_info.append(f"Table merged_data: columns = [{cols}] (disputes + transactions joined)")
        
        # Try to load classification and resolution results if available
        try:
            from pathlib import Path
            csv_dir = Path("csv")
            
            if (csv_dir / "classified_disputes.csv").exists():
                classified = pd.read_csv(csv_dir / "classified_disputes.csv")
                tables['classified_disputes'] = classified
                cols = ", ".join(classified.columns)
                schema_info.append(f"Table classified_disputes: columns = [{cols}]")
            
            if (csv_dir / "resolutions.csv").exists():
                resolutions = pd.read_csv(csv_dir / "resolutions.csv")
                tables['resolutions'] = resolutions
                cols = ", ".join(resolutions.columns)
                schema_info.append(f"Table resolutions: columns = [{cols}]")
                
        except Exception:
            pass  # Ignore if files don't exist
        
        schema_text = "\n".join(schema_info)
        return tables, schema_text
    
    def get_prompt_template(self, schema_text):
        """Create the prompt template with current schema"""
        return ChatPromptTemplate.from_template(
            f"""
            You are an expert Python data analyst.
            I have the following pandas DataFrames already loaded in a dict named `tables`:
            {schema_text}

            Some transaction IDs may contain duplicates indicated by a suffix like '_DUP1'
            (e.g., T001_DUP1). Treat those as duplicate transactions if the user asks.

            Write ONLY the Python pandas code (no explanations) that answers the question below.
            Your code should:
            - Use the `tables` dict to access the DataFrames, e.g. tables["disputes"], tables["transactions"], tables["merged_data"].
            - Print the final answer or DataFrame as the last line.
            - Avoid plotting; only textual pandas operations.
            - Handle empty DataFrames gracefully with try/except if needed.

            Question: {{question}}
            """
        )
    
    def ask_llm_for_code(self, question: str) -> str:
        """Ask the LLM to generate pandas code for a natural-language question."""
        tables, schema_text = self.get_tables_from_session()
        
        if not tables:
            return "print('No data available. Please upload a dataset first.')"
        
        prompt_template = self.get_prompt_template(schema_text)
        prompt = prompt_template.format(question=question)
        raw = self.llm.invoke(prompt).content.strip()

        # Clean markdown/code fences if present
        cleaned = re.sub(r"```(?:python)?", "", raw, flags=re.IGNORECASE)
        cleaned = cleaned.replace("```", "").strip()

        # Optional basic safety check
        if any(forbidden in cleaned.lower() for forbidden in ["import ", "os.", "sys.", "subprocess"]):
            raise ValueError("⚠️ Unsafe code detected.")
        return cleaned

    def run_generated_code(self, code: str):
        """Execute generated pandas code with our tables dict in a restricted namespace."""
        tables, _ = self.get_tables_from_session()
        
        local_env = {"tables": tables, "pd": pd}
        try:
            exec(code, {}, local_env)
        except Exception as e:
            print(f"⚠️ Error executing code: {e}")
        return local_env.get("result")

    def query_dynamic(self, question: str):
        """Process a natural language query against current session data"""
        result = {}
        question = question.strip()
        code = self.ask_llm_for_code(question)
        result["code"] = code
        output = self.run_generated_code(code)
        result["result"] = output
        return result

# Global instance
dynamic_query_processor = DynamicQueryProcessor()
