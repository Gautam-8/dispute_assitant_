from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

prompt = PromptTemplate(
    input_variables=["description","status"],
    template=(
        "Classify this payment dispute into one of "
        "[DUPLICATE_CHARGE, FAILED_TRANSACTION, FRAUD, REFUND_PENDING, OTHERS]. "
        "Give category only.\n"
        "Description: {description}\nTransaction status: {status}"
    )
)

parser = StrOutputParser()
chain = prompt | llm | parser


def llm_classify(row):
    resp = chain.invoke({"description": row['description'], "status": row['status']})
    return resp



# HARD CODED CLASSIFIER

def hard_code_classifier(row):
    desc = row['description'].lower()
    status = row['status'].lower()

    if "duplicate" in desc or "duplicate" in status:
        return "DUPLICATE_CHARGE"
    elif "failed" in desc or "failed" in status or "failure" in desc or "failure" in status:
        return "FAILED_TRANSACTION"
    elif "fraud" in desc or "fraud" in status or "unauthorized" in desc or "unauthorized" in status:
        return "FRAUD"
    elif "refund" in desc or "refund" in status or "refunded" in desc or "refunded" in status or "cancelled transaction" in desc or "cancelled transaction" in status:
        return "REFUND_PENDING"
    else:
        return "OTHERS"