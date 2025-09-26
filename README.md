# 🚀 AI-Powered Dispute Assistant (Dynamic Version)

An intelligent system for classifying payment disputes, suggesting resolutions, and providing natural language insights for financial support teams. **Now with dynamic file upload and dataset management!**

## 📋 Overview

This application helps resolve payment disputes for fintechs and banks by:

- **📁 Dynamic File Upload**: Upload disputes.csv and transactions.csv files
- **🗂️ Dataset History Management**: Store and switch between multiple datasets
- **🎯 Classifying disputes** into categories (DUPLICATE_CHARGE, FAILED_TRANSACTION, FRAUD, REFUND_PENDING, OTHERS)
- **💡 Suggesting appropriate resolutions** (Auto-refund, Manual review, Escalate to bank, etc.)
- **🔍 Natural language query interface** for support agents
- **📈 Visualizing dispute trends** and analytics with real-time data

## 🏗️ Architecture

- **Frontend**: Streamlit web application
- **Backend Logic**: Python modules for classification and resolution
- **AI Integration**: OpenAI GPT-4o-mini with LangChain
- **Database**: SQLite for queries and analytics
- **Data Processing**: Pandas for CSV analysis

## 📁 Project Structure

```
dispute_assistant/
├── app.py                      # Main Streamlit application
├── main.py                     # Original FastAPI (deprecated)
├── requirements.txt            # Dependencies
├── database.py                 # Database configuration
├── models.py                   # SQLAlchemy models
├── csv/                        # Data files
│   ├── disputes.csv           # Input: Customer disputes
│   ├── transactions.csv       # Input: Transaction data
│   ├── classified_disputes.csv # Output: Task 1 results
│   └── resolutions.csv        # Output: Task 2 results
└── tasks/
    ├── task1.py               # Dispute classification logic
    ├── task2.py               # Resolution suggestion logic
    ├── task3.py               # SQL-based query interface
    └── task3_pandas.py        # Pandas-based query interface
```

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.8+
- OpenAI API key

### 2. Installation

```bash
# Clone the repository
git clone <repository-url>
cd dispute_assistant

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Setup

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

### 4. Run the Application

```bash
streamlit run app.py
```

The application will open in your browser at `http://localhost:8501`

## 🎯 Features

### 📊 Dashboard

- **Key Metrics**: Total disputes, amounts, duplicate transactions
- **Visual Analytics**: Status distribution, channel breakdown, payment types
- **Recent Activity**: Latest disputes table

### 🎯 Task 1: Dispute Classification

- **Rule-based Classifier**: Fast classification using keywords
- **LLM Classifier**: OpenAI GPT-4o-mini for ambiguous cases
- **Confidence Scoring**: Reliability metrics for each classification
- **Export Results**: Download `classified_disputes.csv`

### 💡 Task 2: Resolution Suggestions

- **Automated Actions**: Auto-refund, Manual review, Escalate to bank
- **Justifications**: Clear reasoning for each suggested action
- **Distribution Analysis**: Visual breakdown of suggested actions
- **Export Results**: Download `resolutions.csv`

### 🔍 Task 3: Query Interface

- **Natural Language Queries**: Ask questions in plain English
- **Dual Query Engines**:
  - Pandas (faster, direct CSV analysis)
  - SQL (database queries with LangChain)
- **Pre-built Examples**: Common queries ready to use
- **Code Transparency**: View generated Python/SQL code

### 📈 Analytics & Trends (Bonus Features)

- **Time Series Analysis**: Dispute volume trends over time
- **Classification Trends**: Category patterns by date
- **Heatmaps**: Day-of-week analysis
- **Duplicate Detection**: Advanced fuzzy matching
- **Performance Metrics**: System accuracy and auto-resolution rates

## 💡 Usage Examples

### Classification Queries

```
"How many duplicate charges today?"
"List unresolved fraud disputes"
"Break down disputes by type"
```

### Analytics Queries

```
"What's the total amount for failed transactions?"
"Show me all disputes from mobile channel"
"Count disputes by customer"
```

## 🔧 Technical Details

### Classification Logic

1. **Rule-based**: Keyword matching for common patterns
2. **LLM Fallback**: GPT-4o-mini for edge cases
3. **Confidence Scoring**: Rule-based (0.9) vs LLM (0.6)

### Resolution Engine

- Maps categories to actions based on business rules
- Considers transaction status and dispute context
- Provides justification for transparency

### Query Processing

- **Pandas Mode**: Direct DataFrame operations for speed
- **SQL Mode**: LangChain SQL agent for complex queries
- **Duplicate Detection**: Identifies transactions with `_DUP` suffixes

## 📈 Performance

- **Classification Accuracy**: Tracks high-confidence predictions
- **Auto-Resolution Rate**: Percentage of automatically resolvable disputes
- **Query Response Time**: Optimized for real-time insights

## 🛠️ Development

### Adding New Features

1. **New Classification Categories**: Update `task1.py` hard_code_classifier
2. **New Resolution Actions**: Modify `task2.py` suggest_action
3. **Custom Queries**: Extend query examples in `app.py`

### Database Schema

The application uses these main tables:

- `disputes`: Customer dispute records
- `transactions`: Payment transaction data
- `classified_disputes`: Classification results
- `resolutions`: Resolution suggestions

## 🔒 Security

- Environment variables for API keys
- Input validation for queries
- Safe code execution with restricted namespaces

## 📝 Assignment Compliance

✅ **Task 1**: Dispute classification with ML/rule-based models  
✅ **Task 2**: Resolution suggestions with justifications  
✅ **Task 3**: CLI/notebook interface with prompt handling  
✅ **Bonus**: Duplicate detection, visualizations, case history  
✅ **Outputs**: All required CSV files generated  
✅ **Interface**: Modern web UI replacing CLI

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

---

**Built with ❤️ using Streamlit, OpenAI, and Python**
