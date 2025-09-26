# AI-Powered Dispute Assistant

An AI system for classifying payment disputes and suggesting resolutions.

## Overview

This application helps resolve payment disputes by:

- Classifying disputes into categories
- Suggesting appropriate resolutions
- Providing natural language query interface
- Visualizing dispute trends and analytics

## Files

- `app.py` - Main Streamlit application
- `requirements.txt` - Dependencies
- `tasks/` - Task modules for classification, resolution, and querying
- `csv/` - Data files (disputes.csv, transactions.csv, outputs)

## Setup

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Create `.env` file:

```
OPENAI_API_KEY=your_openai_api_key_here
```

3. Run the app:

```bash
streamlit run app.py
```

## Features

- **Upload Data**: Upload disputes.csv and transactions.csv files
- **Task 1**: Classify disputes into categories with confidence scores
- **Task 2**: Generate resolution suggestions with justifications
- **Task 3**: Natural language query interface
- **Analytics**: Visualize dispute trends and patterns

## Outputs

- `classified_disputes.csv` - Classification results
- `resolutions.csv` - Resolution suggestions
