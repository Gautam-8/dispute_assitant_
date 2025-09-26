import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
import sys
import io

# Add tasks to path for imports
sys.path.append(str(Path(__file__).parent / "tasks"))

from task1 import hard_code_classifier, llm_classify
from task2 import suggest_action
from task3 import query_db, init_disputes_db
from task3_pandas import query_pandas
from dataset_manager import dataset_manager
from database import engine
from models import Base
from dynamic_query import dynamic_query_processor

# ---------- Configuration ----------
st.set_page_config(
    page_title="AI-Powered Dispute Assistant",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize database tables
Base.metadata.create_all(bind=engine)

# ---------- Paths ----------
BASE_DIR = Path(__file__).resolve().parent
CSV_DIR = BASE_DIR / "csv"

# ---------- Dynamic Data Loading ----------
def load_current_dataset():
    """Load current active dataset from database only"""
    current_dataset = dataset_manager.get_current_dataset()
    
    if current_dataset:
        # Load from database
        disputes_df, transactions_df = dataset_manager.load_dataset_by_id(current_dataset['id'])
        
        # Convert timestamps
        if not disputes_df.empty:
            disputes_df['created_at'] = pd.to_datetime(disputes_df['created_at'])
        if not transactions_df.empty:
            transactions_df['timestamp'] = pd.to_datetime(transactions_df['timestamp'])
        
        return disputes_df, transactions_df, current_dataset
    else:
        # Return empty DataFrames if no dataset uploaded
        return pd.DataFrame(), pd.DataFrame(), None

def merge_data(disputes_df, transactions_df):
    """Merge disputes and transactions data"""
    if disputes_df.empty or transactions_df.empty:
        return pd.DataFrame()
    
    merged = disputes_df.merge(
        transactions_df[["txn_id", "status"]],
        on="txn_id",
        how="left",
        suffixes=("_disp", "_txn")
    )
    return merged

# Load current dataset
if 'current_dataset_id' not in st.session_state:
    st.session_state.current_dataset_id = None

disputes, transactions, current_dataset_info = load_current_dataset()
merged_data = merge_data(disputes, transactions)

# Store in session state for consistency
st.session_state.disputes = disputes
st.session_state.transactions = transactions
st.session_state.merged_data = merged_data
st.session_state.current_dataset_info = current_dataset_info

# Load results (if available)
@st.cache_data
def load_results():
    """Load classification and resolution results if available"""
    try:
        classified = pd.read_csv(CSV_DIR / "classified_disputes.csv")
        resolutions = pd.read_csv(CSV_DIR / "resolutions.csv")
        return classified, resolutions
    except FileNotFoundError:
        return None, None

classified_disputes, resolutions = load_results()

# ---------- Sidebar ----------
st.sidebar.title("🔧 Navigation")

# Dataset Management Section
st.sidebar.markdown("---")
st.sidebar.subheader("📁 Dataset Management")

# Current dataset info
if current_dataset_info:
    st.sidebar.success(f"🟢 Active: {current_dataset_info['name']}")
    st.sidebar.caption(f"📊 {current_dataset_info['disputes_count']} disputes, {current_dataset_info['transactions_count']} transactions")
    st.sidebar.caption(f"📅 {current_dataset_info['upload_time'].strftime('%Y-%m-%d %H:%M')}")
else:
    st.sidebar.warning("⚠️ No data loaded - Please upload a dataset")

# Dataset history
dataset_history = dataset_manager.get_dataset_history()
if dataset_history:
    st.sidebar.markdown("**📚 Dataset History:**")
    for dataset in dataset_history[:3]:  # Show last 3
        status = "🟢" if dataset['is_current'] else "⚪"
        if st.sidebar.button(f"{status} {dataset['name']}", key=f"dataset_{dataset['id']}"):
            dataset_manager.set_current_dataset(dataset['id'])
            st.rerun()

# Page selection
st.sidebar.markdown("---")
page = st.sidebar.selectbox(
    "Choose a page:",
    ["📊 Dashboard", "📁 Upload Data", "🎯 Task 1: Classification", "💡 Task 2: Resolutions", "🔍 Task 3: Query Interface", "📈 Analytics & Trends", "🗂️ Dataset History"]
)

# ---------- Helper Functions ----------
def create_dispute_metrics():
    """Create key metrics for dashboard"""
    if disputes.empty:
        return 0, 0, 0, pd.Series(dtype=int)
    
    total_disputes = len(disputes)
    total_amount = disputes['amount'].sum() if 'amount' in disputes.columns else 0
    avg_amount = disputes['amount'].mean() if 'amount' in disputes.columns else 0
    
    # Status breakdown from transactions
    if merged_data.empty:
        status_counts = pd.Series(dtype=int)
    else:
        status_counts = merged_data['status'].value_counts()
    
    return total_disputes, total_amount, avg_amount, status_counts

def detect_duplicates():
    """Detect duplicate transactions"""
    if transactions.empty:
        return pd.DataFrame()
    duplicate_txns = transactions[transactions['txn_id'].str.contains('_DUP', na=False)]
    return duplicate_txns

# ---------- DASHBOARD PAGE ----------
if page == "📊 Dashboard":
    st.title("⚖️ AI-Powered Dispute Assistant Dashboard")
    st.markdown("---")
    
    # Key Metrics
    total_disputes, total_amount, avg_amount, status_counts = create_dispute_metrics()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Disputes", total_disputes)
    
    with col2:
        duplicate_count = len(detect_duplicates())
        st.metric("Duplicate Transactions", duplicate_count)
    
    with col3:
        if not transactions.empty:
            failed_txns = len(transactions[transactions['status'] == 'FAILED'])
            st.metric("Failed Transactions", failed_txns)
        else:
            st.metric("Failed Transactions", 0)
    
    st.markdown("---")
    
    # Charts Row 1
    if not disputes.empty and not status_counts.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Transaction Status Distribution")
            fig_status = px.pie(
                values=status_counts.values,
                names=status_counts.index,
                title="Transaction Status Breakdown"
            )
            st.plotly_chart(fig_status, use_container_width=True, key="status_chart")
    
        with col2:
            st.subheader("Disputes by Channel")
            channel_counts = disputes['channel'].value_counts()
            fig_channel = px.bar(
                x=channel_counts.index,
                y=channel_counts.values,
                title="Disputes by Channel",
                labels={'x': 'Channel', 'y': 'Count'}
            )
            st.plotly_chart(fig_channel, use_container_width=True, key="dashboard_channel_chart")
        
        # Charts Row 2
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Payment Type Distribution")
            txn_type_counts = disputes['txn_type'].value_counts()
            fig_txn_type = px.pie(
                values=txn_type_counts.values,
                names=txn_type_counts.index,
                title="Payment Types"
            )
            st.plotly_chart(fig_txn_type, use_container_width=True, key="dashboard_txn_type_chart")
        
        with col2:
            st.subheader("Disputes by Channel")
            channel_counts = disputes['channel'].value_counts()
            fig_channel = px.bar(
                x=channel_counts.index,
                y=channel_counts.values,
                title="Disputes by Channel",
                labels={'x': 'Channel', 'y': 'Count'}
            )
            st.plotly_chart(fig_channel, use_container_width=True, key="channel_chart")
    else:
        st.info("📊 No data available. Please upload a dataset to see dashboard analytics.")
    
    # Recent Disputes Table
    st.subheader("Recent Disputes")
    if not disputes.empty:
        recent_disputes = disputes.sort_values('created_at', ascending=False).head(10)
        st.dataframe(recent_disputes, use_container_width=True)
    else:
        st.info("No disputes data available. Please upload data using the Upload Data page.")

# ---------- UPLOAD DATA PAGE ----------
elif page == "📁 Upload Data":
    st.title("📁 Upload New Dataset")
    st.markdown("Upload disputes.csv and transactions.csv files to create a new dataset.")
    st.markdown("---")
    
    with st.form("upload_form"):
        st.subheader("📋 Dataset Information")
        
        col1, col2 = st.columns(2)
        with col1:
            dataset_name = st.text_input(
                "Dataset Name *",
                placeholder="e.g., Q3_2024_Disputes",
                help="Give your dataset a meaningful name"
            )
        
        with col2:
            description = st.text_area(
                "Description (Optional)",
                placeholder="Brief description of this dataset...",
                height=100
            )
        
        st.subheader("📄 File Uploads")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Disputes CSV File**")
            disputes_file = st.file_uploader(
                "Choose disputes.csv file",
                type=['csv'],
                help="Must contain: dispute_id, customer_id, txn_id, description, txn_type, channel, amount, created_at"
            )
            
            if disputes_file:
                try:
                    disputes_df = pd.read_csv(disputes_file)
                    st.success(f"✅ Loaded {len(disputes_df)} disputes")
                    
                    # Validate disputes CSV
                    is_valid, error_msg = dataset_manager.validate_disputes_csv(disputes_df)
                    if is_valid:
                        st.success("✅ Disputes CSV is valid")
                        with st.expander("Preview disputes data"):
                            st.dataframe(disputes_df.head(), use_container_width=True)
                    else:
                        st.error(f"❌ {error_msg}")
                        disputes_df = None
                        
                except Exception as e:
                    st.error(f"❌ Error reading disputes file: {e}")
                    disputes_df = None
            else:
                disputes_df = None
        
        with col2:
            st.markdown("**Transactions CSV File**")
            transactions_file = st.file_uploader(
                "Choose transactions.csv file",
                type=['csv'],
                help="Must contain: txn_id, customer_id, amount, status, timestamp, channel, merchant"
            )
            
            if transactions_file:
                try:
                    transactions_df = pd.read_csv(transactions_file)
                    st.success(f"✅ Loaded {len(transactions_df)} transactions")
                    
                    # Validate transactions CSV
                    is_valid, error_msg = dataset_manager.validate_transactions_csv(transactions_df)
                    if is_valid:
                        st.success("✅ Transactions CSV is valid")
                        with st.expander("Preview transactions data"):
                            st.dataframe(transactions_df.head(), use_container_width=True)
                    else:
                        st.error(f"❌ {error_msg}")
                        transactions_df = None
                        
                except Exception as e:
                    st.error(f"❌ Error reading transactions file: {e}")
                    transactions_df = None
            else:
                transactions_df = None
        
        # Submit button
        submit_button = st.form_submit_button("🚀 Upload Dataset", type="primary")
        
        if submit_button:
            if not dataset_name.strip():
                st.error("❌ Please provide a dataset name")
            elif disputes_df is None:
                st.error("❌ Please upload a valid disputes CSV file")
            elif transactions_df is None:
                st.error("❌ Please upload a valid transactions CSV file")
            else:
                try:
                    with st.spinner("Saving dataset to database..."):
                        dataset_id = dataset_manager.save_dataset_to_db(
                            disputes_df, transactions_df, dataset_name.strip(), description.strip()
                        )
                    
                    st.success(f"🎉 Dataset '{dataset_name}' uploaded successfully!")
                    st.success("🔄 This dataset is now active. Refreshing page...")
                    
                    # Clear cache and rerun
                    st.cache_data.clear()
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error saving dataset: {e}")
    
    # Show current dataset info
    if current_dataset_info:
        st.markdown("---")
        st.subheader("📊 Current Active Dataset")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Dataset Name", current_dataset_info['name'])
        with col2:
            st.metric("Disputes", current_dataset_info['disputes_count'])
        with col3:
            st.metric("Transactions", current_dataset_info['transactions_count'])
        
        st.caption(f"📅 Uploaded: {current_dataset_info['upload_time'].strftime('%Y-%m-%d %H:%M:%S')}")
        if current_dataset_info['description']:
            st.caption(f"📝 {current_dataset_info['description']}")

# ---------- TASK 1 PAGE ----------
elif page == "🎯 Task 1: Classification":
    st.title("🎯 Task 1: Dispute Classification")
    st.markdown("Classify disputes into categories using.")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🔧 Actions")
        
        if st.button("🚀 Run Classification", type="primary"):
            if merged_data.empty:
                st.error("❌ No data available for classification. Please upload a dataset first.")
            else:
                with st.spinner("Classifying disputes..."):
                    # Work with current session data
                    current_merged = st.session_state.merged_data.copy()
                    
                    # Apply classifications
                    current_merged["predicted_category"] = current_merged.apply(hard_code_classifier, axis=1)
                    current_merged["confidence"] = 0.9
                    current_merged["explanation"] = "Matched hard-coded rules"

                    # Use LLM for OTHERS category
                    low_conf_mask = current_merged["predicted_category"] == "OTHERS"
                    if low_conf_mask.any():
                        st.info("Using LLM for ambiguous cases...")
                        current_merged.loc[low_conf_mask, "predicted_category"] = current_merged[low_conf_mask].apply(
                            llm_classify, axis=1
                        )
                        current_merged.loc[low_conf_mask, "confidence"] = 0.6
                        current_merged.loc[low_conf_mask, "explanation"] = "LLM classification"

                    # Save results
                    out_path = CSV_DIR / "classified_disputes.csv"
                    current_merged[["dispute_id", "predicted_category", "confidence", "explanation"]].to_csv(
                        out_path, index=False
                    )
                    
                    # Update session state
                    st.session_state.classified_data = current_merged
                    
                    st.success(f"✅ Classification completed! Results saved to {out_path}")
                    st.rerun()
    
    with col1:
        st.subheader("📊 Classification Results")
        
        if classified_disputes is not None:
            # Classification distribution
            category_counts = classified_disputes['predicted_category'].value_counts()
            
            fig = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title="Dispute Categories",
                labels={'x': 'Category', 'y': 'Count'},
                color=category_counts.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig, use_container_width=True, key="classification_bar_chart")
            
            # Confidence distribution
            fig_conf = px.histogram(
                classified_disputes,
                x='confidence',
                title="Classification Confidence Distribution",
                nbins=10
            )
            st.plotly_chart(fig_conf, use_container_width=True, key="confidence_histogram")
            
        else:
            st.info("Run classification to see results here.")
    
    # Show classification results table
    if classified_disputes is not None:
        st.subheader("📋 Classified Disputes")
        
        # Filter options
        col1, col2 = st.columns(2)
        with col1:
            category_filter = st.selectbox(
                "Filter by Category:",
                ["All"] + list(classified_disputes['predicted_category'].unique())
            )
        
        with col2:
            confidence_filter = st.slider(
                "Minimum Confidence:",
                0.0, 1.0, 0.0, 0.1
            )
        
        # Apply filters
        filtered_data = classified_disputes.copy()
        if category_filter != "All":
            filtered_data = filtered_data[filtered_data['predicted_category'] == category_filter]
        filtered_data = filtered_data[filtered_data['confidence'] >= confidence_filter]
        
        st.dataframe(filtered_data, use_container_width=True)
        
        # Download button
        csv = filtered_data.to_csv(index=False)
        st.download_button(
            label="📥 Download Classifications",
            data=csv,
            file_name="classified_disputes.csv",
            mime="text/csv"
        )

# ---------- TASK 2 PAGE ----------
elif page == "💡 Task 2: Resolutions":
    st.title("💡 Task 2: Resolution Suggestions")
    st.markdown("Generate action suggestions based on dispute classifications.")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        st.subheader("🔧 Actions")
        
        # Check if we have classified data (either from file or session state)
        has_classification = (classified_disputes is not None) or ('classified_data' in st.session_state)
        
        if has_classification:
            if st.button("🚀 Generate Resolutions", type="primary"):
                if merged_data.empty:
                    st.error("❌ No data available for resolution suggestions. Please upload a dataset first.")
                else:
                    with st.spinner("Generating resolution suggestions..."):
                        # Use classified data from session state if available, otherwise from CSV
                        if 'classified_data' in st.session_state:
                            resolution_data = st.session_state.classified_data.copy()
                        else:
                            resolution_data = merged_data.copy()
                            if 'predicted_category' not in resolution_data.columns:
                                # Load from classified disputes CSV
                                resolution_data = resolution_data.merge(
                                    classified_disputes[['dispute_id', 'predicted_category']], 
                                    on='dispute_id'
                                )
                        
                        resolution_data[["suggested_action", "justification"]] = resolution_data.apply(
                            lambda r: pd.Series(suggest_action(r)), axis=1
                        )

                        # Save results
                        out_path = CSV_DIR / "resolutions.csv"
                        resolution_data[["dispute_id", "suggested_action", "justification"]].to_csv(
                            out_path, index=False
                        )
                        
                        # Update session state
                        st.session_state.resolution_data = resolution_data
                        
                        st.success(f"✅ Resolutions generated! Results saved to {out_path}")
                        st.rerun()
        else:
            st.warning("⚠️ Please run Task 1 (Classification) first!")
    
    with col1:
        st.subheader("📊 Resolution Analysis")
        
        if resolutions is not None:
            # Action distribution
            action_counts = resolutions['suggested_action'].value_counts()
            
            fig = px.pie(
                values=action_counts.values,
                names=action_counts.index,
                title="Suggested Actions Distribution"
            )
            st.plotly_chart(fig, use_container_width=True, key="resolution_pie_chart")
            
        else:
            st.info("Generate resolutions to see analysis here.")
    
    # Show resolution results
    if resolutions is not None:
        st.subheader("📋 Resolution Suggestions")
        
        # Filter options
        action_filter = st.selectbox(
            "Filter by Action:",
            ["All"] + list(resolutions['suggested_action'].unique())
        )
        
        # Apply filter
        filtered_resolutions = resolutions.copy()
        if action_filter != "All":
            filtered_resolutions = filtered_resolutions[filtered_resolutions['suggested_action'] == action_filter]
        
        st.dataframe(filtered_resolutions, use_container_width=True)
        
        # Download button
        csv = filtered_resolutions.to_csv(index=False)
        st.download_button(
            label="📥 Download Resolutions",
            data=csv,
            file_name="resolutions.csv",
            mime="text/csv"
        )

# ---------- TASK 3 PAGE ----------
elif page == "🔍 Task 3: Query Interface":
    st.title("🔍 Task 3: Natural Language Query Interface")
    st.markdown("Ask questions about disputes using natural language.")
    st.markdown("---")
    
    # Initialize database button
    col1, col2 = st.columns([3, 1])
    
    with col2:
        if st.button("🔄 Initialize Database", help="Load CSV data into SQLite database"):
            with st.spinner("Loading data into database..."):
                try:
                    result = init_disputes_db()
                    st.success("✅ Database initialized successfully!")
                except Exception as e:
                    st.error(f"❌ Error initializing database: {e}")
    
    # Query interface
    st.subheader("💬 Ask Questions")
    
    # Pre-defined example queries
    st.markdown("**Try these example queries:**")
    examples = [
        "How many duplicate charges today?",
        "List unresolved fraud disputes",
        "Break down disputes by type",
        "What's the total count for failed transactions?",
        "Show me all disputes from mobile channel",
        "Count disputes by customer"
    ]
    
    cols = st.columns(3)
    for i, example in enumerate(examples):
        if cols[i % 3].button(f"📝 {example}", key=f"example_{i}"):
            st.session_state.query_text = example
    
    # Query input
    query = st.text_area(
        "Enter your question:",
        value=st.session_state.get('query_text', ''),
        placeholder="e.g., How many duplicate charges today?",
        height=100
    )
    
    # Query method selection
    st.markdown("**Choose Query Method:**")
    col1, col2 = st.columns(2)
    
    with col1:
        query_method = st.radio(
            "Query Method:",
            ["🐼 Pandas (Faster)", "🗄️ SQL Database"],
            help="Pandas: Direct CSV analysis (faster). SQL: Database queries (more robust)"
        )
    
    with col2:
        if query_method == "🐼 Pandas (Faster)":
            st.info("🐼 **Pandas**: Fast, direct analysis. Good for simple queries.")
        else:
            st.info("🗄️ **SQL**: Database queries. Better for complex analysis.")
    
    # Run query button
    if st.button("🔍 Run Query", type="primary", disabled=not query.strip()):
        if query.strip():
            with st.spinner("Processing your question..."):
                try:
                    if query_method == "🐼 Pandas (Faster)":
                        # Use dynamic query processor that works with session state
                        result = dynamic_query_processor.query_dynamic(query)
                        st.session_state.query_result = result
                        st.session_state.query_method = query_method
                        st.rerun()
                    
                    else:  # SQL method
                        result = query_db(query)
                        st.session_state.query_result = {"result": result, "code": "SQL Query"}
                        st.session_state.query_method = query_method
                        st.rerun()
                        
                except Exception as e:
                    if query_method == "🐼 Pandas (Faster)":
                        st.error(f"❌ Pandas query failed: {e}")
                        st.info("💡 **Tip**: Try switching to 'SQL Database' method for complex queries or if the pandas query doesn't work as expected.")
                    else:
                        st.error(f"❌ SQL query failed: {e}")
                        st.info("💡 **Tip**: Try switching to 'Pandas (Faster)' method or rephrase your question.")
    
    # Display query results in full width below the query interface
    st.markdown("---")
    
    if 'query_result' in st.session_state and st.session_state.query_result:
        st.subheader("📊 Query Result")
        
        result = st.session_state.query_result
        query_method_used = st.session_state.get('query_method', 'Unknown')
        
        # Show generated code for pandas queries
        if query_method_used == "🐼 Pandas (Faster)" and 'code' in result:
            with st.expander("🔍 Generated Code"):
                st.code(result['code'], language='python')
        
        # Show result with better formatting in full width
        if result.get('result') is not None:
            # Check if result is effectively empty
            try:
                import pandas as pd
                if isinstance(result['result'], pd.DataFrame) and result['result'].empty:
                    st.info("Query executed but returned empty results.")
                    st.info("💡 **Tip**: No data found. Try switching to 'SQL Database' method or rephrase your question.")
                elif isinstance(result['result'], (list, tuple)) and len(result['result']) == 0:
                    st.info("Query executed but returned empty results.")
                    st.info("💡 **Tip**: No data found. Try switching to 'SQL Database' method or rephrase your question.")
                else:
                    # Better formatting for different result types - FULL WIDTH
                    if isinstance(result['result'], pd.DataFrame):
                        st.dataframe(result['result'], use_container_width=True, height=400)
                    elif isinstance(result['result'], (int, float)):
                        col1, col2, col3 = st.columns([1, 2, 1])
                        with col2:
                            st.metric("Result", result['result'])
                    elif isinstance(result['result'], str):
                        if len(result['result']) > 200:
                            # Long text results in expandable section
                            with st.expander("📄 Full Result", expanded=True):
                                st.markdown(result['result'])
                        else:
                            st.success(f"**Result:** {result['result']}")
                    else:
                        st.write(result['result'])
            except Exception as e:
                # Fallback: just show the result
                st.write(result['result'])
        else:
            st.info("Query executed but no result returned.")
            if query_method_used == "🐼 Pandas (Faster)":
                st.info("💡 **Tip**: If you expected results, try switching to 'SQL Database' method for more comprehensive search.")
            else:
                st.info("💡 **Tip**: Try switching to 'Pandas (Faster)' method or rephrase your question.")
    else:
        st.info("💡 Run a query above to see results here")
    
    # Quick stats section
    st.markdown("---")
    st.subheader("📈 Quick Stats")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_disputes = len(disputes)
        st.metric("Total Disputes", total_disputes)
    
    with col2:
        if classified_disputes is not None:
            fraud_count = len(classified_disputes[classified_disputes['predicted_category'] == 'FRAUD'])
            st.metric("Fraud Cases", fraud_count)
        else:
            st.metric("Fraud Cases", "Run Task 1")
    
    with col3:
        duplicate_txns = detect_duplicates()
        st.metric("Duplicate Transactions", len(duplicate_txns))
    
    with col4:
        failed_txns = len(transactions[transactions['status'] == 'FAILED'])
        st.metric("Failed Transactions", failed_txns)

# ---------- ANALYTICS PAGE ----------
elif page == "📈 Analytics & Trends":
    st.title("📈 Analytics & Dispute Trends")
    st.markdown("Visualize dispute patterns and trends over time.")
    st.markdown("---")
    
    # Time series analysis
    st.subheader("📅 Dispute Trends Over Time")
    
    if not disputes.empty:
        # Prepare time series data
        disputes_ts = disputes.copy()
        disputes_ts['date'] = disputes_ts['created_at'].dt.date
        daily_disputes = disputes_ts.groupby('date').size().reset_index(name='count')
        daily_disputes['date'] = pd.to_datetime(daily_disputes['date'])
        
        # Line chart for trends
        fig_trend = px.line(
            daily_disputes,
            x='date',
            y='count',
            title="Daily Dispute Volume",
            markers=True
        )
        fig_trend.update_layout(
            xaxis_title="Date",
            yaxis_title="Number of Disputes"
        )
        st.plotly_chart(fig_trend, use_container_width=True, key="trend_line_chart")
    else:
        st.info("📈 No data available for trends analysis. Please upload a dataset first.")
    
    # Classification trends (if available)
    if classified_disputes is not None:
        st.subheader("🎯 Classification Trends")
        
        # Merge with disputes for timestamps
        classified_with_time = classified_disputes.merge(
            disputes[['dispute_id', 'created_at']], 
            on='dispute_id'
        )
        classified_with_time['date'] = pd.to_datetime(classified_with_time['created_at']).dt.date
        
        # Category trends
        category_trends = classified_with_time.groupby(['date', 'predicted_category']).size().reset_index(name='count')
        category_trends['date'] = pd.to_datetime(category_trends['date'])
        
        fig_cat_trend = px.line(
            category_trends,
            x='date',
            y='count',
            color='predicted_category',
            title="Dispute Categories Over Time",
            markers=True
        )
        st.plotly_chart(fig_cat_trend, use_container_width=True, key="category_trend_chart")
        
        # Heatmap of categories by day of week
        classified_with_time['day_of_week'] = pd.to_datetime(classified_with_time['created_at']).dt.day_name()
        heatmap_data = classified_with_time.groupby(['day_of_week', 'predicted_category']).size().unstack(fill_value=0)
        
        fig_heatmap = px.imshow(
            heatmap_data.T,
            title="Dispute Categories by Day of Week",
            labels=dict(x="Day of Week", y="Category", color="Count"),
            aspect="auto"
        )
        st.plotly_chart(fig_heatmap, use_container_width=True, key="heatmap_chart")
    
    # Transaction Type analysis  
    st.subheader("💳 Transaction Type Analysis")
    
    if not disputes.empty:
        col1, col2 = st.columns(2)
        
        with col1:
            # Transaction type distribution
            txn_type_counts = disputes['txn_type'].value_counts()
            fig_txn_type = px.bar(
                x=txn_type_counts.index,
                y=txn_type_counts.values,
                title="Disputes by Transaction Type",
                labels={'x': 'Transaction Type', 'y': 'Count'}
            )
            st.plotly_chart(fig_txn_type, use_container_width=True, key="analytics_txn_type_chart")
        
        with col2:
            # Channel distribution over time
            fig_channel_time = px.scatter(
                disputes,
                x='created_at',
                y='txn_type',
                color='channel',
                title="Transaction Types Over Time by Channel",
                hover_data=['dispute_id', 'customer_id']
            )
            st.plotly_chart(fig_channel_time, use_container_width=True, key="channel_time_chart")
    
    # Duplicate transaction analysis
    st.subheader("🔄 Duplicate Transaction Analysis")
    
    duplicate_txns = detect_duplicates()
    if len(duplicate_txns) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            # Duplicate transaction status
            dup_status_counts = duplicate_txns['status'].value_counts()
            fig_dup_status = px.pie(
                values=dup_status_counts.values,
                names=dup_status_counts.index,
                title="Duplicate Transaction Status"
            )
            st.plotly_chart(fig_dup_status, use_container_width=True, key="duplicate_status_chart")
        
        with col2:
            # Duplicates by merchant
            dup_merchant = duplicate_txns['merchant'].value_counts().head(10)
            fig_dup_merchant = px.bar(
                x=dup_merchant.values,
                y=dup_merchant.index,
                orientation='h',
                title="Top Merchants with Duplicates"
            )
            st.plotly_chart(fig_dup_merchant, use_container_width=True, key="duplicate_merchant_chart")
        
        st.subheader("🔍 Duplicate Transactions Details")
        st.dataframe(duplicate_txns, use_container_width=True)
    else:
        st.info("No duplicate transactions found.")
    
    # Performance metrics
    st.subheader("⚡ System Performance")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if classified_disputes is not None:
            high_conf = len(classified_disputes[classified_disputes['confidence'] >= 0.8])
            total_classified = len(classified_disputes)
            accuracy_rate = (high_conf / total_classified) * 100 if total_classified > 0 else 0
            st.metric("High Confidence Classifications", f"{accuracy_rate:.1f}%")
        else:
            st.metric("Classification Accuracy", "Run Task 1")
    
    with col2:
        if resolutions is not None:
            auto_resolutions = len(resolutions[resolutions['suggested_action'].str.contains('Auto', na=False)])
            total_resolutions = len(resolutions)
            auto_rate = (auto_resolutions / total_resolutions) * 100 if total_resolutions > 0 else 0
            st.metric("Auto-Resolution Rate", f"{auto_rate:.1f}%")
        else:
            st.metric("Auto-Resolution Rate", "Run Task 2")
    
    with col3:
        total_customers = len(disputes['customer_id'].unique()) if not disputes.empty else 0
        st.metric("Unique Customers", total_customers)

# ---------- DATASET HISTORY PAGE ----------
elif page == "🗂️ Dataset History":
    st.title("🗂️ Dataset History")
    st.markdown("Manage and switch between uploaded datasets.")
    st.markdown("---")
    
    dataset_history = dataset_manager.get_dataset_history()
    
    if not dataset_history:
        st.info("📭 No datasets uploaded yet. Use the Upload Data page to add your first dataset.")
    else:
        st.subheader(f"📚 All Datasets ({len(dataset_history)})")
        
        for i, dataset in enumerate(dataset_history):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 2, 1])
                
                with col1:
                    status_icon = "🟢" if dataset['is_current'] else "⚪"
                    status_text = " (Active)" if dataset['is_current'] else ""
                    st.markdown(f"### {status_icon} {dataset['name']}{status_text}")
                    if dataset['description']:
                        st.caption(f"📝 {dataset['description']}")
                    st.caption(f"📅 {dataset['upload_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                
                with col2:
                    st.metric("Disputes", dataset['disputes_count'])
                
                with col3:
                    st.metric("Transactions", dataset['transactions_count'])
                
                with col4:
                    # Action buttons
                    if not dataset['is_current']:
                        if st.button("🔄 Activate", key=f"activate_{dataset['id']}"):
                            dataset_manager.set_current_dataset(dataset['id'])
                            st.success(f"✅ Activated dataset: {dataset['name']}")
                            st.rerun()
                    else:
                        st.success("✅ Active")
                    
                    if st.button("🗑️ Delete", key=f"delete_{dataset['id']}", 
                               help="Delete this dataset permanently"):
                        if st.session_state.get(f"confirm_delete_{dataset['id']}", False):
                            try:
                                dataset_manager.delete_dataset(dataset['id'])
                                st.success(f"🗑️ Deleted dataset: {dataset['name']}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"❌ Error deleting dataset: {e}")
                        else:
                            st.session_state[f"confirm_delete_{dataset['id']}"] = True
                            st.warning("⚠️ Click delete again to confirm")
                
                st.divider()
        
        # Bulk actions
        st.subheader("🔧 Bulk Actions")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Delete All Non-Active Datasets", type="secondary"):
                inactive_datasets = [d for d in dataset_history if not d['is_current']]
                if inactive_datasets:
                    try:
                        for dataset in inactive_datasets:
                            dataset_manager.delete_dataset(dataset['id'])
                        st.success(f"🗑️ Deleted {len(inactive_datasets)} inactive datasets")
                        st.rerun()
                    except Exception as e:
                        st.error(f"❌ Error during bulk delete: {e}")
                else:
                    st.info("ℹ️ No inactive datasets to delete")
        
        with col2:
            # Export dataset info
            if st.button("📥 Export Dataset List", type="secondary"):
                df_history = pd.DataFrame(dataset_history)
                csv = df_history.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name=f"dataset_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

# ---------- Footer ----------
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        <p>⚖️ AI-Powered Dispute Assistant | Built with Streamlit & OpenAI</p>
    </div>
    """,
    unsafe_allow_html=True
)
