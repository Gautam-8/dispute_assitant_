"""
Dataset Manager for handling dynamic file uploads and database history
"""
import pandas as pd
import streamlit as st
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine
from datetime import datetime
from pathlib import Path
import tempfile
import io

from database import engine, SessionLocal
from models import DatasetHistory, DisputeRecord, TransactionRecord

class DatasetManager:
    def __init__(self):
        self.engine = engine
        self.SessionLocal = SessionLocal

    def validate_disputes_csv(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Validate disputes CSV structure"""
        required_columns = ['dispute_id', 'customer_id', 'txn_id', 'description', 
                          'txn_type', 'channel', 'amount', 'created_at']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Basic data type validation
        try:
            pd.to_datetime(df['created_at'])
            pd.to_numeric(df['amount'])
        except Exception as e:
            return False, f"Data validation error: {str(e)}"
        
        return True, "Valid"

    def validate_transactions_csv(self, df: pd.DataFrame) -> tuple[bool, str]:
        """Validate transactions CSV structure"""
        required_columns = ['txn_id', 'customer_id', 'amount', 'status', 
                          'timestamp', 'channel', 'merchant']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            return False, f"Missing required columns: {', '.join(missing_columns)}"
        
        # Basic data type validation
        try:
            pd.to_datetime(df['timestamp'])
            pd.to_numeric(df['amount'])
        except Exception as e:
            return False, f"Data validation error: {str(e)}"
        
        return True, "Valid"

    def save_dataset_to_db(self, disputes_df: pd.DataFrame, transactions_df: pd.DataFrame, 
                          dataset_name: str, description: str = None) -> int:
        """Save uploaded dataset to database and return dataset_id"""
        
        db = self.SessionLocal()
        try:
            # Mark all existing datasets as not current
            db.query(DatasetHistory).update({DatasetHistory.is_current: 0})
            
            # Create new dataset history record
            dataset_history = DatasetHistory(
                dataset_name=dataset_name,
                upload_timestamp=datetime.utcnow(),
                disputes_filename=f"disputes_{dataset_name}.csv",
                transactions_filename=f"transactions_{dataset_name}.csv",
                disputes_count=len(disputes_df),
                transactions_count=len(transactions_df),
                description=description,
                is_current=1
            )
            
            db.add(dataset_history)
            db.commit()
            db.refresh(dataset_history)
            
            dataset_id = dataset_history.id
            
            # Save disputes records
            for _, row in disputes_df.iterrows():
                dispute_record = DisputeRecord(
                    dataset_id=dataset_id,
                    dispute_id=row['dispute_id'],
                    customer_id=row['customer_id'],
                    txn_id=row['txn_id'],
                    description=row['description'],
                    txn_type=row['txn_type'],
                    channel=row['channel'],
                    amount=float(row['amount']),
                    created_at=pd.to_datetime(row['created_at'])
                )
                db.add(dispute_record)
            
            # Save transaction records
            for _, row in transactions_df.iterrows():
                transaction_record = TransactionRecord(
                    dataset_id=dataset_id,
                    txn_id=row['txn_id'],
                    customer_id=row['customer_id'],
                    amount=float(row['amount']),
                    status=row['status'],
                    timestamp=pd.to_datetime(row['timestamp']),
                    channel=row['channel'],
                    merchant=row['merchant']
                )
                db.add(transaction_record)
            
            db.commit()
            return dataset_id
            
        except Exception as e:
            db.rollback()
            raise e
        finally:
            db.close()

    def get_dataset_history(self) -> list:
        """Get list of all uploaded datasets"""
        db = self.SessionLocal()
        try:
            datasets = db.query(DatasetHistory).order_by(DatasetHistory.upload_timestamp.desc()).all()
            return [{
                'id': d.id,
                'name': d.dataset_name,
                'upload_time': d.upload_timestamp,
                'disputes_count': d.disputes_count,
                'transactions_count': d.transactions_count,
                'description': d.description,
                'is_current': bool(d.is_current)
            } for d in datasets]
        finally:
            db.close()

    def load_dataset_by_id(self, dataset_id: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Load disputes and transactions DataFrames for a specific dataset"""
        db = self.SessionLocal()
        try:
            # Load disputes
            disputes_query = db.query(DisputeRecord).filter(DisputeRecord.dataset_id == dataset_id)
            disputes_data = []
            for record in disputes_query:
                disputes_data.append({
                    'dispute_id': record.dispute_id,
                    'customer_id': record.customer_id,
                    'txn_id': record.txn_id,
                    'description': record.description,
                    'txn_type': record.txn_type,
                    'channel': record.channel,
                    'amount': record.amount,
                    'created_at': record.created_at
                })
            
            disputes_df = pd.DataFrame(disputes_data)
            
            # Load transactions
            transactions_query = db.query(TransactionRecord).filter(TransactionRecord.dataset_id == dataset_id)
            transactions_data = []
            for record in transactions_query:
                transactions_data.append({
                    'txn_id': record.txn_id,
                    'customer_id': record.customer_id,
                    'amount': record.amount,
                    'status': record.status,
                    'timestamp': record.timestamp,
                    'channel': record.channel,
                    'merchant': record.merchant
                })
            
            transactions_df = pd.DataFrame(transactions_data)
            
            return disputes_df, transactions_df
            
        finally:
            db.close()

    def set_current_dataset(self, dataset_id: int):
        """Set a dataset as the current active one"""
        db = self.SessionLocal()
        try:
            # Mark all as not current
            db.query(DatasetHistory).update({DatasetHistory.is_current: 0})
            
            # Mark selected as current
            db.query(DatasetHistory).filter(DatasetHistory.id == dataset_id).update({DatasetHistory.is_current: 1})
            
            db.commit()
        finally:
            db.close()

    def get_current_dataset(self) -> dict:
        """Get the currently active dataset info"""
        db = self.SessionLocal()
        try:
            current = db.query(DatasetHistory).filter(DatasetHistory.is_current == 1).first()
            if current:
                return {
                    'id': current.id,
                    'name': current.dataset_name,
                    'upload_time': current.upload_timestamp,
                    'disputes_count': current.disputes_count,
                    'transactions_count': current.transactions_count,
                    'description': current.description
                }
            return None
        finally:
            db.close()

    def delete_dataset(self, dataset_id: int):
        """Delete a dataset and all its records"""
        db = self.SessionLocal()
        try:
            # Delete related records first
            db.query(DisputeRecord).filter(DisputeRecord.dataset_id == dataset_id).delete()
            db.query(TransactionRecord).filter(TransactionRecord.dataset_id == dataset_id).delete()
            
            # Delete dataset history
            db.query(DatasetHistory).filter(DatasetHistory.id == dataset_id).delete()
            
            db.commit()
        finally:
            db.close()

# Global instance
dataset_manager = DatasetManager()
