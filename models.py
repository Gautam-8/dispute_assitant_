from sqlalchemy import Column, Integer, String, Float, DateTime, Text
from database import Base
from datetime import datetime


class Documentation(Base):
    __tablename__ = "documentation"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, index=True)


class DatasetHistory(Base):
    __tablename__ = "dataset_history"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_name = Column(String, index=True)  # User-friendly name
    upload_timestamp = Column(DateTime, default=datetime.utcnow)
    disputes_filename = Column(String)
    transactions_filename = Column(String)
    disputes_count = Column(Integer)
    transactions_count = Column(Integer)
    description = Column(Text, nullable=True)
    is_current = Column(Integer, default=0)  # 1 for current active dataset


class DisputeRecord(Base):
    __tablename__ = "dispute_records"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, index=True)  # Foreign key to DatasetHistory
    dispute_id = Column(String, index=True)
    customer_id = Column(String)
    txn_id = Column(String, index=True)
    description = Column(Text)
    txn_type = Column(String)
    channel = Column(String)
    amount = Column(Float)
    created_at = Column(DateTime)


class TransactionRecord(Base):
    __tablename__ = "transaction_records"
    
    id = Column(Integer, primary_key=True, index=True)
    dataset_id = Column(Integer, index=True)  # Foreign key to DatasetHistory
    txn_id = Column(String, index=True)
    customer_id = Column(String)
    amount = Column(Float)
    status = Column(String)
    timestamp = Column(DateTime)
    channel = Column(String)
    merchant = Column(String)