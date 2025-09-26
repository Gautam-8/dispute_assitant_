#!/usr/bin/env python3
"""
Simple launcher script for the AI-Powered Dispute Assistant with Dynamic Upload
"""

import subprocess
import sys
from pathlib import Path
import os

def main():
    print("🚀 Starting AI-Powered Dispute Assistant (Dynamic Version)...")
    print("📍 Make sure you have set your OPENAI_API_KEY in .env file")
    print("📁 This version supports dynamic file uploads with database history!")
    print("-" * 60)
    
    # Check if app.py exists
    if not Path("app.py").exists():
        print("❌ app.py not found!")
        return 1
    
    # Check for .env file
    if not Path(".env").exists():
        print("⚠️ Warning: .env file not found!")
        print("💡 Create a .env file with: OPENAI_API_KEY=your_key_here")
        print()
    
    # Create csv directory if it doesn't exist (for backward compatibility)
    Path("csv").mkdir(exist_ok=True)
    
    print("🌟 Features available:")
    print("  📁 Dynamic file upload (disputes.csv + transactions.csv)")
    print("  🗂️ Dataset history and management")
    print("  🔄 Switch between datasets instantly")
    print("  📊 Real-time analytics and classification")
    print("  🔍 Natural language querying")
    print()
    
    # Run the Streamlit app
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to start Streamlit: {e}")
        return 1
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        return 0

if __name__ == "__main__":
    sys.exit(main())
