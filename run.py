"""
Quick-run script for ZK-KGVerify.
Usage: python run.py
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.pipeline import run_full_pipeline

if __name__ == "__main__":
    run_full_pipeline()
