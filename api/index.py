import os
import sys

# Add the project root to the python path so imports resolve correctly
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Import the Flask application instance from app/main.py
from app.main import app
