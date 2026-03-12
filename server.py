import sys
import os
import subprocess
from flask import Flask

# Add the 'src' directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from main import main as run_main
from tools.create_dataset import create_bids_dataset

server = Flask(__name__)


@server.route("/process-data")
def process_data():
    try:
        create_bids_dataset()
        return 'Dataset Parsed'
    except Exception as e:
        return f"An error occurred: {e}", 500


@server.route("/test-classifiers")
def test_classifiers():
    try:
        return run_main()
    except Exception as e:
        return f"An error occurred: {e}", 500


if __name__ == "__main__":
    server.run(host='0.0.0.0')
