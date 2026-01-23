#!/bin/bash
# Start the application and dependencies using docker-compose

# install dependencies
pip install -r requirements.txt

# Start the services
uvicorn app:app --host 0.0.0.0 --port 8000