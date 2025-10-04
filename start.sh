#!/bin/bash

pip install --upgrade pip setuptools wheel

pip install -r requirements.txt

streamlit run app.py --server.port=$PORT --server.headless=true
