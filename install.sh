#!/bin/bash

# Check if the environment 'thuna' exists
if conda env list | grep -q '^thuna\s'; then
    echo "Environment 'thuna' exists. Removing it..."
    conda env remove -n thuna
fi

# Create a new environment 'thuna' with Python 3.10.12
echo "Creating new environment 'thuna' with Python 3.10.12..."
conda create -n thuna python=3.10.12 -y

# Activate the environment 'thuna'
echo "Activating environment 'thuna'..."
source activate thuna  # Use `source activate` for compatibility with both conda and bash scripts

# Install packages from requirements.txt
if [ -f requirements.txt ]; then
    echo "Installing packages from requirements.txt..."
    pip install -r requirements.txt
else
    echo "requirements.txt not found."
fi

echo "Environment 'thuna' is ready, activated, and required packages are installed."

# python main_knowledge.py --cfg config/iu_retrieval.yml --gpu 0 --version 1