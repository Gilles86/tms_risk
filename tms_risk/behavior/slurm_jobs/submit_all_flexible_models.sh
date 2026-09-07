#!/bin/bash

# List of model labels with only the 6th-order polynomials uncommented
model_labels=(
    # "flexible1"
    # "flexible1_null"
    # "flexible1a"
    # "flexible1b"
    # "flexible1.4"
    # "flexible1.4_null"
    # "flexible1.4a"
    # "flexible1.4b"
    
    "flexible1.6"
    "flexible1.6_null"
    "flexible1.6a"
    "flexible1.6b"
    
    # "flexible2"
    # "flexible2a"
    # "flexible2b"
    # "flexible2_null"
    # "flexible2.4"
    # "flexible2.4_null"
    # "flexible2.4a"
    # "flexible2.4b"
    
    "flexible2.6"
    "flexible2.6_null"
    "flexible2.6a"
    "flexible2.6b"
)

# Loop over each model label and submit a job
for MODEL_LABEL in "${model_labels[@]}"; do
    # Submit the job with the model label as an argument
    sbatch fit_model.sh "$MODEL_LABEL"
    echo "Submitted job for model label: $MODEL_LABEL"
done
