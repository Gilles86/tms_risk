#!/bin/bash

# List of all model labels to iterate over and submit
model_labels=(
"10a"
"10b"
"10c"
"11a"
"11b"
'11c'
)

# Loop over each model label and submit a job
for MODEL_LABEL in "${model_labels[@]}"; do
    # Submit the job with the model label as an argument
    sbatch fit_model.sh "$MODEL_LABEL"
    echo "Submitted job for model label: $MODEL_LABEL"
done