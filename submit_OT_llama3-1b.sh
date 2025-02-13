#!/bin/bash

# Script to submit and automatically relaunch Llama-3 1B training jobs in case of failures.
# Handles failures by restarting the bash script.

# Initialize counter
counter=1

# Function to submit a job
submit_job() {
  local job_name="llama3_1b_$counter"
  sbatch --job-name="$job_name" --output=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/${job_name}_%j.out --error=/iopsstor/scratch/cscs/%u/Megatron-LM/logs/slurm/training/${job_name}_%j.err submit-llama3-1b.sh
}

# Initial submission
submit_job

# Get the initial job ID
initial_job_id=$(squeue -u $USER -n llama3_1b_$counter -o %i | head -n 1)

# Monitor the job and relaunch on failure
while true; do
  sleep 120  # Check job status every 2 min

  # Check if the job has finished (COMPLETED, CANCELLED, FAILED, TIMEOUT)
  job_status=$(sacct -j $initial_job_id -n -o State= | tail -n 1)

  if [[ "$job_status" == "FAILED" ]]; then
    echo "Job $initial_job_id failed with status: $job_status"

    # Increment counter for the next job
    counter=$((counter + 1))

    # Relaunch the job
    submit_job

    # Get the new job ID
    initial_job_id=$(squeue -u $USER -n llama3_1b_$counter -o %i | head -n 1)
    echo "Relaunched job with ID: $initial_job_id and name: llama3_1b_$counter"

  elif [[ "$job_status" == "COMPLETED" || "$job_status" == "TIMEOUT" ]]; then
    echo "Job $initial_job_id finished with status: $job_status"
    break
  else
    echo "Job $initial_job_id is still running with status: $job_status"
  fi
done