#!/bin/bash

# Check if the correct number of arguments are provided
if [ $# -lt 2 ]; then
    echo "Usage: $0 <iterations> <command>"
    exit 1
fi

# Get the number of iterations from the first argument
iterations=$1

# Shift the arguments so that $@ now contains the command
shift

timestamp=$(date +"%Y%m%d_%H%M%S")
output_file="/path/to/results_$timestamp.txt"
ckpt_path="/path/to/ckpt"

# Loop for the specified number of iterations
for i in $(seq 1 $iterations)
do
    # Execute the command passed as arguments and capture its output
    echo "Iteration $i: Executing command '$@'"
    output=$($@)
    
    # Store the result in the array
    results+=("$output")
done

# Print the results
echo "All Results:"
for result in "${results[@]}"; do
    echo "$result"
done

# Save the results to a text file
echo "Saving results to $output_file"
printf "%s\n" "${results[@]}" > "$output_file"

