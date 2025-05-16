#!/bin/bash

# Process each episode directory
for episode_dir in data/train_data/*/all_variations/episodes/episode*/nerf_data/; do
    echo "Processing $episode_dir"
    
    # Get the list of numbered directories (sorted numerically)
    dirs=($(find "$episode_dir" -maxdepth 1 -type d -name "[0-9]*" | sort -V))
    
    # Check if directory 0 exists
    if [[ ! -d "${episode_dir}0" ]]; then
        echo "Warning: Directory '0' not found in $episode_dir"
        continue
    fi
    
    # Duplicate directory 0 to a temporary name
    temp_name="temp_duplicate"
    #cp -r "${episode_dir}0" "${episode_dir}${temp_name}"
    
    # Rename all directories starting from 0 to N+1
    for ((i=${#dirs[@]}-1; i>=0; i--)); do
        dir="${dirs[$i]}"
        dir_name=$(basename "$dir")
        
        # Only process numeric directories
        if [[ "$dir_name" =~ ^[0-9]+$ ]]; then
            new_name=$((dir_name + 1))
            echo "renaming ${episode_dir}${new_name}"
            #mv "$dir" "${episode_dir}${new_name}"
        fi
    done
    
    # Rename the temporary duplicate back to 0
    #mv "${episode_dir}${temp_name}" "${episode_dir}0"
    
    echo "Renumbering complete for $episode_dir"
done

echo "All episodes processed"