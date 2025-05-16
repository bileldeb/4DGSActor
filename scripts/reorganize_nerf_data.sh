#!/bin/bash
tasks=("close_jar" "meat_off_grill")
#tasks=("sweep_to_dustpan_of_size" "turn_tap" "slide_block_to_color_target" "put_item_in_drawer" "reach_and_drag" "push_buttons" "stack_blocks")
for task in "${tasks[@]}"; do

    for episode_dir in data/train_data/${task}/all_variations/episodes/episode*/; do
        nerf_dir="${episode_dir}nerf_data/"
        nerf_dir_rgb="${episode_dir}nerf_data_rgb/"
        
        # Check if nerf_data exists
        if [ ! -d "$nerf_dir" ]; then
            echo "Skipping: $nerf_dir not found"
            continue
        fi
        
        # Create temporary directory for reorganization
        temp_dir="${episode_dir}nerf_data_temp/"
        mkdir -p "$temp_dir"
        
        # Get the list of original frame folders (000, 001, etc.)
        frame_folders=($(find "$nerf_dir" -maxdepth 1 -type d -name '[0-9]*' | sort -V))
        N=${#frame_folders[@]}
        
        # Create new folders (0-19) in temporary directory
        for k in {0..20}; do
            mkdir -p "${temp_dir}${k}"
        done
        
        # Move each k.png to corresponding new folder
        for ((i=0; i<N; i++)); do
            original_folder="${frame_folders[$i]}"
            for k in {0..20}; do
                img="${original_folder}/images/${k}.png"
                if [ -f "$img" ]; then
                    cp "$img" "${temp_dir}${k}/${i}.png"
                fi
            done
        done
        
        # Replace old nerf_data with reorganized one
        #rm -rf "$nerf_dir"
        mv "$temp_dir" "${nerf_dir_rgb}"    
        echo "Reorganized $nerf_dir N=$N frames"
    done

done
echo "All episodes processed"