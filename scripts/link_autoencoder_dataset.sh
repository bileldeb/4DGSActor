#!/bin/bash

# Create target directory
mkdir -p data/autoencoder_dataset

# Counter for unique filenames
counter=0

# Process each file
find data/train_data/*/all_variations/episodes/episode*/nerf_data_semantic/*/semantic_features -name "*_f.npy" | while read -r src_file; do
    # Extract components for unique naming
    task=$(echo "$src_file" | cut -d'/' -f3)         
    episode=$(echo "$src_file" | cut -d'/' -f6) 
    cam=$(echo "$src_file" | cut -d'/' -f8)        
    frame=$(echo "$src_file" | cut -d'/' -f10)        
    
    # Generate unique filename
    new_name="${task}_${episode}_${cam}_${frame}"

    #echo ne "Processing: $src_file -> data/autoencoder_dataset/$new_name"
    
    # Create symlink
    ln -s "$(realpath "$src_file")" "data/autoencoder_dataset/$new_name"
    
    # Progress
    ((counter++))
    echo -ne "Created $counter symlinks\r"
done

echo -e "\nDone! Created $counter symlinks in data/autoencoder_dataset"