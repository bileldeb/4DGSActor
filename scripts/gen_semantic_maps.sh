#!/bin/bash




export CUDA_VISIBLE_DEVICES=4
python3 -c "import torch; torch.cuda.empty_cache(); torch.ones( (int(torch.cuda.get_device_properties(0).total_memory * 0.9 // 4)), dtype=torch.float32, device='cuda' ); print('GPU memory reserved')" & export HOLDER_PID=$!

cleanup() {
  echo "Caught Ctrl+C! Performing cleanup..."
  kill $HOLDER_PID && python3 -c "import torch; torch.cuda.empty_cache(); print('GPU memory released')"
  exit
}

trap cleanup SIGINT


RED='\033[0;31m'
GREEN='\033[0;32m'
WHITE='\033[0;37m'

tasks=("close_jar" "open_drawer" "sweep_to_dustpan_of_size" "meat_off_grill" "turn_tap" "slide_block_to_color_target" "push_buttons" "put_item_in_drawer" "reach_and_drag" "stack_blocks")

# Function to check if output exists and is valid
should_process() {
    local dir="$1"
    for d in default small middle large; do 
            [ -d "$dir/$d" ] || { echo "$dir missing $d"; return 1; }
    done
    return 0
}

track() {
    local level=$1
    local input_dir=$2
    local output_dir=$3
    local max_retries=3
    local attempt=0
    local success=0
    
    output_dir="${output_dir}${level}"
    echo -e "${GREEN}Processing: Level=$level"

    export LEVEL=$level
    
    while [[ $attempt -le $max_retries ]]; do
        # Run the python command
        python -W ignore -s demo/demo_automatic.py --chunk_size 4 \
            --img_path "$input_dir" \
            --amp --temporal_setting semionline \
            --size 128 \
            --output "$output_dir" > /dev/null 2>&1
        
        # Verify success
        if [[ -d "$output_dir" && -n "$(ls -A "$output_dir")" ]]; then
            success=1
            break
        fi
        
        ((attempt++))
        if [[ $attempt -le $max_retries ]]; then
            echo -e "${YELLOW}Attempt $attempt failed, retrying..."
            # Clean up failed attempt
            [[ -d "$output_dir" ]] && rm -rf "$output_dir"
        fi
    done
    
    if [[ $success -eq 0 ]]; then
        echo -e "${RED}Error: Failed to process level $level after $((max_retries+1)) attempts"
        return 1
    fi
    
    echo -e "${GREEN}Successfully processed level $level"
    return 0
}


export -f track should_process

MAX_JOBS=4

for task in "${tasks[@]}"; do
    # Create directories for the task

    
    echo -e "${WHITE}===== Processing task: $task ====="



    for episode in {0..13}; do
        # Process front_rgb
        echo -e "${WHITE}===== Processing Episode: $episode ====="

        # Process nerf_data_semantic

        mkdir -p "/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data/train_data/${task}/all_variations/episodes/episode${episode}/nerf_data_semantic"
        running_jobs=0
        for cam_num in {0..20}; do
            dataset_path="/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data/train_data/${task}/all_variations/episodes/episode${episode}/nerf_data_rgb/${cam_num}/"
            output_dir="/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data/train_data/${task}/all_variations/episodes/episode${episode}/nerf_data_semantic/${cam_num}/"

            if should_process "$output_dir"; then
                echo -e "${WHITE} ✅ $test_dir has all required subfolders" 
                continue
            else
                echo -e "${RED} ❌ $test_dir is incomplete"
            fi

            mkdir -p "$output_dir"

            cd third_party/4DLangSplat/submodules/4d-langsplat-tracking-anything-with-deva

            for level in "default" "small" "middle" "large"; do
                
                track "$level" "$dataset_path" "$output_dir" &
                ((running_jobs++))
                sleep 2
            done

            while [ $(jobs -rp | wc -l) -gt 0 ]; do
                sleep 5
            done
            echo '4 levels done'

            echo -e "${GREEN}Concat: Task=$task, Episode=$episode, Cam=$cam_num"
            python -s concat_npy.py --base_dir ${output_dir} #> /dev/null 2>&1

            cd ../../../..
            cd third_party/4DLangSplat/preprocess

            echo -e "${GREEN}gen clip feats: Task=$task, Episode=$episode, Cam=$cam_num"
            precompute_seg_path="${output_dir}video_mask_concat"
            clip_language_feature_name=semantic_features
            python -W ignore generate_clip_features.py --dataset_path $dataset_path \
            --dataset_type custom \
            --precompute_seg ${precompute_seg_path} \
            --output_name ${clip_language_feature_name}

            cd ../../..


        done
        wait
    done
    
    echo -e "${WHITE}===== Finished task: $task ====="
done




cleanup
echo -e "${WHITE}All tasks completed successfully!"