cd third_party/4DLangSplat/autoencoder
export CUDA_VISIBLE_DEVICES=4

#!/bin/bash

# Base paths
base_dir="/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data"
combined_dataset_path="$base_dir/autoencoder_dataset"
feature_name="semantic_features"
bottleneck_dim=3

# Function to process a feature type
process_feature_type() {
    local feature_type=$1
    
    # Set architecture based on feature type
    local encoder_dims decoder_dims feature_dims
    case $feature_type in
        clip|dino)
            encoder_dims="512 128 64 32"
            decoder_dims="16 32 64 128 256 512 1024"
            feature_dims=1024
            ;;
        combined)
            encoder_dims="1024 512 128 64 32"
            decoder_dims="16 32 64 128 256 512 1024 2048"
            feature_dims=2048
            ;;
        *)
            echo "Unknown feature type: $feature_type" >&2
            return 1
            ;;
    esac
    
    echo "========================================"
    echo "Processing feature type: $feature_type"
    echo "Encoder dims: $encoder_dims ${bottleneck_dim}"
    echo "Decoder dims: $decoder_dims"
    echo "========================================"
    
    # Training
    #echo "Starting training for $feature_type..."
    #python train.py \
    #    --lr 7e-4 \
    #    --dataset_path "${combined_dataset_path}" \
    #    --model_name "${feature_type}" \
    #    --feature_dims $feature_dims \
    #    --encoder_dims $encoder_dims "${bottleneck_dim}" \
    #    --decoder_dims $decoder_dims \
    #    --hidden_dims "${bottleneck_dim}" \
    #    --language_name "${feature_name}" \
    #    --feature_type "${feature_type}"\
    #    --batch_size 1024 \
    #    --num_epochs 100 \
    #
    #[ $? -ne 0 ] && { echo "Training failed for $feature_type" >&2; return 1; }
    
    # Testing
    echo "Starting testing for $feature_type..."
    find "$base_dir"/train_data/*/all_variations/episodes/episode*/nerf_data_semantic/*/ -maxdepth 0 -type d | \
    while read -r dataset_path; do
        echo "Testing on: $dataset_path"
        #
        python test.py \
            --dataset_path "${dataset_path}" \
            --model_name "${feature_type}" \
            --feature_dims $feature_dims \
            --encoder_dims $encoder_dims "${bottleneck_dim}" \
            --decoder_dims $decoder_dims \
            --hidden_dims "${bottleneck_dim}" \
            --language_name "${feature_name}" \
            --feature_type "${feature_type}"
        
        [ $? -ne 0 ] && echo "Test failed for $dataset_path" >&2

    done
    
    echo "Completed processing for $feature_type"
    echo "========================================"
}

# Process all feature types
for feature_type in clip dino combined; do
    process_feature_type "$feature_type" || break  # Stop on error if desired
done

echo "All feature types processed"
















cd ../../..


