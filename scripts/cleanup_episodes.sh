tasks=("close_jar" "meat_off_grill")
#tasks=("open_drawer" "sweep_to_dustpan_of_size" "meat_off_grill" "turn_tap" "slide_block_to_color_target" "put_item_in_drawer" "reach_and_drag" "push_buttons" "stack_blocks")
exit

for task in "${tasks[@]}"; do
    echo -e "${WHITE}===== Processing task: $task ====="

    # Step 1: Identify all bad episodes (where M != N)
    bad_episodes=()
    for episode in {0..19}; do
        base_dir="/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data/train_data/${task}/all_variations/episodes/episode${episode}/"
        if [ ! -d "$base_dir" ]; then
            continue
        fi
        
        N=$(ls -1 "$base_dir/front_rgb" 2>/dev/null | wc -l)
        M=$(ls -1 "$base_dir/nerf_data" 2>/dev/null | wc -l)
        
        if [ "$M" -gt "$N" ]; then
            bad_episodes+=("$episode")
            echo "Bad episode found: $episode (N=$N, M=$M)"
        fi
    done

    # Step 2: Select exactly 6 episodes to remove (prioritizing bad ones)
    episodes_to_remove=("${bad_episodes[@]}")
    remaining_slots=$((6 - ${#bad_episodes[@]}))
    
    # If we need more episodes to reach 6, add random good ones
    if [ "$remaining_slots" -gt 0 ]; then
        # Find all good episodes
        good_episodes=()
        for episode in {0..19}; do
            if [[ ! " ${bad_episodes[@]} " =~ " ${episode} " ]]; then
                good_episodes+=("$episode")
            fi
        done
        
        # Randomly select remaining episodes to remove
        if [ ${#good_episodes[@]} -gt 0 ]; then
            shuffle=($(shuf -e "${good_episodes[@]}"))
            for ((i=0; i<remaining_slots && i<${#shuffle[@]}; i++)); do
                episodes_to_remove+=("${shuffle[$i]}")
            done
        fi
    fi

    # Step 3: Remove selected episodes
    echo "Removing episodes: ${episodes_to_remove[@]}"
    for episode in "${episodes_to_remove[@]}"; do
        base_dir="/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data/train_data/${task}/all_variations/episodes/episode${episode}/"
        echo "$base_dir"
        rm -rf "$base_dir"
    done
    
    # Step 4: Rename remaining episodes to maintain continuous numbering
    remaining_episodes=()
    for episode in {0..19}; do
        base_dir="/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data/train_data/${task}/all_variations/episodes/episode${episode}/"
        if [ -d "$base_dir" ]; then
            remaining_episodes+=("$episode")
        fi
    done

    # Sort remaining episodes
    sorted_episodes=($(printf '%s\n' "${remaining_episodes[@]}" | sort -n))
    
    # Rename to 0-13
    for new_episode in {0..13}; do
        old_episode="${sorted_episodes[$new_episode]}"
        if [ -z "$old_episode" ]; then
            echo "Warning: Not enough episodes remaining for task $task"
            break
        fi
        
        if [ "$old_episode" -ne "$new_episode" ]; then
            old_dir="/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data/train_data/${task}/all_variations/episodes/episode${old_episode}/"
            new_dir="/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data/train_data/${task}/all_variations/episodes/episode${new_episode}/"
            mv "$old_dir" "$new_dir"
        fi
    done

    # Remove any remaining episodes beyond 14
    for episode in {14..19}; do
        base_dir="/home/bing_TUM/bilel/MasterThesis/ManiGaussian/data/train_data/${task}/all_variations/episodes/episode${episode}/"
        if [ -d "$base_dir" ]; then
            rm -rf "$base_dir"
        fi
    done
done