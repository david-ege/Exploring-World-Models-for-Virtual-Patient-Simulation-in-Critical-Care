#GRID SEARCH TERMINAL COMMAND
#!/bin/bash
cd ~/thesis_work/hirid_jepa
source ~/miniconda3/etc/profile.d/conda.sh
conda activate exploration

GRID_DIR="results/grid_search_$(date +%d_%m_%H%M)"
mkdir -p $GRID_DIR
echo "Saving results to $GRID_DIR"

for wd in 1e-3 1e-4; do
    for hidden in 128 256; do
        for dropout in 0.3 0.4; do
            echo ""
            echo "=== wd=$wd hidden=$hidden dropout=$dropout ==="
            python train.py \
                --hidden $hidden \
                --dropout $dropout \
                --wd $wd \
                --results_dir $GRID_DIR \
                2>&1 | tee -a $GRID_DIR/training_log.txt
        done
    done
done

echo ""
echo "Grid search complete. Results in $GRID_DIR"