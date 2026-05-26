#GRID SEARCH TERMINAL COMMAND
#!/bin/bash
# run_grid.sh

cd ~/thesis_work/hirid_jepa
source ~/miniconda3/etc/profile.d/conda.sh
conda activate exploration

RESULTS_FILE="grid_results_$(date +%d_%m_%H%M).txt"
echo "Grid Search Results" > $RESULTS_FILE
echo "==================" >> $RESULTS_FILE

for target in 12 48 72; do
    for context in 36 72; do
        for hidden in 128 256; do
            for layers in 1 2; do
                echo "" >> $RESULTS_FILE
                echo "target=$target context=$context hidden=$hidden layers=$layers" >> $RESULTS_FILE
                echo "=== target=$target context=$context hidden=$hidden layers=$layers ==="
                python train.py \
                    --target $target \
                    --context $context \
                    --hidden $hidden \
                    --layers $layers \
                    2>&1 | tee -a $RESULTS_FILE
            done
        done
    done
done

echo ""
echo "Grid search complete. Results saved to $RESULTS_FILE"