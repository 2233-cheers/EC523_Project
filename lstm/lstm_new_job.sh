#!/bin/bash -l

#$ -P ec523                
#$ -l h_rt=30:00:00             # Wall-clock time limit
#$ -l gpus=1                    # Request 1 GPU
#$ -l gpu_c=7.0                 # Request GPU with compute capability 7.0 (e.g. P100)
#$ -l mem_per_core=4G           # Memory per core
#$ -pe omp 4                    # Request 4 CPU cores
#$ -N music-lstm-cross                # Job name
#$ -j y                         # Merge stdout and stderr
#$ -o $JOB_NAME_$JOB_ID.log

#$ -M wsoxl@bu.edu         
#$ -m bea                       # b = begin, e = end, a = abort (failures)

# Load required modules
module load miniconda
module load academic-ml/fall-2024

# Activate conda environment
conda activate fall-2024-pyt

# Run your training script
echo "Running training script on SCC with GPU CC 6.0..."
cd /projectnb/ec523/projects/Proj_MusicGen/lstm/final
echo "Now in working directory: $(pwd)"
python composer2event.py --csv ../../data/maestro-v3.0.0.csv --root_dir ../../data/Composer/
python model.py > /projectnb/ec523/projects/Proj_MusicGen/lstm/final/model_out.log 2> /projectnb/ec523/projects/Proj_MusicGen/lstm/final/model_err.log
python generate.py --gen_all --steps 10000 --temp 1 --seed_file start.txt > /projectnb/ec523/projects/Proj_MusicGen/lstm/final/gen_out.log 2> /projectnb/ec523/projects/Proj_MusicGen/lstm/final/gen_err.log
echo "Job finished."
