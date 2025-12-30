#!/bin/bash

# # Example values for PSGC
# DATASET="pubmed"
# ARCH="GCN"
# LEARN_RATE=0.004           # sys.argv[3]
# L2_PARAM=6e-3          # sys.argv[4]
# RATIO=0.01                # sys.argv[5]
# USE_RANK_APPROX="False"   # sys.argv[6]
# NUM_LAYERS=3              # sys.argv[7]
# HIDDEN_DIM=150            # sys.argv[8]
# DROPOUT=0.0               # sys.argv[9]
# LAYER_NORM=0              # sys.argv[10] (0 or 1)
# RATIO2=0.7                # sys.argv[11]
# ORDER=2                   # sys.argv[12]
# DEVICE=1                  # sys.argv[13]
# LR_UPDATE_WEIGHT=0.01     # sys.argv[14]
# RANDOM_SAMPLE=10          # sys.argv[15]
# EPOCHS=100                # sys.argv[16]
# RUNS=10                   # sys.argv[17]
# SEED=123                  # sys.argv[18]
# # Run the experiment
# python main_runner.py \
#   $DATASET $ARCH $LEARN_RATE $L2_PARAM $RATIO $USE_RANK_APPROX \
#   $NUM_LAYERS $HIDDEN_DIM $DROPOUT $LAYER_NORM $RATIO2 $ORDER $DEVICE \
#   $LR_UPDATE_WEIGHT $RANDOM_SAMPLE $EPOCHS $RUNS $SEED  


# # Arguments for GCN sketch experiment
# DATASET="pubmed"
# ARCH="GCN"
# LEARN_RATE=0.004           # sys.argv[3]
# L2_PARAM=6e-3          # sys.argv[4]
# RATIO=0.01                # sys.argv[5]
# USE_RANK_APPROX="False"   # sys.argv[6]
# NUM_LAYERS=3              # sys.argv[7]
# HIDDEN_DIM=150            # sys.argv[8]
# DROPOUT=0.0               # sys.argv[9]
# LAYER_NORM=0              # sys.argv[10] (0 or 1)
# RATIO2=0.7                # sys.argv[11]
# ORDER=2                   # sys.argv[12]
# DEVICE=1                  # sys.argv[13]
# LR_UPDATE_WEIGHT=0.01     # sys.argv[14]
# RANDOM_SAMPLE=10          # sys.argv[15]
# EPOCHS=100                # sys.argv[16]
# RUNS=10                   # sys.argv[17]
# SEED=123                  # sys.argv[18]
# # Run the experiment
# python main_runner.py \
#   $DATASET $ARCH $LEARN_RATE $L2_PARAM $RATIO $USE_RANK_APPROX \
#   $NUM_LAYERS $HIDDEN_DIM $DROPOUT $LAYER_NORM $RATIO2 $ORDER $DEVICE \
#   $LR_UPDATE_WEIGHT $RANDOM_SAMPLE $EPOCHS $RUNS $SEED  


# Arguments for GAT sketch experiment
DATASET="pubmed"
ARCH="GAT"
LEARN_RATE=0.004           # sys.argv[3]
L2_PARAM=6e-3          # sys.argv[4]
RATIO=0.02                # sys.argv[5]
USE_RANK_APPROX="False"   # sys.argv[6]
NUM_LAYERS=3              # sys.argv[7]
HIDDEN_DIM=150            # sys.argv[8]
DROPOUT=0.0               # sys.argv[9]
LAYER_NORM=0              # sys.argv[10] (0 or 1)
RATIO2=0.7                # sys.argv[11]
ORDER=2                   # sys.argv[12]
DEVICE=1                  # sys.argv[13]
LR_UPDATE_WEIGHT=0.01     # sys.argv[14]
RANDOM_SAMPLE=400          # sys.argv[15]
EPOCHS=200                # sys.argv[16]
RUNS=10                   # sys.argv[17]
SEED=2                 # sys.argv[18]
# Run the experiment
python main_runner.py \
  $DATASET $ARCH $LEARN_RATE $L2_PARAM $RATIO $USE_RANK_APPROX \
  $NUM_LAYERS $HIDDEN_DIM $DROPOUT $LAYER_NORM $RATIO2 $ORDER $DEVICE \
  $LR_UPDATE_WEIGHT $RANDOM_SAMPLE $EPOCHS $RUNS $SEED
# Arguments for PSGC sketch experiment