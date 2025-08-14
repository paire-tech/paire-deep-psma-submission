# This script is used to train the nnUNet baseline model on the FDG dataset.
# -> It uses the nnUNetResEncUNetLPlans for training.

# ----------------------------------------------------------------------------------------------------------
# Using the Default nnUNet
# ----------------------------------------------------------------------------------------------------------
# Step 1: Preprocessing
# python scripts/01a_preprocess_baseline.py --data-dir /data/DEEP_PSMA_CHALLENGE_DATA/CHALLENGE_DATA --tracer-name PSMA --dataset-id 801 -y
# ...or if the dataset is already preprocessed, you can just generate the plans:
# python scripts/02_plan.py --dataset-id 801 --plan ExperimentPlanner -y

# Step 2: Training
# CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 801 --fold 0 --plan nnUNetPlans -y
# CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 801 --fold 1 --plan nnUNetPlans -y
# CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 801 --fold 2 --plan nnUNetPlans -y
# CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 801 --fold 3 --plan nnUNetPlans -y
# CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 801 --fold 4 --plan nnUNetPlans -y


# ----------------------------------------------------------------------------------------------------------
# Using the Residual Encoder nnUNet (Large)
# ----------------------------------------------------------------------------------------------------------
# Step 1: Preprocessing
# python scripts/01a_preprocess_baseline.py --data-dir /data/DEEP_PSMA_CHALLENGE_DATA/CHALLENGE_DATA --tracer-name PSMA --dataset-id 801 -y
# ...or if the dataset is already preprocessed, you can just generate the plans:
# python scripts/02_plan.py --dataset-id 801 --plan nnUNetPlannerResEncL -y

# Step 2: Training
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 801 --fold 0 --plan nnUNetResEncUNetLPlans -y
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 801 --fold 1 --plan nnUNetResEncUNetLPlans -y
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 801 --fold 2 --plan nnUNetResEncUNetLPlans -y
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 801 --fold 3 --plan nnUNetResEncUNetLPlans -y
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 801 --fold 4 --plan nnUNetResEncUNetLPlans -y
