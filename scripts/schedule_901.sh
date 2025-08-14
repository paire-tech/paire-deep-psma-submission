# This script is used to train the nnUNet baseline model on the PSMA dataset.
# -> It uses the nnUNetResEncUNetLPlans for training.

# Step 1: Preprocessing
# python scripts/01a_preprocess_baseline.py --data-dir /data/DEEP_PSMA_CHALLENGE_DATA/CHALLENGE_DATA --tracer-name PSMA --dataset-id 901 -y
# ...or if the dataset is already preprocessed, you can just generate the plans:
# python scripts/02_plan.py --dataset-id 901 --plan ExperimentPlanner -y
# python scripts/02_plan.py --dataset-id 901 --plan nnUNetPlannerResEncL -y

# Step 2: Training
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 901 --fold 0 -y --resume # --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 901 --fold 1 -y # --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 901 --fold 2 -y # --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 901 --fold 3 -y # --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 901 --fold 4 -y # --plan nnUNetResEncUNetLPlans