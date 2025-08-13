# This script is used to schedule the execution of multiple experiments.
python scripts/02_plan.py --dataset-id 901 -y
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 901 --fold 0 -y --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 901 --fold 1 -y --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 901 --fold 2 -y --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 901 --fold 3 -y --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=0 python scripts/02_train.py --dataset-id 901 --fold 4 -y --plan nnUNetResEncUNetLPlans