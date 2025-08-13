# This script is used to schedule the execution of multiple experiments.
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 902 --fold 0 -y --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 902 --fold 1 -y --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 902 --fold 2 -y --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 902 --fold 3 -y --plan nnUNetResEncUNetLPlans
CUDA_VISIBLE_DEVICES=1 python scripts/02_train.py --dataset-id 902 --fold 4 -y --plan nnUNetResEncUNetLPlans