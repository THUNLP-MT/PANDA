gpu=0

CUDA_VISIBLE_DEVICES=0 python panda_cot.py --task_name sentiment --mode eval_fs --data_size 1000 --retrieve_k 6 \
    --leap_insight_path "path to insight pool"
