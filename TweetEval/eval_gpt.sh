gpu_cnt=0
CUDA_VISIBLE_DEVICES=$gpu_cnt python panda.py --task_name sentiment --mode eval_fs --data_size 1000  --retrieve_k 6  --few_shot_k $shot_k\
    --panda_insight_path "path to insight pool"
