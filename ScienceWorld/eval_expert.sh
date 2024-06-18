L=5
model_path=path/to/expert_model
gpu=5
output_path=path/to/output
L=6
# task_nums=("5,6,7,8" "9,10,11,12" "13,14,15,16" "17,18,19,20" "21,22,23,24" "25,26,27,28,29")
task_nums=("0" "1" "2" "3" "4")
for ((i=0; i<L; i++)); do
    task_num=${task_nums[i]}
    ((gpu=i))
    echo $task_num "on" $gpu
    TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=$gpu python eval_agent_fast_only.py \
        --task_nums $task_num \
        --set test \
        --cut_off --no_stop\
        --simplification_str easy \
        --env_step_limit 100 \
        --lm_path $model_path \
        --output_path $output_path &
    sleep 5
done