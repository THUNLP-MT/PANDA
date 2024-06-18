# for task in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
# for task in 7 12 16 18 21
for task in 28
do
    # TOKENIZERS_PARALLELISM=false python InverseRL_agent_react.py \
    #     --task_nums $task \
    #     --set train \
    #     --no_stop --cut_off --do_ablation_train\
    #     --env_step_limit 100 \
    #     --simplification_str easy \
    #     --prompt_file ReAct_baseline/prompt.jsonl \
    #     --output_path Inverse_Train_ReAct_logs/gpt-3.5-turbo-1106 \
    #     --model_name gpt-3.5-turbo-1106
    if [[ "$task" -eq 7 || "$task" -eq 21 || "$task" -eq 28 ]]; then
        max_train_num=50
    else
        max_train_num=200
    fi
    trials_w_obs_files=$(echo ./trials_v2/iteration_3/task$task-0-*.json)

    TOKENIZERS_PARALLELISM=false python InverseRL_agent_react.py \
        --task_nums $task \
        --set train \
        --max_train_num $max_train_num \
        --no_stop --cut_off --do_ablation_train\
        --env_step_limit 100 \
        --simplification_str easy \
        --prompt_file ReAct_baseline/prompt.jsonl \
        --output_path Inverse_Train_ReAct_logs/ablation \
        --trials_w_samples_path ./trials_v2/iteration_3/task${task}_action_samples.json \
        --trials_w_obs $trials_w_obs_files
done