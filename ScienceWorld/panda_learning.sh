# for task in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
ana_num=1
# for task in 7

# 25, 28
# for task in 0 21 29 10 2 17 13 15 23
# for task in 24 16 12 18 4 11 6 1 7
for task in 28
# for task in 25 28
do
    if [[ "$task" -eq 7 || "$task" -eq 21 || "$task" -eq 29 || "$task" -eq 28 || "$task" -eq 10 ||  "$task" -eq 17 ||  "$task" -eq 15 ||  "$task" -eq 23 || "$task" -eq 6 || "$task" -eq 11 || "$task" -eq 16 || "$task" -eq 18 || "$task" -eq 24 ]]; then
        max_train_num=50
    else
        max_train_num=200
    fi
    trials_w_obs_files=$(echo ./trials_v2/iteration_3/task$task-0-*.json)
    CUDA_VISIBLE_DEVICES=4,5 TOKENIZERS_PARALLELISM=false python panda_learning.py \
        --task_nums $task \
        --max_train_num $max_train_num \
        --set train \
        --no_stop --cut_off --do_train\
        --env_step_limit 100 \
        --analysis_num 3 \
        --simplification_str easy \
        --prompt_file ReAct_baseline/prompt.jsonl \
        --output_path Inverse_Train_ReAct_logs/gpt-3.5-turbo-1106/summary_0129_leap \
        --model_name gpt-3.5-turbo-1106 \
        --trials_w_samples_path ./trials_v2/iteration_3/task${task}_action_samples.json \
        --trials_w_obs $trials_w_obs_files &
    sleep 5

    # CUDA_VISIBLE_DEVICES=4,5 TOKENIZERS_PARALLELISM=false python InverseRL_agent_react.py \
    #     --task_nums $task \
    #     --max_train_num $max_train_num \
    #     --set train \
    #     --no_stop --cut_off --do_train\
    #     --env_step_limit 100 \
    #     --analysis_num 2 \
    #     --simplification_str easy \
    #     --prompt_file ReAct_baseline/prompt.jsonl \
    #     --output_path Inverse_Train_ReAct_logs/gpt-3.5-turbo-1106/summary_0129_leap \
    #     --model_name gpt-3.5-turbo-1106 \
    #     --trials_w_samples_path ./trials_v2/iteration_3/task${task}_action_samples.json \
    #     --trials_w_obs $trials_w_obs_files &
    # sleep 5
done

# for task in 24 16 12 18 4 11 1
# do
#     if [[ "$task" -eq 7 || "$task" -eq 21 || "$task" -eq 6 || "$task" -eq 11 || "$task" -eq 16 || "$task" -eq 18 || "$task" -eq 24 ]]; then
#         max_train_num=50
#     else
#         max_train_num=200
#     fi
#     trials_w_obs_files=$(echo ./trials_v2/iteration_3/task$task-0-*.json)
#     CUDA_VISIBLE_DEVICES=4,5 TOKENIZERS_PARALLELISM=false python InverseRL_agent_react.py \
#         --task_nums $task \
#         --max_train_num $max_train_num \
#         --set train \
#         --no_stop --cut_off --do_train\
#         --env_step_limit 100 \
#         --analysis_num 2 \
#         --simplification_str easy \
#         --prompt_file ReAct_baseline/prompt.jsonl \
#         --output_path Inverse_Train_ReAct_logs/summary_0131_leap_v2 \
#         --model_name gpt-3.5-turbo-1106 \
#         --trials_w_samples_path ./trials_v2/iteration_3/task${task}_action_samples.json \
#         --trials_w_obs $trials_w_obs_files &
#     sleep 5

#      CUDA_VISIBLE_DEVICES=4,5 TOKENIZERS_PARALLELISM=false python InverseRL_agent_react.py \
#         --task_nums $task \
#         --max_train_num $max_train_num \
#         --set train \
#         --no_stop --cut_off --do_train\
#         --env_step_limit 100 \
#         --analysis_num 1 \
#         --simplification_str easy \
#         --prompt_file ReAct_baseline/prompt.jsonl \
#         --output_path Inverse_Train_ReAct_logs/summary_0131_leap_v2 \
#         --model_name gpt-3.5-turbo-1106 \
#         --trials_w_samples_path ./trials_v2/iteration_3/task${task}_action_samples.json \
#         --trials_w_obs $trials_w_obs_files &
#     sleep 5
# done