# for task in 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29
for task in 6
do
    TOKENIZERS_PARALLELISM=false python InverseRL_agent_react.py \
        --task_nums $task \
        --no_stop --cut_off --do_eval --eval_with_ablation_inverse_insight\
        --ablation_inverse_insights_path ./Inverse_Train_ReAct_logs/gpt-3.5-turbo-1106/task6-ablation_inverse_insights/task6_ablation1_expert_insights.json \
        --env_step_limit 100 \
        --simplification_str easy \
        --prompt_file ReAct_baseline/prompt.jsonl \
        --output_path Inverse_Eval_ReAct_logs/gpt-3.5-turbo-1106 \
        --model_name gpt-3.5-turbo-1106
    
    TOKENIZERS_PARALLELISM=false python InverseRL_agent_react.py \
        --task_nums $task \
        --no_stop --cut_off --do_eval --eval_with_ablation_inverse_insight\
        --ablation_inverse_insights_path ./Inverse_Train_ReAct_logs/gpt-3.5-turbo-1106/task6-ablation_inverse_insights/task6_ablation2_expert_insights.json \
        --env_step_limit 100 \
        --simplification_str easy \
        --prompt_file ReAct_baseline/prompt.jsonl \
        --output_path Inverse_Eval_ReAct_logs/gpt-3.5-turbo-1106 \
        --model_name gpt-3.5-turbo-1106

    TOKENIZERS_PARALLELISM=false python InverseRL_agent_react.py \
        --task_nums $task \
        --no_stop --cut_off --do_eval --eval_with_ablation_inverse_insight\
        --ablation_inverse_insights_path ./Inverse_Train_ReAct_logs/gpt-3.5-turbo-1106/task6-ablation_inverse_insights/task6_ablation3_expert_insights.json \
        --env_step_limit 100 \
        --simplification_str easy \
        --prompt_file ReAct_baseline/prompt.jsonl \
        --output_path Inverse_Eval_ReAct_logs/gpt-3.5-turbo-1106 \
        --model_name gpt-3.5-turbo-1106
done