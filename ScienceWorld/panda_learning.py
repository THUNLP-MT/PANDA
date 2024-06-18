 
import argparse
import os
import re
import time
import datetime
import random
from scienceworld import ScienceWorldEnv
import json
import sys
from data_utils import sanitizeStr
from eval_utils import findValidActionNew, load_variation, findValidActionNewWithTopK
from utils import completion_with_backoff
import tiktoken
import openai
from sentence_transformers import SentenceTransformer
openai.api_key = os.environ["OPENAI_API_KEY"]
import logging
from logging import INFO, WARN

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

# Call language model
def llm_gpt(prompt, logger=None, stop=["\n"], model_name="gpt-3.5-turbo", is_reflection=False):
    if "instruct" in model_name:
        input_prompt = prompt
    else:
        if is_reflection:
            input_prompt = [{"role": "user", "content": prompt}]
        else:
            input_prompt = [{"role": "user", "content": prompt}]
    if is_reflection:
        max_tokens = 150
    else:
        max_tokens = 50

    response = completion_with_backoff(logger=logger,
        model=model_name,
        messages=input_prompt,
        n=1, 
        temperature=0,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop
        )
    if logger:
        logger.info(f"This call's response: {response}")
    
    return response
    

def get_file_name(args, task_num, cur_time, eval_id=None):
    if (len(args["output_path"]) > 0):
        # args["output_path"] = os.path.join(args["output_path"], cur_time)
        output_path = os.path.join(args["output_path"], cur_time)
        # Make path if it doesn't exist
        if (not os.path.exists(output_path)):
            os.makedirs(output_path)

    if eval_id:
        filenameOutPrefixSeed = os.path.join(output_path, "task" + str(task_num) +f'_eval_{eval_id}')
    else:
        filenameOutPrefixSeed = os.path.join(output_path, "task" + str(task_num))
    return filenameOutPrefixSeed
  
def read_json(file_name):
    if file_name=="":
        return None
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def load_sbert(sbert_path='./sentence-transformers_paraphrase-MiniLM-L6-v2', device="cuda"):
    sbert_model = SentenceTransformer(sbert_path).to(device)
    return sbert_model



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jar_path", type=str, default="") 
    parser.add_argument("--task_nums", default="0")  # use comma to split 
    parser.add_argument("--env_step_limit", type=int, default=100)
    parser.add_argument("--simplification_str", default="easy")
    parser.add_argument("--max_episode_per_file", type=int, default=9999)
    parser.add_argument("--set", default="test_mini")
    parser.add_argument("--output_path", default="BC_ReAct/")
    parser.add_argument("--no_stop", action="store_true", default=False)
    parser.add_argument("--prompt_file", default="ReAct_baseline/prompt.jsonl")
    parser.add_argument("--model_name", default="gpt-3.5-turbo-1106")
    parser.add_argument("--cut_off", action="store_true", default=True)
    parser.add_argument("--do_train", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument("--is_debug", action="store_true", default=False)
    parser.add_argument("--eval_with_bc_insight", action="store_true", default=False)
    parser.add_argument("--eval_with_inverse_insight", action="store_true", default=False)
    parser.add_argument("--top_k", type=int, default=5)

    parser.add_argument("--insight_config", type=int, default=0)
    parser.add_argument("--retrieve_k", type=int, default=1)
    parser.add_argument("--panda_insights_path", default="")
    parser.add_argument("--raw_preference_path", default="")

    parser.add_argument("--do_ablation_train", action="store_true", default=False)

    parser.add_argument("--analysis_num", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--max_train_num", type=int, default=50)

    parser.add_argument("--trials_w_samples_path", default="./trials_v2/iteration_0/action_samples.json")
    parser.add_argument("--trials_w_obs", type=str, default="./training_swift_trials/task18-0-61.json")
    args = parser.parse_args()
    params = vars(args)
    return params

def init_logger(args, task_num, cur_time, eval_id=None, log_level=INFO):
    filenameOutPrefixSeed = get_file_name(args, task_num, cur_time, eval_id=eval_id)
    logger = logging.getLogger()
    formatter = logging.Formatter("[%(asctime)s][%(levelname)s\t] %(message)s",
                                    datefmt='%Y-%m-%d %H:%M:%S')
    logger.setLevel(log_level)

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)


    logging_dir = args["output_path"]
    if logging_dir:
        os.makedirs(logging_dir, exist_ok=True)
        now = int(round(time.time() * 1000))
        timestr = time.strftime('%Y-%m-%d_%H-%M', time.localtime(now / 1000))
        filename = f"{filenameOutPrefixSeed}.log"
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(fh)
    return logger, filenameOutPrefixSeed

def env_redo_laststep(env, new_action, actions_buffer):
    obs, info = env.reset()
    for action in actions_buffer:
        obs, reward, done, info = env.step(action['action'])
    return env.step(new_action)   

def env_rerun_trials(env, actions_buffer):
    obs, info = env.reset()
    for action in actions_buffer:
        obs, reward, done, info = env.step(action['action'])
    return# return obs, reward, done, info


def find_last_no_up(trials):
    t_trials = trials[::-1]
    cnt = 0
    t_score = t_trials[0]['score']
    while(cnt < len(t_trials) and t_trials[cnt]['score']==t_score):
        cnt += 1
    return len(t_trials)-cnt

def panda_learning(args, env, logger, task_num, variations_id, analysis_num=2):
    # with open(args["trials_w_samples_path"], 'r') as f:
    #     trials_w_samples = json.load(f)
    # task_id = '18'
    task_id = str(task_num)
    taskName = env.getTaskNames()[int(task_id)]
    actions_samples = read_json(args["trials_w_samples_path"])[task_id]
    # variations_id = list(actions_samples.keys())
    trials_file = read_json(args["trials_w_obs"])

    #check the consistency of 'action_samples' with 'trials_file'
    check_passed = True
    for v_id in variations_id:
        if not check_passed:
            break
        t_trial = trials_file[v_id]['history']['history']
        sample_trial = actions_samples[v_id]
        if len(t_trial) != len(sample_trial)+1:
            check_passed = False
            break
        for i in range(len(t_trial)-1):
            if sample_trial[i][0] != t_trial[i+1]['action']:
                check_passed = False
                break            
    assert check_passed, "Not pass the consistency-check of 'action_samples' with 'trials_file'"

    prompt_0shot_ablation2 = """The expert's trial up to now is as follows:
{}
Now it's time to act again, the expert prefers to {} rather than to {}. Please explain the reason why the expert holds on this preference.
Expert_insight:
"""
    prompt_0shot_ablation1 = """The expert's trial up to now is as follows:
{}
Now it's time to act again, the expert prefers to {}. Please explain the reason why the expert holds on this preference.
Expert_insight:
"""
    prompt_0shot_ablation3 = """The expert's trial up to now is as follows:
{}
Now it's time to act again, the expert prefers to {} rather than to {} or to {}. Please explain the reason why the expert holds on this preference.
Expert_insight:"""


    if "gpt-3.5-turbo" in args["model_name"]:
        max_len = 4096
    elif args["model_name"] == "gpt-4":
        max_len = 8192
    else:
        max_len = 4096

    EARLY_STOP = 5

    accumulated_expert_insight = dict()
    gpt_call_num = 0
    # random.seed(args['seed'])
    # variations_id = random.sample(variations_id, min(args['max_train_num'], len(variations_id)))
    v_id_cnt = 0
    for v_id in variations_id:
        v_id_cnt += 1
        logger.info(f"COMPLETED {v_id_cnt}/{len(variations_id)}!!!")
        current_expert_insight = []
        # env.load(taskName, int(v_id), args['simplification_str'])
        task_description = trials_file[v_id]['history']['taskDescription']
        t_trial = trials_file[v_id]['history']['history']
        t_trial = t_trial[:find_last_no_up(t_trial)+EARLY_STOP]
        sample_trial = actions_samples[v_id][:find_last_no_up(t_trial)+3]
        trial_len = len(sample_trial)

        action = "look around"
        # obs = f"{t_trial[0]['freelook']}; {t_trial[0]['inventory']}; {task_description}"
        obs = f"{t_trial[0]['freelook']}\n {task_description}"
        obs = clean(obs)
        current_trial = f"Here is the task.\n{obs}\n> "
        for i in range(trial_len):
            if analysis_num == 2:
                current_prompt = prompt_0shot_ablation2.format(current_trial, sample_trial[i][0], sample_trial[i][1])
            elif analysis_num == 1:
                current_prompt = prompt_0shot_ablation1.format(current_trial, sample_trial[i][0])
            elif analysis_num == 3:
                current_prompt = prompt_0shot_ablation3.format(current_trial, sample_trial[i][0], sample_trial[i][1], sample_trial[i][2])
            _, current_prompt = compress_prompt("", current_prompt, model_name=args["model_name"], max_len=max_len, max_output_tokens=150)
            logger.info(f"PROMPT[{task_id}-{v_id}-{i}]:\n{current_prompt}")
            expert_insight = llm_gpt(current_prompt, logger=logger, is_reflection=True)
            gpt_call_num += 1
            #store the insight
            current_expert_insight.append({f"[action]: {action}; [observation]: {obs}": expert_insight})
            logger.info(f"INSIGHT[{task_id}-{v_id}-{i}]\n-> {current_expert_insight[-1]}")
            # obs = f"{t_trial[i+1]['observation']}; {t_trial[min(i+2,trial_len)]['freelook']}; {t_trial[min(i+2,trial_len)]['inventory']}"
            obs = f"{t_trial[i+1]['observation']}; {t_trial[min(i+2,trial_len)]['inventory']}; {t_trial[min(i+2,trial_len)]['freelook']}"
            obs = clean(obs)
            action = t_trial[i+1]['action']
            current_trial += f"{action}\n{obs}\n> "
            
        accumulated_expert_insight.update({f"{task_id}-{v_id}": current_expert_insight})
        # break
    logger.info(f"Calling OpenAI-API {gpt_call_num} times.")

    return accumulated_expert_insight, gpt_call_num



def ablation_panda_learning(args, env, task_num, variations_id, sample_num):
    # with open(args["trials_w_samples_path"], 'r') as f:
    #     trials_w_samples = json.load(f)
    task_id = str(task_num)
    taskName = env.getTaskNames()[int(task_id)]
    actions_samples = read_json(args["trials_w_samples_path"])[task_id]
    # variations_id = list(actions_samples.keys())
    trials_file = read_json(args["trials_w_obs"])
    #check the consistency of 'action_samples' with 'trials_file'
    check_passed = True
    for v_id in variations_id:
        if not check_passed:
            break
        t_trial = trials_file[v_id]['history']['history']
        sample_trial = actions_samples[v_id]
        if len(t_trial) != len(sample_trial)+1:
            check_passed = False
            break
        for i in range(len(t_trial)-1):
            if sample_trial[i][0] != t_trial[i+1]['action']:
                check_passed = False
                break            
    assert check_passed, "Not pass the consistency-check of 'action_samples' with 'trials_file'"

    accumulated_expert_insight = dict()
    gpt_call_num = 0
    for v_id in variations_id:
        current_expert_insight = []
        # env.load(taskName, int(v_id), args['simplification_str'])
        task_description = trials_file[v_id]['history']['taskDescription']
        t_trial = trials_file[v_id]['history']['history']
        sample_trial = actions_samples[v_id]
        trial_len = len(sample_trial)

        action = "look around"
        obs = f"{t_trial[0]['freelook']}\n {task_description}"
        obs = clean(obs)
        for i in range(trial_len):
            if sample_num == 1:
                expert_insight = f"The expert prefer to {sample_trial[i][0]}."
            elif sample_num == 2:
                expert_insight = f"The expert prefer to {sample_trial[i][0]} rather than to {sample_trial[i][1]}."
            elif sample_num == 3:
                expert_insight = f"The expert prefer to {sample_trial[i][0]} rather than to {sample_trial[i][1]} or to {sample_trial[i][2]}."
            else:
                raise Exception
            # gpt_call_num += 1
            #store the insight
            current_expert_insight.append({f"[action]: {action}; [observation]: {obs}": expert_insight})
            # logger.info(f"INSIGHT[{task_id}-{v_id}-{i}]\n-> {current_expert_insight[-1]}")
            obs = f"{t_trial[i+1]['observation']}; {t_trial[min(i+2,trial_len)]['inventory']}; {t_trial[min(i+2,trial_len)]['freelook']}"
            # obs = f"{t_trial[i+1]['observation']}; {t_trial[min(i+2,trial_len)]['freelook']}; {t_trial[min(i+2,trial_len)]['inventory']}"
            obs = clean(obs)
            action = t_trial[i+1]['action']
            # current_trial += f"{action}\n{obs}\n> "
            
        accumulated_expert_insight.update({f"{task_id}-{v_id}": current_expert_insight})
        # break
    # logger.info(f"Calling OpenAI-API {gpt_call_num} times.")

    return accumulated_expert_insight


def compress_prompt(init_prompt, prompt, model_name, max_len, max_output_tokens=100):
    encoding = tiktoken.encoding_for_model(model_name)
    # Cut the prompt to make it shorter than maximun token numbers
    while len(encoding.encode(init_prompt + prompt)) > max_len - max_output_tokens - 10:
        index1 = init_prompt.find('>')

        # If init prompt doesn't have actions, cut game prompt
        if index1 == -1:
            index1_prompt = prompt.find('>')
            index2_prompt = prompt.find('>', index1_prompt+1)
            prompt = prompt[:index1_prompt] + prompt[index2_prompt:]

        # Cut initial prompt
        else:
            index2 = init_prompt.find('>', index1+1)
            if index2 == -1:
                init_prompt = init_prompt[:index1]
            else:
                init_prompt = init_prompt[:index1] + init_prompt[index2:]
    return init_prompt, prompt

# Example user input console, to play through a game.
def eval(args, env, task_num, logger, inverse_insights, raw_preferences, sbert, cur_time, variation_ids, eval_ids):
    taskNames = env.getTaskNames()
    taskName = taskNames[task_num]
    # env.load(taskName, variation_id, args['simplification_str'])
    # variations = load_variation(env, args, task_num, logger)
    filenameOutPrefixSeed = get_file_name(args, task_num, cur_time)+f"_eval_{eval_ids}"

    # Load init prompt
    with open(args["prompt_file"], 'r') as f:
        d = json.load(f)
    
    # Load encoding tool to count token numbers
    encoding = tiktoken.encoding_for_model(args["model_name"])
    # plans = get_plans(args)

    scores = []
    sum_call_gpt_nums = 0
    for variation in variation_ids:
        call_gpt_num = 0
        # train_data = []
        env.load(taskName, variation, args["simplification_str"], generateGoldPath=True)
        task_description = env.taskdescription()[18:]
        recent_actions = ["look around"]
 
        obs, info = env.reset()

        done = False
        score = 0.0
        last_score = 0.0
        step = 0

        # The env has an internal step count, some actions like look around are free
        # however, the t5 model only generates the action "look around", which will result in a dead loop below
        # so the max_steps here is only used to avoid the model generating the same action forever
        max_steps = args["env_step_limit"] * 2
 
        init_prompt = 'Interact with a household to solve a task.{}Here is an example.\n' + d[str(task_num)]
        # init_prompt = 'Act to solve a task. Here is an example.\n' + d[str(task_num)]
        prompt = '\n\nHere is the task.\n' + clean(obs) + '\n' + task_description + '\n>'
        insights_prompt = """\n
In addition, these are some insights that may be helpful for you to improve the success rate:\n{}\n
"""#.format('\n'.join(outter_insights))    # 20 tokens
#         insights_prompt = """There are some similar states and the corresponding insights gained from the expert, which is considered to be a good understanding of the task:
# {}\n
# """
        # if args["model_name"] == "gpt-3.5-turbo":
        if "gpt-3.5-turbo" in args["model_name"]:
            max_len = 4096
        elif args["model_name"] == "gpt-4":
            max_len = 8192
        else:
            max_len = 4097
        obs = clean(obs + f'\n {task_description}')
        last_no_think_obs = obs
        while not done:  

            init_prompt, prompt = compress_prompt(init_prompt, prompt, model_name=args["model_name"], max_len=max_len, max_output_tokens=150+25) # insight_len + insight_instruct_len

            # retrieval_key = [f"[action]: {recent_actions[-1]};[observation]: {last_no_think_obs}"]
            retrieval_key = [f"[observation]: {last_no_think_obs}"]
            # last_retrieval_key = retrieval_key
            
            retrieved_insights = []
            if inverse_insights:
                retrieved_inverse_insights, _ = retrieve_insights(inverse_insights, raw_preferences, retrieval_key, sbert, is_expert=True, retrieve_k=args['retrieve_k'], config=args['insight_config'], logger=logger)
                retrieved_insights.extend(retrieved_inverse_insights)

            if len(retrieved_insights)>0:

                cur_prompt = init_prompt.format('') + '\n' + insights_prompt.format("\n".join(retrieved_insights)) + '\n\n' + prompt
                    
            else:
                cur_prompt = init_prompt.format('') + prompt

            logger.info("Prompt: " + cur_prompt)
            action = llm_gpt(cur_prompt, logger=logger, stop=['\n'], model_name=args["model_name"])
            call_gpt_num += 1
            action = action.strip()
            logger.info(f"api_response: {action}")
            if action.startswith('think:'):
                obs = 'OK.'
            else:
                # Get valid actions at this point
                action = findValidActionNew([action], env, info['look'], recent_actions, None, logger, is_gpt=True)
                obs, reward, done, info = env.step(action)

                score = info['score']

                if score < 0:
                    # Our own solution for dealing with such cases
                    if args["no_stop"]:
                        done = True
                        score = last_score
                    else:
                        done = True
                        score = 0
                last_score = score
            
            if "No known action matches that input" not in obs and "Unknown action" not in obs:
                obs += (';'+info['inv']+';'+info['look'])
                obs = clean(obs)

            # Add action and observaton to game prompt
            prompt += f' {action}\n{obs}\n>'
            if not action.startswith('think'):
                recent_actions.append(action)
                if "No known action matches that input" not in obs and "Unknown action" not in obs:
                    last_no_think_obs = obs
            #logger.info("Input string: " + str(input_str))
            logger.info(f"Variation: {variation}, Step: {step}, Action: {action}")
            logger.info("Obs: " + obs)
            logger.info(f"Score: {score}")
            logger.info("")

            step += 1
            if (step >= max_steps) or done:
                if step >= max_steps:
                    logger.info(f"[Stopped] step{step} >= max_steps{max_steps}")
                else:
                    logger.info(f"[Stopped] done[state] from the environment or no_stop")
                break
  

            logger.info("Recent Actions: " + str(recent_actions))

            # Early stopping if we're in a loop
            # if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
            if len(recent_actions) >= 10 and len(set(recent_actions[-10:])) <= 6:
                logger.info("Many recent actions in history are the same -- model is likely in a loop, stopping early.")
                break


        sum_call_gpt_nums += call_gpt_num
        # Store results
        logger.info(f"current trial's {args['model_name']} calling num: {call_gpt_num}, current all calling num: {sum_call_gpt_nums}")
        env.storeRunHistory(variation, notes = {'mode':"react_baseline", 'lm': None} )
        env.saveRunHistoriesBufferIfFull(filenameOutPrefixSeed, maxPerFile=args["max_episode_per_file"])

        scores.append(score)

        logger.info("Run completed...")
        logger.info("Scores: " + str(scores))
        time.sleep(2)

    # Episodes are finished -- manually save any last histories still in the buffer
    env.saveRunHistoriesBufferIfFull(filenameOutPrefixSeed, maxPerFile=args["max_episode_per_file"], forceSave=True)

    avg = sum(scores) / len(scores)
    logger.info("Average score: " + str(avg))

    f = open(filenameOutPrefixSeed + "-score.txt", "a")
    f.write("\n" + "Task name:" + taskName + "Scores: " + str(scores) + " Average score: " + str(avg) + " Args: " + str(args) + "\n")
    f.close()

    logger.info("Shutting down server...")
    # env.shutdown()

    logger.info("Completed.")
    return scores, avg, sum_call_gpt_nums


def remove_task_name_key(dictionary):
    if dictionary is None:
        return None
    new_dicts = []
    # v_idxs = []
    for key in dictionary:
        # v_idxs.append(key)
        level_dict = dictionary[key]
        new_dicts.extend(level_dict)
        # v_idxs.extend([key]*len(level_dict))
    return new_dicts


def retrieve_insights(accumulate_insights, raw_preferences, retrieval_key, sbert, is_expert=True, retrieve_k=1, config=1, logger=None):
    insights = []

    if len(accumulate_insights)==0:
        return [], None


    insights_pool_keys = [list(i.keys())[0] for i in accumulate_insights]
    insights_pool_keys = [i[i.find('[observation]'):] for i in insights_pool_keys]


    retrieval_vector = sbert.encode(retrieval_key, show_progress_bar=False)
    insights_pool_vectors = sbert.encode(insights_pool_keys, batch_size=min(len(insights_pool_keys), 128), show_progress_bar=False)
    similarity_matrix = cosine_similarity(retrieval_vector, insights_pool_vectors)[0]
    top_indice = np.argsort(similarity_matrix)[::-1]

    insight_template = "TASK-CONTEXT: {}\nINSIGHT: {}"

    if is_expert:
        
        for i in range(retrieve_k):
            if raw_preferences:
                insights.append(str(list(raw_preferences[top_indice[i]].values())[0]) +' '+ str(list(accumulate_insights[top_indice[i]].values())[0]))
            else:
                insights.append(str(list(accumulate_insights[top_indice[i]].values())[0]))
            logger.info(f"retrievalKey: {retrieval_key}")
            logger.info(f"RetrievedKEY: {insights_pool_keys[top_indice[i]]}")
            logger.info(f"RetrievedINSIGHTS: {list(accumulate_insights[top_indice[i]].values())[0]}")
    else:
        for i in range(retrieve_k):
            insights.append(list(accumulate_insights[top_indice[i]].values())[0])
    return insights, similarity_matrix


def main():
    args = parse_args()
    print(args)
    is_debug, do_train, do_eval, do_ablation_train = args['is_debug'], args['do_train'], args['do_eval'], args['do_ablation_train']
    analysis_num = args['analysis_num']
    sum_gpt_call_num = 0
    env = ScienceWorldEnv("", args["jar_path"], envStepLimit = args["env_step_limit"])
    for task_num in [int(args['task_nums'])]:
        cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        if do_train or do_ablation_train:
            task_id = str(task_num)
            taskName = env.getTaskNames()[int(task_id)]
            actions_samples = read_json(args["trials_w_samples_path"])[task_id]
            variations_id = list(actions_samples.keys())

            random.seed(args['seed'])
            variations_id = random.sample(variations_id, min(args['max_train_num'], len(variations_id)))

        if do_train:
            logger, file_name_prefix = init_logger(args, task_num, cur_time)
            logger.info(args)
            accummulated_insights, _ = panda_learning(args, env, logger, task_num, variations_id, analysis_num)
            with open(f"{file_name_prefix}_leap{analysis_num}_expert_insights.json", 'w') as f:
                json.dump(accummulated_insights, f)
        
        if do_ablation_train:
            logger, file_name_prefix = init_logger(args, task_num, cur_time)
            logger.info(args)
            for ablation_num in [1,2,3]:
                accummulated_insights = ablation_panda_learning(args, env, task_num, variations_id, ablation_num)
                with open(f"{file_name_prefix}_ablation{ablation_num}_expert_insights.json", 'w') as f:
                    json.dump(accummulated_insights, f)


        if do_eval:
            sbert = load_sbert(device='cuda')
            if args['eval_with_inverse_insight']:
                inverse_insights = remove_task_name_key(read_json(args['panda_insights_path']))
                raw_preferences = remove_task_name_key(read_json(args['raw_preference_path']))
            else:
                inverse_insights = []
            taskNames = env.getTaskNames()
            taskName = taskNames[task_num]
            env.load(taskName, 0, args['simplification_str'])
            
            variations = list(env.getVariationsTest())
            test_len = min(10, len(variations))
            random.seed(1)
            random.shuffle(variations)
            # test_len = 1
            eval_variation_ids = variations[:test_len]
            avg_scores = []
            test_rounds = 5
            for i in range(test_rounds):
                logger, file_name_prefix = init_logger(args, task_num, cur_time, eval_id=i)
                logger.info("obs as retrieval_key")
                logger.info(args)
                scores, avg, cur_gpt_call_num = eval(args, env, task_num, logger, inverse_insights, raw_preferences, sbert, cur_time, eval_variation_ids, i)
                logger.info(f"Scores are {scores}.\nAvg score= {avg}")
                avg_scores.append(avg)
                sum_gpt_call_num += cur_gpt_call_num
            logger.info(f"After {test_rounds} times experiments, Scores are {avg_scores}.\nAvg score= {sum(avg_scores)/len(avg_scores)}")

if __name__ == "__main__":
    main()