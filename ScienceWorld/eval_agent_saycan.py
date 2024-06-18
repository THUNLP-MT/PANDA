 
import argparse
import os
import re
import time
import random
from scienceworld import ScienceWorldEnv
import json
import sys
from data_utils import sanitizeStr
from eval_utils import findValidActionNew, load_variation
from utils import completion_with_backoff_saycan_thumt as completion_with_backoff
# from slow_agent.utils import completion_with_backoff
import tiktoken
# import openai

# openai.api_key = os.environ["OPENAI_API_KEY"]
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from logging import INFO, WARN
from sentence_transformers import SentenceTransformer
# sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
from sklearn.metrics.pairwise import cosine_similarity
def clean(s):
    clean_toks = ['\n', '\t']
    for tok in clean_toks:
        s = s.replace(tok, ' ')
    return s

# Call language model
def llm_gpt(prompt, stop=["\n"], model_name="gpt-3.5-turbo"):
    response = completion_with_backoff(
      model=model_name,
      messages=[{"role": "user", "content": prompt}],
      n=5, 
      temperature=0.5, 
      max_tokens=50,
      top_p=1,
      frequency_penalty=0.0,
      presence_penalty=0.0,
      stop=stop
    )
    return response
    # return [i["message"]["content"].strip() for i in response["choices"]]
    # return response["choices"][0]["message"]["content"]

# def llm(prompt, stop=["\n"]):
#     response = openai.Completion.create(
#       model="text-davinci-002",
#       prompt=prompt,
#       temperature=0,
#       max_tokens=50,
#       top_p=1,
#       frequency_penalty=0.0,
#       presence_penalty=0.0,
#       stop=stop
#     )
#     return response["choices"][0]["text"]

def get_file_name(args, task_num):
    if (len(args["output_path"]) > 0):
        args["output_path"] = args["output_path"] + "/"

        # Make path if it doesn't exist
        if (not os.path.exists(args['output_path'])):
            os.makedirs(args["output_path"])

    # filenameOutPrefix = args["output_path"] + "transformer-" + args["mode"] + "-eval-" + str(args["lm_path"].split('/')[-1]) + "-task" + str(task_num)
    filenameOutPrefixSeed = args["output_path"] + "task" + str(task_num)

    return filenameOutPrefixSeed
  


# Example user input console, to play through a game.
def eval(args, task_num, leap_insights, sbert, filenameOutPrefixSeed, logger):

    # Initialize environment
    # env = ScienceWorldEnv("", args["jar_path"], envStepLimit = args["env_step_limit"], threadNum = 0)
    env = ScienceWorldEnv("", args["jar_path"], envStepLimit = args["env_step_limit"])
    taskNames = env.getTaskNames()
    taskName = taskNames[task_num]
    env.load(taskName, 0, args['simplification_str'])
    variations = load_variation(env, args, task_num, logger)
    filenameOutPrefixSeed = get_file_name(args, task_num)

    # Load init prompt
    with open(args["prompt_file"], 'r') as f:
        d = json.load(f)
    
    # Load encoding tool to count token numbers
    encoding = tiktoken.encoding_for_model(args["model_name"])
    # plans = get_plans(args)

    scores = []

    for variation in variations:

        # train_data = []
        env.load(taskName, variation, args["simplification_str"], generateGoldPath=True)
        task_description = env.taskdescription()[18:]
        # task_description = env.taskdescription()  
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
 
        init_prompt = 'Interact with a household to solve a task. Here is an example.\n' + d[str(task_num)]
        prompt = '\n\nHere is the task.\n' + clean(obs) + '\n' + task_description + '\n>'

        # Different models have different maximun token numbers
        if args["model_name"] == "gpt-3.5-turbo-1106":
            max_len = 4096
        elif args["model_name"] == "gpt-4":
            max_len = 8192
        else:
            max_len = 4097

        insights_prompt = """\n\nIn addition, these are some insights that may be helpful for you to improve the success rate:\n{}"""

        obs = clean(obs + f'\n {task_description}')
        last_obs = obs
        while not done:        
            
            retrieved_insights = []

            if leap_insights:
                retrieval_key = [f"[observation]: {last_obs}"]
                retrieved_inverse_insights, _ = retrieve_insights(leap_insights, retrieval_key, sbert, retrieve_k=args['retrieve_k'], logger=logger)
                retrieved_insights.extend(retrieved_inverse_insights)

            if len(retrieved_insights)>0:
                leap_prompt = insights_prompt.format("\n".join(retrieved_insights))
            else:
                leap_prompt = ""

            # Cut the prompt to make it shorter than maximun token numbers
            while len(encoding.encode(init_prompt + leap_prompt + prompt)) > max_len - 60:
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

            logger.info("Prompt: " + init_prompt + leap_prompt + prompt)
            # action = llm(init_prompt + prompt, stop=['\n']).strip()
            action = llm_gpt(init_prompt + leap_prompt + prompt, stop=['\n'], model_name=args["model_name"])
            print(action)

            # Get valid actions at this point
            action = findValidActionNew(action, env, info['look'], recent_actions, sbert, logger)
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
            
            obs += (';'+info['inv']+';'+info['look'])
            obs = clean(obs)
            prompt += f' {action}\n{obs}\n>'
            
            recent_actions.append(action) 
            
            #logger.info("Input string: " + str(input_str))
            logger.info(f"Variation: {variation}, Step: {step}, Action: {action}")
            logger.info("Obs: " + obs)
            logger.info(f"Score: {score}")
            logger.info("")

            step += 1
            if (step >= max_steps) or done:
                break
  

            logger.info("Recent Actions: " + str(recent_actions))

            # Early stopping if we're in a loop
            if len(recent_actions) >= 5 and len(set(recent_actions[-5:])) == 2:
                logger.info("Many recent actions in history are the same -- model is likely in a loop, stopping early.")
                break


        # Store results
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
    return avg



def retrieve_insights(accumulate_insights, retrieval_key, sbert, retrieve_k=1, logger=None):
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
        
    for i in range(retrieve_k):
        insights.append(str(list(accumulate_insights[top_indice[i]].values())[0]))
        logger.info(f"retrievalKey: {retrieval_key}")
        logger.info(f"RetrievedKEY: {insights_pool_keys[top_indice[i]]}")
        logger.info(f"RetrievedINSIGHTS: {list(accumulate_insights[top_indice[i]].values())[0]}")

    return insights, similarity_matrix



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jar_path", type=str, default="") 
    parser.add_argument("--task_nums", default="0")  # use comma to split 
    parser.add_argument("--env_step_limit", type=int, default=100)
    parser.add_argument("--simplification_str", default="easy")
    parser.add_argument("--max_episode_per_file", type=int, default=9999)
    parser.add_argument("--set", default="test_mini")
    parser.add_argument("--output_path", default="")
    parser.add_argument("--no_stop", action="store_true", default=False)
    parser.add_argument("--prompt_file", default="ReAct_baseline/prompt.jsonl")
    parser.add_argument("--model_name", default="gpt-3.5-turbo")

    parser.add_argument("--retrieve_k", type=int, default=1)
    parser.add_argument("--panda_insights_path", type=str, default="")
    args = parser.parse_args()
    params = vars(args)
    return params



def read_json(file_name):
    if file_name=="":
        return None
    with open(file_name, 'r') as file:
        data = json.load(file)
    return data

def load_sbert(sbert_path='./sentence-transformers_paraphrase-MiniLM-L6-v2', device="cuda"):
    sbert_model = SentenceTransformer(sbert_path).to(device)
    return sbert_model


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
#
#   Main
#

def init_logger(args, task_num, log_level=INFO):
    filenameOutPrefixSeed = get_file_name(args, task_num)
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
    return logger

def main():
    args = parse_args()
    print(args) 
    test_rounds = 5
    sbert = load_sbert(device='cuda')
    if args['panda_insights_path']:
        leap_insights = remove_task_name_key(read_json(args['panda_insights_path']))
    else:
        leap_insights = None

    task_nums = args["task_nums"].split(",")
    for task_num in task_nums:
        logger = init_logger(args, task_num)
        logger.info(args)
        all_scores = []
        filenameOutPrefixSeed = get_file_name(args, task_num)
        for i in range(test_rounds):
            score = eval(args, int(task_num), leap_insights, sbert, filenameOutPrefixSeed, logger)
            all_scores.append(score)
        
        f = open(filenameOutPrefixSeed + "-score.txt", "a")
        f.write("\n\n" + "Task name:" + task_num + f"{test_rounds}-rounds-AvgScores: " + str(sum(all_scores)/len(all_scores)) + str(all_scores))
        f.close()

if __name__ == "__main__":
    main()