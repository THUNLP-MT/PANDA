import numpy as np
import pandas as pd
import pathlib
from sklearn.metrics import accuracy_score, f1_score
import argparse
from transformers import pipeline
from utils import completion_with_backoff, load_sbert
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import logging, os, time, json
import datetime
from TweetEvalDataset import TweetEvalDataset

MODEL_NAME = "gpt-3.5-turbo-1106"

MODEL_OUTPUT2NUM_MAPPINGS = {
    'sentiment': {
        "LABEL_0": 0,
        "LABEL_1": 1,
        "LABEL_2": 2
    }
}

def read_json(filename):
    with open(filename, "r") as file:
        data = json.load(file)
    return data

def retrieve_insights(insight_pool, retrieval_key, sbert, retrieve_k=1, logger=None):
    retrieval_key = [retrieval_key]

    insight_pool_keys = [i['text'] for i in insight_pool]
    retrieval_vector = sbert.encode(retrieval_key, show_progress_bar=False)
    insights_pool_vectors = sbert.encode(insight_pool_keys, batch_size=min(len(insight_pool_keys), 128), show_progress_bar=False)
    similarity_matrix = cosine_similarity(retrieval_vector, insights_pool_vectors)[0]
    top_indice = np.argsort(similarity_matrix)[::-1]

    insights = []
    retrieved_keys = []
    for i in range(retrieve_k):
        insights.append(insight_pool[top_indice[i]]['insight'])
        retrieved_keys.append(insight_pool_keys[top_indice[i]])
        if logger:
            logger.info(f'Retrieval-key:\n{retrieval_key}\nRetrieved-key:\n{insight_pool_keys[top_indice[i]]}')
    return insights, retrieved_keys


def retrieve_pref(insight_pool, retrieval_key, sbert, retrieve_k=1, logger=None):
    retrieval_key = [retrieval_key]

    insight_pool_keys = [i['text'] for i in insight_pool]
    retrieval_vector = sbert.encode(retrieval_key, show_progress_bar=False)
    insights_pool_vectors = sbert.encode(insight_pool_keys, batch_size=min(len(insight_pool_keys), 128), show_progress_bar=False)
    similarity_matrix = cosine_similarity(retrieval_vector, insights_pool_vectors)[0]
    top_indice = np.argsort(similarity_matrix)[::-1]

    insights = []
    retrieved_keys = []
    for i in range(retrieve_k):
        insights.append(eval(insight_pool[top_indice[i]]['preferences'])[0])
        retrieved_keys.append(insight_pool_keys[top_indice[i]])
        if logger:
            logger.info(f'Retrieval-key:\n{retrieval_key}\nRetrieved-key:\n{insight_pool_keys[top_indice[i]]}')
    return insights, retrieved_keys

def evaluate_sota(dat: TweetEvalDataset, model_name:str="", MAPPING="", logger=None):
    #evaluate sota and store it's outputs samples
    # import os
    # os.environ["http_proxy"] = "http://127.0.0.1:10809"
    # os.environ["https_proxy"] = "http://127.0.0.1:10809"
    # model_name = "cardiffnlp/twitter-roberta-base-sentiment"
    model = pipeline("text-classification", model=model_name, truncation=True, max_length=128, top_k=5)
    predictions = model(dat.get_data())
    predictions = [str([MAPPING[j['label']] for j in i]) for i in predictions]

    expert_trials = [{'text': dat.data[idx], 'preferences': pred} for idx,pred in enumerate(predictions)]
    # df[expert_key] = predictions
    sota_answers = [eval(i)[0] for i in predictions]
    logger.info(get_results(dat, sota_answers))
    return expert_trials


def evaluate_gpt(dat: TweetEvalDataset, insight_pool=None, retrieve_k=1, config='zs', logger=None, verbose=10):
    cnt = 0
    last_time = time.time()
    # for idx, row in df.iterrows():
    gpt_results = []
    gpt_processed_res = []
    data_queries = dat.get_data_prompt(config=config)

    insights_template = "These are some insights that may be helpful for you to improve the success rate:\n{}\n\n"
    # insights_template = "{}\n\n"
    sbert = load_sbert()

    # insight_pool = read_json("logs/sentiment/gather_pref/2024-02-07 01:53:15/sentiment_1000_expert_trials.json")
    # logger.info("insight_path: logs/sentiment/gather_pref/2024-02-07 01:53:15/sentiment_1000_expert_trials.json")
    for idx, query in enumerate(data_queries):
        prefix = dat.get_prompt_prefix(config=config)
        # prefix = few_shot_w_learnings_prompt
        if not insight_pool:
            prompt = prefix + query
        else:
            insights, retrieved_text = retrieve_insights(insight_pool, retrieval_key=dat.data[idx], retrieve_k=retrieve_k, sbert=sbert, logger=logger)
            # expert_labels, retrieved_text = retrieve_pref(insight_pool, retrieval_key=dat.data[idx], retrieve_k=retrieve_k, sbert=sbert, logger=logger)
            # insights_pref = [dat.prompt_template.format(retrieved_text[i]) + f' {expert_labels[i]}' for i in range(retrieve_k)]
            insights = ["Text: {}\nINSIGHT: {}".format(retrieved_text[i], insights[i]) for i in range(retrieve_k)]
            # prompt = prefix + insights_template.format('\n'.join(insights)) + query
            prompt = insights_template.format('\n\n'.join(insights)) + prefix + query
            # prompt = prefix + insights_template.format(retrieved_text[0], '\n'.join(insights)) + query
            # insights = ["Text: {}\nINSIGHT: {}".format(retrieved_text[i], insights[i]) for i in range(retrieve_k)]
            # prompt = insights_template.format('\n\n'.join(insights_pref)) + prefix + query

        logger.info('PROMPT: \n' + prompt)
        gpt_raw = completion_with_backoff(model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                n=1, 
                temperature=0,
                max_tokens=50,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                top_p=1,
                stop=['\n']
                )
        # df.at[idx, answer_key] = gpt_raw
        processed_res = dat.post_process_func(gpt_raw)
        gpt_results.append({'prompt': prompt, 'res': gpt_raw, 'processed_res': processed_res})
        gpt_processed_res.append(processed_res)
        logger.info('RESPONSE: \n' + gpt_raw)
        cnt += 1
        if cnt%verbose==0:
            cur_time = time.time()
            logger.info(f"{cnt}/{len(dat.labels)}, cost {cur_time-last_time:.3f}s.")
            last_time = cur_time
    
    try:
        logger.info(get_results(dat, gpt_processed_res))
    except:
        logger.info("gpt-results are not cleaned to get results.")

    return gpt_results
    # return df

def get_results(dat: TweetEvalDataset, sota_answers) -> pd.DataFrame:
    y_true = dat.get_labels()
    # y_pred = sota_answers.to_numpy()
    y_pred = sota_answers
    
    f1_macro = f1_score(y_true=y_true, y_pred=y_pred, average="macro")
    f1_micro = f1_score(y_true=y_true, y_pred=y_pred, average="micro")
    acc = accuracy_score(y_true=y_true, y_pred=y_pred)
    
    return [
            {"metric": "F1_macro", "value": f1_macro},
            {"metric": "F1_micro", "value": f1_micro},
            {"metric": "acc", "value": acc},
        ]

def leap_learning(dat: TweetEvalDataset, expert_trials, learning_k:int=2, verbose:int=10, logger=None):
    cnt = 0
    last_time = time.time()
    # learnings_key = f"gpt-3.5-turbo-1106_learn_num{learning_k}_gt"
    insights_pool = []
    for idx in range(len(expert_trials)):
        # prompt = row[query_key]
        # expert_outputs = eval(row[expert_key])
        expert_trial = expert_trials[idx]
        query, expert_prediction = dat.prompt_template.format(expert_trial['text']), eval(expert_trial['preferences'])

        expert_prediction_text = [dat.mapping[i] for i in expert_prediction]
        # if learning_k==2:
        #     suffix = f"The expert prefers {expert_prediction_text[0]}({expert_prediction[0]}) rather than {expert_prediction_text[1]}({expert_prediction[1]}). Please explain the reason why the expert holds on this preference. Do not provide the answer, only the explanation."
        # elif learning_k==3:
        #     suffix = f"The expert prefers {expert_prediction_text[0]}({expert_prediction[0]}) rather than {expert_prediction_text[1]}({expert_prediction[1]}) or {expert_prediction_text[2]}({expert_prediction[2]}). Please explain the reason why the expert holds on this preference. Do not provide the answer, only the explanation."
        # else:
        #     raise NotImplementedError
        # if learning_k==2:
        #     suffix = f"Now it's time to response again, the expert prefers {expert_prediction_text[0]}({expert_prediction[0]}) rather than {expert_prediction_text[1]}({expert_prediction[1]}). Please deduce the essential approaches to solving such query from the expert's preferences. Summarize the query-analysis strategies rather than analyzing this specific problem."
        # elif learning_k==3:
        #     suffix = f"Now it's time to response again, the expert prefers {expert_prediction_text[0]}({expert_prediction[0]}) rather than {expert_prediction_text[1]}({expert_prediction[1]}) or {expert_prediction_text[2]}({expert_prediction[2]}). Please deduce the essential approaches to solving such query from the expert's preferences. Summarize the query-analysis strategies rather than analyzing this specific problem."
        # else:
        #     raise NotImplementedError         # To determine the sentiment of a given text,
        if learning_k==1:
            suffix = f"The expert prefers {expert_prediction_text[0]}({expert_prediction[0]}). Please explain the reason why the expert holds on this preference.\nTo determine the {dat.dataset_name} of a given text,"
        elif learning_k==2:
            suffix = f"The expert prefers {expert_prediction_text[0]}({expert_prediction[0]}) rather than {expert_prediction_text[1]}({expert_prediction[1]}). Please explain the reason why the expert holds on this preference.\nTo determine the {dat.dataset_name} of a given text,"
        elif learning_k==3:
            suffix = f"The expert prefers {expert_prediction_text[0]}({expert_prediction[0]}) rather than {expert_prediction_text[1]}({expert_prediction[1]}) or {expert_prediction_text[2]}({expert_prediction[2]}). Please explain the reason why the expert holds on this preference.\nTo determine the {dat.dataset_name} of a given text,"
        else:
            raise NotImplementedError


        learning_prompt = f"{query}\n\n{suffix}"
        logger.info("PROMPT: \n" + learning_prompt)
        gpt_raw = completion_with_backoff(model=MODEL_NAME,
                messages=[{"role": "user", "content": learning_prompt}],
                n=1, 
                temperature=0,
                max_tokens=150,
                frequency_penalty=0.0,
                presence_penalty=0.0,
                top_p=1,
                stop=[]
                )
        cur_insight = f"To determine the {dat.dataset_name} of a given text, {gpt_raw}"
        logger.info('INSIGHT: \n' + cur_insight)
        insights_pool.append({'text': expert_trial['text'], 'query_prompt': learning_prompt,
                              'insight': cur_insight})
        # print(gpt_raw, raw['chatgpt_raw'])
        # df.at[idx, learnings_key] = gpt_raw
        cnt += 1
        if cnt%verbose==0:
            cur_time = time.time()
            logger.info(f"{cnt}/{len(expert_trials)}, cost {cur_time-last_time:.3f}s.")
            last_time = cur_time
    return insights_pool

def init_logger(args, log_level=logging.INFO):
    def get_file_name(root_output_path, cur_time):
        output_path = os.path.join(root_output_path, cur_time)
        # Make path if it doesn't exist
        if (not os.path.exists(output_path)):
            os.makedirs(output_path)
        return output_path
    cur_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    root_output_path = os.path.join(args.output_path, args.task_name, args.mode)

    filenameOutPrefixSeed = get_file_name(root_output_path, cur_time)
    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s",
                                    datefmt='%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_file_path = os.path.join(filenameOutPrefixSeed, 'status')
    if filenameOutPrefixSeed:
        filename = f"{log_file_path}.log"
        fh = logging.FileHandler(filename)
        fh.setLevel(log_level)
        fh.setFormatter(formatter)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.addHandler(fh)
    return logger, filenameOutPrefixSeed

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default='sentiment')

    parser.add_argument("--mode", type=str, default='gather_pref') # learning_{num}, eval_fs, eval_zs, eval_expert, eval_cot, gather_pref
    parser.add_argument("--data_size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument('--learning_config', type=str, default='near') # [near:{a,b}, far:{a,c}]
    parser.add_argument('--learning_k', type=int, default=2)
    parser.add_argument('--leap_insight_path', type=str, default=None)

    parser.add_argument('--retrieve_k', type=int, default=1)

    parser.add_argument('--output_path', type=str, default='./logs_analysis')
    parser.add_argument('--model_name', type=str, default="cardiffnlp/twitter-roberta-base-sentiment")
    parser.add_argument('--expert_trials_path', type=str, default=None)

    parser.add_argument('--few_shot_k', type=int, default=3)

    args = parser.parse_args()
    params = argparse.Namespace()
    for arg in vars(args):
        setattr(params, arg, getattr(args, arg))
    return params

def main():
    args = parse_args()
    print(args)
    if args.mode.startswith('learning'):
        dat = TweetEvalDataset(args.task_name, file_path='./datasets', fold='train', sample_size=10)
        logger, result_path = init_logger(args)
        logger.info(args)
        expert_trials = read_json(args.expert_trials_path)
        insight_pool = leap_learning(dat, expert_trials, learning_k=args.learning_k, logger=logger)
        with open(f"{result_path}/{args.task_name}_{args.data_size}_insight_leap{args.learning_k}.json", "w") as file:
            json.dump(insight_pool, file)

    elif args.mode in ['gather_pref', 'eval_expert']:
        if args.mode == 'gather_pref':
            fold = 'train'
        else:
            fold = 'test'
        dat = TweetEvalDataset(args.task_name, file_path='./datasets', fold=fold, sample_size=args.data_size)
        logger, result_path = init_logger(args)
        logger.info(args)
        expert_trials = evaluate_sota(dat, model_name=args.model_name, MAPPING=MODEL_OUTPUT2NUM_MAPPINGS[args.task_name], logger=logger)
        # if args.mode == 'gather_pref':
        with open(f"{result_path}/{args.task_name}_{args.data_size}_expert_trials.json", "w") as file:
            json.dump(expert_trials, file)

    elif args.mode in ['eval_zs', 'eval_fs', 'eval_zs_cot', 'eval_fs_cot']:
        config = args.mode.strip('_eval')
        dat = TweetEvalDataset(args.task_name, file_path='./datasets', fold='test', sample_size=args.data_size, shot_k=args.few_shot_k)
        logger, result_path = init_logger(args)
        logger.info(args)

        if args.leap_insight_path:
            insight_pool = read_json(args.leap_insight_path)
        else:
            insight_pool = None

        gpt_results = evaluate_gpt(dat, insight_pool=insight_pool, retrieve_k=args.retrieve_k, config=config, logger=logger)

        with open(f"{result_path}/{args.task_name}_{args.data_size}_gpt_res.json", "w") as file:
            json.dump(gpt_results, file)

    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()