from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

from utils import *

sentiment_prompt = """Output the sentiment of the given text. Choose your answer from provided list and map your answer with following {{negative: 0, neutral: 1, positive: 2}} and return an integer as a result. Before output the answer, think it step by step firstly.
Text: {}
Let's think step by step (end your answer with "the answer is #integer#"):"""

sentiment_exemplars = [{'text': 'Dark Souls 3 April Launch Date Confirmed With New Trailer: Embrace the darkness.', 'label': 1,
                        'response': """1. The text mentions the launch date being confirmed for Dark Souls 3, which is likely to be exciting news for fans of the game. This could indicate a positive sentiment.
2. The phrase "Embrace the darkness" could be interpreted as a call to action or a thematic element of the game, which may not necessarily indicate a negative sentiment.
3. Overall, the text seems to convey a sense of anticipation and excitement for the upcoming release.
Based on this analysis, the sentiment of the given text can be categorized as neutral. 
Therefore, the answer is 1.
"""},
                      {'text': '"National hot dog day, national tequila day, then national dance day... Sounds like a Friday night."', 'label': 2,
                       'response': """Step 1: Identify key words and phrases
- "National hot dog day"
- "national tequila day"
- "national dance day"
- "Sounds like a Friday night"
Step 2: Analyze the tone and context
- The text seems to be expressing excitement or anticipation for the events mentioned.
- The mention of "Friday night" suggests a positive and fun atmosphere.
Step 3: Determine the sentiment
- The overall sentiment of the text is positive.
Step 4: Map the sentiment to the provided list
- Positive: 2
So, the answer is 2.
"""},
                      {'text': "When girls become bandwagon fans of the Packers because of Harry.   Do y'all even know who Aaron Rodgers is?  Or what a 1st down is? ", 'label': 0,
                       'response': """Step 1: The text seems to be mocking girls who become fans of the Packers because of Harry, implying that they may not actually know much about football.
Step 2: The tone of the text is somewhat negative and condescending.
Step 3: Based on the above analysis, the sentiment of the text is negative.
Step 4: Mapping the sentiment "negative" to the provided list, we get negative: 0.
Step 5: Therefore, the integer result for the sentiment of the text is 0. 
So, the answer is 0."""}]


PROMPT_TEMPLATES = {"sentiment": sentiment_prompt}
EXEMPLARS = {'sentiment': sentiment_exemplars}
POST_PROCESS_FUNC = {'sentiment': potprocess_sentiment_cot}

class TweetEvalDataset:
    def __init__(self, dataset_name, file_path, fold='test', sample_size=500, shot_k=3):
        self.file_path = file_path
        self.dataset_name = dataset_name
        assert dataset_name in ['sentiment'], f"{dataset_name} is not implemented yet."
        self.fold = fold
        self.prompt_template = PROMPT_TEMPLATES[dataset_name]
        self.exemplars = EXEMPLARS[dataset_name][:shot_k]
        self.mapping = self._load_mapping()
        self.labels = self._load_labels()
        self.data = self._load_data()

        self.post_process_func = POST_PROCESS_FUNC[dataset_name]
        sample_size = min(sample_size, len(self.labels))
        if sample_size < len(self.labels):
            self.data, self.labels = self._stratified_sample(sample_size)

    def _load_mapping(self):
        mapping_path = f"{self.file_path}/{self.dataset_name}/mapping.txt"
        mapping = {}
        with open(mapping_path, "r") as file:
            for line in file:
                index, label = line.strip().split("\t")
                mapping[int(index)] = label
        return mapping
    
    def _load_labels(self):
        labels_path = f"{self.file_path}/{self.dataset_name}/{self.fold}_labels.txt"
        labels = []
        with open(labels_path, "r") as file:
            for line in file:
                labels.append(int(line.strip()))
        return labels
    
    def _load_data(self):
        data_path = f"{self.file_path}/{self.dataset_name}/{self.fold}_text.txt"
        data = []
        with open(data_path, "r") as file:
            for line in file:
                data.append(line.strip())
        return data
    
    def _stratified_sample(self, sample_size):
        sss = StratifiedShuffleSplit(n_splits=1, train_size=sample_size, random_state=42)
        train_index, _ = next(sss.split(self.data, self.labels))
        sampled_data = [self.data[i] for i in train_index]
        sampled_labels = [self.labels[i] for i in train_index]
        return sampled_data, sampled_labels

    def get_label_text(self, label):
        return self.mapping[label]
    
    def get_data_label(self, index):
        text = self.data[index]
        label = self.labels[index]
        return text, label
    
    def get_data_labels(self):
        data_labels = []
        for i in range(len(self.data)):
            text, label = self.get_data_label(i)
            data_labels.append((text, label))
        return data_labels

    def get_data(self):
        data = []
        for i in range(len(self.data)):
            text, label = self.get_data_label(i)
            data.append(text)
        return data
    
    def get_data_prompt(self, config='zs'): # ['zs', 'cot']
        query = []
        for i in range(len(self.data)):
            text, label = self.get_data_label(i)
            if config in ['zs', 'fs']:
                query.append(self.prompt_template.format(text))
            else:
                raise NotImplementedError
        return query

    def get_prompt_prefix(self, config='zs'): #fs,zs, cot, zs-cot, w_insights
        if config.startswith('zs'):
            return ""
        elif config == 'fs':
            # exemplars = [i['insight_prefix'] + self.prompt_template.format(i['text'])+f' {i["label"]}' for i in self.exemplars]
            exemplars = [self.prompt_template.format(i['text'])+f'\n{i["response"]}' for i in self.exemplars]
            return '\n\n'.join(exemplars) + '\n\n'
        # elif config == 'w_insights':
        else:
            raise NotImplementedError

    def get_labels(self):
        labels = []
        for i in range(len(self.data)):
            text, label = self.get_data_label(i)
            labels.append(label)
        return labels
    
    def analyze_labels(self):
        label_counts = Counter(self.labels)
        total_count = len(self.labels)
        label_ratios = {label: count / total_count for label, count in label_counts.items()}
        print("=="*5 + self.fold + "=="*5)
        print("Total count:", total_count)
        print("Label counts:")
        for label, count in label_counts.items():
            print(f"Label {label}: {count}")
        
        print("Label ratios:")
        for label, ratio in label_ratios.items():
            print(f"Label {label}: {ratio:.2%}")


from sklearn.metrics.pairwise import cosine_similarity
import numpy as np       
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

    for i in range(retrieve_k):
        insights.append(insight_pool[top_indice[i]]['insight'])
        if logger:
            logger.info(f'Retrieval-key:\n{retrieval_key}\nRetrieved-key:\n{insight_pool_keys[top_indice[i]]}')
    return insights


if __name__ == "__main__":
    import random
    def select_random_number(x):
        numbers = {0, 1, 2}
        numbers.discard(x)
        return random.choice(list(numbers))

    # dataset1 = TweetEvalDataset("irony", "./datasets", fold='test', sample_size=783)
    dataset2 = TweetEvalDataset("sentiment", "./datasets", fold='val', sample_size=1000)
    # dataset2 = TweetEvalDataset("sentiment", "./datasets", fold='test', sample_size=500)
    # print(dataset1.get_prompt_prefix('fs'))
    # for i in [dataset2]:
        # print(len(i.data))
    idx0 = [i for i in range(len(dataset2.labels)) if dataset2.labels[i]==0]
    idx1 = [i for i in range(len(dataset2.labels)) if dataset2.labels[i]==1]
    idx2 = [i for i in range(len(dataset2.labels)) if dataset2.labels[i]==2]

    cnt = 0
    while cnt < 8:
        cnt += 1
        for idx in [idx0, idx1, idx2]:
            print("'text': '{}', 'label': {},".format(dataset2.data[idx[cnt]], dataset2.labels[idx[cnt]]))
    # leap_insight_path = "/data/private/liuan/multiInsight/tweeteval/logs/sentiment/learning/2024-02-04 01:07:05/sentiment_500_insight_leap2.json"
    # insight_pool = read_json(leap_insight_path)
    # sbert = load_sbert()
    # insights_template = "These are some insights that may be helpful for you to improve the success rate:\n{}\n\n"
    # for dat in sentiment_exemplars:
    #     text = dat['text']
    #     insights = retrieve_insights(insight_pool, retrieval_key=text, sbert=sbert)
    #     print(insights_template.format('\n'.join(insights)))
