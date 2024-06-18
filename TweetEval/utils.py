import backoff
import json
import requests
from json import JSONDecodeError
from requests.auth import HTTPBasicAuth
from requests import ConnectTimeout, ConnectionError, ReadTimeout
from requests.exceptions import Timeout
from sentence_transformers import SentenceTransformer
import openai, os
openai.api_key = os.getenv("OPENAI_API_KEY")
RETRIES = 20
TIMEOUT = 1000
@backoff.on_exception(backoff.expo, (TypeError, KeyError, JSONDecodeError, ReadTimeout, ConnectionError, Timeout, ConnectTimeout), max_tries=RETRIES)
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs) 

def load_sbert(sbert_path='/data/private/liuan/multiInsight/SwiftSage/baselines/sentence-transformers_paraphrase-MiniLM-L6-v2', device="cuda"):
    sbert_model = SentenceTransformer(sbert_path).to(device)
    return sbert_model


def potprocess_sentiment(gpt_raw):
    res = str(gpt_raw).lower()
    if '0' in res or 'negative' in res:
        res = 0
    elif '1' in res or 'neutral' in res:
        res = 1
    elif '2' in res or 'positive' in res:
        res = 2
    else:
        res = -1
    return res

def potprocess_sentiment_cot(gpt_raw):
    res = gpt_raw.split('\n')[-1]
    res = str(res).lower()
    if '0' in res or 'negative' in res:
        res = 0
    elif '1' in res or 'neutral' in res:
        res = 1
    elif '2' in res or 'positive' in res:
        res = 2
    else:
        res = -1
    return res