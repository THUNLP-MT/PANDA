
# from openai_key import OPENAI_KEY
# import openai 
import os
import json
import random
import requests
from requests.auth import HTTPBasicAuth
from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
)  # for exponential backoff
import requests
import json
from json import JSONDecodeError
from requests.auth import HTTPBasicAuth
from requests import ConnectTimeout, ConnectionError, ReadTimeout
import backoff
from requests.exceptions import Timeout
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")


@retry(wait=wait_random_exponential(min=1, max=10), stop=stop_after_attempt(10))
def completion_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs) 
    triplets_by_task = load_triplets()
    prompt = sample_few_shot(triplets_by_task, "0")
    print(prompt)
