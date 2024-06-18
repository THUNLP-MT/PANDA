# PANDA
This repo contains the codes for our work 
[🐼PANDA: Preference Adaptation for Enhancing
Domain-Specific Abilities of LLMs](https://arxiv.org/abs/2402.12835).

The required package can be installed by running the following command.
```sh
pip install -r requirements.txt
```

## ScienceWorld
Firstly, switch to the ScienceWorld workspace:
```sh
cd ScienceWorld
```

0. To directly run experiments with PANDA (You can also run PANDA from scratch by starting from step1).
```sh
./run_eval_react_panda.sh
./run_eval_reflexion_panda.sh
./run_eval_saycan_panda.sh
```
1. Step1: Gather expert trials to construct preferences data.
```sh
./gather_trials.sh
```
2. Step2: PANDA-Learning from the expert preferences.
```sh
./panda_learning.sh
```
3. Step3: Test with PANDA-Insight:
```sh
./run_eval_react_panda.sh
./run_eval_reflexion_panda.sh
./run_eval_saycan_panda.sh
```
## TweetEval
Firstly, switch to the ScienceWorld workspace:
```sh
cd TweetEval
```
0. Step0: Download datasets file from [cardifnlp/tweeteval](https://github.com/cardiffnlp/tweeteval) and put it in the `dataset` folder and the expert models from [cardifnlp/models](https://huggingface.co/cardiffnlp) and put it in the `models` folder.

1. Step1: Gather expert trials to construct preferences data.
```sh
./gather_trials.sh
```
2. Step2: PANDA-Learning from the expert preferences.
```sh
./panda_learning.sh
```
3. Step3: Test with PANDA-Insight:
```sh
./eval_gpt.sh
./eval_gpt_cot.sh
```
## Acknowledgement
Our codes for scienceworld are adapted from [yuchenlin/SwiftSage](https://github.com/yuchenlin/SwiftSage). Thanks for their kind open-sourced code.

## Citation 
If you find our project helpful to your research, please consider citing:
```bib
@inproceedings{liu2024panda,
  title={PANDA: Preference Adaptation for Enhancing Domain-Specific Abilities of LLMs},
  author={Liu, An and Yang, Zonghan and Zhang, Zhenhe and Hu, Qingyuan and Li, Peng and Yan, Ming and Zhang, Ji and Huang, Fei and Liu, Yang},
  booktitle={Findings of the Association for Computational Linguistics: ACL 2024},
  year={2024}
}
```