import sys, json, copy
import pandas as pd
import logging
import os
from tqdm import tqdm
import argparse
logging.disable(logging.CRITICAL)
sys.path.append("src/")
import FrugalGPT

parser = argparse.ArgumentParser(description="Train FrugalGPT for entity matching")
parser.add_argument('--configpath', type=str, default='config/serviceinfo_abt-buy.json', help='Path to the service configuration file')
parser.add_argument('--dataname', type=str, default='wdc', help='Dataset name (e.g., abt-buy, wdc, amazon-google, dblp-scholar, walmart-amazon)')
parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'em'], help='Evaluation metric (default: f1)')
parser.add_argument('--date', type=str, default='0101', help='Date string for naming (format: MMDD, default: 0101)')
parser.add_argument('--budgets', type=str, default='0.00001,0.00005,0.0001,0.0005,0.001', 
                   help='Comma-separated list of budgets (default: "0.00001,0.00005,0.0001,0.0005,0.001")')
parser.add_argument('--skip', type=int, default=0, help='Skip the first few budgets if they have already been trained (default: 0)')
args = parser.parse_args()

supported_LLM = FrugalGPT.getservicename(configpath=args.configpath)
print("supported LLMs:", supported_LLM)
supported_LLM_names = [llm.split("/")[1] for llm in supported_LLM]
print("supported_LLM_names:", supported_LLM_names)

# ## Step 1: Prepare the dataset

dataname = args.dataname
print(f"Using dataset: {dataname}")

# read from data/{dataname}/Queried_{dataname}_all_models_clean_train.csv and data/{dataname}/Queried_{dataname}_all_models_clean_test.csv
dataset_df = pd.read_csv(f'data/{dataname}/Queried_{dataname}_all_models_clean_train.csv', header=0)
print(dataset_df.head())

train_data = []
for index, row in dataset_df.iterrows():
    query = row['query']
    ref_answer = row['ref_answer']
    _id = index
    model_answer = {}
    for model_name in supported_LLM_names:
        model_answer[model_name] = row[model_name]
    train_data.append([query, ref_answer, _id, model_answer])

print(train_data[3])

# get the answer of the model llama-3-8B
print(train_data[3][3]['llama-3-8B'])

print(len(train_data))

# ## Step 2: Train the FrugalGPT strategy for different budgets

service_names = [
    'openaichat/gpt-4o-mini',
    'openaichat/gpt-4o',
    'google/gemini-1.5-flash-002',
    'google/gemini-1.5-pro-002',
    'google/gemini-1.0-pro',
    'azure/Phi-3-mini-4k-instruct',
    # 'azure/Phi-3.5-mini-instruct',
    'azure/Phi-3-small-8k-instruct',
    'azure/Phi-3-medium-4k-instruct',
    'deepinfra/llama-3-8B',
    'deepinfra/llama-3-70B',
    'deepinfra/mixtral-8x7B',
]

genparams = FrugalGPT.GenerationParameter(max_tokens=50, temperature=0.1, stop=['\n'])

def compute_tradeoffs(
    train_data,
    budget_list,
    name="HEADLINES",  # test
    service_names=[
        'openaichat/gpt-4o-mini',
        'openaichat/gpt-4o',
        'openaichat/gpt-4-turbo',
        'togetherai/meta-llama/Meta-Llama-3-70B-Instruct-Turbo',
        'togetherai/google/gemma-2-9b-it',
    ],
    prefix="",
    skip=0,
    MyCascade=FrugalGPT.LLMCascade(
        score_noise_injection=False,
        db_path="db/SCIQ.sqlite",
        metric="f1",
    ),
    cascade_depth=3,
):
    for idx, budget in tqdm(enumerate(budget_list)):
        # train the model
        user_budget = budget
        # MyCascade.load(loadpath=f"strategy/{name}/", budget=user_budget)

        try:
            MyCascade.load(loadpath=f"strategy/{name}/", budget=user_budget)
            print("Already trained. Skipped.")
            continue
        except:
            print("cannot find, start new training")
        if idx < skip:
            continue
        if idx == 0:
            result = MyCascade.train(
                train_data,
                budget=user_budget,
                service_names=service_names,
                no_scorer_train=False,
                prefix=prefix,
                cascade_depth=cascade_depth,
            )
        else:
            result = MyCascade.train(
                train_data,
                budget=user_budget,
                service_names=service_names,
                no_scorer_train=True,
                prefix=prefix,
                cascade_depth=cascade_depth,
            )
        MyCascade.save(savepath=f"strategy/{name}/")
    return

name = f'{dataname}_{args.date}'
budget_list = [float(budget.strip()) for budget in args.budgets.split(',')]
print(f"Using name: {name}")
print(f"Using budget list: {budget_list}")

MyCascade = FrugalGPT.LLMCascade(
    score_noise_injection=False,
    db_path=f"db/{dataname}.sqlite",
    batch_build=True,
    metric=args.metric,
)

train_data_sample = train_data[0:]  # [0:100]
print(len(train_data_sample))

compute_tradeoffs(
    train_data=train_data_sample,
    budget_list=budget_list,
    name=name,
    service_names=service_names,
    # prefix=prefix,
    skip=args.skip,  # you can manually skip the first few budgets if they have already been trained.
    MyCascade=MyCascade,
    cascade_depth=3,
)

print(f"Training completed. Model saved to strategy/{name}/")
print(f"To evaluate the performance, please run:")
print(f"python evaluate_entity_matching.py --dataname {dataname} --date {args.date} --metric {args.metric} --budgets \"{args.budgets}\" --configpath {args.configpath}")
