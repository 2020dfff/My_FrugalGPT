import sys, json, copy
import pandas as pd
import logging
import os
from tqdm import tqdm
logging.disable(logging.CRITICAL)
sys.path.append("src/")
import FrugalGPT
import argparse
import torch

# Disable logging
logging.disable(logging.CRITICAL)

# Argument parser
parser = argparse.ArgumentParser(description="Run FrugalGPT for entity matching")
parser.add_argument('--device', type=int, default=0, help='CUDA device number (default: 0)')
parser.add_argument('--configpath', type=str, required=True, help='Path to the service configuration file')
parser.add_argument('--dataname', type=str, required=True, help='Dataset name (e.g., abt-buy, wdc, amazon-google, dblp-scholar, walmart-amazon)')
parser.add_argument('--metric', type=str, default='f1', choices=['f1', 'em'], help='Evaluation metric (default: f1)')
args = parser.parse_args()

# check device
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load supported LLMs
supported_LLM = FrugalGPT.getservicename(configpath=args.configpath)
print("supported LLMs:", supported_LLM)
supported_LLM_names = [llm.split("/")[1] for llm in supported_LLM]
print("supported_LLM_names:", supported_LLM_names)

# Dataset name
dataname = args.dataname

# Service names
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

# Generation parameters
genparams = FrugalGPT.GenerationParameter(max_tokens=50, temperature=0.1, stop=['\n'])

# Read training data
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
print(train_data[3][3]['llama-3-8B'])
print(len(train_data))

# Read testing data
dataset_df_test = pd.read_csv(f'data/{dataname}/Queried_{dataname}_all_models_clean_test.csv', header=0)
print(dataset_df_test.head())
test_data = []
for index, row in dataset_df_test.iterrows():
    query = row['query']
    ref_answer = row['ref_answer']
    _id = index
    model_answer = {}
    for model_name in supported_LLM_names:
        model_answer[model_name] = row[model_name]
    test_data.append([query, ref_answer, _id, model_answer])
print(test_data[3])
print(test_data[3][3]['llama-3-8B'])
print(len(test_data))

# Experiment name and budget list
name = f'{dataname}_0407'
budget_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001]

# Function to generate DataFrame from cascade
def generate_dataframe_from_cascade(MyCascade, budget_list, train_data, test_data, genparams, name, metric):
    data = []
    for budget in tqdm(budget_list):
        MyCascade.load(loadpath=f"strategy/{name}/", budget=budget)
        train_result = MyCascade.get_completion_batch(
            queries=train_data, 
            genparams=genparams, 
            budget=budget
        )
        test_result, average_cost, metric_value = MyCascade.get_completion_batch_test(
            queries=test_data, 
            genparams=genparams,  
            budget=budget, 
            metric=metric
        )
        average_test_cost = test_result['cost'].mean()
        max_cost_per_test_query = test_result['cost'].max()
        average_train_cost = train_result['cost'].mean()
        max_cost_per_train_query = train_result['cost'].max()

        train_acc_cost = FrugalGPT.compute_score(train_result)
        test_acc_cost = FrugalGPT.compute_score(test_result)

        row = {
            "Test_F1-score" if metric == "f1" else "Test_Accuracy": metric_value,
            "Test_cost": average_cost,
            "Max_test_cost": max_cost_per_test_query,
            "Test_size": len(test_data),
            "Train_acc": train_acc_cost['em'],
            "Train_cost": average_train_cost,
            "Max_train_cost": max_cost_per_train_query,
            "Train_size": len(train_data),
            "Budget": budget,
            "Method": "FrugalGPT",
            "Provider": "FrugalGPT",
            "Marker": 1,
        }
        data.append(row)
    df = pd.DataFrame(data)
    return df

# Run the evaluation
MyCascade_eval = FrugalGPT.LLMCascade()
frugalgpt_df = generate_dataframe_from_cascade(MyCascade_eval, budget_list, train_data, test_data, genparams, name, args.metric)
print(frugalgpt_df)
frugalgpt_df.to_csv(f"summary/entity-matching/{args.metric}/summary_{dataname}_e8_frugalgpt_2025_0423.csv")