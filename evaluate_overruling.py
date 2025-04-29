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

parser = argparse.ArgumentParser(description="Run FrugalGPT with specified device")
parser.add_argument('--device', type=int, default=0, help='CUDA device number (default: 0)')
args = parser.parse_args()
device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

logging.disable(logging.CRITICAL)
sys.path.append("src/")

supported_LLM = FrugalGPT.getservicename()
print("supported LLMs:", supported_LLM)
supported_LLM_names = [llm.split("/")[1] for llm in supported_LLM]
print("supported_LLM_names:", supported_LLM_names)

dataname = "OVERRULING"
# dataname = "AGNEWS"
# dataname = "HEADLINES"
# dataname = "SCIQ"

service_names = [
    'openaichat/gpt-4o-mini',
    'openaichat/gpt-4o',
    'google/gemini-1.5-flash-002',
    'google/gemini-1.5-pro-002',
    'google/gemini-1.0-pro',
    'azure/Phi-3-mini-4k-instruct',
    'azure/Phi-3.5-mini-instruct',
    'azure/Phi-3-small-8k-instruct',
    'azure/Phi-3-medium-4k-instruct',
    'deepinfra/llama-3-8B',
    'deepinfra/llama-3-70B',
    'deepinfra/mixtral-8x7B',
]

genparams = FrugalGPT.GenerationParameter(max_tokens=50, temperature=0.1, stop=['\n'])


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

# read from data/{dataname}/Queried_{dataname}_all_models_clean_train.csv and data/{dataname}/Queried_{dataname}_all_models_clean_test.csv
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
# get the answer of the model llama-3-8B
print(test_data[3][3]['llama-3-8B'])
print(len(test_data))


name = f'{dataname}_1125' #0116
budget_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001] # 
# budget_list = [0.0000268302083333333, 0.0000503473032407407, 0.000307842083333333, 0.000440937326388888] # 0.00000705800925925925, overruling
# budget_list = [0.00002645277, 0.0000597315, 0.00031284855, 0.000529866075] # 0.0000071974, # agnews
# budget_list = [0.0000302061632492115, 0.0000501699921135646, 0.000125359781151419, 0.000264852511829652] # 0.00000710551853312302, sciq
# budget_list = [0.00000729, 0.000026296875, 0.00004179045, 0.00014493809, 0.000248624] # headlines


def generate_dataframe_from_cascade(MyCascade,budget_list, train_data, test_data, genparams,name):
    # Initialize an empty list to store the rows for the DataFrame
    data = []

    # Iterate through the budget list
    for budget in tqdm(budget_list):
        # Load the strategy for the given budget
        MyCascade.load(loadpath=f"strategy/{name}/", budget=budget)
        # print("loaded from path:",f"strategy/{name}/")
        # print("now the budget is:",budget)

        # Get the completion batch for test data
        train_result = MyCascade.get_completion_batch(queries=train_data, genparams=genparams, budget=budget)
        test_result, average_cost, acc = MyCascade.get_completion_batch_test(queries=test_data, genparams=genparams,  budget=budget)
        # print("cost", test_result['cost'])
        average_test_cost = test_result['cost'].mean()
        max_cost_per_test_query = test_result['cost'].max()
        average_train_cost = train_result['cost'].mean()
        max_cost_per_train_query = train_result['cost'].max()
        

        train_acc_cost = FrugalGPT.compute_score(train_result)
        test_acc_cost = FrugalGPT.compute_score(test_result)

        # Create a row with the schema
        row = {
            # "Test_acc": test_acc_cost['em'],
            # "Test_cost": average_test_cost, # test_result['cost']
            "Test_acc": acc,
            "Test_cost": average_cost,
            "Max_test_cost": max_cost_per_test_query,
            "Test_size": len(test_data),
            "Train_acc": train_acc_cost['em'],
            "Train_cost": average_train_cost, # train_acc_cost['cost'],
            "Max_train_cost": max_cost_per_train_query,
            "Train_size": len(train_data),
            "Budget": budget,
            "Method": "FrugalGPT",
            "Provider": "FrugalGPT",
            "Marker": 1,  # Marker is always 1 for this function
        }

        # Append the row to the data list
        data.append(row)
        # print(row)

    # Create the DataFrame from the data list
    df = pd.DataFrame(data)
    return df

MyCascade_eval = FrugalGPT.LLMCascade()
frugalgpt_df = generate_dataframe_from_cascade(MyCascade_eval, budget_list, train_data, test_data, genparams, name)
print(frugalgpt_df)
# frugalgpt_df.to_csv(f"summary/summary_{dataname}_e8_frugalgpt_2024.csv")
# frugalgpt_df.to_csv(f"summary/summary_{dataname}_e8_frugalgpt_2025_final.csv")
frugalgpt_df.to_csv(f"summary/summary_{dataname}_e8_frugalgpt_2025_0410.csv")
