import sys, json, copy
import pandas as pd
import logging
import os
from tqdm import tqdm
logging.disable(logging.CRITICAL)
sys.path.append("src/")
import FrugalGPT

supported_LLM = FrugalGPT.getservicename()
print("supported LLMs:", supported_LLM)
supported_LLM_names = [llm.split("/")[1] for llm in supported_LLM]
print("supported_LLM_names:", supported_LLM_names)

# ## Step 1: Prepare the dataset

dataname = "OVERRULING"

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
    'azure/Phi-3.5-mini-instruct',
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

name = f'{dataname}_1125'
budget_list = [0.00001, 0.00005, 0.0001, 0.0005, 0.001] #  , 0.0015

MyCascade = FrugalGPT.LLMCascade(
    score_noise_injection=False,
    db_path=f"db/{dataname}.sqlite",
    batch_build=True,
)

train_data_sample = train_data[0:]  # [0:100]
print(len(train_data_sample))

compute_tradeoffs(
    train_data=train_data_sample,
    budget_list=budget_list,
    name=name,
    service_names=service_names,
    # prefix=prefix,
    skip=0,  # you can manually skip the first few budgets if they have already been trained.
    MyCascade=MyCascade,
    cascade_depth=3,
)

# ## Step 3: Evaluate and save the performance

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
        test_result = MyCascade.get_completion_batch(queries=test_data, genparams=genparams, budget=budget)
        # print("cost", test_result['cost'])
        average_test_cost = test_result['cost'].mean()
        max_cost_per_test_query = test_result['cost'].max()
        average_train_cost = train_result['cost'].mean()
        max_cost_per_train_query = train_result['cost'].max()
        

        train_acc_cost = FrugalGPT.compute_score(train_result)
        test_acc_cost = FrugalGPT.compute_score(test_result)

        # Create a row with the schema
        row = {
            "Test_acc": test_acc_cost['em'],
            "Test_cost": average_test_cost, # test_result['cost']
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
frugalgpt_df.to_csv(f"summary/summary_{dataname}_e8_frugalgpt_2024.csv")
