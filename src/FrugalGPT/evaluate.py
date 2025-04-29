from service.utils import evaluate

# def compute_score(data,metric='em'):
    
#     evaluate

#     score={"em":0}
#     if(metric=="em"):
#         exact_match_percentage = data.apply(calculate_score, axis=1).mean()
#         score['em'] = exact_match_percentage
#     score['cost'] = data['cost'].mean()
#     return score

def compute_score(data, metric='em'):
    score = {"em": 0}
    mean_cost = data['cost'].mean()

    if metric == "em":
        total_records = len(data)
        filtered_data = data[data['cost'] <= mean_cost]
        exact_match_count = filtered_data.apply(calculate_score, axis=1).sum()
        exact_match_percentage = exact_match_count / total_records
        score['em'] = exact_match_percentage

    score['cost'] = mean_cost
    return score

    # Define the scoring function
def calculate_score(row):
    # Perform scoring logic based on 'answer' and 'ref_answer'
    # For example, you can use a simple equality check and assign a score of 1 for a match, and 0 for a mismatch
    score = evaluate(row['answer'], row['ref_answer'])
    #print("lens",len(row['answer']),len(row['ref_answer']))
    return score