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
    score = {"em": 0, "f1": 0}
    mean_cost = data['cost'].mean()

    total_records = len(data)
    if metric == "em":
        filtered_data = data[data['cost'] <= mean_cost]
        exact_match_count = filtered_data.apply(lambda row: calculate_score(row, metric='em'), axis=1).sum()
        exact_match_percentage = exact_match_count / total_records
        score['em'] = exact_match_percentage
    elif metric == "f1":
        total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
        for _, row in data.iterrows():
            tp, fp, tn, fn = calculate_score(row, metric='f1')
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
        
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        score['f1'] = f1_score
        score['precision'] = precision
        score['recall'] = recall

        # here add em score as a comparison metric
        exact_match_count = data.apply(lambda row: calculate_score(row, metric='em'), axis=1).sum()
        exact_match_percentage = exact_match_count / total_records
        score['em'] = exact_match_percentage

    score['cost'] = mean_cost
    return score

# Define the scoring function
def calculate_score(row, metric='em'):
    try:
        if metric == 'em':
            return evaluate(row['answer'], row['ref_answer'], metric='em')
        elif metric == 'f1':
            tp, fp, tn, fn = evaluate(row['answer'], row['ref_answer'], metric='f1')
            return tp, fp, tn, fn
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    except Exception as e:
        print(f"Error in calculate_score: {e}, row: {row}")
        if metric == 'em':
            return 0
        elif metric == 'f1':
            return 0, 0, 0, 0