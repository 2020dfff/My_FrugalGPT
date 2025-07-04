# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 21:06:55 2022

@author: CLJ
"""

import os, time
import smart_open
import json
import re
import urllib.request

import string
from collections import Counter

def compute_cost(
                 input_size,
                 output_size,
                 service_info):
        cost = 0        
        cost_output = service_info["cost_output"]
        cost_input = service_info["cost_input"]
        cost_fixed = service_info["cost_fixed"]
        fixed_size = service_info["fixed_size"]
        
        cost = cost_input*input_size + cost_fixed
        if(output_size>fixed_size):
            cost += cost_output*(output_size - fixed_size)    
        return cost 

def debug(message,
          debug_mode=True):
    if(debug_mode):
        print(message)

def load_http_text(url):
    with urllib.request.urlopen(url) as f:
        return f.read().decode("utf-8")


def load_text(path):
    if path.startswith("http://") or path.startswith("https://"):
        return load_http_text(path)
    else:
        with smart_open.open(path) as f:
            return f.read()


def load_json(path):
    return json.loads(load_text(path))


def load_json_lines(path):
    print("json load path:",path)
    return [json.loads(line) for line in load_text(path).split("\n") if line]


def dump_json(data, path):
    with smart_open.open(path, "w") as f:
        json.dump(data, f)

def dump_json_lines(data, path):
    with smart_open.open(path, mode='w') as f:
        for i in range(len(data)):
            f.write(json.dumps(data[i]))
            f.write("\n")
    


def dump_dataframe(df, path):
    with smart_open.open(path, "w") as f:
        df.to_csv(f, index=False)


def ensure_path_exists(path):
    if "://" in path:
        # Buckets like GS/S3 don't need to pre-create the prefix/folder
        return

    if not os.path.exists(path):
        os.makedirs(path)


def word_count(text):
    # Count words in text, this isn't well-defined but we count regex full words and
    # single-char non-words (e.g. punctuation), similar to word tokenizers
    return len(re.findall(r"\w+|[^\w\s]", text))

def write_json(path,data):
    json_object = json.dumps(data, indent=4)
    with open(path,'w') as f:
        f.write(json_object)
    return 

def evaluate_batch(answers, labels):
    isExist = os.path.exists("temp/")
    if not isExist:        
        # Create a new directory because it does not exist
        os.makedirs("temp/")
    prefix = str(int(time.time()*1000000))
    pred_path = "temp/temp_completion_json"+prefix+".json"
    write_json(pred_path,answers)
    # print("Debug: write to",pred_path)
    true_path = "temp/temp_true_json"+prefix+".json"
    # print("Debug: write to",true_path)
    write_json(true_path,labels)
    metrics = eval(pred_path,true_path) 
    os.remove(true_path)
    os.remove(pred_path)
	
    costs = [answers['cost'][key] for key in answers['cost']]            
    metrics['cost_list'] = costs
    metrics['cost'] = sum(costs)
    metrics['avg_cost'] = sum(costs)/len(costs)
    
    # 确保metrics中包含f1_list, precision_list和recall_list
    if 'f1' in metrics and 'f1_list' not in metrics:
        # 如果有f1但没有f1_list，则计算它们
        all_keys = list(answers['answer'].keys())
        f1_list = []
        precision_list = []
        recall_list = []
        
        for key in all_keys:
            pred = answers['answer'][key]
            # 找到对应的label
            label = None
            for lb in labels:
                if str(lb['_id']) == str(key):
                    label = lb['answer']
                    break
                    
            if label is not None:
                f1, precision, recall, _, _, _, _ = f1_score_binary(pred, label)
                f1_list.append(f1)
                precision_list.append(precision)
                recall_list.append(recall)
            else:
                f1_list.append(0)
                precision_list.append(0)
                recall_list.append(0)
        
        metrics['f1_list'] = f1_list
        metrics['precision_list'] = precision_list
        metrics['recall_list'] = recall_list
        
    return metrics

# def evaluate(prediction, ground_truth,metric="em"):
#     if (metric == "em"):
#         return int(exact_match_score(prediction, ground_truth, normal_method=""))

def evaluate(prediction, ground_truth, metric="em"):
    if metric == "em":
        return int(exact_match_score(prediction, ground_truth, normal_method=""))
    elif metric == "f1":
        f1, precision, recall, tp, fp, tn, fn = f1_score_binary(prediction, ground_truth)
        # print(f"Debug - evaluate: pred={prediction}, ground={ground_truth}, tp={tp}, fp={fp}, tn={tn}, fn={fn}")
        return tp, fp, tn, fn
    else:
        raise ValueError(f"Unsupported metric: {metric}")

def normalize_answer(s,normal_method=""):

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
#        print("text is:",text)
        return text.lower()

    def mc_remove(text):
        return text
        # a1 = re.findall('\([a-zA-Z]\)', text)
        # #print("text is",text)
        # #print("a1",a1)
        # if(len(a1)==0):
        #     return ""
        # return re.findall('\([a-zA-Z]\)', text)[-1]
    if(normal_method=="mc"):
        return mc_remove(s)
    if not isinstance(s, str):
        s = str(s)
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    ZERO_METRIC = (0, 0, 0)

    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return ZERO_METRIC
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def f1_score_binary(prediction, ground_truth):
    """
    Calculate F1-score for binary classification tasks (yes/no).
    """
    # Ensure inputs are normalized
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)

    # Initialize counts
    tp = 0  # True Positive
    fp = 0  # False Positive
    tn = 0  # True Negative
    fn = 0  # False Negative

    # Update counts based on prediction and ground truth
    if normalized_prediction == 'yes' and normalized_ground_truth == 'yes':
        tp += 1
    elif normalized_prediction == 'yes' and normalized_ground_truth == 'no':
        fp += 1
    elif normalized_prediction == 'no' and normalized_ground_truth == 'no':
        tn += 1
    elif normalized_prediction == 'no' and normalized_ground_truth == 'yes':
        fn += 1
    else:
        # Handle invalid predictions (not 'yes' or 'no')
        if normalized_ground_truth == 'yes':
            fn += 1  # Treat as a False Negative
        elif normalized_ground_truth == 'no':
            fp += 1  # Treat as a False Positive


    # Calculate Precision, Recall, and F1-score
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return f1, precision, recall, tp, fp, tn, fn


def exact_match_score(prediction, ground_truth, normal_method=""):
    return (normalize_answer(prediction,normal_method=normal_method) == normalize_answer(ground_truth,normal_method=normal_method))


def update_answer(metrics, prediction, gold):
    em = exact_match_score(prediction, gold)
    em_mc = exact_match_score(prediction, gold, normal_method="mc")
    #em_batch = exact_match_score(prediction, gold, normal_method="batch")
    '''
    if(em==False):
        print("Not match, pred:",prediction,"With true:", gold)
    '''
    f1, prec, recall = f1_score(prediction, gold)
    metrics['em'] += float(em)
    metrics['em_mc'] += float(em_mc)
    #metrics['em_batch'] += float(em_batch)

    metrics['f1'] += f1
    metrics['prec'] += prec
    metrics['recall'] += recall
    metrics['em_list'].append(float(em))
    metrics['em_mc_list'].append(float(em_mc))
    #metrics['em_batch_list'].append(float(em_batch))

    return em, prec, recall

def update_sp(metrics, prediction, gold):
    cur_sp_pred = set(map(tuple, prediction))
    gold_sp_pred = set(map(tuple, gold))
    tp, fp, fn = 0, 0, 0
    for e in cur_sp_pred:
        if e in gold_sp_pred:
            tp += 1
        else:
            fp += 1
    for e in gold_sp_pred:
        if e not in cur_sp_pred:
            fn += 1
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0
    metrics['sp_em'] += em
    metrics['sp_f1'] += f1
    metrics['sp_prec'] += prec
    metrics['sp_recall'] += recall
    return em, prec, recall

def eval(prediction_file, gold_file):
    # print("Debug: prediction file is",prediction_file)
    # print("Debug: gold file is",gold_file)
    with open(prediction_file) as f:
        prediction = json.load(f)
    with open(gold_file) as f:
        gold = json.load(f)

    metrics = {'em': 0, 'f1': 0, 'prec': 0, 'recall': 0,
        'sp_em': 0, 'sp_f1': 0, 'sp_prec': 0, 'sp_recall': 0,
        'joint_em': 0, 'joint_f1': 0, 'joint_prec': 0, 'joint_recall': 0,
        'em_list':[],
        'em_mc':0,
        'em_mc_list':[],
        'f1_list':[],  # 添加f1_list
        'precision_list':[],  # 添加precision_list
        'recall_list':[],  # 添加recall_list
        #'em_batch':0,
        #'em_batch_list':[],
        }
    
    # 添加F1相关统计
    total_tp, total_fp, total_tn, total_fn = 0, 0, 0, 0
    
    for dp in gold:
        cur_id = str(dp['_id'])
        # cur_id = dp['_id']
        can_eval_joint = True
        #print(prediction['answer'])

        #prediction['answer'][cur_id]
        if cur_id not in prediction['answer']:
            # print("Debug: Now the cur_id is",cur_id)
            # print("Debug: Now the prediction is",prediction)
            print('missing answer {}'.format(cur_id))
            can_eval_joint = False
            metrics['em_list'].append(0)
            metrics['f1_list'].append(0)
            metrics['precision_list'].append(0)
            metrics['recall_list'].append(0)
        else:
            # 计算EM和F1
            pred = prediction['answer'][cur_id]
            gt = dp['answer']
            
            # EM计算
            em = exact_match_score(pred, gt)
            metrics['em'] += float(em)
            metrics['em_list'].append(float(em))
            
            # F1计算
            f1, precision, recall, tp, fp, tn, fn = f1_score_binary(pred, gt)
            metrics['f1'] += f1
            metrics['prec'] += precision
            metrics['recall'] += recall
            metrics['f1_list'].append(f1)
            metrics['precision_list'].append(precision)
            metrics['recall_list'].append(recall)
            
            # 累积F1统计
            total_tp += tp
            total_fp += fp
            total_tn += tn
            total_fn += fn
            
            # EM-MC计算
            em_mc = exact_match_score(pred, gt, normal_method="mc")
            metrics['em_mc'] += float(em_mc)
            metrics['em_mc_list'].append(float(em_mc))
            
        if cur_id not in prediction['sp']:
            # print('missing sp fact {}'.format(cur_id))
            can_eval_joint = False
        else:
            sp_em, sp_prec, sp_recall = update_sp(
                metrics, prediction['sp'][cur_id], dp['supporting_facts'])

        if can_eval_joint:
            joint_prec = precision * sp_prec
            joint_recall = recall * sp_recall
            if joint_prec + joint_recall > 0:
                joint_f1 = 2 * joint_prec * joint_recall / (joint_prec + joint_recall)
            else:
                joint_f1 = 0.
            joint_em = em * sp_em

            metrics['joint_em'] += joint_em
            metrics['joint_f1'] += joint_f1
            metrics['joint_prec'] += joint_prec
            metrics['joint_recall'] += joint_recall

    N = len(gold)
    for k in metrics.keys():
        if k not in ["em_list", "em_mc_list", "f1_list", "precision_list", "recall_list"]:
            metrics[k] /= N
    
    # 计算整体F1分数
    if total_tp + total_fp > 0:
        overall_precision = total_tp / (total_tp + total_fp)
    else:
        overall_precision = 0
    
    if total_tp + total_fn > 0:
        overall_recall = total_tp / (total_tp + total_fn)
    else:
        overall_recall = 0
    
    if overall_precision + overall_recall > 0:
        overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
    else:
        overall_f1 = 0
    
    # 添加整体F1统计
    metrics['overall_f1'] = overall_f1
    metrics['overall_precision'] = overall_precision
    metrics['overall_recall'] = overall_recall
    metrics['tp'] = total_tp
    metrics['fp'] = total_fp
    metrics['tn'] = total_tn
    metrics['fn'] = total_fn

   #print(metrics)
    return metrics


