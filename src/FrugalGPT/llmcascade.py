from .llmvanilla import LLMVanilla
from service.modelservice import GenerationParameter
import pandas
from service.utils import evaluate
from .scoring import Score
from sklearn.model_selection import train_test_split
from .llmchain import LLMChain
import json, os
import random
from tqdm.notebook import tqdm
import tiktoken

MODEL_PRICE = {
    'gpt-4o-mini':   [0.15, 0.6, 0],
    'gpt-4-turbo':   [10, 30, 0],
    'gpt-4o':        [5, 15, 0],
    'gemini-1.5-flash-002':  [0.075, 0.3, 0],
    'gemini-1.5-pro-002':    [3.5, 10.5, 0],
    'gemini-1.0-pro':    [0.5, 1.5, 0],
    'Phi-3-mini-4k-instruct':     [0.13, 0.52, 0],
    'Phi-3.5-mini-instruct':   [0.13, 0.52, 0],
    'Phi-3-small-8k-instruct':    [0.15, 0.6, 0],
    'Phi-3-medium-4k-instruct':  [0.17, 0.68, 0],
    'llama-3-8B':     [0.055, 0.055, 0],
    'llama-3-70B':    [0.35, 0.4, 0],
    'mixtral-8x7B':  [0.24, 0.24, 0]
    }

def price_of(q, model):
    encoding = tiktoken.get_encoding('cl100k_base')
    in_token_num = len(encoding.encode(q))
    # for classification tasks, we assume the model always give out answer with len=1
    out_token_num = 1
    in_price = MODEL_PRICE[model][0] * in_token_num / 1e6
    out_price = MODEL_PRICE[model][1] * out_token_num / 1e6
    request_price = MODEL_PRICE[model][2]
    return in_price + out_price + request_price

def price_all(q, models):
    costs = []
    for m in models:
        costs.append(price_of(q, m))
    return costs

def scorer_text(text):
    #return text

    newtext = "Q:"+text.split("Q:")[-1]    
    return newtext

def tempsave(label, response, score,name):
    return


class LLMCascade(object):
    def __init__(self, 
                 #service_names =['openaichat/gpt-3.5-turbo','openaichat/gpt-4'],
                 metric="em",
                 db_path='db/HEADLINES.sqlite',
                 score_noise_injection=False,
                 batch_build = False,
                 ):
        # Initialization code for the FrugalGPT class
        self.prefix = " "
        self.MyLLMEngine = LLMVanilla(db_path=db_path)    
        self.MyScores = dict()
        self.LLMChain = LLMChain(metric=metric)
        #self.service_names =['openaichat/gpt-3.5-turbo','openaichat/gpt-4'],
        self.eps=1e-8
        self.score_noise_injection = score_noise_injection
        self.batch_build = batch_build
        return 

    def load(self,loadpath="strategy/HEADLINES/",budget=0.01):
        self.LLMChain = LLMChain()
        self.LLMChain.setbudget(budget=budget)
        self.LLMChain.loadstrategy(loadpath+"cascade_strategy.json")        
        model_names = self.loadmodelnames(loadpath)
        #print("model names",model_names)
        self.scorer = dict()
        for name in model_names:
            path1 = loadpath+name+"/"
            self.MyScores[name]=Score()
            self.MyScores[name].load(path1)
            self.scorer[name]  = self.MyScores[name].get_model()
        #print("scoer keys:",self.scorer.keys())
        return
    
    def loadmodelnamesold(self,loadpath):
        directories = []
        for entry in os.scandir(loadpath):
            if entry.is_dir():
                for sub_entry in os.scandir(entry.path):
                    if sub_entry.is_dir():
                        subdirectory_name = os.path.relpath(sub_entry.path, loadpath)
                        directories.append(subdirectory_name)
        keys = directories
        return keys

    def loadmodelnames(self, loadpath):
      directories = []
      for dirpath, dirnames, _ in os.walk(loadpath):
          if not dirnames:  # If there are no more subdirectories, it's the last directory
              subdirectory_name = os.path.relpath(dirpath, loadpath)
              directories.append(subdirectory_name)
      keys = directories
      return keys  # Return the directories list if needed


    def save(self, savepath="strategy/HEADLINES/"):
        # Save both Scores and LLChains to the disk
        if not os.path.exists(savepath):
            os.makedirs(savepath)
        self.LLMChain.savestrategy(savepath+"cascade_strategy.json")
        if(self.no_scorer_train):
            return
        for name in self.MyScores:
            path1 = savepath+name+"/"
            self.MyScores[name].save(path1)
        return
    
    def train(self,
              trainingdata=None,
              budget=0.1,
              cascade_depth=3,
              service_names =['openaichat/gpt-3.5-turbo','openaichat/gpt-4'],
              metric="em",
              genparams=GenerationParameter(max_tokens=50, temperature=0.1, stop=['\n']),
              no_scorer_train=False,
              score_type='DistilBert',
              prefix="",
              ):
        self.no_scorer_train = no_scorer_train
        self.score_type = score_type
        self.prefix = prefix
        # Three major steps
        # Step 1: evaluate all services on the given dataset
        train, test = train_test_split(trainingdata, test_size=0.2)
        print("train and test size",len(train),len(test))
        model_perf_train = self.evaluateall(train,service_names=service_names,metric=metric,genparams=genparams)
        model_perf_test = self.evaluateall(test,service_names=service_names,metric=metric,genparams=genparams)
        # print("model_perf_train",model_perf_train)
        # Step 2: Build the scorer
        if(no_scorer_train):
            #print("directly get the scorers")
            scorers = self.get_scorers()
            #print("")
        else:
            scorers = self.build_scorers(model_perf_train)
        # Step 3: Build the cascade
        #self.build_cascade(model_perf_test, scorers = scorers, budget=budget, cascade_depth=cascade_depth,metric=metric)
        self.build_cascade(model_perf_train, scorers = scorers, budget=budget, cascade_depth=cascade_depth,metric=metric)
        return model_perf_test
    
    # def get_completion(self, query,genparams):
    #     LLMChain = self.LLMChain
    #     MyLLMEngine = self.MyLLMEngine
    #     cost = 0 
    #     LLMChain.reset()
    #     prefix = self.prefix
    #     while(1):
    #         service_name, score_thres = LLMChain.nextAPIandScore()
    #         if(service_name==None):
    #             break
    #         res = MyLLMEngine.get_completion(query=query,service_name=service_name,genparams=genparams)
    #         cost += MyLLMEngine.get_cost()
    #         t1 = query+" "+res
    #         t2 = t1.removeprefix(prefix)
    #         score = self.MyScores[service_name].get_score(scorer_text(t2))
    #         if(self.score_noise_injection==True):
    #           score+=random.random()*self.eps
    #         #print("score and score thres:",service_name,score,score_thres)
    #         if(score>1-score_thres):
    #             break
    #     self.cost = cost
    #     return res

    # def get_completion_batch(self, queries, genparams):
    #     result = list()
    #     for query in queries:
    #         ans1 = self.get_completion(query=query[0], genparams=genparams)
    #         cost = self.get_cost()
    #         result.append({'_id':query[2],'answer':ans1,'ref_answer':query[1],'cost':cost})
    #     result = pandas.DataFrame(result)
    #     return result
    
    def get_completion(self, query, genparams, budget):
        LLMChain = self.LLMChain
        MyLLMEngine = self.MyLLMEngine
        cost = 0 
        LLMChain.reset()
        prefix = self.prefix
        while(1):
            service_name, score_thres = LLMChain.nextAPIandScore()
            if(service_name is None):
                break
            # answer get from data
            res = query[3][service_name.split("/")[1]]
            # cost += MyLLMEngine.get_cost()
            # if cost < budget:
            cost += price_of(query[0], service_name.split("/")[1])

            # print("now service_name",service_name)
            # print("query",query)
            # print("res",res)
            t1 = query[0] + " " + str(res)
            t2 = t1.removeprefix(prefix)
            score = self.MyScores[service_name].get_score(scorer_text(t2))
            # print("score and score thres:",score,score_thres)
            if self.score_noise_injection:
                score += random.random() * self.eps
            if score > 1 - score_thres:
                # print("stop at",service_name)
                break
            # if cost > budget:
                # print("Stop at", service_name)
                # return res
                # break
        self.cost = cost
        return res

    # def get_completion_batch(self, queries, genparams):
    #     result = list()
    #     for query in queries:
    #         ans1 = self.get_completion(query=query[0], genparams=genparams)
    #         cost = self.get_cost()
    #         result.append({'_id': query[2], 'answer': ans1, 'ref_answer': query[1], 'cost': cost})
    #     result = pandas.DataFrame(result)
    #     return result
    
    def get_completion_batch(self, queries, genparams, budget):
        result = list()
        overall_cost = 0
        # max_cost = 0
        # for query in tqdm(queries, desc="Collecting results"):
        for query in queries:
            ans1 = self.get_completion(query, genparams=genparams, budget=budget)
            cost = self.get_cost()
            # print("cost",cost)
            overall_cost += cost
            # if cost > max_cost:
                # max_cost = cost
            result.append({'_id': query[2], 'answer': ans1, 'ref_answer': query[1], 'cost': cost})
        average_cost = overall_cost / len(queries)
        print("average cost", average_cost)
        # print("max cost", max_cost)
        result = pandas.DataFrame(result)
        return result
        
    def get_completion_test(self, query, genparams, budget, metric="em"):
        LLMChain = self.LLMChain
        MyLLMEngine = self.MyLLMEngine
        cost = 0 
        LLMChain.reset()
        prefix = self.prefix
        is_correct = False
        res = None
        last_res = None
        last_cost = 0

        # initialize the variables
        tp, fp, tn, fn = 0, 0, 0, 0

        while(1):
            service_name, score_thres = LLMChain.nextAPIandScore()
            if service_name is None:
                break
            # answer get from data
            res = query[3][service_name.split("/")[1]]
            new_cost = cost + price_of(query[0], service_name.split("/")[1])
            if new_cost > budget:
                print("Budget exceeded, stop at", service_name)
                if metric == "f1":
                    tp, fp, tn, fn = evaluate(prediction=last_res, ground_truth=query[1], metric="f1")
                elif metric == "em":
                    is_correct = evaluate(prediction=last_res, ground_truth=query[1], metric="em")
                res = last_res
                cost = last_cost
                break
            cost = new_cost
            last_res = res
            last_cost = cost
            t1 = query[0] + " " + str(res)
            t2 = t1.removeprefix(prefix)
            score = self.MyScores[service_name].get_score(scorer_text(t2))
            if self.score_noise_injection:
                score += random.random() * self.eps
            if score > 1 - score_thres:
                # Evaluate the result
                if metric == "f1":
                    tp, fp, tn, fn = evaluate(prediction=res, ground_truth=query[1], metric="f1")
                elif metric == "em":
                    is_correct = evaluate(prediction=res, ground_truth=query[1], metric="em")
                    # print("evaluation result", is_correct)
                break
        self.cost = cost
        return res, is_correct, tp, fp, tn, fn
        
    def get_completion_batch_test(self, queries, genparams, budget, metric="em"):
        result = list()
        overall_cost = 0
        total_tp = 0
        total_fp = 0
        total_tn = 0
        total_fn = 0
        total_correct_count = 0
        total_wrong_count = 0

        for query in queries:
            ans1, is_correct, tp, fp, tn, fn = self.get_completion_test(query, genparams=genparams, budget=budget, metric=metric)
            cost = self.get_cost()
            overall_cost += cost

            # print("cost",cost)

            if metric == "f1":
                total_tp += tp
                total_fp += fp
                total_fn += fn
                total_tn += tn
                
            elif metric == "em":
                if is_correct:
                    total_correct_count += 1
                else:
                    total_wrong_count += 1

            result.append({'_id': query[2], 'answer': ans1, 'ref_answer': query[1], 'cost': cost})        
        # print("average cost", average_cost)
        # print("max cost", max_cost)

        if metric == "f1":
            print("Total TP:", total_tp, "Total FP:", total_fp, "Total TN:", total_tn, "Total FN:", total_fn)
        elif metric == "em":
            print("Total Correct Count:", total_correct_count, "Total Wrong Count:", total_wrong_count)

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        result = pandas.DataFrame(result)
        average_cost = result['cost'].mean()
        if metric == "f1":
            return result, average_cost, f1_score
        elif metric == "em":
            accuracy = total_correct_count / len(queries)
            return result, average_cost, accuracy


    def get_cost(self):
        return self.cost

    def _get_response(self,
                      data,
                      genparams,
                      service_name,
                      ):
        # data is a list
        # data[i][0]: query, data[i][1]: answer
        result = list()
        MyLLMEngine = self.MyLLMEngine
        for i in range(len(data)):
            query = data[i][0]
            temp = dict()
            temp['true_answer']= data[i][1] 
            temp['_id'] = data[i][2]
            temp['query'] = query
            temp['answer'] = data[i][3][service_name.split("/")[1]]
            # temp['answer'] = MyLLMEngine.get_completion(query=query,service_name=service_name,genparams=genparams)
            # temp['latency'] = MyLLMEngine.get_latency()
            # temp['cost'] = MyLLMEngine.get_cost()
            temp['cost'] = MyLLMEngine.compute_cost(query, data[i][3][service_name.split("/")[1]], service_name=service_name)
            result.append(temp)
        return result
 
    def  build_scorers(self,model_perf_train):
        self.scorer = dict()
        for name in model_perf_train:
            self.MyScores[name], self.scorer[name] = self._build_scorer(model_perf_train[name])
        return self.scorer

    def get_scorers(self):
        return self.scorer

    def _build_scorer(self,res_and_eval):
        #train, test = train_test_split(res_and_eval, test_size=0.2)
        #print("res_and_eval",res_and_eval)
        #traintext = list((res_and_eval['query']+res_and_eval['answer']).apply(scorer_text))
        prefix = self.prefix
        res_and_eval['query'] = res_and_eval['query'].apply(lambda x: str(x) if not isinstance(x, str) else x)
        res_and_eval['answer'] = res_and_eval['answer'].apply(lambda x: str(x) if not isinstance(x, str) else x)
        #traintext = list((res_and_eval['query'] + res_and_eval['answer']).apply(lambda x: scorer_text(x).removeprefix(prefix)))
        traintext = list((res_and_eval['query'] +" "+ res_and_eval['answer']).apply(lambda x: scorer_text(x.removeprefix(prefix))))



        trainlabel = list(res_and_eval['quality'])
        MyScore = Score(score_type=self.score_type)
        model = MyScore.train(traintext,trainlabel)
        return MyScore, model
    
    def get_scores(self, data, name):
        eps=self.eps
        model = self.scorer[name]
        scores_dict = dict()
        rawdata = data[['_id','query','answer']].to_dict(orient='records')
        for ptr in rawdata:
            text0 = ptr['query']+" "+ptr['answer']
            text0 = text0.removeprefix(self.prefix)
            text = scorer_text(text0)
            score1 = self.MyScores[name].get_score(text)
            if(self.score_noise_injection==True):
              score1+=random.random()*eps
            scores_dict[ptr['_id']] = score1
        return scores_dict

    def evaluateall(self,train,service_names,metric,genparams):
        api_responses = dict()
        for name in service_names:
            # step 1: get the answers from all API
            response = self._get_response(data=train, genparams=genparams,service_name=name)
            # step 2: evaluate the performance
            res_and_eval = self._evaluate(response, metric=metric)
            api_responses[name] = res_and_eval
        return api_responses

    def _evaluate(self,response, metric='em'):
        for i in range(len(response)):
            ptr = response[i]
            score = evaluate(prediction = ptr['answer'], ground_truth=ptr['true_answer'], metric=metric)
            response[i]['quality'] = score
        result = pandas.DataFrame(response)
        return result
    
    def build_cascade(self,model_perf_test, scorers, budget, cascade_depth,metric):
        LLMChain1 = LLMChain(metric=metric,L_max=cascade_depth)
        LLMChain1.setbudget(budget=budget)
        responses = dict()
        scores = dict()
        if(self.batch_build):
            try:
                responses = self.responses
                labels = self.labels
                scores = self.scores         
                print("scores",scores)

                LLMChain1.train(responses,labels,scores)
                self.LLMChain = LLMChain1
                return

            except:
                print("first train")

        for key in model_perf_test:
            labels = model_perf_test[key][['_id','true_answer']].rename(columns={'true_answer': 'answer'}).to_dict(orient='records')
            responses[key] = table2json(model_perf_test[key])
            scores[key] = self.get_scores(model_perf_test[key],name=key)
            tempsave(labels,responses[key],scores[key],key)
        # print("responses",responses)  
        # print("labels",labels) 
        # print("scores",scores)
        self.responses = responses
        self.labels = labels
        self.scores = scores         
        LLMChain1.train(responses,labels,scores)
        self.LLMChain = LLMChain1
        return
    
def table2json(df):
    # Convert DataFrame to the desired dictionary format
    result_dict = {}
    for _, row in df.iterrows():
        answer = row['answer']
        cost = row['cost']
        _id = row['_id']
        quality = row['quality']
    
        # Update "answer" key
        if 'answer' not in result_dict:
            result_dict['answer'] = dict()
        result_dict['answer'][_id]= answer
    
        # Update "cost" key
        if 'cost' not in result_dict:
            result_dict['cost'] = dict()
        result_dict['cost'][_id]= cost
    
        # Update "quality" key
        if 'quality' not in result_dict:
            result_dict['quality'] = dict()
        result_dict['quality'][_id]= quality

    result_dict['sp'] = dict()
    result_dict['logprobs'] = dict()
    return result_dict

class strategy():
    def __init__(self):
        return
