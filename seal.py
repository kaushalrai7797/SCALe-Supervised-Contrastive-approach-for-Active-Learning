from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader
import time
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import math
# import warnings filter
from warnings import simplefilter
from datasets import load_dataset
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

import pandas as pd
import numpy as np
#from parallel import DataParallelModel, DataParallelCriterion

import torch
import random
import torch
import pdb



def set_seed(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)

set_seed(0)



def max_entropy_acquisition_function(logits_b_K_C):
    return entropy_val(logit_mean(logits_b_K_C, dim=1, keepdim=False), dim=-1)

def entropy_val(logits, dim: int, keepdim: bool = False):
    return -torch.sum((torch.log(logits) * logits).double(), dim=dim, keepdim=keepdim)
def logit_mean(logits, dim: int, keepdim: bool = False):
    r"""Computes $\log \left ( \frac{1}{n} \sum_i p_i \right ) =
    \log \left ( \frac{1}{n} \sum_i e^{\log p_i} \right )$.

    We pass in logits.
    """
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(logits.shape[dim])

def entropy(logits):
    """
    Entropy-based uncertainty.
    see http://robertmunro.com/Uncertainty_Sampling_Cheatsheet_PyTorch.pdf
    :param prob_dist: output probability distribution
    :return:
    """

    if type(logits) is list:
        logits_B_K_C = torch.stack(logits, 1)  # prob
    else:
        logits_B_K_C = logits.unsqueeze(1)  # det

    entropy_scores_ = max_entropy_acquisition_function(logits_B_K_C)
    return entropy_scores_.cpu().numpy()






def sentence_pairs_generation(sentences, labels, pairs):
	# initialize two empty lists to hold the (sentence, sentence) pairs and
	# labels to indicate if a pair is positive or negative

  numClassesList = np.unique(labels)
  idx = [np.where(labels == i)[0] for i in numClassesList]

  for idxA in range(len(sentences)):      
    currentSentence = sentences[idxA]
    label = labels[idxA]
    idxB = np.random.choice(idx[np.where(numClassesList==label)[0][0]])
    posSentence = sentences[idxB]
		  # prepare a positive pair and update the sentences and labels
		  # lists, respectively
    pairs.append(InputExample(texts=[currentSentence, posSentence], label=1.0))

    negIdx = np.where(labels != label)[0]
    negSentence = sentences[np.random.choice(negIdx)]
		  # prepare a negative pair of images and update our lists
    pairs.append(InputExample(texts=[currentSentence, negSentence], label=0.0))
  
	# return a 2-tuple of our image pairs and labels
  return (pairs)


#SST-2
# Load SST-2 dataset into a pandas dataframe.

def sst_2():
    train_df = pd.read_csv('data/SST-2/train.tsv', delimiter='\t')
    train_df, val_df = train_test_split(train_df,test_size=0.1,random_state=42)
    # Load the test dataset into a pandas dataframe.
    eval_df = pd.read_csv('data/SST-2/dev.tsv', delimiter='\t')
    label = "label"

    return train_df,eval_df,label


def imdb():
    train_df = pd.read_csv('data/IMDB/train/train.tsv')
    train_df, val_df = train_test_split(train_df,test_size=0.1,random_state=42)
    # Load the test dataset into a pandas dataframe.
    eval_df = pd.read_csv('data/IMDB/test/test.tsv')
    label = "sentiment"

    return train_df,eval_df,label

def trec():
    trec_6 = {}
    data_all = load_dataset('trec')
    trec_6['sentence'] = data_all['train']['text']
    trec_6['label'] = data_all['train']['label-coarse']

    train_df_trec = pd.DataFrame(trec_6)
    train_df_trec, val_df_trec = train_test_split(train_df_trec,test_size=546,random_state=42)
    # Load the test dataset into a pandas dataframe.
    test_df_trec = {}
    test_df_trec['sentence'] = data_all['test']['text']
    test_df_trec['label'] = data_all['test']['label-coarse']
    eval_df_trec = pd.DataFrame(test_df_trec)
    label = "label" 
    return train_df_trec,eval_df_trec,label


def aquisition_step(model_lr,X,algo="Entropy"):
    print("Algo *****"+algo)
    if algo=="Entropy":
        logits = model_lr.predict_proba(X)
        logits = torch.Tensor(logits)
        unc_scores = entropy(logits)
        new_ids = np.argsort(unc_scores)[::-1][:aquisition_size]
    else:
        new_ids = np.random.choice(len(X),aquisition_size) 

    return new_ids
  

    



print("SST-2 Dataset")
train_df,eval_df,label = sst_2() #experiment between sst, trec, imdb
# random.shuffle(train_df)
# train_df_label = train_df.iloc[:aquisition_size]
# train_df_unlabel = train_df.iloc[aquisition_size:]

# train_df_unlabel, train_df_label = train_test_split(train_df,test_size=,random_state=42)

# pd.concat([train_df[train_df[label]==0].sample(aquisition_size//2), train_df[train_df[label]==1].sample(aquisition_size//2)])




text_col=train_df.columns.values[0] 
category_col=train_df.columns.values[1]

x_eval = eval_df[text_col].values.tolist()
y_eval = eval_df[category_col].values.tolist()

aquisition_size = 100
st_model = 'paraphrase-mpnet-base-v2' #@param ['paraphrase-mpnet-base-v2', 'all-mpnet-base-v1', 'all-mpnet-base-v2', 'stsb-mpnet-base-v2', 'all-MiniLM-L12-v2', 'paraphrase-albert-small-v2', 'all-roberta-large-v1']
num_itr = 5 #@param ["1", "2", "3", "4", "5", "10"] {type:"raw"}
plot2d_checkbox = False #@param {type: 'boolean'}
set_fit_accs = {}
no_fit_accs = {}

orig_model = SentenceTransformer(st_model)
model = SentenceTransformer(st_model)
train_loss = losses.CosineSimilarityLoss(model)

def fit(train_label,train_unlabel,algo="Entropy"):
    train_label = train_label.reset_index(drop=True)
    train_unlabel = train_unlabel.reset_index(drop=True)

    # Equal samples per class training
    #train_df_sample = train_df.sample(num_training*6)
    #pd.concat([train_df[train_df[label]==0].sample(num_training), train_df[train_df[label]==1].sample(num_training)])
    # train_df_sample = train_df[train_df[label]==0].sample(num_training)
    # for i in range(1,max(train_df[label]+1)):
    #     print(i)
    #     train_df_sample = pd.concat([train_df_sample, train_df[train_df[label]==i].sample(num_training)])
    x_train = train_label[text_col].values.tolist()
    y_train = train_label[category_col].values.tolist()

    x_train_unlabel = train_unlabel[text_col].values.tolist()
    y_train_unlabel = train_unlabel[category_col].values.tolist()



    train_examples = [] 
    for x in range(num_itr):
        train_examples = sentence_pairs_generation(np.array(x_train), np.array(y_train), train_examples)
    
    # S-BERT adaptation 
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=64)
    
    #pdb.set_trace()
    model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=1, warmup_steps=10, show_progress_bar=True)

    # # No Fit
    # X_train_noFT = orig_model.encode(x_train)
    # X_eval_noFT = orig_model.encode(x_eval)

    # sgd =  LogisticRegression()
    # sgd.fit(X_train_noFT, y_train)
    #y_pred_eval_sgd = sgd.predict(X_eval_noFT)
    #no_fit_accs[num_training*2] = accuracy_score(y_eval, y_pred_eval_sgd)
    
    # With Fit (SetFit)
    X_train = model.encode(x_train)
    X_eval = model.encode(x_eval)
    
    sgd =  LogisticRegression()
    sgd.fit(X_train, y_train)
    y_pred_eval_sgd = sgd.predict(X_eval)

    acc = accuracy_score(y_eval, y_pred_eval_sgd) 

    X_train_unlabel = model.encode(x_train_unlabel)
    new_ids_label = aquisition_step(sgd,X_train_unlabel,algo)
    new_ids_unlabel = [i for i in train_unlabel.index.tolist() if i not in new_ids_label]
    

    train_label_new = train_unlabel.iloc[new_ids_label]
    train_label = pd.concat([train_label,train_label_new])
    train_unlabel = train_unlabel.iloc[new_ids_unlabel]


    #* num_classes
    print('Acc. SetFit', accuracy_score(y_eval, y_pred_eval_sgd))
    return train_label,train_unlabel,acc





results_random =[[] for i in range(10)]
results_entropy =[[] for i in range(10)]

print("Random")
for j in range(5):
    label_ids = np.random.choice(len(train_df),aquisition_size)
    unlabel_ids = [i for i in range(len(train_df)) if i not in label_ids]
    train_label = train_df.iloc[label_ids]
    train_unlabel = train_df.iloc[unlabel_ids]

    for i in range(10):
        train_label,train_unlabel ,acc= fit(train_label,train_unlabel,"random")
        print(len(train_label))
        results_random[i].append(acc)

print("Entropy")
for j in range(5):
    label_ids = np.random.choice(len(train_df),aquisition_size)
    unlabel_ids = [i for i in range(len(train_df)) if i not in label_ids]
    train_label = train_df.iloc[label_ids]
    train_unlabel = train_df.iloc[unlabel_ids]
    for i in range(10):
        train_label,train_unlabel ,acc= fit(train_label,train_unlabel)
        #print(acc)
        results_entropy[i].append(acc)
 

# for i in range(80,500,80):
#     print(i)
#     fit(i)
pdb.set_trace()
