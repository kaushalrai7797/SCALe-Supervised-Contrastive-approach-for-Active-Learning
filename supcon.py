from datasets import load_dataset, load_metric
from transformers import TrainingArguments, Trainer, AutoTokenizer, AutoModel,AutoModelForSequenceClassification
#----for roberta-----#
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaModel
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
from sklearn.metrics import classification_report
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
import random
from SupCsTrainer import *
import pdb
import pandas as pd

from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

GLUE_TASKS = ["cola", "mnli", "mnli-mm", "mrpc", "qnli", "qqp", "rte", "sst2", "stsb", "wnli"]



task = "trec"
model_name = "bert-base-uncased"


actual_task = "mnli" if task == "mnli-mm" else task
if task in GLUE_TASKS:
    dataset = load_dataset("glue", actual_task)
    metric = load_metric('glue', actual_task)
else:
    dataset = load_dataset(task)
    metric = load_metric("accuracy")


num_labels = 3 if actual_task =='mnli' else 2
if actual_task =='stsb': num_labels = 1

if task=="trec":
    num_labels=6
if task == "ag_news":
    num_labels = 4

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)


class Model_Classifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_labels, dropout):
        super(Model_Classifier, self).__init__()
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained(model_name)
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.dropout = dropout

        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(self.embedding_dim, self.num_labels),
            #nn.Dropout(self.dropout),
            #nn.ReLU(),
            #nn.Linear(self.hidden_dim, self.num_labels)
        )

    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[1]
        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits, last_hidden_state_cls


model = Model_Classifier(768, 20, num_labels, dropout=0.1)
#pdb.set_trace()
#model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels,output_hidden_states=True)

task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "imdb": ("text", None),
    "trec": ("text", None),
    "ag_news":("text", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
}


sentence1_key, sentence2_key = task_to_keys[task]

def preprocess_function(examples):
    if sentence2_key is None:
        return tokenizer(examples[sentence1_key], truncation=True)
    return tokenizer(examples[sentence1_key], examples[sentence2_key], truncation=True)

encoded_dataset = dataset.map(preprocess_function, batched=True)

if task=='trec':
    encoded_dataset = encoded_dataset.rename_column("label-coarse","label")
# elif task == 'ag_news':
#     encoded_dataset = encoded_dataset.rename_column("label","labels")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if task != "stsb":
        predictions = np.argmax(predictions, axis=1)
    else:
        predictions = predictions[:, 0]
    return metric.compute(predictions=predictions, references=labels)

validation_key = "validation_mismatched" if task == "mnli-mm" else "validation_matched" if task == "mnli" else "validation"


if task =="trec":
    validation_key = "test"
    train_dataset = encoded_dataset["train"]
    val_dataset = encoded_dataset[validation_key]
if task == "ag_news":
    val_dataset = encoded_dataset["test"]
    train_dataset = encoded_dataset["train"]
    train_dataset = train_dataset.shuffle(seed=42)
    # val_dataset = train_dataset.select(range(6000))
    train_dataset = train_dataset.select(range(6000,120000))

acquisition_size = len(encoded_dataset['train'])//100
train_dataset.shuffle(seed=42)
train_dataset = train_dataset.select(range(acquisition_size))
#label = np.array(encoded_dataset["train"]["label"])


# zero_ind = np.where(label==0)[0]
# one_ind = np.where(label==1)[0]
# inds = random.choices(one_ind,k=acquisition_size//2)
# inds = inds + random.choices(zero_ind,k=acquisition_size//2)
# random.shuffle(inds)
# train_dataset = train_dataset.select(inds)
# assert len(train_dataset)==acquisition_size 

CL_args = TrainingArguments(
        output_dir = './results',
        save_total_limit = 1,
        num_train_epochs=25,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=1,  
        logging_steps = 50,
        learning_rate = 5e-05,
        evaluation_strategy = 'steps',
        per_device_eval_batch_size=64,
        eval_steps = 100,
        lr_scheduler_type='constant',
        #warmup_steps=50, 
        weight_decay=0.01,               
        logging_dir='./logs',
        label_names=['labels']
    )

SupCL_trainer = SupCsTrainer(   
            w_drop_out= [0.1,0.1],
            temperature= 0.1,
            def_drop_out=0.1,
            pooling_strategy='pooler',
            model = model,
            args = CL_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics
        )

SupCL_trainer.train()
SupCL_trainer.save_model('./sst2_baseline')

model_name = './sst2_baseline'#"./results/checkpoint-500/"


# pdb.set_trace()
#------ Add classification layer ---------#
#model = RobertaForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)
# model = AutoModelForSequenceClassification.from_pretrained(model_name,num_labels=num_labels)
# ---- Freeze the base model -------#

# for param in model.base_model.parameters():
#     param.requires_grad = False


# pdb.set_trace()
# embeddings_train = []
# labels_train = [] 
# for i in range(len(train_dataset)):
#     input_ids = torch.tensor(tokenizer.encode(train_dataset['sentence'][i])).unsqueeze(0)  # Batch size 1
#     inputs = {
#     "input_ids": input_ids}
#     outputs = model(**inputs,output_hidden_states=True)
#     last_hidden_states = outputs[1][0][0].numpy().mean(axis=0)
#     embeddings_train.append(last_hidden_states)
#     labels_train.append(train_dataset['label'][i])


# embeddings = []
# labels = [] 



# for i in range(len(test_dataset)):
#     input_ids = torch.tensor(tokenizer.encode(test_dataset['sentence'][i])).unsqueeze(0)  # Batch size 1
#     inputs = {
#     "input_ids": input_ids}
#     outputs = model(**inputs,output_hidden_states=True)
#     last_hidden_states = outputs[1][0][0].numpy().mean(axis=0)
#     embeddings.append(last_hidden_states)   
#     labels.append(test_dataset['label'][i])


# sgd =  LogisticRegression()


# pdb.set_trace()

    


    
    
    






# args = TrainingArguments(report_to="wandb",output_dir = './results',per_device_eval_batch_size=64,evaluation_strategy = 'steps',save_total_limit = 1,num_train_epochs=20,per_device_train_batch_size=32,gradient_accumulation_steps=1,logging_steps = 200,learning_rate = 1e-03,eval_steps = 20,warmup_steps=50,weight_decay=0.01,logging_dir='./logs')

# trainer = Trainer(model,args,train_dataset=train_dataset,eval_dataset=test_dataset,tokenizer=tokenizer,compute_metrics=compute_metrics)
    
# trainer.train()
# wandb.finish()



