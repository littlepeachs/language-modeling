from statistics import mean
import numpy as np
from numpy.random import sample
from scipy import spatial
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.utils import shuffle
import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torchtext.legacy.data import Field
import pandas as pd
import argparse
from copy import *
from torchtext import data
import os
from torchtext.vocab import Vectors

# gpus = [0, 1, 2, 3]
# torch.cuda.set_device('cuda:{}'.format(gpus[0]))
# wordsList = np.load('./glove_data/wordsList.npy')

# wordsList = wordsList.tolist()  # Originally loaded as numpy array
# wordVectors = np.load('./glove_data/wordVectors.npy')
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

dataset_type = ["train","valid","test"]#
count = 0
 
# hyperparameter
embed_size = 300   
hidden_size = 832  
num_layers = 2
num_epochs = 30   
batch_size = 256       
learning_rate = 0.0065
seq_length = 10

parser = argparse.ArgumentParser()
parser.add_argument('--shuffle_or_continuous',type=str, default='shuffle') #use as k-shot k=train_size
parser.add_argument('--zero_or_former',type=str,default='zero')#,'delta_tuning'
params = parser.parse_args()
batch_order = params.shuffle_or_continuous #shuffle=1,continuous=0
zero_or_former = params.zero_or_former #1 means true

def preprocess_data():
    idx = 0
    total_sequence = []
    word2idx={}
    idx2word={}
    for name in dataset_type:
        sequence = []
        with open("penn-treebank/ptb.{}.txt".format(name)) as file:
            for line in file:
                words = line.split(' ')
                if words[-1]=='\n':
                    words[-1]='<eos>'
                # for word in words:
                #     if word not in word2idx:
                #         word2idx[word]=idx
                #         idx2word[idx]=word
                #         idx+=1
                sequence+=words
        total_sequence.append(sequence)
    # total_sequence = [].append(total_sequence)
    return total_sequence # ,word2idx,idx2word

vector = Vectors(name='glove_data/glove.6B.300d.txt')
# 利用torchtext建词汇表，词向量作为参数
TEXT =Field(sequential=True)


sequence=preprocess_data() #,word2idx,idx2word
allword =[sequence[0]+sequence[1]+sequence[2]]

TEXT.build_vocab(allword, vectors=vector)
vocab_size = len(TEXT.vocab)
word2idx = TEXT.vocab.stoi
idx2word = TEXT.vocab.itos
train_sequence = sequence[0]
valid_sequence = sequence[1]
# vocab_size = len(word2idx)
train_length = len(train_sequence)
valid_length = len(valid_sequence)

class LSTM_LM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTM_LM,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.RNN(embed_size,hidden_size,num_layers,batch_first = True,dropout=0.5)
        self.linear1 = nn.Linear(hidden_size, vocab_size)
        self.sigmoid =nn.Sigmoid()
        self.linear2 = nn.Linear(2048, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(p=0.71)
    def forward(self,x,h): #h is hidden vector
        embed = self.embed(x)
        embed = self.dropout(embed)
        out,(h,c) = self.lstm(embed,h)
        
        out = self.dropout(out)
        out = self.linear1(out)
        # out = self.sigmoid(out)
        # out = self.linear2(out)
        out = self.softmax(out)
        return out,(h,c)


def get_batch(sequence,total_length,start,batch_size,seq_length):
    batch = []
    next_sentences = []
    start  = start*batch_size*seq_length
    for i in range(batch_size):
        posi = start+i*seq_length
        seq = []
        next_sen = []
        for j in range(posi,posi+seq_length+1):
            if j>posi and j<total_length:
                next_sen.append(word2idx[sequence[j]])
            if j<posi+seq_length and j<total_length:
                seq.append(word2idx[sequence[j]])
            if j>=total_length:
                break
        if len(seq)==seq_length:
            batch.append(seq)
            next_sentences.append(next_sen)
    # for i in range(len(batch)):
    #     for j in range(len(batch[0])):
    #         if batch[i][j] in wordsList:
    #             batch[i][j] = wordVectors[wordsList.index(batch[i][j])]
    #         else:
    #             batch[i][j]=np.random.normal(loc=0,scale=1,size=(300)).astype("float32")
    batch = torch.tensor(batch).to(device)
    next_sentences = torch.tensor(next_sentences).to(device)
    return batch,next_sentences

model = LSTM_LM(vocab_size, embed_size, hidden_size, num_layers)
pretrain_weight = np.array(TEXT.vocab.vectors)
model.embed.weight.data.copy_(torch.from_numpy(pretrain_weight))

criterion = nn.NLLLoss(reduce=mean)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
    factor=0.2, patience=2, threshold=0.001, threshold_mode='abs')


model.to(device)


def evaluation():
    total_loss = 0
    model.eval()
    with torch.no_grad():
        count=0
        step = valid_length//(batch_size*seq_length)
        print("total step:{}".format(step))
        print("type:{} , hidden:{} ".format(batch_order,zero_or_former))
        continueous_batch_posi =np.arange(step)
        shuffle_batch_posi = np.arange(step)
        np.random.shuffle(shuffle_batch_posi)
        h_0 = torch.zeros(num_layers,batch_size,hidden_size).to(device)
        # c_0 = torch.zeros(num_layers,batch_size,hidden_size).to(device)

        for each_step in range(step):
            if batch_order=='shuffle':
                posi = shuffle_batch_posi[each_step]
            elif batch_order=='continuous':
                posi = continueous_batch_posi[each_step]
            batch,next_sentences = get_batch(valid_sequence,valid_length,posi,batch_size,seq_length)
            out,h = model(batch,h_0)
            if zero_or_former == 'former':
                if len(batch)==batch_size:
                    h_0 =deepcopy(h)
                    # c_0 = deepcopy(c)
                else:
                    h_0 = torch.zeros(num_layers,len(batch),hidden_size).to(device)
                    # c_0 = torch.zeros(num_layers,len(batch),hidden_size).to(device)

            loss = 0
            for i in range(out.shape[1]):
                sample_loss = criterion(out[:,i,:],next_sentences[:,i])
                loss+=sample_loss
            loss/=seq_length
            count+=1
            total_loss += loss.item()
            if(count%100==0):
                print("step:{} ,NLL batch_loss: {}".format(count,loss.item()))
                print("ppl : {}".format(np.exp(loss.item()))) #perplexity
        ave = total_loss/count
        # scheduler.step(ave)
        ppl = np.exp(ave)
        print("----------------------------------")
        print("VALID NLL loss: {}".format(ave))
        print("VALID ppl : {}".format(ppl))
        print("----------------------------------")
        return ave,ppl

def train():
    train_nll = []
    train_ppl = []
    valid_nll=[]
    valid_ppl = []
    for epoch in range(num_epochs):
        count=0
        total_loss=0
        model.train()
        print("epoch:{}, type:{} , hidden:{} ".format(epoch,batch_order,zero_or_former))
        step = train_length//(batch_size*seq_length)
        print("total step:{}".format(step))
        
        continueous_batch_posi =np.arange(step)
        shuffle_batch_posi = np.arange(step)
        np.random.shuffle(shuffle_batch_posi)
        h_0 = torch.zeros(num_layers,batch_size,hidden_size).to(device)
        # c_0 = torch.zeros(num_layers,batch_size,hidden_size).to(device)

        for each_step in range(step):
            if batch_order=='shuffle':
                posi = shuffle_batch_posi[each_step]
            elif batch_order=='continuous':
                posi = continueous_batch_posi[each_step]
            model.zero_grad()
            batch,next_sentence = get_batch(train_sequence,train_length,posi,batch_size,seq_length)
            
            out,h = model(batch,h_0)
            if zero_or_former == 'former':
                if len(batch)==batch_size:
                    h_0 =deepcopy(h.detach())
                    # c_0 = deepcopy(c.detach()) # we need deep copy and get it down the calculate graph
                else:
                    h_0 = torch.zeros(num_layers,len(batch),hidden_size).to(device)
                    # c_0 = torch.zeros(num_layers,len(batch),hidden_size).to(device)
            loss = 0
            for i in range(out.shape[1]):
                sample_loss = criterion(out[:,i,:],next_sentence[:,i])
                loss+=sample_loss
            loss/=seq_length
            loss.backward()
            optimizer.step()
            clip_grad_norm_(model.parameters(),0.5)
            count+=1
            total_loss+=loss.item()
            if(count%100==0):
                print("step:{} ,NLL batch_loss: {}".format(count,loss.item()))
                print("ppl : {}".format(np.exp(loss.item()))) #perplexity
        # scheduler.step(total_loss/count)
        print("epoch:{}".format(epoch))
        print("NLL loss: {}".format(total_loss/count))
        print("ppl : {}".format(np.exp(total_loss/count)))
        train_nll.append(total_loss/count)
        train_ppl.append(np.exp(total_loss/count))
        ave_nll,ppl = evaluation()
        valid_nll.append(ave_nll)
        valid_ppl.append(ppl)
         # early stop
        # if epoch>5:
        #     if(valid_ppl[-1]>valid_ppl[-2] and valid_ppl[-2]>valid_ppl[-3]):
        #         break

    dataframe = pd.DataFrame({'train_nll':train_nll,'train_ppl':train_ppl,'valid_nll':valid_nll,"valid_ppl":valid_ppl})
    dataframe.to_csv('rnn_train_valid_loss_{}_{}_naive.csv'.format(batch_order,zero_or_former),index=False,sep=',')
    with open("rnn_832_{}_{}_naive.pt".format(batch_order,zero_or_former),'wb') as f:
        torch.save(model,f)


train()
