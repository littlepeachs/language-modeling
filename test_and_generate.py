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
device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")

dataset_type = ["train","valid","test"]#
count = 0
 
# hyperparameter
embed_size = 300   
hidden_size = 832  
num_layers = 2
num_epochs = 30   
batch_size = 256       
learning_rate = 0.0005
seq_length = 10

class LSTM_LM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(LSTM_LM,self).__init__()
        self.embed = nn.Embedding(vocab_size,embed_size)
        self.lstm = nn.LSTM(embed_size,hidden_size,num_layers,batch_first = True)#,dropout=0.5
        self.linear1 = nn.Linear(hidden_size, vocab_size)
        self.sigmoid =nn.Sigmoid()
        self.linear2 = nn.Linear(2048, vocab_size)
        self.softmax = nn.LogSoftmax(dim=2)
        self.dropout = nn.Dropout(p=0.71)
    def forward(self,x,h): #h is hidden vector
        embed = self.embed(x)
        # embed = self.dropout(embed)
        out,(h,c) = self.lstm(embed,h)
        
        # out = self.dropout(out)
        out = self.linear1(out)
        # out = self.sigmoid(out)
        # out = self.linear2(out)
        out = self.softmax(out)
        return out,(h,c)


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

criterion = nn.NLLLoss(reduce=mean)
def test():
    with open('lstm_832_shuffle_zero_layer1_has_dropout.pt','rb') as f:
        model = torch.load(f)
        model.to(device)
    # optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)    
    model.eval()
    with open("penn-treebank/ptb.test.txt") as file:
        test_sequence = []
        for line in file:
            words = line.split(' ')
            if len(words)<=5:
                continue
            elif len(words)>=20:
                words =  words[:20]
            if words[-1]=='\n':
                words[-1]='<eos>'
            for i in range(len(words)):
                words[i] = word2idx[words[i]]
            eos_token = word2idx["<pad>"]
            for i in range(len(words),20):
                words.append(eos_token)
            test_sequence.append(words)

    h_0,c_0 = torch.zeros(num_layers,1,hidden_size).to(device),torch.zeros(num_layers,1,hidden_size).to(device)
    with torch.no_grad():
        step=0
        test_nll = []
        for sentence in test_sequence:
            sen_loss = 0
            step+=1
            gennerate_sen =sentence[:5]
            posi = 0 # record the sub-sentence position
            # for word_id in sentence[:5]:
            #     embed_sen.append(wordVectors[word_id])
            for i in range(15): # generate 15 words
            #     # print("success")
                posi+=1
                embed_sen = [gennerate_sen[-5:]]
                embed_sen = torch.tensor(embed_sen).to(device)
            #     embed_sen = embed_sen.unsqueeze(0)
                out,(h,c) = model(embed_sen,(h_0,c_0))

                loss = criterion(out[0],torch.tensor(sentence[posi:posi+5]).to(device))
                sen_loss+=loss.item()

                out = out[0,-1,:].sort(descending=True)
                out_word_logit = out[0][:40].detach().cpu().numpy()
                out_word_logit = np.exp(out_word_logit)
                out_word_logit /= out_word_logit.sum()
                out_word_token = out[1][:40].cpu().detach().numpy()
                index = np.random.choice(out_word_token, p=out_word_logit.ravel())

                gennerate_sen.append(index)
            sen_loss/=15
            if(step%1000==0):
                print("sentence loss:{}".format(sen_loss))
            test_nll.append(sen_loss)
        test_nll = np.mean(test_nll)
        print("test_ave_nll: {},test_ave_ppl: {}".format(test_nll,np.exp(test_nll)))
                
def decode(input): # input_sentence type:string use 4 word to gennerate
    input_sentence = input.split()
    if len(input_sentence)<4:
        print("need longer")
        return
    with open('lstm_832_shuffle_zero_boost.pt','rb') as f:
        model = torch.load(f)
        model.to(device)
    model.eval()
    embed_sen =[]
    gennerate_sen = []
    real_sen = []
    for word in input_sentence:
        gennerate_sen.append(word2idx[word])
    # for word_id in gennerate_sen:
    #     embed_sen.append(wordVectors[word_id])
    
    for i in range(20):
        embed_sen = [gennerate_sen]
        embed_sen = torch.tensor(embed_sen).to(device)
        # embed_sen = embed_sen.unsqueeze(0)
        h_0,c_0 = torch.zeros(num_layers,1,hidden_size).to(device),torch.zeros(num_layers,1,hidden_size).to(device)
        out,(h,c) = model(embed_sen,(h_0,c_0))
        out = out[0,-1,:].sort(descending=True)
        out_word_logit = out[0][:5].detach().cpu().numpy()
        out_word_logit = np.exp(out_word_logit)
        out_word_logit /= out_word_logit.sum()
        out_word_token = out[1][:5].cpu().detach().numpy()
        while True:
            index = np.random.choice(out_word_token, p=out_word_logit.ravel())
            if index == word2idx['<unk>'] or len(idx2word[index])==1:
                continue
            else:
                gennerate_sen.append(index)
                break
        # h_0,c_0 = deepcopy(h.detach()),deepcopy(c_0.detach())
    
    for i in range(len(gennerate_sen)):
        real_sen.append(idx2word[gennerate_sen[i]])
    print(" ".join(real_sen))

test()

# decode("it is widely accepted that")
