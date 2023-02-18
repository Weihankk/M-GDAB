#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 15:27:03 2022

@author: weihan
"""
import argparse
import torch
import random
import math
import gc
import pickle
import os

import numpy as np
import pandas as pd
import torch.nn as nn
import scipy.sparse as sp
import torch.optim as optim
import torch.nn.init as init
import torch.nn.functional as F

from tqdm import tqdm
from itertools import product
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.metrics import precision_recall_curve,roc_curve,auc,balanced_accuracy_score

import warnings
warnings.filterwarnings("ignore")

np.random.seed(123)

device = "cuda:3" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# For debug
print("Reading data")
GTR_adj = pd.read_csv("./Process_data/GTR.txt", sep = "\t", index_col=0, header=0)
GPR_adj = pd.read_csv("./Process_data/GPR.txt", sep = "\t", index_col=0, header=0)

GTR_adj = torch.from_numpy(GTR_adj.values)
GTR_adj = GTR_adj.to(torch.float32)
GPR_adj = torch.from_numpy(GPR_adj.values)
GPR_adj = GPR_adj.to(torch.float32)

G_P = pd.read_csv("./Process_data/G_P.txt", sep = "\t", index_col=0, header=0)
g_n = len(G_P)

G_D = pd.read_csv("./Process_data/G_D.txt", sep="\t", index_col=0, header=0)
D_D = pd.read_csv("./Process_data/D_D.txt", sep="\t", index_col=0, header=0)

parser = argparse.ArgumentParser()
parser.add_argument('--hidden_dim', type=int, default=64)
parser.add_argument('--dropout', type=float, default=0.05)
parser.add_argument('--lr', type=float, default=0.0004)
parser.add_argument('--epochs', type=int, default=1000)
parser.add_argument('--kfold', type=int, default=5)
parser.add_argument('--ChebConvK', type=int, default=4)
args = parser.parse_args()


class Net(nn.Module):
    def __init__(self, n_representation, hidden_dims=64, dropout=0.3):
        super().__init__()
        
        self.n_representation = n_representation
        self.linear1 = nn.Linear(self.n_representation, hidden_dims)
        self.linear2 = nn.Linear(hidden_dims, hidden_dims)
        self.linear3 = nn.Linear(hidden_dims, hidden_dims)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
    
    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = F.relu(self.linear2(x))
        x = self.dropout(x)
        x = self.linear3(x)
        return x
    
    def init_weights(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)

class mlp(nn.Module):
    def __init__(self, nhid, nclass, dropout):
        super().__init__()
        
        self.mlp1 = nn.Linear(nhid, nclass)
        self.mlp2 = nn.Linear(nclass, nclass)
        self.dropout = dropout
        
        self.reset_parameters()
        
    def reset_parameters(self):
        nn.init.normal_(self.mlp1.weight,std=0.05)
        nn.init.normal_(self.mlp2.weight,std=0.05)
        
    def forward(self, x, GPR):
        if GPR:
            x=torch.from_numpy(x)
            x = F.tanh(self.mlp1(x))
            x = self.mlp2(x)
        else:
            x=torch.from_numpy(x)
            x = self.mlp1(x)
            # x = self.mlp2(x)
        return x

class Sample_Set(Dataset):
    '''
    train dataSet
    '''
    
    def __init__(self,file_train_sample, ggi,dds,node2index=None):
        self.pairs_train = get_pairs(file_train_sample)
        self.all_genes, self.all_diseases = get_items(self.pairs_train)
        self.dropout = 0.5
        self.nega_weight = 10
        if node2index is None:
            self.node2index = assign_index(self.all_genes, self.all_diseases)
        else:
            self.node2index = node2index
        self.ggi=ggi
        self.dds=dds
    
    def reassign_samples(self):
        '''
        renew generate positive samples and negative samples
        :return:
        '''
        self.adj_matrix = adjacency_matrix(self.pairs_train, self.ggi, self.dds, self.node2index)
        self.negative_samples = generate_negative(self.all_genes, self.all_diseases, self.pairs_train)
        self.samples = merge_samples(list(self.pairs_train), self.negative_samples)
        
    def get_adj_matrix(self):
        '''
        get positive_samples_target adjacency matrix
        :return:adjacency matrix ,type is coo_matrix
        '''
        return self.adj_matrix
    
    def get_node2index(self):
        '''
        get node's index
        :return: node's index ,type is dict
        '''
        return self.node2index
    
    def __getitem__(self, index):
        if index < len(self.pairs_train):
            target_flag = 1
        else:
            target_flag = 0
        item = self.samples[index]
        sample = node_vector(item, self.node2index)
        target = target_vector(target_flag)
        return sample, target
    
    def __len__(self):
        return len(self.samples)

class pre_GNN(nn.Module):
    def __init__(self, feature, hidden1, hidden2,dropout=0.2,Kvalue=4):
        super(pre_GNN, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.encoder_o1 = ChebConv(hidden1, hidden1,K=Kvalue)
        self.encoder_o2 = ChebConv(hidden1, hidden1,K=Kvalue)
    
    def forward(self, x_o, adj):
        x1_o = F.leaky_relu(self.encoder_o1(x_o, adj))
        x1_o = self.dropout(x1_o)
        x=self.encoder_o2(x1_o,adj)
        return(x)

class Link_Prediction(nn.Module):
    def __init__(self, n_representation, hidden_dims=[64, 32], dropout=0.3):
        super(Link_Prediction, self).__init__()
        self.n_representation = n_representation
        self.linear1 = nn.Linear(2*self.n_representation, hidden_dims[0])
        self.linear2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.linear3 = nn.Linear(hidden_dims[1], 1)
        self.dropout = nn.Dropout(p=dropout)
        self.sigmoid = nn.Sigmoid()
        self.init_weights()
        
    def forward(self, x1, x2):
        x = torch.cat((x1,x2),1) # N * (2 node_dim)
        
        x = F.relu(self.linear1(x)) # N * hidden1_dim
        x = self.dropout(x)
        x = F.relu(self.linear2(x)) # N * hidden2_dim
        x = self.dropout(x)
        x = self.linear3(x) # N * 2
        x = self.sigmoid(x) # N * ( probility of each event )
        return x
    
    def init_weights(self):
        init.xavier_uniform_(self.linear1.weight)
        init.xavier_uniform_(self.linear2.weight)
        init.xavier_uniform_(self.linear3.weight)
        
def get_pairs(file):
    # get gid and did as a pair from file and add it in a set
    df = file  # sep='\t'
    df.reset_index(inplace=True)
    pairs = set()
    for i in range(0, len(df)):
        pair = (str(df['gene'][i]), df['dis'][i])
        pairs.add(pair)
    return(pairs)

def get_items(pairs):
    # get gene_set and  disease_set from pairs set
    gene_set = set()
    disease_set = set()
    for pair in pairs:
        gene_set.add(pair[0])
        disease_set.add(pair[1])
    
    return gene_set, disease_set
    # generate negative sample
    
def generate_negative(genes, diseases, pos_samples):
    # generate negative sample
    pairs = []
    genes, diseases = list(genes), list(diseases)
    pairs=set([i for i in product(*[genes,diseases])]) -pos_samples
    return list(pairs)

def assign_index(all_genes, all_diseases):
    # Set node index
    node_index = dict()
    for gene in all_genes:
        node_index[gene] = len(node_index)
    for disease in all_diseases:
        node_index[disease] = len(node_index)
    return node_index

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def adjacency_matrix(pos_samples, ggi, dds, node2index):
    values = [1.0 for i in range(0, len(node2index))]
    vertex_1 = list(range(0, len(node2index)))
    vertex_2 = list(range(0, len(node2index)))
    """ Generate associations based on positive sample sets"""
    for ps in pos_samples:
        values.append(1.0) ## ps[2]
        vertex_1.append(node2index[ps[0]])
        vertex_2.append(node2index[ps[1]])
    
    """ Generate associations based on gene to gene associations"""
    df = ggi
    df.columns=['gid1','gid2','score']
    df=df.reset_index()
    
    for i in range(0, len(df)):
        g1 = node2index[str(df['gid1'][i])]
        g2 = node2index[str(df['gid2'][i])]
        values.append(df['score'][i])
        vertex_1.append(g1)
        vertex_2.append(g2)
    
    """ Generate associations based on disease to disease associations"""
    df = dds
    df.columns=['mid1','mid2','score']
    df.reset_index(inplace=True)
    for i in range(0, len(df)):
        d1 = node2index[int(df['mid1'][i])]
        d2 = node2index[int(df['mid2'][i])]
        values.append(df['score'][i])
        vertex_1.append(d1)
        vertex_2.append(d2)
    
    """build graph, coo_matrix((data, (i, j)), [shape=(M, N)])"""
    adj = sp.coo_matrix((values, (vertex_1, vertex_2)), shape=(len(node2index), len(node2index)), dtype=np.float32)
    
    """ build symmetric adjacency matrix"""
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = normalize(adj)  # + sp.eye(adj.shape[0])
    return adj

def merge_samples(p_samples, n_samples):
    '''
    merge negative and positive samples by order
    :param nega_weight: the rate of negative  and  positive samples
    :param p_samples: positive samples
    :param n_samples: negative samples
    :return: all train samples
    '''
    pos_sample=[]
    neg_sample=[]
    for w in p_samples:
        pos_sample.append(w)
    
    for i in n_samples:
        neg_sample.append(i)
    
    neg_sample=random.sample(neg_sample,len(pos_sample))
    samples=pos_sample+neg_sample
    #print(len(pos_sample),len(neg_sample))
    return samples

def node_vector(item, node2index):
    n1, n2 = item[0], item[1]
    tensor1 = torch.tensor(node2index[n1])
    tensor2 = torch.tensor(node2index[n2])
    return tensor1, tensor2

def target_vector(flag):
    """ Generate label vector by flag"""
    tensor = torch.LongTensor(1).zero_()
    tensor[0] = flag
    return tensor

def extract_gene(GTR, n):
    a = GTR[:n,:]
    b = GTR[n:,:].T
    fuse_GTR = np.matmul(a,b)
    return(fuse_GTR)

def fuseNet(fuse_GPR,fuse_GTR):
    GPR = fuse_GPR.detach().numpy()
    GTR = fuse_GTR.detach().numpy()
    fuse_GPR =np.matmul(GPR, GTR.T)
    
    from sklearn.metrics.pairwise import pairwise_distances
    #print('gpr', GPR)
    fuse_GPR = pairwise_distances(fuse_GPR, metric="manhattan")
    #print('fuse_gpr_con', fuse_GPR)
    
    mean_ = np.mean(fuse_GPR ,axis=1)
    mean_dims = np.expand_dims(mean_, axis=1)
    b = np.repeat(mean_dims, fuse_GPR.shape[1], axis=1)
    fuse_GPR[fuse_GPR<=b] = 0
    fuse_GPR[fuse_GPR>b] = 1
    #print('fuse_gpr_final', fuse_GPR)
    
    return(fuse_GPR)

def partition(ls, size):
    return [ls[i:i+size] for i in range(0, len(ls), size)]

def reconstruct_gene(gene_dis):
    a,b=np.where(gene_dis==1)
    return pd.DataFrame({'gene':a.tolist(),'dis':b.tolist(),'score':[1 for i in range(len(a))]})

def reconstruct(gene_dis):
    score=[]
    gene=[]
    dis=[]
    for i in range(len(gene_dis)):
        for j in range(gene_dis.shape[1]):
            score.append(gene_dis[i,j])
            gene.append(i)
            dis.append(j)
    return pd.DataFrame({'gene':gene,'dis':dis,'score':score})

def generate_train_test(TrainPositiveFeature,data_gene_gene,data_dis_dis):
    gene = []
    dis = []
    for i in TrainPositiveFeature:
        gene.append(i[0])
        dis.append(i[1])
    train_gene_di = pd.DataFrame({'gene': gene, 'dis': dis})
    gene_in = set(gene)
    dis_in = set(dis)
    
    data_gene_gene = data_gene_gene[data_gene_gene['gene'].isin(gene_in)]
    data_gene_gene = data_gene_gene[data_gene_gene['dis'].isin(gene_in)]
    
    indexs_dis = []
    for index, row in data_dis_dis.iterrows():
        if row['gene'] not in dis_in:
            indexs_dis.append(index)
        if row['dis'] not in dis_in:
            indexs_dis.append(index)
    
    data_dis_dis.drop(list(set(indexs_dis)),inplace=True)
    return train_gene_di,data_gene_gene,data_dis_dis
    
def get_kfold_data(k, NewRandomList, gene_dislist,data_gene_gene,data_dis_dis):
    counter = k
    PositiveSample = list(set(gene_dislist))
    Num = 0
    NewPositiveSampleFeature = []
    counter1 = 0
    while counter1 < len(NewRandomList):
        PairP = []
        PairP.extend(PositiveSample[Num:Num + len(NewRandomList[counter1])])
        NewPositiveSampleFeature.append(PairP)
        Num = Num + len(NewRandomList[counter1])
        counter1 = counter1 + 1
    
    TestPositiveFeature = []
    TrainPositiveFeature = []
    counter2 = 0
    #print(counter)
    while counter2 < len(NewRandomList):
        if counter2 == counter:
            TestPositiveFeature.extend(NewPositiveSampleFeature[counter2])
        else:
            TrainPositiveFeature.extend(NewPositiveSampleFeature[counter2])
        counter2 = counter2 + 1
    #print(len(TrainPositiveFeature),len(TestPositiveFeature))
    train_gene_di, train_data_gene_gene, train_data_dis_dis=generate_train_test(TrainPositiveFeature,data_gene_gene.copy(),data_dis_dis.copy())
    test_gene_di, test_data_gene_gene, test_data_dis_dis=generate_train_test(TestPositiveFeature,data_gene_gene.copy(),data_dis_dis.copy())
    return(train_gene_di, train_data_gene_gene, train_data_dis_dis,test_gene_di, test_data_gene_gene, test_data_dis_dis)

def load_pretrain_vector(node2index,args):
    'load node feature embedding'
    embed_gene=pd.read_csv('./Process_data/Gene_features.txt', sep="\t", header=0, index_col=0).reset_index(drop=True)
    embed_dis=pd.read_csv('./Process_data/Disease_features.txt',sep="\t", header=0, index_col=0).reset_index(drop=True)
    
    a=[float(i) for i in range(0,embed_gene.shape[0])]
    b=[float(j) for j in range(embed_gene.shape[0],embed_gene.shape[0]+embed_dis.shape[0])]
    
    embed_gene.insert(0,"id",a)
    embed_dis.insert(0,'id',b)
    embed=pd.concat([embed_gene,embed_dis])
    # print('embed',embed)
    # print(node2index.keys())
    embedding = [ [] for i in range(0,len(node2index.keys() )) ]
    for key in node2index.keys() :
        index = node2index[key]
        embedding[index] = embed.loc[embed['id']==float(key)].iloc[:,1:].values.tolist()[0]
    embedding = np.array(embedding)
    return embedding

def load_embedding(node2index,args):
    embedding = load_pretrain_vector(node2index,args)  #
    embedding=torch.tensor(embedding, dtype=torch.float)
    return embedding

def sparse_to_torch_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    edges_s = sparse_mx.nonzero()
    edge_index_s = torch.tensor(np.vstack((edges_s[0], edges_s[1])), dtype=torch.long)
    return edge_index_s

def auroc(prob,label):
    y_true=label.data.numpy().flatten()
    y_scores=prob.data.numpy().flatten()
    fpr,tpr,thresholds=roc_curve(y_true,y_scores)
    auroc_score=auc(fpr,tpr)
    return auroc_score,fpr,tpr

def auprc(prob,label):
    y_true=label.data.numpy().flatten()
    y_scores=prob.data.numpy().flatten()
    precision,recall,thresholds=precision_recall_curve(y_true,y_scores)
    auprc_score=auc(recall,precision)
    return auprc_score,precision,recall

def test(lp_model,gcn,test_embedding,testloader,testset,fold):
    lp_model.eval()
    gcn.eval()
    with torch.no_grad():
        adj_matrix = testset.get_adj_matrix()
        adj_matrix=sparse_to_torch_tensor(adj_matrix)
        
        rp_data=Data(x=test_embedding, edge_index=adj_matrix)
        #rp_data = rp_data.to(device)

        rp_matrix = gcn(rp_data.x, rp_data.edge_index)
        #rp_matrix=rp_matrix.to(device)
        for i, data in enumerate(testloader, 0):
            inputs, labels = data
            input1, input2 = inputs[0], inputs[1]
            rps1 = F.embedding(input1.to(device), rp_matrix.to(device))
            rps2 = F.embedding(input2.to(device), rp_matrix.to(device))
            outputs = lp_model(rps1, rps2).cpu()
    
            labels = labels.float()
            test_auc,fpr,tpr=auroc(outputs,labels)
            test_aupr,precision,recall=auprc(outputs,labels)
            out=outputs.data.numpy().flatten()
            #pd.DataFrame(out).to_csv('./K{}_result/out.csv'.format(fold),index=0,header=0)
            #pd.DataFrame(input1.numpy()).to_csv('./K{}_result/input1.csv'.format(fold),index=0,header=0)
            #pd.DataFrame(input2.numpy()).to_csv('./K{}_result/input2.csv'.format(fold),index=0,header=0)
            #a=pd.DataFrame(input1.numpy())
            #b=pd.DataFrame(input2.numpy())
            #pd.concat([a,b],axis=1).to_csv('./K{}_result/edges'.format(fold),index=0,header=0)


            print("test_auc:{},test_aupr:{}".format(test_auc,test_aupr))
            predlabel=outputs.data.numpy()>0.5
            predlabel=predlabel.astype(np.int32)
            labels=labels.int().numpy()
            acc,prec,re,f1=prediction(predlabel,labels)
            print("acc:{},prec:{},re:{},f1:{}".format(acc,prec,re,f1))

    return test_auc,test_aupr,fpr,tpr,precision,recall, acc,prec,re,f1

from  collections import Iterable
def flatten(items,ignore_types=(str,bytes)):
    for x in items:
        if isinstance(x,Iterable) and not isinstance(x,ignore_types):
            yield from flatten(x)
        else:
            yield x

def prediction(predlabel,labels):
    predlabel_s=[]
    labels_s=[]
    for x in flatten(predlabel):
        predlabel_s.append(x)
    for x in flatten(labels):
        labels_s.append(x)

    from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
    acc=accuracy_score(labels_s, predlabel_s)
    precision=precision_score(labels_s, predlabel_s)
    recall=recall_score(labels_s, predlabel_s)
    f1=f1_score(labels_s, predlabel_s)
    return acc,precision,recall,f1

def save(v, filename):
    f = open(filename, "wb")
    pickle.dump(v,f)
    f.close()
    return(filename)

def load(filename):
    f = open(filename, "rb")
    r = pickle.load(f)
    f.close()
    return(r)

if __name__ == '__main__':
    GTR_Net = Net(GTR_adj.shape[0], args.hidden_dim, args.dropout)
    GPR_Net = Net(GPR_adj.shape[0], args.hidden_dim, args.dropout)
    
    fuse_GTR_mlp = mlp(GTR_adj.shape[0]-g_n, args.hidden_dim, args.dropout)
    fuse_GPR_mlp = mlp(GPR_adj.shape[0]-g_n, args.hidden_dim, args.dropout)
    
    kl_loss = nn.KLDivLoss()
    
    optimizer = optim.Adam(list(GTR_Net.parameters()) + list(GPR_Net.parameters()) + list(fuse_GPR_mlp.parameters()) + list(fuse_GTR_mlp.parameters()), lr=args.lr)
    
    for i in range(args.kfold):
        GTR_Net.train()
        GPR_Net.train()
        
        fuse_GTR_mlp.train()
        fuse_GPR_mlp.train()
        
        X_GTR = GTR_Net(GTR_adj)    
        X_GPR = GPR_Net(GPR_adj)
        
        fuse_GTR = extract_gene(X_GTR.detach().numpy(),g_n)
        fuse_GPR = extract_gene(X_GPR.detach().numpy(),g_n)
        
        fuse_GTR = fuse_GTR_mlp(fuse_GTR, False)
        fuse_GPR = fuse_GPR_mlp(fuse_GPR, True)
        
        output = kl_loss(F.log_softmax(fuse_GPR), F.softmax(fuse_GTR))
        
        optimizer.zero_grad()
        
        output.backward()
        
        optimizer.step()
        
        print('KL loss',output)
        
    GTR_Net.eval()
    GPR_Net.eval()
    
    fuse_GTR_mlp.eval()
    fuse_GPR_mlp.eval()    
    
    X_GTR = GTR_Net(GTR_adj)            
    X_GPR = GPR_Net(GPR_adj)   
    
    fuse_GTR = extract_gene(X_GTR.detach().numpy(), g_n)
    fuse_GPR = extract_gene(X_GPR.detach().numpy(), g_n)   
    
    fuse_GTR = fuse_GTR_mlp(fuse_GTR, False)
    fuse_GPR = fuse_GPR_mlp(fuse_GPR, True)
    
    fuse = fuseNet(fuse_GPR, fuse_GTR)
    gene_gene = fuse

    gene_dis = G_D
    dis_dis = D_D
    
    data_gene_dis = reconstruct_gene(gene_dis.values)
    data_gene_gene = reconstruct_gene(gene_gene)        
    data_dis_dis = reconstruct(dis_dis.values)
    data_gene_dis = data_gene_dis.loc[data_gene_dis['score']==1]
    
    a = [i for i in range(0, gene_dis.shape[1])]
    b = [j for j in range(gene_dis.shape[0], gene_dis.shape[0]+gene_dis.shape[1])]
    
    c = dict(list(zip(a, b)))
    data_gene_dis['dis'] = data_gene_dis['dis'] + g_n
    data_dis_dis['dis'] = data_dis_dis['dis'] + g_n
    data_dis_dis['gene'] = data_dis_dis['gene'] + g_n
    
    gene = data_gene_dis['gene'].tolist()
    dis = data_gene_dis['dis'].tolist()
    gene_dislist = list(zip(gene, dis))
    #print('data_gene_dis',data_gene_dis)
    
    shuf = random.sample(range(0, len(gene_dislist)), len(gene_dislist))
    NewRandomList = partition(shuf, math.ceil(len(shuf) / args.kfold))
    
    os.system("mkdir Split_data")
    save(gene_gene, "Split_data/gene_gene.txt")
    save(dis_dis, "Split_data/dis_dis.txt")
    save(gene_dis, "Split_data/gene_dis.txt")
    save(g_n, "Split_data/g_n.txt")
    save(NewRandomList, "Split_data/NewRandomList.txt")
    save(gene_dislist, "Split_data/gene_dislist.txt")
