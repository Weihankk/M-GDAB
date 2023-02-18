from Split_dataset import *
np.random.seed(123)
w = 4
os.system("mkdir K{}_result".format(w+1))
recall_s = []
precision_s = []
fprs = []
tprs = []
test_aucs = []
test_auprs=[]
accs=[]
precs=[]
res=[]
f1s=[]
#device = "cuda:3" if torch.cuda.is_available() else "cpu"
#print(f"Using {device} device")

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)


gene_gene = load("Split_data/gene_gene.txt")
dis_dis = load("Split_data/dis_dis.txt")
gene_dis = load("Split_data/gene_dis.txt")
g_n = load("Split_data/g_n.txt")
NewRandomList = load("Split_data/NewRandomList.txt")
gene_dislist = load("Split_data/gene_dislist.txt")

print('*' * 25, 'The', w + 1, 'fold', '*' * 25)
data_gene_gene = reconstruct_gene(gene_gene)
data_dis_dis = reconstruct(dis_dis.values)
a = [i for i in range(0, gene_dis.shape[1])]
b = [j for j in range(gene_dis.shape[0], gene_dis.shape[0]+gene_dis.shape[1])]
c = dict(list(zip(a, b)))
data_dis_dis['dis'] = data_dis_dis['dis'] + g_n
data_dis_dis['gene'] = data_dis_dis['gene'] + g_n
f_train, f_ggi, f_dds,test_gene_di, test_data_gene_gene, test_data_dis_dis = get_kfold_data(w, NewRandomList, gene_dislist,data_gene_gene,data_dis_dis)
print('----')
trainset = Sample_Set(f_train, f_ggi, f_dds)
testset = Sample_Set(test_gene_di, test_data_gene_gene, test_data_dis_dis)
print('loaded set')
node2index = trainset.get_node2index()
test_node2index = testset.get_node2index()

node_count = len(node2index)
node_dim = args.hidden_dim
n_repr = args.hidden_dim
gcn = pre_GNN(node_count, node_dim, n_repr, dropout=args.dropout, Kvalue=args.ChebConvK)
lp_model = Link_Prediction(n_repr, dropout=args.dropout)

lp_model = lp_model.to(device)

print('load embedding')
embedding=load_embedding(node2index,args)
test_embedding=load_embedding(test_node2index,args)
trainset.reassign_samples()
testset.reassign_samples()
trainloader = MultiEpochsDataLoader(trainset, batch_size=len(trainset), shuffle=True, num_workers=1)
testloader = MultiEpochsDataLoader(testset, batch_size=len(testset), shuffle=True, num_workers=1)
class_weight = torch.FloatTensor([1, 1])
optimizer_lp_model = optim.Adam(list(gcn.parameters()) + list(lp_model.parameters()), lr=args.lr)
losses=[]

adj_matrix = trainset.get_adj_matrix()
adj_matrix=sparse_to_torch_tensor(adj_matrix)
print('strat training!')
for epoch in tqdm(range(0, args.epochs)):
    running_loss = 0.0
    lp_model.train()
    gcn.train()

    gc.collect()
    torch.cuda.empty_cache()

    rp_data=Data(x=embedding, edge_index=adj_matrix)
    rp_matrix=gcn(rp_data.x, rp_data.edge_index)
    rp_matrix = rp_matrix.to(device)

    for i, data in enumerate(trainloader, 0):
        inputs, labels = data
        input1, input2 = inputs[0], inputs[1]
        input1 = input1.to(device)
        input2 = input2.to(device)
        rps1 = F.embedding(input1, rp_matrix)
        rps2 = F.embedding(input2, rp_matrix)
        
        rps1 = rps1.to(device)
        rps2 = rps2.to(device)

        outputs = lp_model(rps1, rps2)

        labels = labels.float()
        weight = class_weight[labels.long()]
        criterion = torch.nn.BCELoss(weight=weight, reduction='mean')
        cross_loss = criterion(outputs.cpu(), labels)

        loss = cross_loss
        optimizer_lp_model.zero_grad()
        loss.backward()
        optimizer_lp_model.step()
        running_loss += loss.item()
        losses.append(loss.item())

    if epoch % 50 ==0:
        print('Epoch: {}, Loss: {:.5f} '.format(epoch + 1, loss.item()))
        auc_value,f,t=auroc(outputs.cpu(),labels)
        aupr_value,p,r=auprc(outputs.cpu(),labels)
        print("auc:{},aupr:{}".format(auc_value,aupr_value))

    # output model
    torch.save(lp_model.state_dict(), "./K{}_result/lp_model.pth".format(w+1))
    torch.save(gcn.state_dict(), "./K{}_result/gcn.pth".format(w+1))

test_auc,test_aupr,fpr,tpr,precision,recall, acc,prec,re,f1=test(lp_model,gcn,test_embedding,testloader,testset,w+1)
test_aucs.append(test_auc)
test_auprs.append(test_aupr)
recall_s.append(recall)
precision_s.append(precision)
accs.append(acc)
precs.append(prec)
res.append(re)
f1s.append(f1)
fprs.append(fpr)
tprs.append(tpr)

pd.DataFrame(test_aucs).to_csv("./K{}_result/test_auc_scores.csv".format(w+1),index=False,header=False)
pd.DataFrame(test_auprs).to_csv("./K{}_result/test_auprs.csv".format(w+1),index=False,header=False)
pd.DataFrame(accs).to_csv("./K{}_result/accs.csv".format(w+1), index=False,header=False)
pd.DataFrame(precs).to_csv("./K{}_result/tprecs.csv".format(w+1), index=False,header=False)
pd.DataFrame(res).to_csv("./K{}_result/tres.csv".format(w+1), index=False,header=False)
pd.DataFrame(f1s).to_csv("./K{}_result/tf1s.csv".format(w+1), index=False,header=False)
pd.DataFrame(precision_s).to_csv("./K{}_result/tprecision_s.csv".format(w+1),index=False,header=False)
pd.DataFrame(recall_s).to_csv("./K{}_result/trecall_s.csv".format(w+1),index=False,header=False)
pd.DataFrame(fprs).to_csv("./K{}_result/tfprs.csv".format(w+1),index=False,header=False)
pd.DataFrame(tprs).to_csv("./K{}_result/ttpr.csv".format(w+1),index=False,header=False)

