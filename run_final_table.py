"""
FINAL UNIFIED TABLE: All models under identical conditions
==========================================================
Single script, same val split, same neighbors, same everything.

Models:
  Traditional:     RF, SVM, XGBoost, MLP
  Traditional+D:   RF+Delta, SVM+Delta, XGB+Delta, MLP+Delta
  GNN:             GIN Baseline
  GNN Residual:    GIN+ResidOnly, GIN+Delta
  GNN Ablation:    GIN+Full, GIN-NoResid, GIN-NoHA, GIN-NoYA
  Advanced:        GIN+RAG Attention
  Neighbor:        Tanimoto Top-3, Scaffold, Cliff-potential, Random
  Other:           NB Copy, SVM+NeuralCorrection

All on MoleculeACE 30 datasets, same split, same seed.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GINConv, global_add_pool
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit import RDLogger
import warnings, os, importlib.util, time
from collections import OrderedDict

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore")
RDLogger.logger().setLevel(RDLogger.ERROR)
np.random.seed(42)
torch.manual_seed(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FP_DIM = 1024; EMB_DIM = 128
print(f"Device: {DEVICE}")

# ============================================================
# Mol features
# ============================================================
ATOM_FEATURES = {
    'atom_type': list(range(1,54)), 'degree': [0,1,2,3,4,5],
    'formal_charge': [-2,-1,0,1,2],
    'hybridization': [Chem.rdchem.HybridizationType.SP,
                      Chem.rdchem.HybridizationType.SP2,
                      Chem.rdchem.HybridizationType.SP3],
    'aromatic': [False, True], 'num_hs': [0,1,2,3,4],
}
def one_hot(v, a):
    e=[0]*(len(a)+1)
    if v in a: e[a.index(v)]=1
    else: e[-1]=1
    return e
def atom_feats(atom):
    f=[]
    for k in ['atom_type','degree','formal_charge','hybridization','aromatic','num_hs']:
        f+=one_hot(getattr(atom,{'atom_type':'GetAtomicNum','degree':'GetDegree',
            'formal_charge':'GetFormalCharge','hybridization':'GetHybridization',
            'aromatic':'GetIsAromatic','num_hs':'GetTotalNumHs'}[k])(),ATOM_FEATURES[k])
    return f
ATOM_FEAT_DIM=sum(len(v)+1 for v in ATOM_FEATURES.values())

def mol_to_graph(smi):
    mol=Chem.MolFromSmiles(smi)
    if mol is None: return None
    x=torch.FloatTensor([atom_feats(a) for a in mol.GetAtoms()])
    edges=[]
    for b in mol.GetBonds():
        i,j=b.GetBeginAtomIdx(),b.GetEndAtomIdx(); edges+=[[i,j],[j,i]]
    ei=torch.LongTensor(edges).t().contiguous() if edges else torch.zeros((2,0),dtype=torch.long)
    return Data(x=x,edge_index=ei)

def smiles_to_fp(smi):
    mol=Chem.MolFromSmiles(smi)
    if mol is None: return None
    fp=AllChem.GetMorganFingerprintAsBitVect(mol,2,nBits=FP_DIM)
    arr=np.zeros(FP_DIM,dtype=np.float32); DataStructs.ConvertToNumpyArray(fp,arr)
    return arr

def get_scaffold(smi):
    mol=Chem.MolFromSmiles(smi)
    if mol is None: return None
    try: return MurckoScaffold.MurckoScaffoldSmiles(mol=mol,includeChirality=False)
    except: return None

def tanimoto_sim(a,b):
    a_=a.astype(bool).astype(float); b_=b.astype(bool).astype(float)
    inter=a_@b_.T
    return inter/np.maximum(a_.sum(1,keepdims=True)+b_.sum(1,keepdims=True).T-inter,1e-8)

# ============================================================
# Data
# ============================================================
def get_data_dir():
    spec=importlib.util.find_spec("MoleculeACE")
    return os.path.join(os.path.dirname(spec.origin),"Data","benchmark_data")

def prepare_data(name):
    df=pd.read_csv(os.path.join(get_data_dir(),name))
    graphs,fps,ys,cliffs,splits,scaffolds=[],[],[],[],[],[]
    for _,row in df.iterrows():
        g=mol_to_graph(row["smiles"]); fp=smiles_to_fp(row["smiles"])
        if g is not None and fp is not None:
            graphs.append(g); fps.append(fp); ys.append(row["y"])
            cliffs.append(int(row["cliff_mol"])); splits.append(row["split"])
            scaffolds.append(get_scaffold(row["smiles"]))
    fps=np.array(fps); ys=np.array(ys,dtype=np.float32)
    cliffs=np.array(cliffs,dtype=bool); splits=np.array(splits)
    scaffolds=np.array(scaffolds,dtype=object)
    tr,te=splits=="train",splits=="test"
    return ([graphs[i] for i in range(len(graphs)) if tr[i]],
            fps[tr],ys[tr],cliffs[tr],scaffolds[tr],
            [graphs[i] for i in range(len(graphs)) if te[i]],
            fps[te],ys[te],cliffs[te],scaffolds[te])

# ============================================================
# Neighbor strategies (all return nb_fps, nb_ys)
# ============================================================
def nb_tanimoto_top3(fps_q,fps_db,y_db,scaff_q=None,scaff_db=None,exclude_self=False):
    sim=tanimoto_sim(fps_q,fps_db)
    if exclude_self: np.fill_diagonal(sim,-1)
    n=len(fps_q); nb_fps=np.zeros((n,FP_DIM),dtype=np.float32); nb_ys=np.zeros(n,dtype=np.float32)
    for i in range(n):
        topk=np.argsort(-sim[i])[:3]; w=sim[i][topk]; w=w/(w.sum()+1e-8)
        nb_fps[i]=(fps_db[topk]*w[:,None]).sum(0); nb_ys[i]=(y_db[topk]*w).sum()
    return nb_fps,nb_ys

def nb_scaffold(fps_q,fps_db,y_db,scaff_q,scaff_db,exclude_self=False):
    sim=tanimoto_sim(fps_q,fps_db)
    if exclude_self: np.fill_diagonal(sim,-1)
    n=len(fps_q); nb_fps=np.zeros((n,FP_DIM),dtype=np.float32); nb_ys=np.zeros(n,dtype=np.float32)
    for i in range(n):
        same=np.array([j for j in range(len(fps_db)) if scaff_q[i] and scaff_db[j]==scaff_q[i]])
        if len(same)>=3:
            ss=sim[i][same]; topk=same[np.argsort(-ss)[:3]]
        else:
            topk=np.argsort(-sim[i])[:3]
        w=sim[i][topk]; w=w/(w.sum()+1e-8)
        nb_fps[i]=(fps_db[topk]*w[:,None]).sum(0); nb_ys[i]=(y_db[topk]*w).sum()
    return nb_fps,nb_ys

def nb_cliff_potential(fps_q,fps_db,y_db,scaff_q=None,scaff_db=None,exclude_self=False):
    sim=tanimoto_sim(fps_q,fps_db)
    if exclude_self: np.fill_diagonal(sim,-1)
    n=len(fps_q); nb_fps=np.zeros((n,FP_DIM),dtype=np.float32); nb_ys=np.zeros(n,dtype=np.float32)
    for i in range(n):
        similar=np.where(sim[i]>=0.4)[0]
        if len(similar)>=3:
            mean_a=y_db[similar].mean(); devs=np.abs(y_db[similar]-mean_a)
            topk=similar[np.argsort(-devs)[:3]]
        else:
            topk=np.argsort(-sim[i])[:3]
        w=sim[i][topk]; w=w/(w.sum()+1e-8)
        nb_fps[i]=(fps_db[topk]*w[:,None]).sum(0); nb_ys[i]=(y_db[topk]*w).sum()
    return nb_fps,nb_ys

def nb_random(fps_q,fps_db,y_db,scaff_q=None,scaff_db=None,exclude_self=False):
    sim=tanimoto_sim(fps_q,fps_db)
    n=len(fps_q); nb_fps=np.zeros((n,FP_DIM),dtype=np.float32); nb_ys=np.zeros(n,dtype=np.float32)
    for i in range(n):
        chosen=np.random.choice(len(fps_db),3,replace=False)
        w=sim[i][chosen]; w=w/(w.sum()+1e-8)
        nb_fps[i]=(fps_db[chosen]*w[:,None]).sum(0); nb_ys[i]=(y_db[chosen]*w).sum()
    return nb_fps,nb_ys

def get_emb_nb(fps_q,fps_db,h_db,y_db,k=3,exclude_self=False):
    sim=tanimoto_sim(fps_q,fps_db)
    if exclude_self: np.fill_diagonal(sim,-1)
    n=len(fps_q); h_nb=torch.zeros(n,h_db.shape[1]); y_nb=np.zeros(n,dtype=np.float32)
    for i in range(n):
        topk=np.argsort(-sim[i])[:k]; w=sim[i][topk]; w=w/(w.sum()+1e-8)
        wt=torch.FloatTensor(w)
        h_nb[i]=(h_db[topk]*wt.unsqueeze(-1)).sum(0); y_nb[i]=(y_db[topk]*w).sum()
    return h_nb,y_nb

# ============================================================
# GNN models
# ============================================================
class GINEncoder(nn.Module):
    def __init__(self,in_dim,h=EMB_DIM,n_layers=3,drop=0.2):
        super().__init__()
        self.convs,self.bns=nn.ModuleList(),nn.ModuleList()
        for i in range(n_layers):
            d=in_dim if i==0 else h
            self.convs.append(GINConv(nn.Sequential(nn.Linear(d,h),nn.ReLU(),nn.Linear(h,h))))
            self.bns.append(nn.BatchNorm1d(h))
        self.drop=drop
    def forward(self,data):
        x,ei,batch=data.x,data.edge_index,data.batch
        for c,b in zip(self.convs,self.bns): x=F.dropout(F.relu(b(c(x,ei))),self.drop,self.training)
        return global_add_pool(x,batch)

class GINBase(nn.Module):
    def __init__(self,in_dim,h=EMB_DIM):
        super().__init__()
        self.encoder=GINEncoder(in_dim,h)
        self.head=nn.Sequential(nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h//2,1))
    def forward(self,data): return self.head(self.encoder(data)).squeeze(-1)
    def encode(self,data): return self.encoder(data)

# Correction nets (all take h_b, h_a, y_a, m_b in some combination)
class CorrectionNet(nn.Module):
    def __init__(self,use_ha=True,use_res=True,use_ya=True,emb=EMB_DIM):
        super().__init__()
        self.use_ha=use_ha; self.use_res=use_res; self.use_ya=use_ya
        d=emb
        if use_ha: d+=emb
        if use_res: d+=emb
        if use_ya: d+=1
        d+=1  # m_b
        self.net=nn.Sequential(nn.Linear(d,emb),nn.ReLU(),nn.Dropout(0.3),
                                nn.Linear(emb,emb//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(emb//2,1))
    def forward(self,h_b,h_a,y_a,m_b):
        parts=[h_b]
        if self.use_ha: parts.append(h_a)
        if self.use_res: parts.append(h_b-h_a)
        if self.use_ya: parts.append(y_a.unsqueeze(-1))
        parts.append(m_b.unsqueeze(-1))
        return self.net(torch.cat(parts,-1)).squeeze(-1)

class DeltaNet(nn.Module):
    def __init__(self,emb=EMB_DIM):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(emb*3,emb),nn.ReLU(),nn.Dropout(0.3),
                                nn.Linear(emb,emb//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(emb//2,1))
    def forward(self,h_b,h_a,y_a):
        return y_a+self.net(torch.cat([h_b,h_a,h_b-h_a],-1)).squeeze(-1)

class RAGNet(nn.Module):
    def __init__(self,emb=EMB_DIM,k=5):
        super().__init__()
        self.q=nn.Linear(emb,emb); self.k_proj=nn.Linear(emb,emb)
        self.v=nn.Linear(emb+1,emb); self.scale=emb**0.5
        self.net=nn.Sequential(nn.Linear(emb*3+1,emb),nn.ReLU(),nn.Dropout(0.3),
                                nn.Linear(emb,emb//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(emb//2,1))
    def forward(self,h_b,h_nbs,y_nbs,m_b):
        Q=self.q(h_b).unsqueeze(1); K=self.k_proj(h_nbs)
        V=self.v(torch.cat([h_nbs,y_nbs.unsqueeze(-1)],-1))
        attn=F.softmax((Q@K.transpose(-2,-1))/self.scale,dim=-1)
        ctx=(attn@V).squeeze(1)
        aw=attn.squeeze(1); h_a=(aw.unsqueeze(-1)*h_nbs).sum(1)
        return self.net(torch.cat([h_b,ctx,h_b-h_a,m_b.unsqueeze(-1)],-1)).squeeze(-1)

class FPDeltaNet(nn.Module):
    def __init__(self,dim=FP_DIM,h=512):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim*3,h),nn.ReLU(),nn.Dropout(0.3),
                                nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.3),nn.Linear(h//2,1))
    def forward(self,fp_b,fp_a,y_a):
        return y_a+self.net(torch.cat([fp_b,fp_a,fp_b-fp_a],-1)).squeeze(-1)

class MLPBase(nn.Module):
    def __init__(self,dim=FP_DIM,h=256):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(dim,h),nn.ReLU(),nn.Dropout(0.2),
                                nn.Linear(h,h//2),nn.ReLU(),nn.Dropout(0.2),nn.Linear(h//2,1))
    def forward(self,x): return self.net(x).squeeze(-1)

# ============================================================
# Unified training
# ============================================================
def train_gin(g_tr,y_tr,g_va,y_va,ep=150,lr=1e-3,pat=25,bs=64):
    m=GINBase(ATOM_FEAT_DIM).to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=0.5,min_lr=1e-5)
    vb=Batch.from_data_list(g_va).to(DEVICE); vy=torch.FloatTensor(y_va).to(DEVICE)
    best,bst,w=float("inf"),None,0; n=len(g_tr)
    for _ in range(ep):
        m.train(); p=np.random.permutation(n)
        for s in range(0,n,bs):
            idx=p[s:s+bs]; b=Batch.from_data_list([g_tr[i] for i in idx]).to(DEVICE)
            yt=torch.FloatTensor(y_tr[idx]).to(DEVICE)
            l=nn.MSELoss()(m(b),yt); opt.zero_grad(); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl=nn.MSELoss()(m(vb),vy).item()
        sch.step(vl)
        if vl<best: best,bst,w=vl,{k:v.clone() for k,v in m.state_dict().items()},0
        else:
            w+=1
            if w>=pat: break
    m.load_state_dict(bst); return m

def get_embs(m,gs,bs=256):
    m.eval(); es=[]
    for s in range(0,len(gs),bs):
        b=Batch.from_data_list(gs[s:s+bs]).to(DEVICE)
        with torch.no_grad(): es.append(m.encode(b).cpu())
    return torch.cat(es,0)

def train_corr(net,h_tr,y_tr,h_nb,nba,m_tr,h_va,y_va,h_nbv,nbav,m_va,
               ep=200,lr=5e-4,pat=25,bs=64):
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=0.5,min_lr=1e-5)
    ds=TensorDataset(h_tr,torch.FloatTensor(y_tr),h_nb,torch.FloatTensor(nba),torch.FloatTensor(m_tr))
    loader=DataLoader(ds,batch_size=bs,shuffle=True)
    Hv,Yv=h_va.to(DEVICE),torch.FloatTensor(y_va).to(DEVICE)
    HNv,NAv,Mv=h_nbv.to(DEVICE),torch.FloatTensor(nbav).to(DEVICE),torch.FloatTensor(m_va).to(DEVICE)
    best,bst,w=float("inf"),None,0
    for _ in range(ep):
        net.train()
        for hb,yb,hnb,nab,mb in loader:
            hb,yb,hnb,nab,mb=[t.to(DEVICE) for t in [hb,yb,hnb,nab,mb]]
            l=nn.MSELoss()(mb+net(hb,hnb,nab,mb),yb); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl=nn.MSELoss()(Mv+net(Hv,HNv,NAv,Mv),Yv).item()
        sch.step(vl)
        if vl<best: best,bst,w=vl,{k:v.clone() for k,v in net.state_dict().items()},0
        else:
            w+=1
            if w>=pat: break
    net.load_state_dict(bst); return net

def train_delta(net,h_tr,y_tr,h_nb,y_nb,h_va,y_va,h_nbv,y_nbv,
                ep=200,lr=5e-4,pat=25,bs=64):
    opt=torch.optim.Adam(net.parameters(),lr=lr,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=0.5,min_lr=1e-5)
    ds=TensorDataset(h_tr,torch.FloatTensor(y_tr),h_nb,torch.FloatTensor(y_nb))
    loader=DataLoader(ds,batch_size=bs,shuffle=True)
    Hv,Yv=h_va.to(DEVICE),torch.FloatTensor(y_va).to(DEVICE)
    HNv,YNv=h_nbv.to(DEVICE),torch.FloatTensor(y_nbv).to(DEVICE)
    best,bst,w=float("inf"),None,0
    for _ in range(ep):
        net.train()
        for hb,yb,hnb,ynb in loader:
            hb,yb,hnb,ynb=[t.to(DEVICE) for t in [hb,yb,hnb,ynb]]
            l=nn.MSELoss()(net(hb,hnb,ynb),yb); opt.zero_grad(); l.backward(); opt.step()
        net.eval()
        with torch.no_grad(): vl=nn.MSELoss()(net(Hv,HNv,YNv),Yv).item()
        sch.step(vl)
        if vl<best: best,bst,w=vl,{k:v.clone() for k,v in net.state_dict().items()},0
        else:
            w+=1
            if w>=pat: break
    net.load_state_dict(bst); return net

def train_mlp(fps_tr,y_tr,fps_va,y_va,ep=300,lr=1e-3,pat=30,bs=64):
    m=MLPBase().to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-4)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=0.5,min_lr=1e-5)
    Xt,Yt=torch.FloatTensor(fps_tr).to(DEVICE),torch.FloatTensor(y_tr).to(DEVICE)
    Xv,Yv=torch.FloatTensor(fps_va).to(DEVICE),torch.FloatTensor(y_va).to(DEVICE)
    ds=TensorDataset(Xt,Yt); loader=DataLoader(ds,batch_size=bs,shuffle=True)
    best,bst,w=float("inf"),None,0
    for _ in range(ep):
        m.train()
        for xb,yb in loader:
            l=nn.MSELoss()(m(xb),yb); opt.zero_grad(); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl=nn.MSELoss()(m(Xv),Yv).item()
        sch.step(vl)
        if vl<best: best,bst,w=vl,{k:v.clone() for k,v in m.state_dict().items()},0
        else:
            w+=1
            if w>=pat: break
    m.load_state_dict(bst); return m

def train_fp_delta(fps_tr,y_tr,nb_fps_tr,nb_y_tr,fps_va,y_va,nb_fps_va,nb_y_va,
                    ep=300,lr=5e-4,pat=30,bs=64):
    m=FPDeltaNet().to(DEVICE)
    opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=1e-3)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,patience=10,factor=0.5,min_lr=1e-5)
    ds=TensorDataset(torch.FloatTensor(fps_tr),torch.FloatTensor(y_tr),
                      torch.FloatTensor(nb_fps_tr),torch.FloatTensor(nb_y_tr))
    loader=DataLoader(ds,batch_size=bs,shuffle=True)
    Fv,Yv=torch.FloatTensor(fps_va).to(DEVICE),torch.FloatTensor(y_va).to(DEVICE)
    Nv,NYv=torch.FloatTensor(nb_fps_va).to(DEVICE),torch.FloatTensor(nb_y_va).to(DEVICE)
    best,bst,w=float("inf"),None,0
    for _ in range(ep):
        m.train()
        for fb,yb,nb,nyb in loader:
            fb,yb,nb,nyb=[t.to(DEVICE) for t in [fb,yb,nb,nyb]]
            l=nn.MSELoss()(m(fb,nb,nyb),yb); opt.zero_grad(); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl=nn.MSELoss()(m(Fv,Nv,NYv),Yv).item()
        sch.step(vl)
        if vl<best: best,bst,w=vl,{k:v.clone() for k,v in m.state_dict().items()},0
        else:
            w+=1
            if w>=pat: break
    m.load_state_dict(bst); return m

# ============================================================
# Run one dataset
# ============================================================
def run_dataset(name):
    (g_tr,fps_tr,y_tr,cliff_tr,scaff_tr,
     g_te,fps_te,y_te,cliff_te,scaff_te)=prepare_data(name)
    if len(g_tr)<20 or len(g_te)<5: return None

    # UNIFIED val split
    n=len(g_tr); rng=np.random.RandomState(42); perm=rng.permutation(n)
    nv=max(int(n*0.15),10); vi,ti=perm[:nv],perm[nv:]
    g_va=[g_tr[i] for i in vi]; fps_va,y_va,scaff_va=fps_tr[vi],y_tr[vi],scaff_tr[vi]
    g_tr2=[g_tr[i] for i in ti]; fps_tr2,y_tr2,scaff_tr2=fps_tr[ti],y_tr[ti],scaff_tr[ti]

    t=lambda x: torch.FloatTensor(x)
    R=OrderedDict()

    # === UNIFIED NEIGHBORS (Tanimoto Top-3) ===
    nb_fps_tr_full,nb_y_tr_full=nb_tanimoto_top3(fps_tr,fps_tr,y_tr,exclude_self=True)
    nb_fps_te,nb_y_te=nb_tanimoto_top3(fps_te,fps_tr,y_tr)
    nb_fps_tr2,nb_y_tr2=nb_tanimoto_top3(fps_tr2,fps_tr2,y_tr2,exclude_self=True)
    nb_fps_va,nb_y_va=nb_tanimoto_top3(fps_va,fps_tr2,y_tr2)

    # ============ TRADITIONAL MODELS ============
    # RF
    rf=RandomForestRegressor(n_estimators=500,min_samples_leaf=2,n_jobs=-1,random_state=42)
    rf.fit(fps_tr,y_tr); R["RF"]=rf.predict(fps_te)

    rf_d=RandomForestRegressor(n_estimators=500,min_samples_leaf=2,n_jobs=-1,random_state=42)
    Xd=np.concatenate([fps_tr,nb_fps_tr_full,fps_tr-nb_fps_tr_full],1)
    rf_d.fit(Xd,y_tr-nb_y_tr_full)
    Xdt=np.concatenate([fps_te,nb_fps_te,fps_te-nb_fps_te],1)
    R["RF+Delta"]=nb_y_te+rf_d.predict(Xdt)

    # SVM
    svm=SVR(kernel='rbf',C=10,gamma='scale')
    svm.fit(fps_tr,y_tr); R["SVM"]=svm.predict(fps_te)

    svm_d=SVR(kernel='rbf',C=10,gamma='scale')
    svm_d.fit(Xd,y_tr-nb_y_tr_full)
    R["SVM+Delta"]=nb_y_te+svm_d.predict(Xdt)

    # XGB
    if HAS_XGB:
        xgb=XGBRegressor(n_estimators=500,max_depth=6,learning_rate=0.1,subsample=0.8,random_state=42,verbosity=0)
        xgb.fit(fps_tr,y_tr); R["XGB"]=xgb.predict(fps_te)
        xgb_d=XGBRegressor(n_estimators=500,max_depth=6,learning_rate=0.1,subsample=0.8,random_state=42,verbosity=0)
        xgb_d.fit(Xd,y_tr-nb_y_tr_full)
        R["XGB+Delta"]=nb_y_te+xgb_d.predict(Xdt)

    # MLP
    mlp=train_mlp(fps_tr2,y_tr2,fps_va,y_va)
    mlp.eval()
    with torch.no_grad(): R["MLP"]=mlp(t(fps_te).to(DEVICE)).cpu().numpy()

    # MLP+Delta (ECFP)
    fpd=train_fp_delta(fps_tr2,y_tr2,nb_fps_tr2,nb_y_tr2,fps_va,y_va,nb_fps_va,nb_y_va)
    fpd.eval()
    with torch.no_grad():
        R["MLP+Delta"]=fpd(t(fps_te).to(DEVICE),t(nb_fps_te).to(DEVICE),t(nb_y_te).to(DEVICE)).cpu().numpy()

    # NB Copy
    R["NB_Copy"]=nb_y_te.copy()

    # ============ GNN MODELS (shared encoder) ============
    gin=train_gin(g_tr2,y_tr2,g_va,y_va)
    gin.eval()
    with torch.no_grad():
        R["GIN"]=gin(Batch.from_data_list(g_te).to(DEVICE)).cpu().numpy()
        m_tr=gin(Batch.from_data_list(g_tr2).to(DEVICE)).cpu().numpy()
        m_va=gin(Batch.from_data_list(g_va).to(DEVICE)).cpu().numpy()
        m_te=gin(Batch.from_data_list(g_te).to(DEVICE)).cpu().numpy()

    h_tr=get_embs(gin,g_tr2); h_va=get_embs(gin,g_va)
    h_te=get_embs(gin,g_te); h_all=get_embs(gin,g_tr)
    h_nb_tr,y_nb_tr=get_emb_nb(fps_tr2,fps_tr2,h_tr,y_tr2,exclude_self=True)
    h_nb_va,y_nb_va=get_emb_nb(fps_va,fps_tr2,h_tr,y_tr2)
    h_nb_te,y_nb_te=get_emb_nb(fps_te,fps_tr,h_all,y_tr)

    # GIN+ResidOnly: m(B) + g(h_B, h_B-h_A, m_B)
    cn_ro=CorrectionNet(use_ha=False,use_res=True,use_ya=False).to(DEVICE)
    cn_ro=train_corr(cn_ro,h_tr,y_tr2,h_nb_tr,y_nb_tr,m_tr,h_va,y_va,h_nb_va,y_nb_va,m_va)
    cn_ro.eval()
    with torch.no_grad():
        R["GIN+ResidOnly"]=( t(m_te).to(DEVICE)+cn_ro(h_te.to(DEVICE),h_nb_te.to(DEVICE),
                              t(y_nb_te).to(DEVICE),t(m_te).to(DEVICE)) ).cpu().numpy()

    # GIN+Full: m(B) + g(h_B, h_A, h_B-h_A, Y_A, m_B)
    cn_full=CorrectionNet(use_ha=True,use_res=True,use_ya=True).to(DEVICE)
    cn_full=train_corr(cn_full,h_tr,y_tr2,h_nb_tr,y_nb_tr,m_tr,h_va,y_va,h_nb_va,y_nb_va,m_va)
    cn_full.eval()
    with torch.no_grad():
        R["GIN+Full"]=( t(m_te).to(DEVICE)+cn_full(h_te.to(DEVICE),h_nb_te.to(DEVICE),
                          t(y_nb_te).to(DEVICE),t(m_te).to(DEVICE)) ).cpu().numpy()

    # GIN+Delta: Y_A + delta(h_B, h_A, h_B-h_A)
    dn=DeltaNet().to(DEVICE)
    dn=train_delta(dn,h_tr,y_tr2,h_nb_tr,y_nb_tr,h_va,y_va,h_nb_va,y_nb_va)
    dn.eval()
    with torch.no_grad():
        R["GIN+Delta"]=dn(h_te.to(DEVICE),h_nb_te.to(DEVICE),t(y_nb_te).to(DEVICE)).cpu().numpy()

    # ============ METRICS ============
    out={"dataset":name,"n_test":len(y_te),"n_cliff":int(cliff_te.sum()),"metrics":OrderedDict()}
    for rn,pred in R.items():
        ov=np.sqrt(mean_squared_error(y_te,pred))
        cl=np.sqrt(mean_squared_error(y_te[cliff_te],pred[cliff_te])) if cliff_te.sum()>1 else np.nan
        out["metrics"][rn]={"overall_rmse":ov,"cliff_rmse":cl}
    return out

# ============================================================
# Main
# ============================================================
def main():
    print("="*110)
    print("  FINAL UNIFIED TABLE: All Models, Same Conditions")
    print("="*110)

    data_dir=get_data_dir()
    datasets=sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    all_res=[]

    for i,d in enumerate(datasets):
        print(f"\n[{i+1}/{len(datasets)}] {d}...",end=" ",flush=True)
        t0=time.time()
        try:
            r=run_dataset(d)
            if r is None: print("SKIP"); continue
            all_res.append(r); m=r["metrics"]
            print(f"({time.time()-t0:.1f}s) "
                  f"SVM={m['SVM']['overall_rmse']:.3f} "
                  f"GIN={m['GIN']['overall_rmse']:.3f} "
                  f"GIN+D={m['GIN+Delta']['overall_rmse']:.3f}")
        except Exception as e:
            print(f"ERROR: {e}"); import traceback; traceback.print_exc()

    keys=list(all_res[0]["metrics"].keys())

    # ============ TABLE 1: Overall RMSE ============
    print("\n\n"+"="*110)
    print("  TABLE 1: OVERALL RMSE (mean across 30 datasets)")
    print("="*110)
    print(f"\n{'Rank':>4s} | {'Model':30s} | {'Overall RMSE':>12s} | {'Cliff RMSE':>10s} | {'Win_OV':>6s} | {'Win_CL':>6s}")
    print("-"*80)

    rows=[]
    for k in keys:
        ov=np.mean([r["metrics"][k]["overall_rmse"] for r in all_res])
        cl=np.mean([r["metrics"][k]["cliff_rmse"] for r in all_res if not np.isnan(r["metrics"][k]["cliff_rmse"])])
        ov_w=sum(1 for r in all_res if k==min(keys,key=lambda kk:r["metrics"][kk]["overall_rmse"]))
        cl_w=sum(1 for r in all_res if not np.isnan(r["metrics"][k]["cliff_rmse"]) and
                 k==min([kk for kk in keys if not np.isnan(r["metrics"][kk]["cliff_rmse"])],
                        key=lambda kk:r["metrics"][kk]["cliff_rmse"]))
        rows.append((k,ov,cl,ov_w,cl_w))
    rows.sort(key=lambda x:x[1])
    for rank,(k,ov,cl,ov_w,cl_w) in enumerate(rows,1):
        print(f"{rank:4d} | {k:30s} | {ov:12.4f} | {cl:10.4f} | {ov_w:>6d} | {cl_w:>6d}")

    # ============ TABLE 2: Base vs +Delta ============
    print("\n\n"+"="*110)
    print("  TABLE 2: BASE vs +DELTA (same backbone)")
    print("="*110)
    pairs=[("RF","RF+Delta"),("SVM","SVM+Delta")]
    if HAS_XGB: pairs.append(("XGB","XGB+Delta"))
    pairs+=[("MLP","MLP+Delta"),("GIN","GIN+Delta")]

    print(f"\n{'Model':8s} | {'Base OV':>8s} | {'+Delta OV':>9s} | {'OV%':>6s} | {'Win':>5s} | {'Base CL':>8s} | {'+Delta CL':>9s} | {'CL%':>6s}")
    print("-"*80)
    for base,delta in pairs:
        bo=[r["metrics"][base]["overall_rmse"] for r in all_res]
        do=[r["metrics"][delta]["overall_rmse"] for r in all_res]
        bc=[r["metrics"][base]["cliff_rmse"] for r in all_res if not np.isnan(r["metrics"][base]["cliff_rmse"])]
        dc=[r["metrics"][delta]["cliff_rmse"] for r in all_res if not np.isnan(r["metrics"][delta]["cliff_rmse"])]
        ov_imp=np.mean([(b-d)/b*100 for b,d in zip(bo,do)])
        cl_imp=np.mean([(b-d)/b*100 for b,d in zip(bc,dc)])
        wins=sum(1 for b,d in zip(bo,do) if d<b)
        print(f"{base:8s} | {np.mean(bo):8.4f} | {np.mean(do):9.4f} | {ov_imp:+5.1f}% | {wins:>2d}/{len(all_res):<2d} | "
              f"{np.mean(bc):8.4f} | {np.mean(dc):9.4f} | {cl_imp:+5.1f}%")

    # ============ TABLE 3: Per-dataset (key models only) ============
    show=["SVM","RF","GIN","GIN+ResidOnly","GIN+Delta","MLP+Delta","NB_Copy"]
    short={"SVM":"SVM","RF":"RF","GIN":"GIN","GIN+ResidOnly":"G+RO",
           "GIN+Delta":"G+D","MLP+Delta":"M+D","NB_Copy":"NB"}

    print("\n\n"+"="*110)
    print("  TABLE 3: PER-DATASET (key models, Overall RMSE)")
    print("="*110)
    print(f"{'':28s}",end="")
    for k in show: print(f" | {short[k]:>5s}",end="")
    print(); print("-"*85)

    for r in all_res:
        m=r["metrics"]; vals=[m[k]["overall_rmse"] for k in show]; bv=min(vals)
        line=f"{r['dataset']:28s}"
        for v in vals:
            mk="*" if abs(v-bv)<1e-6 else " "
            line+=f" | {v:4.3f}{mk}"
        print(line)

    means=[np.mean([r["metrics"][k]["overall_rmse"] for r in all_res]) for k in show]
    bm=min(means); line=f"{'MEAN':28s}"
    for v in means:
        mk="*" if abs(v-bm)<1e-6 else " "
        line+=f" | {v:4.3f}{mk}"
    print("-"*85); print(line)

    # Cliff
    print(f"\n[Cliff RMSE]")
    print(f"{'':28s}",end="")
    for k in show: print(f" | {short[k]:>5s}",end="")
    print(); print("-"*85)

    for r in all_res:
        m=r["metrics"]; vals=[m[k]["cliff_rmse"] for k in show]
        valid=[v for v in vals if not np.isnan(v)]
        if not valid: continue
        bv=min(valid); line=f"{r['dataset']:28s}"
        for v in vals:
            if np.isnan(v): line+=f" |  nan "
            else:
                mk="*" if abs(v-bv)<1e-6 else " "
                line+=f" | {v:4.3f}{mk}"
        print(line)

    cmeans=[np.mean([r["metrics"][k]["cliff_rmse"] for r in all_res
                     if not np.isnan(r["metrics"][k]["cliff_rmse"])]) for k in show]
    bcm=min(cmeans); line=f"{'MEAN':28s}"
    for v in cmeans:
        mk="*" if abs(v-bcm)<1e-6 else " "
        line+=f" | {v:4.3f}{mk}"
    print("-"*85); print(line)


if __name__=="__main__":
    main()
