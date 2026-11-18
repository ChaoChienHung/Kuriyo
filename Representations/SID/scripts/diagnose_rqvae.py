"""Diagnose RQ-VAE training stability and codebook usage on saved item embeddings.

Reads an embeddings.npz (as written by generate_sids), reuses the
production RQVAE and k-means initialization, and prints one JSON line per
report epoch: component losses, per-level code usage, collisions, latent norm.
Nothing is written to disk. Run from the repository root, for example:

  python -m Representations.SID.scripts.diagnose_rqvae --emb <run>/embeddings.npz       --opt adagrad --lr 0.4 --accum 0 --epochs 60 --report 20
"""
import argparse, json, time
from collections import Counter
import numpy as np, torch
from torch.utils.data import DataLoader, TensorDataset
from Representations.SID.rqvae import RQVAE
from Representations.SID.scripts.generate_sids import initialize_codebooks

p=argparse.ArgumentParser()
p.add_argument("--emb",required=True); p.add_argument("--opt",choices=("adagrad","adamw"),required=True); p.add_argument("--lr",type=float,required=True)
p.add_argument("--accum",type=float,default=0.0); p.add_argument("--epochs",type=int,required=True)
p.add_argument("--report",type=int,default=50); p.add_argument("--standardize",action="store_true"); p.add_argument("--reset-dead",action="store_true"); p.add_argument("--seed",type=int,default=42)
a=p.parse_args()
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.set_num_threads(4)
emb=np.load(a.emb)["embeddings"]
if a.standardize: emb=((emb-emb.mean(0))/(emb.std(0)+1e-8)).astype(np.float32)
model=RQVAE(emb.shape[1])
idx=np.random.default_rng(a.seed).permutation(len(emb))[:1024]
initialize_codebooks(model, emb[idx], a.seed)
if a.opt=="adagrad": opt=torch.optim.Adagrad(model.parameters(), lr=a.lr, initial_accumulator_value=a.accum)
elif a.opt=="adamw": opt=torch.optim.AdamW(model.parameters(), lr=a.lr)
X=torch.from_numpy(emb)
loader=DataLoader(TensorDataset(X),batch_size=1024,shuffle=True,generator=torch.Generator().manual_seed(a.seed))
def stats():
    model.eval()
    with torch.no_grad():
        loss, codes, parts = model(X)
        lat = model.encoder(X)
    codes=codes.numpy()
    usage=[len(set(codes[:,l])) for l in range(3)]
    groups=Counter(map(tuple,codes))
    model.train()
    return dict(loss=float(loss), **{k:float(v) for k,v in parts.items()}, usage=usage,
                unique_3code=len(groups), collision_groups=sum(c>1 for c in groups.values()),
                items_in_collisions=sum(c for c in groups.values() if c>1),
                largest_group=max(groups.values()), latent_norm=float(lat.norm(dim=-1).mean()))
print(json.dumps({"epoch":0, **stats()}), flush=True)
t=time.time()
for ep in range(1,a.epochs+1):
    for (b,) in loader:
        opt.zero_grad(); loss,_,_=model(b); loss.backward(); opt.step()
    if a.reset_dead:
        with torch.no_grad():
            r=model.encoder(X); g=torch.Generator().manual_seed(ep)
            for cb in model.codebooks:
                d=r.square().sum(-1,keepdim=True)+cb.square().sum(-1)-2*r@cb.T
                i=d.argmin(-1); used=torch.bincount(i,minlength=cb.shape[0])>0
                dead=(~used).nonzero().flatten()
                if len(dead): cb.data[dead]=r[torch.randint(len(r),(len(dead),),generator=g)]
                r=r-cb[d.argmin(-1)]
    if ep==1 or ep%a.report==0 or ep==a.epochs:
        print(json.dumps({"epoch":ep, **stats(), "sec":round(time.time()-t)}), flush=True)
print("zero-predictor reconstruction mse", float((X**2).mean()))
