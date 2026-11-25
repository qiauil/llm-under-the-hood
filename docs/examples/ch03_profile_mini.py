#!/usr/bin/env python3
"""Mini-GPT resource ledger and PyTorch Profiler experiment.

PyTorch Linear stores [out,in], while the textbook uses row vectors and
mathematical weights [in,out]; F.linear therefore applies the transpose.
"""
import argparse, json, math, platform
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

CFG = dict(L=2, d=128, H=4, H_KV=2, d_h=32, d_ff=352, V=256, T_max=128)

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.weight=nn.Parameter(torch.ones(d)); self.eps=eps
    def forward(self,x): return x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+self.eps)*self.weight

def rope(x):
    # x [B,H,T,D]; positions start at zero and only Q/K are rotated.
    D=x.shape[-1]; pos=torch.arange(x.shape[-2],device=x.device,dtype=torch.float32)
    inv=10000**(-torch.arange(0,D,2,device=x.device,dtype=torch.float32)/D)
    ang=pos[:,None]*inv[None,:]; c,s=ang.cos().to(x.dtype),ang.sin().to(x.dtype)
    a,b=x[...,0::2],x[...,1::2]
    return torch.stack((a*c-b*s,a*s+b*c),dim=-1).flatten(-2)

class Attention(nn.Module):
    def __init__(self,c):
        super().__init__(); d,hk,dh=c['d'],c['H_KV'],c['d_h']
        self.c=c; self.q=nn.Linear(d,d,bias=False); self.k=nn.Linear(d,hk*dh,bias=False)
        self.v=nn.Linear(d,hk*dh,bias=False); self.o=nn.Linear(d,d,bias=False)
    def forward(self,x):
        B,T,_=x.shape; c=self.c
        q=rope(self.q(x).view(B,T,c['H'],c['d_h']).transpose(1,2))
        k=rope(self.k(x).view(B,T,c['H_KV'],c['d_h']).transpose(1,2))
        v=self.v(x).view(B,T,c['H_KV'],c['d_h']).transpose(1,2)
        groups=c['H']//c['H_KV']; k=k.repeat_interleave(groups,1); v=v.repeat_interleave(groups,1)
        score=q@k.transpose(-1,-2)/math.sqrt(c['d_h'])
        mask=torch.ones(T,T,device=x.device,dtype=torch.bool).triu(1)
        p=score.masked_fill(mask,float('-inf')).softmax(-1)
        return self.o((p@v).transpose(1,2).contiguous().view(B,T,c['d']))

class Block(nn.Module):
    def __init__(self,c):
        super().__init__(); d,f=c['d'],c['d_ff']; self.n1=RMSNorm(d); self.a=Attention(c); self.n2=RMSNorm(d)
        self.g=nn.Linear(d,f,bias=False); self.u=nn.Linear(d,f,bias=False); self.down=nn.Linear(f,d,bias=False)
    def forward(self,x):
        x=x+self.a(self.n1(x)); n=self.n2(x); return x+self.down(F.silu(self.g(n))*self.u(n))

class MiniGPT(nn.Module):
    def __init__(self,c):
        super().__init__(); self.c=c; self.embed=nn.Embedding(c['V'],c['d']); self.blocks=nn.ModuleList([Block(c) for _ in range(c['L'])]); self.norm=RMSNorm(c['d'])
    def forward(self,tokens):
        x=self.embed(tokens)
        for block in self.blocks: x=block(x)
        return F.linear(self.norm(x),self.embed.weight) # tied LM Head

def theory(c,B,T):
    P=c['V']*c['d']+c['L']*(2*c['d']**2+2*c['d']*c['H_KV']*c['d_h']+3*c['d']*c['d_ff']+2*c['d'])+c['d']
    block=4*B*T*c['d']**2+4*B*T*c['d']*c['H_KV']*c['d_h']+6*B*T*c['d']*c['d_ff']+4*B*T*T*c['d']
    forward=c['L']*block+2*B*T*c['d']*c['V']
    return dict(parameters=P,forward_main_flops=forward,train_three_x_main_flops=3*forward,
                teaching_bf16_adam_state_bytes=12*P,
                cpu_fp32_adam_state_bytes=16*P,
                memory_note='parameter-related logical state only; excludes activations, temporaries and runtime')

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--device',choices=['cpu','cuda'],default='cpu'); ap.add_argument('--batch',type=int,default=2); ap.add_argument('--seq',type=int,default=32); ap.add_argument('--warmup',type=int,default=3); ap.add_argument('--steps',type=int,default=3); ap.add_argument('--output',required=True); a=ap.parse_args()
    if a.device=='cuda' and not torch.cuda.is_available(): raise SystemExit('CUDA requested but unavailable')
    torch.manual_seed(20261031); device=torch.device(a.device); dtype=torch.bfloat16 if a.device=='cuda' else torch.float32
    model=MiniGPT(CFG).to(device=device,dtype=dtype); opt=torch.optim.AdamW(model.parameters(),lr=1e-3)
    tokens=torch.randint(0,CFG['V'],(a.batch,a.seq+1),device=device)
    def step():
        opt.zero_grad(set_to_none=True); logits=model(tokens[:,:-1]); loss=F.cross_entropy(logits.reshape(-1,CFG['V']).float(),tokens[:,1:].reshape(-1)); loss.backward(); opt.step(); return loss
    for _ in range(a.warmup): step() # warm-up and materialize AdamW states
    activities=[torch.profiler.ProfilerActivity.CPU]
    if a.device=='cuda': activities.append(torch.profiler.ProfilerActivity.CUDA)
    with torch.profiler.profile(activities=activities,record_shapes=True,profile_memory=True,with_flops=True) as prof:
        for _ in range(a.steps): loss=step()
    if a.device=='cuda': torch.cuda.synchronize()
    events=prof.key_averages(); prof_flops=sum(int(getattr(e,'flops',0) or 0) for e in events)
    top=sorted(events,key=lambda e:getattr(e,'self_cpu_time_total',0),reverse=True)[:12]
    result={'status':'measured_cpu' if a.device=='cpu' else 'measured_cuda','seed':20261031,'environment':{'python':platform.python_version(),'torch':torch.__version__,'device':str(device),'dtype':str(dtype),'cuda_available':torch.cuda.is_available()},'config':CFG|{'B':a.batch,'T':a.seq,'warmup':a.warmup,'profiled_steps':a.steps},'theory':theory(CFG,a.batch,a.seq),'actual':{'parameters':sum(p.numel() for p in model.parameters()),'loss_after_profiled_step':float(loss.detach().cpu()),'profiler_reported_flops_partial_total':prof_flops,'profiler_reported_flops_partial_per_step':prof_flops/a.steps,'note':'with_flops covers only supported operators; it is not the complete textbook FLOPs count'},'top_cpu_ops':[{'name':e.key,'self_cpu_time_us':e.self_cpu_time_total,'cpu_memory_bytes':e.cpu_memory_usage,'flops':int(getattr(e,'flops',0) or 0)} for e in top]}
    out=Path(a.output); out.parent.mkdir(parents=True,exist_ok=True); out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8'); print(json.dumps(result,ensure_ascii=False,indent=2))
if __name__=='__main__': main()
