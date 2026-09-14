"""Audit saved predictions only; fit on gate partition, report internal dev AP."""
import argparse,json,multiprocessing as mp,time,hashlib
from pathlib import Path
import numpy as np
P=L=Y=PD=LD=YD=None
GRID=[0.,.0003353501,.0015,.005,.01,.025,.05,.1,.2,.35,.5,.75,1.]
def prob(p,l,g):return np.clip(p+g*(l-p),1e-7,1-1e-7)
def derivative(p,l,y,g):
    z=prob(p,l,g);return float(np.mean((z-y)*(l-p)/(z*(1-z))))
def solve(p,l,y):
    d0=derivative(p,l,y,0);d1=derivative(p,l,y,1)
    if d0>=0:return 0.,d0,d1
    if d1<=0:return 1.,d0,d1
    lo=0.;hi=1.
    for _ in range(35):
        mid=(lo+hi)/2
        if derivative(p,l,y,mid)>0:hi=mid
        else:lo=mid
    return (lo+hi)/2,d0,d1

def ap(y,p):
    yy=y[np.argsort(-p)]>0;n=yy.sum()
    if not n:return 0.
    precision=np.cumsum(yy)/np.arange(1,len(yy)+1)
    return float(np.maximum.accumulate(precision[::-1])[::-1][yy].sum()/n*100)
def loss(p,l,y,g):
    z=prob(p,l,g);pos=float(np.mean(-y*np.log(z)));neg=float(np.mean(-(1-y)*np.log1p(-z)))
    return {'bce':pos+neg,'positive_contribution':pos,'negative_contribution':neg}
def column(task):
    c,trained,global_opt=task
    p=P[:,c].astype('float64');l=L[:,c].astype('float64');y=Y[:,c].astype('float64');pd=PD[:,c].astype('float64');ld=LD[:,c].astype('float64');yd=YD[:,c]
    optimum,d0,d1=solve(p,l,y)
    # Unsupported class fallback follows deployed policy, while mathematical optimum is reported separately.
    supported=bool(y.sum());deploy=optimum if supported else global_opt
    records=[]
    for g in GRID:
        records.append({'weight':g,**loss(p,l,y,g),'dev_AP':ap(yd,pd+g*(ld-pd))})
    return {'class_index':c,'gate_positives':int(y.sum()),'dev_positives':int(yd.sum()),'optimum':optimum,'supported':supported,'deployed_optimum':deploy,'derivative_at_zero':d0,'derivative_at_one':d1,'trained_weight':trained,'trained_loss':loss(p,l,y,trained),'optimum_loss':loss(p,l,y,optimum),'dev_AP_trained':ap(yd,pd+trained*(ld-pd)),'dev_AP_direct':ap(yd,pd+deploy*(ld-pd)),'grid':records}
def global_derivative(g):
    total=0.;n=0
    for start in range(0,len(Y),8192):
        p=P[start:start+8192].astype('float64');l=L[start:start+8192].astype('float64');y=Y[start:start+8192];z=prob(p,l,g);total+=np.sum((z-y)*(l-p)/(z*(1-z)));n+=z.size
    return float(total/n)
def main():
    global P,L,Y,PD,LD,YD
    parser=argparse.ArgumentParser();parser.add_argument('--seed',type=int,required=True);parser.add_argument('--root',type=Path,required=True);a=parser.parse_args();src=Path('/work/bbyrd1/class-gate-study-20260914');run=src/f'runs/seed-{a.seed}';out=a.root/'results';out.mkdir(exist_ok=True,parents=True)
    gate=np.load(run/'gate-expert-probabilities.npz');dev=np.load(run/'development-expert-probabilities.npz');P=np.clip(gate['stage5'],1e-7,1-1e-7);Y=gate['targets'];PD=dev['stage5'];YD=dev['targets'];report={'seed':a.seed,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'grid':GRID,'source':str(run),'fit_partition':'gate','report_partition':'development','experts':{},'numerics':'Gate expert endpoints clipped to [1e-7,1-1e-7] before float64 convex BCE audit; slight numerical distinction from deployed output clipping. 35 bisection iterations; dev AP uses original unclipped expert probabilities'}
    for expert in ['phrase','shuffled']:
        L=np.clip(gate[expert],1e-7,1-1e-7);LD=dev[expert];selected=json.loads((run/f'gate-{expert}.json').read_text());d0=global_derivative(0);d1=global_derivative(1);lo=0.;hi=1.
        if d0>=0:opt=0.
        elif d1<=0:opt=1.
        else:
            for _ in range(35):
                mid=(lo+hi)/2
                if global_derivative(mid)>0:hi=mid
                else:lo=mid
            opt=(lo+hi)/2
        print('GLOBAL',expert,opt,d0,d1,flush=True)
        with mp.get_context('fork').Pool(8) as pool:cols=list(pool.imap(column,[(c,selected['class']['weights'][c],opt) for c in range(135)]))
        report['experts'][expert]={'global_optimum':opt,'global_derivative_zero':d0,'global_derivative_one':d1,'trained_global':selected['global']['weight'],'columns':cols}
        (out/f'audit-seed{a.seed}.partial.json').write_text(json.dumps(report,indent=2));print('COMPLETE EXPERT',expert,flush=True)
    report['completed_utc']=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime());(out/f'audit-seed{a.seed}.json').write_text(json.dumps(report,indent=2));print('COMPLETE',a.seed,flush=True)
if __name__=='__main__':main()
