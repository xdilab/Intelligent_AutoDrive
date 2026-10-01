"""Local resumable transfer of exact frames/metadata, without old mtimes."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib, json, subprocess, time

ROOT=Path('/data/repos/wiki/artifacts/contextual-roi-1b-20260930/input-staging')
REMOTE='/work/bbyrd1/contextual-roi-1b-20260930'
SOURCE=Path('/data/datasets/ROAD_plusplus/rgb-images')

def main():
    hashes=json.loads((ROOT/'frame-hashes.json').read_text());total=0
    for i,(rel,digest) in enumerate(hashes.items()):
        raw=(SOURCE/rel).read_bytes();assert hashlib.sha256(raw).hexdigest()==digest,rel;total+=len(raw)
        if i%10000==0:print('SOURCE_HASHED',i,len(hashes),flush=True)
    subprocess.run(['ssh','ncshare',f'mkdir -p {REMOTE}/frames {REMOTE}/inputs/gate {REMOTE}/results'],check=True)
    def copy(shard):
        log=ROOT/f'transfer-{shard}.log'
        with log.open('a') as f:
            for attempt in range(3):
                rc=subprocess.run(['rsync','-r','--no-times','--partial','--stats','--timeout=120',
                    '--files-from='+str(ROOT/f'files-{shard}.txt'),str(SOURCE)+'/',f'ncshare:{REMOTE}/frames/'],stdout=f,stderr=subprocess.STDOUT).returncode
                if rc==0:return shard
            raise RuntimeError(f'Input transfer shard{shard} failed')
    with ThreadPoolExecutor(max_workers=8) as pool:
        for result in as_completed([pool.submit(copy,i) for i in range(8)]):print('TRANSFERRED',result.result(),flush=True)
    gate=Path('/data/repos/wiki/artifacts/contextual-gate-20260919')
    files=[gate/'data'/f'gate-{k}.npy' for k in ['boxes','targets','frame','partition']]+[gate/'data/gate.jsonl']
    subprocess.run(['rsync','-r','--no-times',*[str(p) for p in files],f'ncshare:{REMOTE}/inputs/gate/'],check=True)
    subprocess.run(['rsync','-r','--no-times',str(gate/'preparation.json'),str(gate/'protocol.json'),f'ncshare:{REMOTE}/inputs/gate/'],check=True)
    subprocess.run(['rsync','-r','--no-times',str(ROOT/'frame-hashes.json'),f'ncshare:{REMOTE}/inputs/'],check=True)
    report={'passed':True,'time':time.time(),'frames':len(hashes),'bytes':total,
            'frame_hashes_sha256':hashlib.sha256((ROOT/'frame-hashes.json').read_bytes()).hexdigest(),
            'gate_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in files+[gate/'preparation.json',gate/'protocol.json']},
            'scope':'local SHA verification and successful rsync; remote SHA verification required before GPU extraction'}
    ready=ROOT/'inputs-transferred.json';ready.write_text(json.dumps(report,indent=2))
    subprocess.run(['rsync','-r','--no-times',str(ready),f'ncshare:{REMOTE}/results/'],check=True)
    print('INPUT_TRANSFER_COMPLETE',json.dumps(report),flush=True)

if __name__=='__main__':main()
