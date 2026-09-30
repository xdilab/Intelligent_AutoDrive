"""Local authenticated HEAD checks; send only scoped expiring asset URLs to NCShare."""
from pathlib import Path
import json,subprocess
from huggingface_hub import get_hf_file_metadata,hf_hub_url
cfg=json.loads((Path(__file__).parent/'protocol.json').read_text());r=cfg['root'];assets=[]
for kind in ['base','clip','text']:
 c=cfg['encoder'];repo=c[kind+'_repo'];name=c[kind+'_file'];rev=c[kind+'_revision']
 d=get_hf_file_metadata(hf_hub_url(repo,name,revision=rev));assert len(d.etag)==64
 assets.append({'kind':kind,'repo':repo,'revision':rev,'filename':name,'bytes':d.size,'sha256':d.etag,'download_url':d.location})
remote="import os,sys; p='/work/bbyrd1/contextual-roi-1b-20260930/.asset-downloads.json'; fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_TRUNC,0o600); os.fchmod(fd,0o600); os.write(fd,sys.stdin.buffer.read()); os.close(fd)"
# Fixed remote command, no credentials or signed URLs in command-line arguments.
import shlex
subprocess.run(['ssh','-o','BatchMode=yes','ncshare','python3 -c '+shlex.quote(remote)],input=json.dumps(assets).encode(),check=True)
print('Provisioned scoped short-lived download links for',len(assets),'assets; no Hugging Face token copied.')
