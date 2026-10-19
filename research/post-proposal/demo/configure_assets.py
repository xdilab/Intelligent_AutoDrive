"""Register exact official visual assets without downloading or altering them."""
import argparse
import json
from pathlib import Path
from export_sample import sha

EXPECTED={'base':'df8cdbe9c3c9f65fbba777afc9ee6fbf58e6d68c4aec2827cac8e2a98d0a8446',
          'clip':'e60e0a6e05daec48285254b17b0b91c7d814b3ac74e538b2c62201553eeec6b1'}

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--base',type=Path,required=True)
    ap.add_argument('--clip',type=Path,required=True)
    ap.add_argument('--root',type=Path,required=True)
    a=ap.parse_args();files=[]
    for kind in ['base','clip']:
        path=getattr(a,kind).resolve();digest=sha(path)
        if digest!=EXPECTED[kind]:raise ValueError(f'Unexpected {kind} checkpoint SHA-256')
        files.append({'kind':kind,'path':str(path),'sha256':digest,'bytes':path.stat().st_size})
    (a.root/'results').mkdir(parents=True,exist_ok=True)
    target=a.root/'results/assets.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps({'passed':True,'files':files},indent=2))
    print(target)

if __name__=='__main__':main()
