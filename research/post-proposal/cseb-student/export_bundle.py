"""Export the selected student and its runtime weights without training assets."""
import argparse,json,shutil,sys
from pathlib import Path
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'demo'))
from export_sample import sha

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--run',type=Path,required=True);ap.add_argument('--yolo',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    manifest=json.loads((a.run/'manifest.json').read_text());ck=torch.load(a.run/'distilled-best.pt',map_location='cpu',weights_only=False)
    if ck['manifest_sha256']!=sha(a.run/'manifest.json'):raise ValueError('Checkpoint/manifest mismatch')
    backbone=Path(manifest['backbone'])
    if sha(backbone)!=manifest['backbone_sha256']:raise ValueError('Backbone mismatch')
    model=Path(__file__).with_name('model.py')
    if sha(model)!=manifest['source_sha256']['model.py']:raise ValueError('Student source mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    torch.save(ck['model'],a.output/'student.pt')
    shutil.copy2(backbone,a.output/'backbone.pth');shutil.copy2(a.yolo,a.output/'yolo.pt')
    bundle={'format':1,'architecture':'shared-resnet50-layer3-scene-gru256','sampling_hz':10,'labels':manifest['labels'],
            'model_sha256':sha(model),'files_sha256':{n:sha(a.output/n) for n in ['student.pt','backbone.pth','yolo.pt']},
            'teacher_checkpoint_sha256':manifest['teacher_sha256'],'student_checkpoint_sha256':sha(a.run/'distilled-best.pt'),
            'training_manifest_sha256':sha(a.run/'manifest.json'),'condition':'distilled','epoch':ck['epoch']}
    (a.output/'bundle.json').write_text(json.dumps(bundle,indent=2)+'\n');print(json.dumps({'bundle':str(a.output),'bytes':sum(p.stat().st_size for p in a.output.iterdir())}))
if __name__=='__main__':main()
