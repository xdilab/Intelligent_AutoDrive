"""Small CPU integration: cache batches, metrics, checkpoints and completed-run resume."""
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
import numpy as np
import torch

class Training(unittest.TestCase):
    def test_training_checkpoint_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            r=Path(tmp);d=r/'data';d.mkdir();rng=np.random.default_rng(3)
            for split in ['train','dev']:
                arrays={'crop':rng.normal(size=(8,1024)).astype('float16'),'context':rng.normal(size=(8,1024)).astype('float16'),'scene':rng.normal(size=(2,16,1024)).astype('float16'),'frame':np.arange(8,dtype='int32')%2,'boxes':np.tile(np.array([.1,.2,.8,.9],np.float32),(8,1)),'targets':np.zeros((8,184),np.uint8)}
                arrays['targets'][:4,[0,98]]=1
                for name,value in arrays.items():np.save(d/f'{split}-{name}.npy',value)
            (r/'cache-ready.json').write_text(json.dumps({'passed':True}))
            (r/'protocol.json').write_text(json.dumps({'epochs':1,'batch_size':4,'learning_rate':.0001,'weight_decay':.01,'save_every':1}))
            labels=[f't{i}' for i in range(86)];(d/'shared-frames.json').write_text(json.dumps({'labels':{'triplet':labels}}));(d/'train-counts.json').write_text(json.dumps({'rows':[{'label':label,'z':1 if i<39 else (-.75 if i<67 else -.25)} for i,label in enumerate(labels)]}))
            torch.save({'embeds':torch.randn(184,512)},d/'phrase_embeds.pt');torch.save(torch.full((184,),.5),d/'flat_alphas.pt')
            cmd=[sys.executable,str(Path(__file__).parent/'train_cached.py'),'--root',str(r),'--fusion','attention','--contrastive','.001','--seed','0','--device','cpu']
            subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,timeout=120)
            dest=r/'runs/attention-contrastive-seed0';ck=torch.load(dest/'resume.pt',map_location='cpu',weights_only=False)
            self.assertEqual(ck['epoch'],1);self.assertEqual(ck['position'],0);self.assertTrue(ck['optimizer']['state'])
            report=json.loads((dest/'complete.json').read_text());self.assertTrue(report['passed']);self.assertEqual(report['epochs'][0]['tail']['deep28']['classes'],28)
            before=(dest/'best.pt').read_bytes();subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,timeout=120);self.assertEqual(before,(dest/'best.pt').read_bytes())
if __name__=='__main__':unittest.main()
