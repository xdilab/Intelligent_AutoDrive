from pathlib import Path
import hashlib,json,ast,concurrent.futures
root=Path(__file__).parent;cf=Path('/data/repos/ROAD_Reason/experiments/exp12_phrase_head/crop_full')
files=[cf/f'crop_feats_train.shard{i}of8.pkl' for i in range(8)]+[cf/f'crop_feats_val.shard{i}of4.pkl' for i in range(4)]
def digest(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return p.name,{'bytes':p.stat().st_size,'sha256':h.hexdigest()}
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: hashes=dict(pool.map(digest,files))
(root/'shard-checksums.json').write_text(json.dumps(hashes,indent=2)+'\n')
# Package unchanged baseline AP functions and label expansion, avoiding unrelated dataset dependencies.
src=Path('/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline/modules/evaluation.py');tree=ast.parse(src.read_text());names={'voc_ap','get_gt_of_cls','compute_iou','evaluate_detections','evaluate'}
blocks=[ast.get_source_segment(src.read_text(),n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
u=Path('/data/repos/PedestrianIntent++/ROAD_plus_plus_Baseline/modules/utils.py');ut=ast.parse(u.read_text());blocks += [ast.get_source_segment(u.read_text(),n) for n in ut.body if isinstance(n,ast.FunctionDef) and n.name=='get_individual_labels']
(root/'baseline-core.py').write_text('import numpy as np\nimport logging\nlogger=logging.getLogger(__name__)\n\n'+'\n\n'.join(blocks)+'\n')
print('Packaged 12 shard checksums and unchanged baseline AP functions.')
