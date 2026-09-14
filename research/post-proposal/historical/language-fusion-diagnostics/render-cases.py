"""Render measured detection cases with coordinate provenance; no synthetic images."""
from pathlib import Path
import json,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image
A=Path(__file__).parent;D=A/'collected';rows=json.loads((D/'examples.json').read_text());report=json.loads((D/'report.json').read_text());assert report['parity_pass']
variants=['head-flat','head-phrase','stage5','stage6'];names=['Flat head','Phrase head','Stage 5','Stage 6'];out=[]
for c,category in [('LarVeh-Stop-Jun','phrase_help_lost'),('Bus-Stop-VehLane','phrase_help_preserved')]:
 candidates=[r for r in rows if r['class']==c and r['category']==category]
 if not candidates:continue
 r=sorted(candidates,key=lambda r:(r['seed'],r['frame']))[0];stem=r['frame'];video,fid=stem.rsplit('_',1);source=Path('/data/datasets/ROAD_plusplus/rgb-images')/video/f'{int(fid):05d}.jpg'
 if not source.exists():raise FileNotFoundError(source)
 im=Image.open(source).convert('RGB');w,h=im.size
 fig,axs=plt.subplots(1,2,figsize=(13,5),gridspec_kw={'width_ratios':[1.65,1]},layout='constrained');ax=axs[0];ax.imshow(im);ax.axis('off')
 for key,color,label in [('gt_box','#44dd44','Ground truth'),('candidate_box','#00ffff','Phrase-matched YOLO box')]:
  x1,y1,x2,y2=r[key];ax.add_patch(Rectangle((x1*w,y1*h),(x2-x1)*w,(y2-y1)*h,fill=False,edgecolor=color,linewidth=2.2,label=label,linestyle='-' if key=='gt_box' else '--'))
 ax.legend(loc='lower left',fontsize=8,facecolor='black',labelcolor='white',framealpha=.75)
 ax.set_title(f'{stem} | seed {r["seed"]} | IoU {r["candidate_iou"]:.3f}',fontsize=10)
 axs[1].axis('off');cells=[]
 for v,name in zip(variants,names):
  rank=r[v+'_gt_rank'];cells.append([name,f'{r[v+"_score_same_candidate"]:.4f}',f'{rank:,}', 'Yes' if rank<=r['budget'] else 'No'])
 table=axs[1].table(cellText=cells,colLabels=['Model','Box score','GT rank','In budget?'],loc='center',cellLoc='center',colWidths=[.30,.24,.24,.22]);table.auto_set_font_size(False);table.set_fontsize(10);table.scale(1,2)
 axs[1].set_title(f'Equal budget: top {r["budget"]:,} predictions\nwithin this class, across all evaluation frames',fontsize=10)
 axs[1].text(0,.15,'Box score: same cyan candidate in every model.\nGT rank: first matched detection for this GT;\nthe winning box may differ between models.\nBudget is diagnostic, not a deployment threshold.',transform=axs[1].transAxes,fontsize=9,va='top')
 title='Phrase recovery lost by fusion' if category=='phrase_help_lost' else 'Phrase recovery preserved by fusion'
 fig.suptitle(f'{title}: {c}',fontsize=14)
 path=A/f'case-{c.lower()}-{category}.png';fig.savefig(path,dpi=170);plt.close(fig);out.append(dict(r,image=path.name,source_frame=str(source),source_frame_sha256=hashlib.sha256(source.read_bytes()).hexdigest()))
(A/'selected-visual-cases.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
