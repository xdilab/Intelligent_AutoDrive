from pathlib import Path
from PIL import Image
import json,subprocess
W=Path(__file__).resolve().parent;r=json.load(open(W/'registration.json'));p=json.load(open(W/'before.json'))['structuredContent'];R=[];dims=[]
for record in r['records']:
 n=record['stage'];s=p['slides'][record['slide']-1];file=W/f'stage{n}.drawio.png';subprocess.run(['python3','/home/brandon/.codex/skills/drawio-skill/scripts/repair_png.py',str(file)],check=True)
 im=Image.open(file).convert('RGBA');dims.append(im.size);im.thumbnail((2000,2000));bg=Image.new('RGBA',im.size,'white');bg.alpha_composite(im);bg.convert('RGB').save(W/f'stage{n}.png')
assert len(set(dims))==1
r['export_dimensions_px']=dims[0];r['slide_frame_pt']['height']=820*dims[0][1]/dims[0][0];frame=r['slide_frame_pt']
for n,m in [(4,5),(5,6)]:Image.blend(Image.open(W/f'stage{n}.png'),Image.open(W/f'stage{m}.png'),.5).save(W/f'overlay-stage{n}-stage{m}.png')
for record in r['records']:
 no=record['slide'];s=p['slides'][no-1];e=next(e for e in s['pageElements'] if 'image'in e);id=e['objectId'];file=W/(record['file']+'.png')
 req=[{'deleteObject':{'objectId':id}},{'createImage':{'objectId':id,'url':str(file),'elementProperties':{'pageObjectId':s['objectId'],'size':{'width':{'magnitude':frame['width'],'unit':'PT'},'height':{'magnitude':frame['height'],'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateX':frame['x'],'translateY':frame['y'],'unit':'PT'}}}},{'updatePageElementAltText':{'objectId':id,'title':f'Aligned draw.io evolution stage {record["stage"]}','description':f'Editable source artifacts/defense-30min/alignment/{record["file"]}; exact shared station coordinates and source hashes in alignment/registration.json. Prior lineage: drawio-revision/provenance.json. Style/alignment contract: tools/defense-diagram-reference.md.'}}]
 # Reference-only provenance, preserving all existing cue content and citations.
 np=s['slideProperties']['notesPage'];nid=np['notesProperties']['speakerNotesObjectId'];old=''.join(t.get('textRun',{}).get('content','') for e in np['pageElements'] if e['objectId']==nid for t in e.get('shape',{}).get('text',{}).get('textElements',[]))
 extra='\n\nALIGNMENT PROVENANCE (not spoken): '+f'alignment/{record["file"]}; fixed canvas and slide transform in alignment/registration.json. Source captions retained here: '+' | '.join(v for v in record['removed_duplicate_title_or_caption'] if v)
 req.append({'insertText':{'objectId':nid,'insertionIndex':len(old)-1,'text':extra}})
 # Align title/subtitle baselines without changing their styles.
 for el in s['pageElements']:
  if 'shape' not in el:continue
  t=''.join(x.get('textRun',{}).get('content','') for x in el['shape'].get('text',{}).get('textElements',[])).strip();id2=el['objectId'];tr=dict(el['transform']);unit=12700 if tr.get('unit')=='EMU' else 1;y=tr.get('translateY',0)/unit
  xy=None
  if no in [8,9] and 80<y<120:xy=(75,99)
  if no in [8,9] and 120<=y<150:xy=(75,145)
  if no in [8,9,11] and 440<y<450:xy=(tr['translateX']/unit,453)
  if no==8:
   xy={'stage0_methodology_copy_0':(75,210),'stage0_methodology_copy_1':(365,210),'stage0_methodology_copy_2':(400,275),'stage0_methodology_copy_3':(400,292)}.get(id2,xy)
  if xy:
   tr['translateX']=xy[0]*unit;tr['translateY']=xy[1]*unit;req.append({'updatePageElementTransform':{'objectId':id2,'applyMode':'ABSOLUTE','transform':tr}})
 R.append({'slide':no,'image':str(file),'requests':req})
(W/'registration.json').write_text(json.dumps(r,indent=2));(W/'slide-batches.json').write_text(json.dumps(R));print('Ready:',len(R),'slides; same dimensions',dims[0],frame)
