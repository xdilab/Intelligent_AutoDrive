"""Build the source-audited thesis citation inventory (stdlib only)."""
from pathlib import Path
import csv, html, json, re

ROOT = Path(__file__).resolve().parent
entries = []

def add(key, title, author, year, venue, url, role, where, group='core', kind='inproceedings', **extra):
    entries.append(dict(key=key,title=title,author=author,year=str(year),venue=venue,url=url,role=role,where=where,group=group,kind=kind,**extra))

def arxiv(identifier, key, year, venue, role, where, group='core', kind='inproceedings', **extra):
    saved=ROOT / 'primary-metadata.json'
    if saved.exists():
        record=json.loads(saved.read_text())[identifier]
        add(key,record['title'],record['author'],year,venue,'https://arxiv.org/abs/'+identifier,role,where,group,kind,eprint=identifier,**extra)
        return
    path = Path('/tmp/internvideo-citation.html') if identifier=='2403.15377' else Path(f'/tmp/cite-{identifier}.html')
    source=path.read_text()
    def meta(name): return [html.unescape(x) for x in re.findall(r'name="citation_'+name+r'" content="([^"]*)',source)]
    assert meta('title') and meta('author'), identifier
    add(key,meta('title')[0],' and '.join(meta('author')),year,venue,'https://arxiv.org/abs/'+identifier,role,where,group,kind,eprint=identifier,**extra)

arxiv('2102.11585','singh2023road',2023,'IEEE Transactions on Pattern Analysis and Machine Intelligence','ROAD event vocabulary, direct/composed prediction, and 3D-RetinaNet baseline.','Thesis [1], §§2.1, 2.5, 3.3; slides 6, 14–15, 44.',kind='article',volume='45',number='1',pages='1036--1054',doi='10.1109/TPAMI.2022.3150906')
arxiv('2411.01683','khan2024roadwaymo',2024,'arXiv preprint arXiv:2411.01683','ROAD-Waymo annotations and ROAD++ framework; use local release audit for measured counts.','Thesis [2], §§2.1, 3.1; slides 6–9, 44.',kind='misc',note='First submitted 2024; title follows version 3, revised July 8, 2026.')
arxiv('2210.01597','giunchiglia2023roadr',2023,'Machine Learning','Logical requirements and neuro-symbolic ROAD background.','Thesis [3], §2.1; slide 6; missing from deck reference slide.',kind='article',volume='112',pages='3261--3291',doi='10.1007/s10994-023-06322-z')
arxiv('1910.09217','kang2020decoupling',2020,'International Conference on Learning Representations','Separate representation and classifier learning; motivation, not an identical reproduction of cRT.','Thesis [4], §2.3; slides 32, 44.')
arxiv('1708.02002','lin2017focal',2017,'IEEE International Conference on Computer Vision','RetinaNet and focal loss.','Thesis [5], §§2.3, 3.3, 3.6, 5.3; baseline and head-training slides; missing from deck reference slide.')
arxiv('1703.06870','he2017maskrcnn',2017,'IEEE International Conference on Computer Vision','RoIAlign attribution; the thesis does not implement a Mask R-CNN segmentation pipeline.','Thesis [6], §§2.3, 3.6; slides 23–24, 44.')
arxiv('2103.00020','radford2021clip',2021,'Proceedings of the 38th International Conference on Machine Learning','CLIP image–text alignment and text-derived classifier directions.','Thesis [7], §§2.4, 5.3; slides 32–37, 44.',volume='139',pages='8748--8763')
arxiv('2403.15377','wang2024internvideo2',2024,'European Conference on Computer Vision','Video foundation model family and published larger-model reference results.','Thesis [8], §§2.4, 5.2–5.3; slides 32–36, 44.')
add('opengvlabInternvideo2ClipS','InternVideo2_CLIP_S','{OpenGVLab}',2026,'Model repository and configuration','https://huggingface.co/OpenGVLab/InternVideo2_CLIP_S','Exact checkpoint identity and configuration; cite separately from the model-family paper.','Thesis [9], §5.2; slide 36.','software','misc',note='Accessed September 7, 2026. Year is access year; release date and immutable checkpoint revision are not established here.')
add('jocher2023yolov8','Ultralytics YOLOv8','Jocher, Glenn and Chaurasia, Ayush and Qiu, Jing',2023,'Software and model documentation','https://docs.ultralytics.com/models/yolov8/','YOLOv8x detector implementation; cite as software, not a conference paper.','Thesis [10], §3.4; slides 17–20, 34–35, 44.','software','misc',note='Documentation accessed September 7, 2026. Record the actual training package version separately.')
arxiv('2410.23077','zhang2024roadchallenge',2024,'arXiv preprint arXiv:2410.23077','ECCV 2024 ROAD++ Track 1 winning system and detector-training recipe context.','Thesis §3.4 currently lacks this reference; slides 17, 44.',kind='misc')
add('wolpert1992stacked','Stacked Generalization','Wolpert, David H.',1992,'Neural Networks','https://doi.org/10.1016/S0893-6080(05)80023-1','Out-of-sample predictions for second-level learning. Video grouping and the 305-to-135 road-event MLP are project-specific.','Thesis §3.7 missing citation; slides 26–27, 44, 46.','core','article',volume='5',number='2',pages='241--259',doi='10.1016/S0893-6080(05)80023-1',verification_url='https://cafri-labs.github.io/lab-manual/papers/wolpert1992.pdf')
arxiv('1705.07750','carreira2017i3d',2017,'IEEE Conference on Computer Vision and Pattern Recognition','Inflated 3D convolution and Kinetics action recognition; ROAD supplies the specific ResNet50-I3D adaptation.','Thesis §3.3 and Figure 3; slides 14–15, 20–24; add component attribution.')
arxiv('1512.03385','he2016resnet',2016,'IEEE Conference on Computer Vision and Pattern Recognition','Residual backbone underlying ResNet50-I3D.','Thesis Figure 3 and baseline architecture description; slide 14.')
arxiv('1612.03144','lin2017fpn',2017,'IEEE Conference on Computer Vision and Pattern Recognition','Feature pyramids and the multiscale feature representation.','Thesis §3.3, Figure 3, §3.6 P3 features; slides 14, 23–24.')
arxiv('1412.6980','kingma2015adam',2015,'International Conference on Learning Representations','Adam optimizer used by completed heads and proposed classifier defaults.','Thesis §5.3; exp11 train_head.py/train_comp_mlp.py and exp12 train_clip_head.py.')
add('rasouli2017jaad','Are They Going to Cross? A Benchmark Dataset and Baseline for Pedestrian Crosswalk Behavior','Rasouli, Amir and Kotseruba, Iuliia and Tsotsos, John K.',2017,'IEEE International Conference on Computer Vision Workshops','https://data.nvision.eecs.yorku.ca/JAAD_dataset/','Primary JAAD benchmark citation recommended by the dataset authors.','Slide 5 currently cites only the dataset website; add paper to references.','core',pages='206--213')
add('rasouli2019pie','PIE: A Large-Scale Dataset and Models for Pedestrian Intention Estimation and Trajectory Prediction','Rasouli, Amir and Kotseruba, Iuliia and Kunic, Toni and Tsotsos, John K.',2019,'IEEE/CVF International Conference on Computer Vision','https://openaccess.thecvf.com/content_ICCV_2019/html/Rasouli_PIE_A_Large-Scale_Dataset_and_Models_for_Pedestrian_Intention_Estimation_ICCV_2019_paper.html','PIE pedestrian intention and trajectory background; distinguish intent from observed actions.','Slide 5, including notes; missing from deck reference slide.','core',pages='6262--6271')
arxiv('1912.04838','sun2020waymo',2020,'IEEE/CVF Conference on Computer Vision and Pattern Recognition','Underlying Waymo Open Dataset video and acquisition provenance, distinct from ROAD-Waymo event annotations.','Thesis §2.1 and benchmark provenance; slide 7.','supporting')
add('everingham2010voc','The PASCAL Visual Object Classes (VOC) Challenge','Everingham, Mark and Van Gool, Luc and Williams, Christopher K. I. and Winn, John and Zisserman, Andrew',2010,'International Journal of Computer Vision','https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/pubs/everingham10.html','Detection matching, ranking, precision–recall and AP background; exact AP variant also needs the VOC2010 resource and ROAD evaluator.','Thesis evaluation protocol; slide 12.','supporting','article',volume='88',number='2',pages='303--338',doi='10.1007/s11263-009-0275-4')
add('voc2010protocol','The PASCAL Visual Object Classes Challenge 2010 (VOC2010)','{PASCAL VOC Organizers}',2010,'Official evaluation documentation','https://www.robots.ox.ac.uk/~vgg/projects/pascal/VOC/voc2010/','Documents the change from 11-point AP to using all data points; pair with the actual evaluator source.','Slide 12 notes; thesis evaluation details.','software','misc',note='Accessed September 7, 2026.')
arxiv('1705.06950','kay2017kinetics',2017,'arXiv preprint arXiv:1705.06950','Kinetics-400 dataset named in the published video-model reference comparison and baseline pretraining.','Slide 36 Kinetics-400 reference; baseline README pretrained-weight path.','supporting',kind='misc')
arxiv('1212.0402','soomro2012ucf101',2012,'arXiv preprint arXiv:1212.0402','UCF101 dataset underlying the quoted reference benchmark.','Slide 36 UCF101 published reference result.','supporting',kind='misc')
arxiv('1912.01703','paszke2019pytorch',2019,'Advances in Neural Information Processing Systems','Implementation framework for baseline and trained heads.','Implementation/reproducibility section; exp11/exp12 torch imports.','supporting',volume='32')
arxiv('2311.17049','vasu2024mobileclip',2024,'IEEE/CVF Conference on Computer Vision and Pattern Recognition','Text-tower architecture identified in the released InternVideo2_CLIP_S configuration. Cite if describing its internals; checkpoint source remains authoritative.','Checkpoint config text_config.model_type=mobileclip_text_model.','supporting',pages='15963--15974')
arxiv('2010.11929','dosovitskiy2021vit',2021,'International Conference on Learning Representations','Vision Transformer patch-token background.','Thesis Figure 7 / encoder internals, if explaining patch embeddings.','supporting')
arxiv('1706.03762','vaswani2017attention',2017,'Advances in Neural Information Processing Systems','Transformer attention background; not a source for the project-specific feature tap.','Encoder architecture explanation if expanded.','supporting',volume='30')
arxiv('2109.01134','zhou2022coop',2022,'International Journal of Computer Vision','Learned context prompts for CLIP; related-work contrast with frozen phrases. Current native snapshot does not contain this paper, although earlier review logs say it was added.','Related-work comparison in Chapter 2, if restored/retained.','related',kind='article',doi='10.1007/s11263-022-01653-1')
arxiv('2204.03574','nayak2023csp',2023,'International Conference on Learning Representations','Learnable primitive prompt tokens and unseen compositions; distinguish from fixed-vocabulary supervised road-event recognition.','Related-work comparison in Chapter 2, if restored/retained.','related',verification_url='https://iclr.cc/virtual/2023/poster/12162')
add('rasouli2017agreeing','Agreeing to Cross: How Drivers and Pedestrians Communicate','Rasouli, Amir and Kotseruba, Iuliia and Tsotsos, John K.',2017,'IEEE Intelligent Vehicles Symposium','https://data.nvision.eecs.yorku.ca/JAAD_dataset/','Companion JAAD interaction study recommended by dataset authors.','Slide 5 interaction discussion; optional deeper background.','related',pages='264--269')
add('roadDatasetRepository','ROAD dataset and baseline resources','{ROAD Dataset Authors}',2026,'Dataset and software repository','https://github.com/gurkirt/road-dataset','Dataset release and implementation provenance alongside the ROAD paper.','Reproducibility/source appendix; slide 44 notes.','software','misc',note='Access year 2026; pin the actual local release and commit before submission.')
add('roadEccv2024','ROAD++ Challenge at ECCV 2024','{ROAD++ Challenge Organizers}',2024,'Official challenge website','https://sites.google.com/view/road-eccv2024/challenge','Track definitions and evaluation setting.','Slide 17 official challenge link.','software','misc',note='Accessed September 7, 2026.')
add('internvideo2ModelZoo','InternVideo2 Multimodality Model Zoo','{OpenGVLab}',2026,'Official model repository','https://github.com/OpenGVLab/InternVideo/blob/main/InternVideo2/multi_modality/MODEL_ZOO.md','Checkpoint/result naming and model-family reference tables. Does not establish that the S checkpoint equals the S14 or L14 benchmark row.','Slide 36 notes and checkpoint audit.','software','misc',note='Access year 2026; mutable main-branch resource.')
add('torchvisionRoiAlign','torchvision.ops.roi_align','{TorchVision Contributors}',2026,'Software API documentation','https://docs.pytorch.org/vision/stable/generated/torchvision.ops.roi_align.html','API-level details for implementation reproducibility; Mask R-CNN remains the scientific RoIAlign citation.','Stage 2 implementation appendix, if documenting aligned/sampling settings.','software','misc',note='Accessed September 7, 2026; record the installed package version rather than treating the current stable documentation version as the experiment version.')

def author_short(s):
    a=s.split(' and ')
    def fmt(n):
        if n.startswith('{'): return n.strip('{}')
        if ',' not in n: return n
        last,first=n.split(',',1)
        return ' '.join(x[0]+'.' for x in first.split() if x)+ ' '+last
    return fmt(a[0])+' et al.' if len(a)>6 else ', '.join(fmt(x) for x in a)

def bib_escape(s):
    return str(s).replace('&',r'\&').replace('_',r'\_').replace('%',r'\%')

md=['# Thesis citation inventory','', 'As of September 7, 2026. Coverage: current native thesis, main deck slides 1–45 plus stacking appendix 46, local proposal drafts, and directly relevant implementation sources. **The SharePoint abstracts spreadsheet is pending access (HTTP 403); this inventory is not yet exhaustive across that source.**', '', 'Entries have stable citation keys. Numbers below are inventory numbers, not replacements for the thesis’s current IEEE citation numbering. Include an entry in a submitted reference list when the text, figure, table, software description, or background discussion actually cites it.', '']
bib=['% Thesis citation inventory, September 7, 2026. SharePoint workbook pending.','% Includes core, supporting, software and related-work entries. Cite selectively by key.','']
groups={'core':'Direct thesis and presentation sources','supporting':'Supporting architecture, benchmark and implementation sources','software':'Software, checkpoint and protocol resources','related':'Associated related work; verify inclusion in final narrative'}
ordered=[e for g in groups for e in entries if e['group']==g]
for g,label in groups.items():
    md += ['## '+label,'']
    for e in [e for e in ordered if e['group']==g]:
        i=ordered.index(e)+1
        line=f'[{i}] {author_short(e["author"])}, “{e["title"]},” *{e["venue"]}*'
        for k,prefix in [('volume',', vol. '),('number',', no. '),('pages',', pp. ')]:
            if e.get(k):line+=prefix+e[k].replace('--','–')
        line+=', '+e['year']+'.'
        if e.get('doi'):line+=' DOI: '+e['doi']+'.'
        line+=f' [Source]({e["url"]}).'
        md += [line,'',f'**Key:** `{e["key"]}`  ',f'**Association:** {e["role"]}  ',f'**Placement/evidence:** {e["where"]}']
        if e.get('note'):md += ['**Note:** '+e['note']]
        if e.get('verification_url'):md += [f'**Verification:** [Primary text/author record]({e["verification_url"]}).']
        md+=['']
for e in ordered:
    fields={'author':e['author'],'title':'{'+e['title']+'}','year':e['year']}
    if e['kind']=='article':fields['journal']=e['venue']
    elif e['kind']=='inproceedings':fields['booktitle']=e['venue']
    else:fields['howpublished']=e['venue']
    for k in ['volume','number','pages','doi','eprint','url','note']:
        if e.get(k):fields[k]=e[k]
    if e.get('eprint'):fields['archivePrefix']='arXiv'
    fields['keywords']=e['group']
    bib.append('@'+e['kind']+'{'+e['key']+',\n'+',\n'.join('  '+k+' = {'+(v if k in ['url','doi'] else bib_escape(v))+'}' for k,v in fields.items())+'\n}\n')
ROOT.joinpath('master-bibliography.md').write_text('\n'.join(md)+'\n')
ROOT.joinpath('thesis-references.bib').write_text('\n'.join(bib)+'\n')
ROOT.joinpath('citation-inventory.json').write_text(json.dumps(ordered,indent=2,ensure_ascii=False)+'\n')
ROOT.joinpath('primary-metadata.json').write_text(json.dumps({e['eprint']:{'title':e['title'],'author':e['author']} for e in ordered if e.get('eprint')},indent=2,ensure_ascii=False)+'\n')
with ROOT.joinpath('citation-inventory.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=['key','group','title','author','year','venue','url','role','where'],extrasaction='ignore');w.writeheader();w.writerows(ordered)
assert len({e['key'] for e in ordered})==len(ordered)
print(f'Wrote {len(ordered)} unique entries: '+', '.join(f'{g}={sum(e["group"]==g for e in ordered)}' for g in groups))
