from pathlib import Path
import xml.etree.ElementTree as E
import html,json,hashlib
A=Path(__file__).resolve().parent
W,H=1320,1190
mx=E.Element('mxfile');d=E.SubElement(mx,'diagram',name='Two contextual fusion variants');gm=E.SubElement(d,'mxGraphModel',pageWidth=str(W),pageHeight=str(H));root=E.SubElement(gm,'root');E.SubElement(root,'mxCell',id='0');E.SubElement(root,'mxCell',id='1',parent='0')
svg=[f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}"><rect width="100%" height="100%" fill="white"/><defs><marker id="a" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8Z" fill="#334155"/></marker></defs>'];seq=0

def node(label,x,y,w,h,fill='#ffffff',stroke='#334155',size=19,bold=False,kind='plain'):
 global seq
 seq+=1;id=f'n{seq}';sw=3 if kind=='trained' else 1.5
 cell=E.SubElement(root,'mxCell',id=id,value=label.replace('\n','<br>'),vertex='1',parent='1',style=f'rounded=1;arcSize=8;html=1;whiteSpace=wrap;fillColor={fill};strokeColor={stroke};strokeWidth={sw};fontFamily=Helvetica;fontSize={size};fontStyle={int(bold)};');E.SubElement(cell,'mxGeometry',x=str(x),y=str(y),width=str(w),height=str(h),attrib={'as':'geometry'})
 svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
 lines=label.split('\n')
 for i,line in enumerate(lines):svg.append(f'<text x="{x+w/2}" y="{y+h/2+(i-(len(lines)-1)/2)*(size+7)+size*.34}" text-anchor="middle" font-family="Arial" font-size="{size}" font-weight="{700 if bold else 400}" fill="#203040">{html.escape(line)}</text>')
 return id

def arrow(points):
 global seq
 seq+=1;c=E.SubElement(root,'mxCell',id=f'e{seq}',edge='1',parent='1',style='edgeStyle=none;noEdgeStyle=1;rounded=1;html=1;endArrow=block;strokeColor=#334155;strokeWidth=2;');g=E.SubElement(c,'mxGeometry',relative='1',attrib={'as':'geometry'});E.SubElement(g,'mxPoint',x=str(points[0][0]),y=str(points[0][1]),attrib={'as':'sourcePoint'});E.SubElement(g,'mxPoint',x=str(points[-1][0]),y=str(points[-1][1]),attrib={'as':'targetPoint'});a=E.SubElement(g,'Array',attrib={'as':'points'})
 for x,y in points[1:-1]:E.SubElement(a,'mxPoint',x=str(x),y=str(y))
 svg.append('<polyline points="'+' '.join(f'{x},{y}' for x,y in points)+'" fill="none" stroke="#334155" stroke-width="2" stroke-linejoin="round" marker-end="url(#a)"/>')

node('BOTH models use language attention',30,20,1260,60,'#ffffff','#ffffff',32,True)
node('Only the visual scene-summary step differs. These are the standalone models behind the results.',30,85,1260,40,'#ffffff','#ffffff',20)
for k,x in enumerate([30,690]):
 node('MLP context fusion' if k==0 else 'Attention context fusion',x,145,600,60,'#e8f1f9','#4b88b5',26,True)
 node('Same inputs: crop + context RoI\nfull-scene tokens + box position',x+35,225,530,65,'#e5f0d8','#82b366',20)
 arrow([(x+235,290),(x+235,325)])
 node('Average scene tokens' if k==0 else 'Visual attention over scene tokens\nQuery = crop + position',x+35,325,400,75,'#fff2cc','#d6b656',20,True,'plain' if k==0 else 'trained')
 node('DIFFERENT',x+445,342,145,40,'#ffffff','#ffffff',18,True)
 arrow([(x+235,400),(x+235,435)])
 node('MLP\n[crop; context RoI; scene summary; position]',x+15,435,440,75,'#f8cecc','#b85450',18,False,'trained')
 arrow([(x+235,510),(x+235,545)])
 node('Crop residual + LayerNorm\nVisual RoI v',x+75,545,320,65,'#dae8fc','#4b88b5',20)
 arrow([(x+235,610),(x+235,665)])
 node('LANGUAGE ATTENTION\nVisual RoI reads label phrases',x+35,665,365,85,'#e1d5e7','#8064a2',21,True,'trained')
 node('Text MLP\n+ residual\nPhrase matrix T',x+455,650,140,110,'#eee8f6','#8064a2',18,False,'trained')
 arrow([(x+455,708),(x+400,708)])
 arrow([(x+235,750),(x+235,790)])
 node('Gated residual + LayerNorm\nJoint RoI h',x+75,790,320,65,'#fff2cc','#d6b656',20)
 arrow([(x+235,855),(x+235,895)])
 node('Same classifier → 184 logits\nInputs: joint RoI + crop + phrase scores',x+15,895,570,70,'#f8cecc','#b85450',19,False,'trained')
node('Corrected contrastive training in BOTH: visual v ↔ adapted phrase matrix T, all 184 labels.\nGradients train both visual and text MLPs. Alignment happens BEFORE language attention.',45,1000,1230,75,'#fff8e3','#d6b656',21)
node('Purple = language attention / text  •  Yellow = scene-summary difference / alignment\nThick borders = learned modules  •  Encoder caches are frozen  •  Arrows = forward features\nThe original “classification only” controls still use language attention; they omit the contrastive loss.',35,1090,1250,85,'#ffffff','#ffffff',18)
svg.append('</svg>');(A/'comparison.svg').write_text(''.join(svg));E.indent(mx);E.ElementTree(mx).write(A/'comparison.drawio',encoding='utf-8',xml_declaration=True)
(A/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Both variants use language attention</title><style>body{margin:0;background:#edf1f5;font-family:Arial}main{max-width:1320px;margin:auto;background:white}img{width:100%;display:block}a{display:block;padding:14px}</style><main><img src="comparison.svg" alt="Side-by-side visual context fusion comparison; both models use language attention"><a href="comparison.drawio">Editable draw.io</a></main>')
source=Path('/data/repos/ROAD_Reason/research/post-proposal/contextual-roi-all184/model.py');(A/'provenance.json').write_text(json.dumps({'source':str(source),'sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'scope':'Explanation of implemented standalone variants; simplified intermediate dimensions; not Stage7 readout proposal.'},indent=2))
