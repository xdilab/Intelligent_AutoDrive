from pathlib import Path
import base64
import hashlib
import json
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
doc = ET.Element('mxfile', host='drawio')
diagram = ET.SubElement(doc, 'diagram', name='Inputs and Output')
model = ET.SubElement(diagram, 'mxGraphModel', page='1', pageWidth='1264', pageHeight='400', background='#ffffff')
root = ET.SubElement(model, 'root')
ET.SubElement(root, 'mxCell', id='0')
ET.SubElement(root, 'mxCell', id='1', parent='0')

def box(id, text, x, y, w, h, style):
    parent = '1' if id == 'frame' else 'frame'
    if id.startswith('photo-') or id == 'structured-label':
        parent = 'inputs'
        x, y = x-2, y-50
    if id in ('frame', 'inputs'):
        style += 'container=1;pointerEvents=0;'
    style = style.replace('fontFamily=Montserrat;', 'fontFamily=Arial;')
    cell = ET.SubElement(root, 'mxCell', id=id, value=text, style=style, vertex='1', parent=parent)
    ET.SubElement(cell, 'mxGeometry', x=str(x), y=str(y), width=str(w), height=str(h), **{'as':'geometry'})

textstyle = 'text;html=1;whiteSpace=wrap;align=center;verticalAlign=middle;fontFamily=Montserrat;fontSize=28;fontColor=#004684;'
box('frame', '', 0, 0, 1264, 400, 'fillColor=#ffffff;strokeColor=none;')
box('inputs', '', 2, 50, 424, 270, 'rounded=1;arcSize=6;fillColor=#ffffff;strokeColor=#C2C2C2;strokeWidth=1;')
box('input-label', 'Inputs', 2, 0, 424, 40, textstyle+'fontStyle=1;')
sources=[]
for i, rel in enumerate([0,4,8,13,14]):
    path=ROOT/f'pipeline-image-{i+4:03}.png'
    data=path.read_bytes()
    sources.append({'path':str(path),'sha256':hashlib.sha256(data).hexdigest(),'relative_frame':rel})
    uri='data:image/png,'+base64.b64encode(data).decode()
    box(f'photo-{rel}', '', 19+i*78, 76, 72, 72, 'shape=image;imageAspect=1;aspect=fixed;image='+uri+';')
box('photo-label', 'Earlier frames + current', 12, 158, 404, 40, textstyle+'fontSize=26;')
box('structured-label', 'Pose + trajectory\nEgo-speed + behavior', 12, 223, 404, 74, textstyle+'fontSize=28;')
box('network', 'SparseTemporalPIE', 502, 135, 320, 100, 'rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#dae8fc;strokeColor=#004684;strokeWidth=2;fontFamily=Montserrat;fontColor=#004684;fontSize=29;')
box('output', 'Crossing /\nNot crossing', 898, 135, 360, 100, 'rounded=1;arcSize=6;whiteSpace=wrap;html=1;fillColor=#f5f5f5;strokeColor=#909090;strokeWidth=1;fontFamily=Montserrat;fontColor=#004684;fontSize=31;')
box('output-label', 'Output', 898, 0, 360, 40, textstyle+'fontStyle=1;')
box('targets', 'PIE: intention\nJAAD: action', 898, 249, 360, 74, textstyle+'fontSize=27;')
for id, source, target in [('flow-in','inputs','network'),('flow-out','network','output')]:
    cell=ET.SubElement(root,'mxCell',id=id,style='edgeStyle=orthogonalEdgeStyle;rounded=1;jettySize=16;html=1;strokeColor=#111111;strokeWidth=2;endArrow=block;endFill=1;exitX=1;exitY=0.5;entryX=0;entryY=0.5;',edge='1',parent='1',source=source,target=target)
    ET.SubElement(cell,'mxGeometry',relative='1',**{'as':'geometry'})
ET.indent(doc)
ET.ElementTree(doc).write(ROOT/'inputs-output-minimal.drawio',encoding='utf-8',xml_declaration=True)
(ROOT/'inputs-output-minimal-provenance.json').write_text(json.dumps({'foundation':'/data/repos/SparseTemporalPIE-paper/figures/pipeline_overview_square.drawio','foundation_sha256':hashlib.sha256(Path('/data/repos/SparseTemporalPIE-paper/figures/pipeline_overview_square.drawio').read_bytes()).hexdigest(),'sources':sources,'scope':'Simplified input/output overview, no architecture internals or training-state semantics. Real PIE imagery extracted unchanged from manuscript PDF; illustrative, not model predictions. Thin borders group content, not frozen weights. White frame fixes export bounds.','preserved_caption':'Action/look supplied by dataset annotations; PIE intention and JAAD action.'},indent=2))
