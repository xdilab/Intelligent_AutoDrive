import json,pathlib
W=pathlib.Path('artifacts/defense-proposal-opening-20260928');d=json.load(open(W/'output.json'))['structuredContent'];es={e['objectId']:e for s in d['slides'] for e in s['pageElements']};r=[]
blue={'green':.22,'blue':.42};gold={'red':.99215686,'green':.7254902,'blue':.15294118}
r.append({'updatePageProperties':{'objectId':'def_01','pageProperties':{'pageBackgroundFill':{'solidFill':{'color':{'rgbColor':blue}}}},'fields':'pageBackgroundFill'}})
for oid,y,h,c in [('prop01_logo_band',372,78,{'red':1,'green':1,'blue':1}),('prop01_footer_band',452,88,gold)]:
 r.extend([{'createShape':{'objectId':oid,'shapeType':'RECTANGLE','elementProperties':{'pageObjectId':'def_01','size':{'width':{'magnitude':960,'unit':'PT'},'height':{'magnitude':h,'unit':'PT'}},'transform':{'scaleX':1,'scaleY':1,'translateY':y,'unit':'PT'}}}},{'updateShapeProperties':{'objectId':oid,'shapeProperties':{'shapeBackgroundFill':{'solidFill':{'color':{'rgbColor':c}}},'outline':{'propertyState':'NOT_RENDERED'}},'fields':'shapeBackgroundFill,outline'}},{'updatePageElementsZOrder':{'pageElementObjectIds':[oid],'operation':'SEND_TO_BACK'}}])
r.append({'deleteObject':{'objectId':'d01_p1_i6'}})
def pos(oid,x,y,w,h):
 e=es[oid];sz=e['size'];r.append({'updatePageElementTransform':{'objectId':oid,'applyMode':'ABSOLUTE','transform':{'scaleX':w*12700/sz['width']['magnitude'],'scaleY':h*12700/sz['height']['magnitude'],'translateX':x,'translateY':y,'unit':'PT'}}})
for j,oid in enumerate(['prop11_g3fa38a5ebe3_1_6','prop11_g3fa38a5ebe3_1_5','prop11_g3fa38a5ebe3_1_4','prop11_g3f8a6883d4b_0_57']):pos(oid,635+j*6,151-j*6,145,108.75)
pos('prop11_g3f8a6883d4b_0_59',625,110,185,25)
pos('prop11_g3f8a6883d4b_0_52',695,265,15,28)
pos('prop11_g3f8a6883d4b_0_53',623,300,160,55)
pos('prop11_g3f8a6883d4b_0_54',695,360,15,28)
pos('prop11_g3f8a6883d4b_0_55',596,399,215,65)
for oid,e in es.items():
 if 'thesis_cite' in oid:pos(oid,75,462,805,22)
r.append({'updatePageElementTransform':{'objectId':'rev12_stage4_image','applyMode':'ABSOLUTE','transform':es['r13_def_09_image']['transform']}})
(W/'repair-requests.json').write_text(json.dumps(r));print(len(r))
