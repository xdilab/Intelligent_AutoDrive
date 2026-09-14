from pathlib import Path
from PIL import Image, ImageDraw
import json,math
A=Path(__file__).resolve().parent
base=Image.open(A/'stage5-static.png').convert('RGB').resize((1440,810),Image.Resampling.LANCZOS)
routes=json.loads((A/'animation-routes.json').read_text());scale=.75
frames=[]
def point_at(points,f):
 lengths=[math.dist(a,b) for a,b in zip(points,points[1:])];remain=f*sum(lengths)
 for a,b,l in zip(points,points[1:],lengths):
  if remain<=l:return (a[0]+(b[0]-a[0])*remain/l,a[1]+(b[1]-a[1])*remain/l)
  remain-=l
 return points[-1]
for frame in range(100):
 t=frame/10;im=base.copy();d=ImageDraw.Draw(im)
 for r in routes:
  f=(t-r['phase']*1.3)/1.2
  if 0<=f<=1:
   x,y=point_at(r['points'],f);x*=scale;y*=scale
   d.ellipse((x-7,y-7,x+7,y+7),fill='#b85450',outline='white',width=2)
 frames.append(im)
palette=frames[55].quantize(colors=256)
indexed=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
indexed[0].save(A/'stage5-animated.gif',save_all=True,append_images=indexed[1:],duration=100,loop=0,optimize=False,disposal=1)
frames[55].save(A/'animation-frame.png')
gif=Image.open(A/'stage5-animated.gif');assert gif.is_animated and gif.n_frames>50
print({'gif_frames':gif.n_frames,'loop':gif.info.get('loop'),'size':gif.size,'bytes':(A/'stage5-animated.gif').stat().st_size})
