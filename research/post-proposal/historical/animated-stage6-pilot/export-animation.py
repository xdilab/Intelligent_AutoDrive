from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from diagram_geometry import rounded_route
from PIL import Image, ImageDraw, ImageFilter, ImageColor
import json,math
A=Path(__file__).resolve().parent
base=Image.open(A/'stage6-static.png').convert('RGB').resize((1800,1088),Image.Resampling.LANCZOS)
routes=json.loads((A/'animation-routes.json').read_text());scale=.75
for route in routes:route['points']=rounded_route(route['points'])[1]
badges=json.loads((A/'animation-badges.json').read_text())
borders=json.loads((A/'animation-borders.json').read_text())
panels=json.loads((A/'animation-panels.json').read_text())
frames=[]
def point_at(points,f):
 lengths=[math.dist(a,b) for a,b in zip(points,points[1:])];remain=f*sum(lengths)
 for a,b,l in zip(points,points[1:],lengths):
  if remain<=l:return (a[0]+(b[0]-a[0])*remain/l,a[1]+(b[1]-a[1])*remain/l)
  remain-=l
 return points[-1]
for frame in range(180):
 t=frame/10;im=base.copy()
 section=Image.new('RGBA',im.size);sd=ImageDraw.Draw(section)
 for b in panels:
  start,end=b['window'];alpha=.8*max(0,min(1,(t-start)/.25,(end-t)/.25))
  if alpha<=0:continue
  x,y,w,h=[b[v]*scale for v in ['x','y','w','h']]
  sd.rounded_rectangle((x,y,x+w,y+h),radius=11,outline=(233,173,53,int(255*alpha)),width=2)
 im=Image.alpha_composite(im.convert('RGBA'),section.filter(ImageFilter.GaussianBlur(3)))
 im=Image.alpha_composite(im,section).convert('RGB')
 outline=Image.new('RGBA',im.size);od=ImageDraw.Draw(outline)
 for b in borders:
  start,end=b['window'];f=(t-start)/(end-start)
  if not 0<=f<=1:continue
  knots=[0,1,.25,1,0];k=min(3,int(f*4));u=f*4-k;alpha=knots[k]*(1-u)+knots[k+1]*u
  color=(*ImageColor.getrgb(b['glow_color']),int(alpha*255));x,y,w,h=[b[v]*scale for v in ['x','y','w','h']]
  if b['polygon']:
   pad=min(25,b['w']*.18)*scale;od.line([(x+pad,y),(x+w,y),(x+w-pad,y+h),(x,y+h),(x+pad,y)],fill=color,width=4,joint='curve')
  else:od.rounded_rectangle((x,y,x+w,y+h),radius=11,outline=color,width=4)
 im=Image.alpha_composite(im.convert('RGBA'),outline.filter(ImageFilter.GaussianBlur(3)))
 im=Image.alpha_composite(im,outline).convert('RGB')
 glow=Image.new('RGBA',im.size);gd=ImageDraw.Draw(glow)
 for b in badges:
  if "window" not in b:continue
  start,end=b["window"];strength=.85*max(0,min(1,(t-start)/.25,(end-t)/.25))
  x=(b['x']+b['size']/2)*scale;y=(b['y']+b['size']/2)*scale;r=b['size']*.52*scale
  rgb=(57,168,237) if b['kind']=='frozen' else (255,152,30)
  gd.ellipse((x-r,y-r,x+r,y+r),fill=(*rgb,int(255*strength)))
 im=Image.alpha_composite(im.convert('RGBA'),glow.filter(ImageFilter.GaussianBlur(3))).convert('RGB');d=ImageDraw.Draw(im)
 rings=Image.new('RGBA',im.size);rd=ImageDraw.Draw(rings)
 for b in badges:
  if 'window' not in b:continue
  start,end=b['window'];strength=.85*max(0,min(1,(t-start)/.25,(end-t)/.25))
  x=(b['x']+b['size']/2)*scale;y=(b['y']+b['size']/2)*scale
  r=b['size']*(.48+.27*(1-abs(2*(t%1)-1)))*scale
  rgb=(57,168,237) if b['kind']=='frozen' else (255,152,30)
  rd.ellipse((x-r,y-r,x+r,y+r),outline=(*rgb,int(255*strength)),width=2)
 im=Image.alpha_composite(im.convert('RGBA'),rings).convert('RGB');d=ImageDraw.Draw(im)
 hot=Image.new('RGBA',im.size);hd=ImageDraw.Draw(hot)
 for r in routes:
  start=r['phase']*1.5;end=start+r['duration'];alpha=.95*max(0,min(1,(t-start)/.06,(end-t)/.06))
  if alpha<=0:continue
  pts=[(x*scale,y*scale) for x,y in r['points']];color=(255,152,30,int(255*alpha))
  hd.line(pts,fill=color,width=3,joint='curve')
  x,y=pts[-1];xx,yy=pts[-2];theta=math.atan2(y-yy,x-xx)
  bx,by=x-22.5*scale*math.cos(theta),y-22.5*scale*math.sin(theta);dx,dy=10*scale*-math.sin(theta),10*scale*math.cos(theta)
  hd.polygon([(x,y),(bx+dx,by+dy),(bx-dx,by-dy)],fill=color)
 im=Image.alpha_composite(im.convert('RGBA'),hot.filter(ImageFilter.GaussianBlur(2.25)))
 im=Image.alpha_composite(im,hot).convert('RGB')
 packetglow=Image.new('RGBA',im.size);pg=ImageDraw.Draw(packetglow)
 active=[]
 for r in routes:
  f=(t-r['phase']*1.5)/r.get('duration',1.3)
  if 0<=f<=1:
   x,y=point_at(r['points'],f);x*=scale;y*=scale
   active.append((x,y));pg.ellipse((x-9,y-9,x+9,y+9),fill=(253,185,39,210))

 im=Image.alpha_composite(im.convert('RGBA'),packetglow.filter(ImageFilter.GaussianBlur(4))).convert('RGB');d=ImageDraw.Draw(im)
 for x,y in active:d.ellipse((x-6,y-6,x+6,y+6),fill='#FDB927',outline='#fff2be',width=2)
 if 4.5<=t<=8.1 and int(t*4)%2==0:
  for i in range(3):
   xx=(1440+78*i)*scale;yy=(950+78*i)*scale
   d.rounded_rectangle((xx,yy,xx+68*scale,yy+68*scale),radius=8,outline='#4f8847',width=4)
 frames.append(im)
palette=frames[55].quantize(colors=256)
indexed=[im.quantize(palette=palette,dither=Image.Dither.NONE) for im in frames]
indexed[0].save(A/'stage6-animated.gif',save_all=True,append_images=indexed[1:],duration=100,loop=0,optimize=True,disposal=1)
frames[55].save(A/'animation-frame.png')
gif=Image.open(A/'stage6-animated.gif');assert gif.is_animated and gif.n_frames>50
print({'gif_frames':gif.n_frames,'loop':gif.info.get('loop'),'size':gif.size,'bytes':(A/'stage6-animated.gif').stat().st_size})
