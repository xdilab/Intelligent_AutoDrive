"""Create a portable interactive HTML viewer and optional MP4 from real predictions."""
import argparse
import base64
import io
import json
import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from export_sample import sha


def load_image(sample, record):
    path = sample / 'frames' / (record['key'] + '.jpg')
    if sha(path) != record['image_sha256']:
        raise ValueError('Display image does not match encoded input')
    with Image.open(path) as image:
        return image.convert('RGB')


def font(size):
    return ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf', size)


def video_frame(image, frame, data):
    result = Image.new('RGB', (1600,1000), '#060f1e')
    d = ImageDraw.Draw(result)
    d.line((36,115,1564,115),fill='#e7edf5',width=3)
    d.line((1148,142,1148,914),fill='#39495e')
    d.line((36,934,1564,934),fill='#39495e')
    d.text((36,23), 'CSEB', font=font(64), fill='#e7edf5')
    d.text((230,35), 'Contextual Scene-Attention', font=font(22), fill='#e7edf5')
    d.text((230,65), 'Expert Blend', font=font(22), fill='#e7edf5')
    d.text((1320,42), 'RECORDED DEMO', font=font(18), fill='#9baac0')
    # Preserve the full frame aspect ratio. No independent x/y scaling.
    image.thumbnail((1100,800))
    ox,oy=36,132+(800-image.height)//2
    result.paste(image,(ox,oy))
    detections = [x for x in frame['detections'] if x['confidence']>=.3]
    triplet_agents={label.split('-')[0] for label in data['labels']['triplet']}
    occupied=[]
    for i,actor in reversed(list(enumerate(detections))):
        x,y,x2,y2=actor['box']; box=(ox+x*image.width,oy+y*image.height,ox+x2*image.width,oy+y2*image.height)
        color='#f0d39a' if i==0 else '#a4d7f4'
        d.rectangle(box, outline=color, width=4 if i==0 else 2)
        if i<5:
            label=(max(actor['heads']['triplet'],key=lambda v:v['score'])['label'].replace('-', ' | ')
                   if actor['agent'] in triplet_agents else actor['agent']+' | triplet N/A')
            textwidth=d.textlength(label,font=font(16))
            tx=min(box[0],ox+image.width-textwidth-10);ty=max(oy,box[1]-24)
            while ty>=oy+27 and any(tx<px+pw+3 and tx+textwidth+13>px and ty<py+27 and ty+27>py for px,py,pw in occupied):ty-=27
            occupied.append((tx,ty,textwidth+10))
            if ty<box[1]-27:d.line((tx+8,ty+24,box[0]+8,box[1]),fill=color,width=1)
            d.rectangle((tx,ty,tx+textwidth+10,ty+24),fill='#060f1e')
            d.text((tx+5,ty+2),label,font=font(16),fill=color)
    sx=1170
    d.text((sx,145),'SELECTED DETECTION',font=font(22),fill='#9baac0')
    if detections:
        actor=detections[0]
        d.text((sx,190),actor['agent'],font=font(32),fill='#f0d39a')
        d.text((sx,238),f"YOLO confidence: {actor['confidence']:.3f}",font=font(18),fill='#e7edf5')
        y=305
        for key,title in [('action','Action'),('loc','Location'),('triplet','Composition')]:
            top=max(actor['heads'][key],key=lambda v:v['score'])
            d.text((sx,y),title,font=font(18),fill='#9baac0')
            label=top['label']; words=label.split('-');lines=['']
            for word in words:
                candidate=(lines[-1]+' - '+word).strip(' -')
                if d.textlength(candidate,font=font(19))>390:lines.append(word)
                else:lines[-1]=candidate
            for k,line in enumerate(lines):d.text((sx,y+30+k*27),line,font=font(19),fill='#e7edf5')
            d.text((sx,y+30+len(lines)*27),f"Score {top['score']:.3f}",font=font(18),fill='#f0d39a')
            y+=125
        top=max(actor['heads']['triplet'],key=lambda v:v['score'])
        d.text((sx,733),'Classification / contrastive',font=font(18),fill='#9baac0')
        d.text((sx,770),f"{top['classification']:.3f}  /  {top['contrastive']:.3f}",font=font(28),fill='#e7edf5')
        d.text((sx,820),f"Blend weight: {data['weight']:.2f}",font=font(18),fill='#f0d39a')
    else:d.text((sx,210),'No candidates above 0.30',font=font(18),fill='#e7edf5')
    mode='End-to-end video inference' if data.get('source_type')=='raw-video' else 'Inference on cached encoder features'
    d.text((36,944),f"{frame['video']} | frame {frame['fid']} | {mode}",font=font(18),fill='#9baac0')
    d.text((36,973),'Single-seed qualitative demonstration; no ground-truth labels displayed. Recorded output, not live playback.',font=font(15),fill='#9baac0')
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sample',type=Path,required=True)
    ap.add_argument('--predictions',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--mp4',action='store_true')
    ap.add_argument('--fps',type=float,default=10,help='ROAD-Waymo source video is 10 fps')
    a=ap.parse_args()
    if a.fps<=0:ap.error('fps must be positive')
    manifest=json.loads((a.sample/'manifest.json').read_text())
    data=json.loads(a.predictions.read_text())
    data['labels']=manifest['labels']
    if sha(a.sample/'manifest.json')!=data['manifest_sha256']:
        raise ValueError('Prediction/sample mismatch')
    if [x['key'] for x in data['frames']]!=[x['key'] for x in manifest['frames']]:
        raise ValueError('Frame ordering mismatch')
    a.output.mkdir(parents=True,exist_ok=False)
    video=None
    if a.mp4:
        if any(b['fid']-x['fid']!=1 for x,b in zip(data['frames'],data['frames'][1:])):
            raise ValueError('Video export requires consecutive frames; HTML supports sparse inspection')
        video=subprocess.Popen(['ffmpeg','-hide_banner','-loglevel','error','-f','rawvideo','-pix_fmt','rgb24',
            '-s','1600x1000','-r',str(a.fps),'-i','pipe:0','-an','-c:v','libx264','-preset','fast',
            '-crf','21','-pix_fmt','yuv420p','-movflags','+faststart',str(a.output/'cseb-prototype.mp4')],stdin=subprocess.PIPE)
    try:
        for i,(frame,record) in enumerate(zip(data['frames'],manifest['frames'])):
            image=load_image(a.sample,record)
            if video:
                rendered=video_frame(image.copy(),frame,data)
                video.stdin.write(rendered.tobytes())
                if i==0:rendered.save(a.output/'preview.png')
            image.thumbnail((1280,960))
            buffer=io.BytesIO();image.save(buffer,format='JPEG',quality=85)
            frame['image']='data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode()
    finally:
        if video:
            video.stdin.close()
            if video.wait()!=0:raise RuntimeError('ffmpeg failed')
    # Avoid script-tag termination from any string embedded in the data.
    payload=json.dumps(data,separators=(',',':')).replace('<','\\u003c').replace('>','\\u003e')
    template=Path(__file__).with_name('viewer.html').read_text()
    (a.output/'index.html').write_text(template.replace('__DEMO_DATA__',payload))
    receipt={'checkpoint_sha256':data['checkpoint_sha256'],'manifest_sha256':data['manifest_sha256'],
             'prediction_sha256':sha(a.predictions),'frames':len(data['frames']),'fps':a.fps,
             'artifacts':{p.name:sha(p) for p in a.output.iterdir() if p.is_file()}}
    (a.output/'receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps({'output':str(a.output),'frames':len(data['frames']),'mp4':a.mp4}))


if __name__=='__main__':main()
