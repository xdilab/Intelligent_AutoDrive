"""Run the causal student on a video or camera, at its trained 10Hz cadence."""
import argparse,json,threading,time
from pathlib import Path
import cv2
import numpy as np
import torch
from inference import StudentInference,VideoReader,write_json

class LatestCamera:
    """Continuously drain capture into one frame slot to avoid queued stale frames."""
    def __init__(self,index):
        self.cap=cv2.VideoCapture(index)
        if not self.cap.isOpened():self.cap.release();raise ValueError('Camera could not be opened')
        self.condition=threading.Condition();self.frame=None;self.serial=0;self.done=False;self.stopping=False
        self.thread=threading.Thread(target=self.read,daemon=True);self.thread.start()
    def read(self):
        try:
            while not self.stopping:
                ok,frame=self.cap.read()
                if not ok:break
                with self.condition:
                    self.frame=frame;self.serial+=1;self.condition.notify_all()
        finally:
            with self.condition:self.done=True;self.condition.notify_all()
    def next(self,previous):
        with self.condition:
            ready=self.condition.wait_for(lambda:self.serial>previous or self.done,timeout=5)
            if not ready:raise TimeoutError('No fresh camera frame for5seconds')
            if self.serial==previous:return None,previous
            return cv2.cvtColor(self.frame,cv2.COLOR_BGR2RGB),self.serial
    def close(self):
        self.stopping=True;self.cap.release();self.thread.join(timeout=2)

def overlay(image,out,labels):
    frame=cv2.cvtColor(image,cv2.COLOR_RGB2BGR);h,w=frame.shape[:2]
    for box,agent,q,scores in zip(out['boxes'],out['agents'],out['confidence'],out['scores']):
        x1,y1,x2,y2=(box*np.array([w,h,w,h])).astype(int);name=labels['agent'][int(agent)]
        valid=[i for i,s in enumerate(labels['triplet']) if s.startswith(name+'-')]
        text=labels['triplet'][max(valid,key=lambda i:scores[98+i])] if valid else name
        cv2.rectangle(frame,(x1,y1),(x2,y2),(230,190,100),1)
        cv2.putText(frame,f'{text} {q:.2f}',(max(0,x1),max(16,y1-5)),cv2.FONT_HERSHEY_PLAIN,1,(230,190,100),1)
    return frame

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--bundle',type=Path,required=True);ap.add_argument('--source',required=True,help='Video path or camera index, e.g.0')
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--frames',type=int,default=0,help='0 means until end/quit')
    ap.add_argument('--display',action='store_true');ap.add_argument('--no-pace',action='store_true',help='Process video faster than playback; timestamps remain10Hz')
    a=ap.parse_args()
    if a.frames<0:ap.error('frames must be nonnegative')
    camera=a.source.isdecimal()
    if camera and a.no_pace:ap.error('Camera inference uses the trained10Hz cadence')
    a.output.mkdir(parents=True,exist_ok=False);torch.set_num_threads(8);begin=time.perf_counter()
    runner=StudentInference(a.bundle,a.output/'yolo-settings');load=time.perf_counter()-begin
    reader=LatestCamera(int(a.source)) if camera else VideoReader(Path(a.source));serial=0;count=0;timings=[];started=time.perf_counter();due=started
    try:
        with (a.output/'predictions.jsonl').open('w') as stream:
            while not a.frames or count<a.frames:
                if not a.no_pace:time.sleep(max(0,due-time.perf_counter()))
                frame_start=time.perf_counter()
                if camera:image,serial=reader.next(serial)
                else:image=reader.frame(count+1) if count<reader.sample_count else None
                if image is None:break
                torch.cuda.synchronize();infer_start=time.perf_counter();out=runner.predict_frame(image);torch.cuda.synchronize()
                inference_seconds=time.perf_counter()-infer_start
                count+=1;record={'frame':count,'source_seconds':None if camera else (count-1)/10,'captured_frame':serial if camera else count,**{k:v.tolist() for k,v in out.items()}}
                stream.write(json.dumps(record,separators=(',',':'))+'\n');stream.flush()
                if a.display:
                    cv2.imshow('CSEB student',overlay(image,out,runner.labels))
                    if cv2.waitKey(1)&255 in (27,ord('q')):break
                timings.append({'frame':count,'inference_seconds':inference_seconds,'processing_seconds':time.perf_counter()-frame_start})
                due=max(due+.1,time.perf_counter())
                if count==1 or count%10==0:print(json.dumps({'frames':count,'inference_ms':round(inference_seconds*1000,2)}),flush=True)
    except KeyboardInterrupt:pass
    finally:
        if camera:reader.close()
        else:reader.cap.release()
        if a.display:cv2.destroyAllWindows()
        write_json(a.output/'report.json',{'frames':count,'load_seconds':load,'wall_seconds':time.perf_counter()-started,'timings':timings,
                  'sampling_hz':10,'source':a.source,'camera':camera,'paced':not a.no_pace,'labels':runner.labels,'causal':True})
if __name__=='__main__':main()
