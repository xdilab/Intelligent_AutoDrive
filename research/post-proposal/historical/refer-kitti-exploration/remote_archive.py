import io, requests

class Remote(io.RawIOBase):
 def __init__(self):self.pos=0;self.length=15813146295;self.session=requests.Session()
 def seekable(self):return True
 def seek(self,n,w=0):self.pos=n if w==0 else self.pos+n if w==1 else self.length+n;return self.pos
 def tell(self):return self.pos
 def read(self,n=-1):
  if n<0:n=self.length-self.pos
  if n==0:return b''
  end=min(self.length,self.pos+n)-1
  r=self.session.get('https://s3.eu-central-1.amazonaws.com/avg-kitti/data_tracking_image_2.zip',headers={'Range':f'bytes={self.pos}-{end}'},timeout=60);r.raise_for_status();assert r.status_code==206
  self.pos+=len(r.content);return r.content
