"""Verify shared source bytes against the original study provenance."""
from common import *
def main():
 original=read(ORIGINAL_ROOT/'provenance.json');rows={}
 for key,row in original['inputs'].items():
  assert sha(row['path'])==row['sha256'],key;rows[key]=row
 save(ROOT/'shared-assets-audit.json',dict(time=time.time(),passed=True,original_provenance_sha256=sha(ORIGINAL_ROOT/'provenance.json'),verified_original_assets=rows,new_backbone=dict(path=str(WEIGHTS),sha256=sha(WEIGHTS)),note='Original ResNet backbone retained only as baseline; EfficientNet backbone is separate. Shared candidate/GT/annotation and teacher provenance unchanged.'));print('SHARED ASSETS VERIFIED')
if __name__=='__main__':main()
