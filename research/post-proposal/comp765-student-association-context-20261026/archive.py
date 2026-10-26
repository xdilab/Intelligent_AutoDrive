from assoc_common import *
import shutil

def main():
 assert read(ROOT/'final-audit.json')['passed'];dest=HERE/'results';assert not dest.exists();dest.mkdir()
 paths=list(ROOT.glob('*.json'))+list(ROOT.glob('*.csv'))
 for sub in ['runs','evaluation-inputs','monitor','logs','analysis']:
  paths.extend(x for x in (ROOT/sub).rglob('*') if x.is_file())
 paths.extend((ROOT/'runtime').glob('*/report.json'))
 for src in paths:
  rel=src.relative_to(ROOT);target=dest/rel;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(src,target)
 shutil.copyfile(ROOT/'analysis/FINDINGS.md',dest/'FINDINGS.md')
 save(dest/'archive-manifest.json',dict(time=time.time(),source_root=str(ROOT),files={str(x.relative_to(dest)):sha(x) for x in dest.rglob('*') if x.is_file()},omitted='Large ephemeral score arrays and compressed raw-video predictions; score hashes retained, raw-video predictions remain in live root. Original model assets never copied/modified.'))
 print('Archived',len(paths),'files',flush=True)
if __name__=='__main__':main()
