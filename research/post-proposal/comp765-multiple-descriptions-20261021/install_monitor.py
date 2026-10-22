"""Install study-specific persistent user units; never enables desktop banners."""
import argparse, subprocess, sys
from common import *

def main(root):
    units=Path.home()/'.config/systemd/user';units.mkdir(parents=True,exist_ok=True)
    prefix='research-comp765-descriptions'
    environment='Environment=CUDA_VISIBLE_DEVICES=0\nEnvironment=CUBLAS_WORKSPACE_CONFIG=:4096:8\nEnvironment=PYTHONDONTWRITEBYTECODE=1\nEnvironment=OMP_NUM_THREADS=4\nEnvironment=MKL_NUM_THREADS=4\nEnvironment=PATH=/data/repos/ROAD_Reason/research/post-proposal/monitoring/silent-bin:/usr/bin:/bin\n'
    (units/f'{prefix}.service').write_text(f'[Unit]\nDescription=COMP765 multiple class descriptions controlled study\n[Service]\nType=simple\nWorkingDirectory={HERE}\n{environment}ExecStart={sys.executable} -u {HERE}/pipeline.py --root {root}\nStandardOutput=append:{root}/logs/pipeline.log\nStandardError=append:{root}/logs/pipeline.log\n')
    (units/f'{prefix}-monitor.service').write_text(f'[Unit]\nDescription=Silent COMP765 log and checkpoint monitor\n[Service]\nType=oneshot\n{environment}ExecStart={sys.executable} {HERE}/monitor.py --root {root}\n')
    (units/f'{prefix}-monitor.timer').write_text(f'[Unit]\nDescription=Check COMP765 progress every2minutes\n[Timer]\nOnBootSec=30\nOnUnitActiveSec=120\nAccuracySec=5\nUnit={prefix}-monitor.service\n[Install]\nWantedBy=timers.target\n')
    subprocess.run(['systemctl','--user','daemon-reload'],check=True)
    subprocess.run([sys.executable,str(HERE/'monitor.py'),'--root',str(root),'--self-test'],check=True)
    subprocess.run(['systemctl','--user','enable','--now',f'{prefix}-monitor.timer'],check=True)
    subprocess.run(['systemctl','--user','start',f'{prefix}-monitor.service'],check=True)
    registration=WIKI/'artifacts/research-monitor/local-studies/comp765-multiple-descriptions-20261021.json'
    save(registration,dict(study='COMP765 multiple descriptions',root=str(root),service=f'{prefix}.service',timer=f'{prefix}-monitor.timer',heartbeat=str(root/'monitor/heartbeat.json'),alerts=str(root/'monitor/alerts.json'),checkpoints=str(root/'monitor/checkpoints.json'),desktop_notifications=False,automatic_agent_repair=False,time=time.time()))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=DEFAULT_ROOT);a=p.parse_args();main(a.root)
