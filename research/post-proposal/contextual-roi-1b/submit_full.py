"""Submit the complete authorized study DAG with fail-closed dependencies."""
from pathlib import Path
import json,subprocess,time

ROOT=Path('/work/bbyrd1/contextual-roi-1b-20260930')

def main():
    record=ROOT/'results/full-launch.json';assert not record.exists(),'Existing launch: inspect/resume it; do not duplicate jobs.'
    for name in ['assets.json','benchmark.json','text-benchmark.json']:assert json.loads((ROOT/'results'/name).read_text())['passed']
    jobs={};old=Path('/work/bbyrd1/stage56-full-20260914/results')
    def submit(name,gpu=False,dependency=None,array=None,hours=24):
        label='full56-1b-'+name;cmd=['sbatch','--parsable','--job-name='+label,'--cpus-per-task=8','--mem=96G',f'--time={hours}:00:00','--output='+str(ROOT/'results'/f'{label}-%A-%a.log')]
        if gpu:cmd+=['--partition=gpu-hp','--qos=ncat_h200_hp','--gres=gpu:h200:1']
        else:cmd+=['--partition=common']
        if dependency:cmd+=['--dependency=afterok:'+':'.join(dependency)]
        if array:cmd+=['--array='+array]
        cmd+=[str(ROOT/'code/run_task.sh'),name]
        job=subprocess.check_output(cmd,text=True).strip().split(';')[0];assert job.isdigit();jobs[name]=job
        (old/f'1b-full-{name}-job.txt').write_text(job+'\n')
        tasks=range(int(array.split('-')[1].split('%')[0])+1) if array else [4294967294]
        for task in tasks:
            logfile=f'{label}-{job}-{task}.log';(old/logfile).symlink_to(ROOT/'results'/logfile)
        record.write_text(json.dumps({'root':str(ROOT),'jobs':jobs,'time':time.time(),'complete_submission':False},indent=2))
        print('SUBMITTED',name,job,flush=True);return job
    prepare=submit('prepare',hours=8)
    cache=submit('cache',True,[prepare],'0-7%8',48)
    compact=submit('compact',False,[cache],hours=12)
    preflight=submit('preflight',True,[compact],hours=2)
    train=submit('train',True,[preflight],'0-5%6',48)
    blend=submit('blend',True,[train],'0-2%3',8)
    evaluate=submit('evaluate',True,[blend],'0-8%4',24)
    submit('summarize',False,[evaluate],hours=1)
    record.write_text(json.dumps({'root':str(ROOT),'jobs':jobs,'time':time.time(),'complete_submission':True,
        'eta':'Provisional input restoration/verification1–4h, cache12–24h, first dev result16–30h, all detector results24–48h plus queue; replace with measured phase throughput.'},indent=2))

if __name__=='__main__':main()
