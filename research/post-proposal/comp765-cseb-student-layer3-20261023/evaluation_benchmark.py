"""Complete two official video prefixes, preserve scores for the full run; no AP."""
from common import *
import subprocess
if __name__=='__main__':
 verify();subprocess.run([sys.executable,str(HERE/'evaluate.py'),'--benchmark-videos','2'],check=True)
