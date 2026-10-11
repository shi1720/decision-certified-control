"""At most two independent local benchmark processes; no external services."""
import subprocess,sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
ROOT=Path(__file__).parent
NAMES=['thermal4','mechanical4','nonnormal6','dense8']
def run(name):
 out=ROOT/'results'/name;out.mkdir(parents=True,exist_ok=True)
 with (out/'progress.log').open('w') as f:
  subprocess.run([sys.executable,str(ROOT/'run_study.py'),'--plant',name,'--out','results/'+name],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
 print(name,'finished',flush=True)
if __name__=='__main__':
 with ThreadPoolExecutor(max_workers=2) as ex:list(ex.map(run,NAMES))
