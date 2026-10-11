"""Synchronize the audited result table, compile, and check main-text limit."""
from pathlib import Path
import subprocess
from pypdf import PdfReader
root=Path(__file__).resolve().parent
p=root/'paper'/'additional.tex';s=p.read_text()
start=s.index('\\begin{tabular}{llrrrr}')
end=s.index('\\end{tabular}',start)
head='\\begin{tabular}{llrrrr}\\toprule\nPlant $(n,m)$ & Method & Mean gap & 95\\% interval & Data & ms\\\\\\midrule\n'
s=s[:start]+head+(root/'paper'/'results_table.tex').read_text().strip()+'\n\\bottomrule'+s[end:]
p.write_text(s)
subprocess.run(['tectonic','main.tex'],cwd=root/'paper',check=True)
r=PdfReader(root/'paper'/'main.pdf')
ref=next(i for i,p in enumerate(r.pages) if 'References' in p.extract_text())
assert ref<=10,f'{ref} main pages exceeds conference limit'
print(f'{ref} main pages; {len(r.pages)} total pages')
