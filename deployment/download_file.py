"""Resume large downloads with validated HTTP ranges and bounded concurrency."""
import argparse, concurrent.futures, datetime, os, time, urllib.request
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('url');p.add_argument('target');p.add_argument('size',type=int);a=p.parse_args()
target=Path(a.target);target.parent.mkdir(parents=True,exist_ok=True)
log=Path('/home/pupu/insPart/XD/logs/progress.log')
def note(s):
 with log.open('a') as f:f.write(datetime.datetime.now().astimezone().isoformat(timespec='seconds')+' '+s+'\n')
# Preserve existing sequential download as a valid prefix. Parts are assembled only after verification.
prefix=target.stat().st_size if target.exists() else 0
if prefix>a.size:raise RuntimeError('Target larger than expected')
if prefix==a.size:note(f'{target.name}: 已有完整大小，待校验');raise SystemExit
partdir=target.parent/(target.name+'.parts');partdir.mkdir(exist_ok=True)
chunk=16*1024*1024
jobs=[(i,s,min(s+chunk,a.size)-1) for i,s in enumerate(range(prefix,a.size,chunk))]
def get(job):
 i,start,end=job;part=partdir/str(start)
 if part.exists() and part.stat().st_size==end-start+1:return
 for attempt in range(10):
  try:
   sep='&' if '?' in a.url else '?'
   offset=part.stat().st_size if part.exists() else 0
   current=start+offset
   if current==end+1:return
   req=urllib.request.Request(a.url+sep+'san_range='+str(current),headers={'Range':f'bytes={current}-{end}'})
   with urllib.request.urlopen(req,timeout=60) as r:
    if r.status!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {current}-{end}/'):raise RuntimeError('Unexpected range response')
    with part.open('ab') as f:
     while True:
      b=r.read(1024*1024)
      if not b:break
      f.write(b)
   if part.stat().st_size!=end-start+1:raise RuntimeError('Incomplete range')
   return
  except Exception as e:
   note(f'{target.name}: 分段 {start} 重试 {attempt+1}: {e}');time.sleep(2)
 raise RuntimeError(f'Failed range {start}')
note(f'{target.name}: 启动12路分段下载，保留已有 {prefix/1e6:.1f} MB；源 {a.url}')
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 for _ in pool.map(get,jobs):
  total=prefix+sum(x.stat().st_size for x in partdir.iterdir())
  note(f'{target.name}: 已下载 {total/1e6:.1f}/{a.size/1e6:.1f} MB ({100*total/a.size:.1f}%)')
with target.open('ab') as out:
 for _,start,end in jobs:
  part=partdir/str(start)
  with part.open('rb') as inp:
   while True:
    b=inp.read(1024*1024)
    if not b:break
    out.write(b)
  part.unlink()
partdir.rmdir()
assert target.stat().st_size==a.size
note(f'{target.name}: 下载完成，等待 SHA256/加载校验')
