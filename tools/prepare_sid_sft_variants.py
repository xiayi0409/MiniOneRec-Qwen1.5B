#!/usr/bin/env python3
"""Materialize comparable MiniOneRec SFT inputs for SID variants.
All variants preserve their 3 quantizer codes.  Only collisions receive an
extra ordinal fourth token, following the official constrained/plus exporter.
"""
import argparse, json, shutil
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
DATASET='Industrial_and_Scientific'
SRC=ROOT/'data/Amazon/index'
VARIANTS={
 'faiss_rqkmeans': ('npy', ROOT/'data/rerun_faiss/Industrial_and_Scientific/Industrial_and_Scientific.raw_codes.npy', False),
 'constrained_rqkmeans': ('json', ROOT/'data/rerun_constrained/index/Industrial_and_Scientific.index.json', True),
 'sinkhorn_rqvae': ('npy', ROOT/'data/sinkhorn_rqvae_raw_codes.npy', True),
 'cvq_rqvae': ('npy', ROOT/'data/cvq_rqvae_raw_codes.npy', True),
 'sinkhorn_dead_reset': ('npy', ROOT/'data/sinkhorn_dead_reset_raw_codes.npy', True),
 'faiss_sinkhorn': ('json', ROOT/'data/rerun_faiss_sinkhorn/Industrial_and_Scientific/Industrial_and_Scientific.faiss-rq.index.json', False),
}

def parse_token(t): return int(t.rsplit('_',1)[1][:-1])
def load_codes(kind,path,already_one):
 if kind=='npy':
  x=np.load(path).astype(int)
  return x if already_one else x+1
 raw=json.loads(path.read_text())
 x=np.asarray([[parse_token(t) for t in raw[str(i)][:3]] for i in range(len(raw))],dtype=int)
 return x if already_one else x+1

def dedup_tokens(codes):
 groups=defaultdict(list)
 for i,row in enumerate(codes.tolist()): groups[tuple(row)].append(i)
 out=[]
 for i,row in enumerate(codes.tolist()):
  tokens=[f'<{chr(97+j)}_{v}>' for j,v in enumerate(row)]
  g=groups[tuple(row)]
  if len(g)>1: tokens.append(f'<d_{g.index(i)+1}>')
  out.append(tokens)
 return {str(i):t for i,t in enumerate(out)}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--only',nargs='*'); args=ap.parse_args()
 names=args.only or list(VARIANTS)
 manifest={}
 for name in names:
  kind,path,already_one=VARIANTS[name]
  if not path.exists(): raise FileNotFoundError(path)
  outroot=ROOT/'data'/f'sid_sft_{name}'; idx=outroot/'index'; idx.mkdir(parents=True,exist_ok=True)
  for suffix in ['item.json','train.inter','valid.inter','test.inter']:
   shutil.copy2(SRC/f'{DATASET}.{suffix}',idx/f'{DATASET}.{suffix}')
  codes=load_codes(kind,path,already_one)
  mapping=dedup_tokens(codes)
  assert len(mapping)==len(codes)==3686 and len({tuple(v) for v in mapping.values()})==len(mapping)
  (idx/f'{DATASET}.index.json').write_text(json.dumps(mapping,ensure_ascii=False,indent=2))
  rawc=Counter(map(tuple,codes.tolist()))
  manifest[name]={
   'source':str(path), 'n_items':len(codes), 'raw_unique_sid':len(rawc),
   'raw_collision_rate':1-len(rawc)/len(codes), 'final_unique_sid':len(mapping),
   'extra_disambiguation_tokens':sum(len(v)==4 for v in mapping.values()),
   'data_root':str(outroot),
  }
  print(name,manifest[name])
 (ROOT/'results'/'sid_sft_variant_manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
if __name__=='__main__': main()
