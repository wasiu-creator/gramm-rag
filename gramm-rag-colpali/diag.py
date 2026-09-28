import json, random, torch, numpy as np
from pathlib import Path
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
MID='vidore/colqwen2.5-v0.2'
m=ColQwen2_5.from_pretrained(MID, torch_dtype=torch.bfloat16, device_map='cuda:0').eval()
p=ColQwen2_5_Processor.from_pretrained(MID)
src=json.loads(Path('/work/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json').read_text())
qm=json.loads(Path('/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json').read_text())['rows']
IMG=Path('/work/gramm-rag-colpali/page_images_full/mmlongbench')
import ast
# how many gold pages exceed the doc's page count?
bad=0; tot=0
for i,r in enumerate(qm):
    gp=ast.literal_eval(src[i]['evidence_pages'])
    for g in gp:
        tot+=1
        if g>r['doc_pages'] or g<1: bad+=1
print(f'gold page numbers out of range: {bad}/{tot}')

rng=random.Random(0)
cands=[i for i,r in enumerate(qm) if r['doc_pages']>4 and ast.literal_eval(src[i]['evidence_pages'])]
rng.shuffle(cands)
off_win=[0,0,0]  # official scorer wins at offset -1,0,+1
n=0
for i in cands:
    if n>=15: break
    r=qm[i]; doc=r['doc_id']; stem=doc[:-4] if doc.lower().endswith('.pdf') else doc
    gp=[g for g in ast.literal_eval(src[i]['evidence_pages']) if 1<=g<=r['doc_pages']]
    if not gp: continue
    others=[x for x in range(1,r['doc_pages']+1) if x not in gp]
    if not others: continue
    qb=p.process_queries([src[i]['question']]).to('cuda:0')
    with torch.no_grad(): qe=m(**qb)
    for k,off in enumerate((-1,0,1)):
        gpg=gp[0]+off
        f=IMG/f'{stem}_p{gpg}.png'; rf=IMG/f'{stem}_p{rng.choice(others)}.png'
        if not f.exists() or not rf.exists(): continue
        ims=[Image.open(x).convert('RGB') for x in (f,rf)]
        ib=p.process_images(ims).to('cuda:0')
        with torch.no_grad(): ie=m(**ib)
        s=p.score_multi_vector(qe.float().cpu(), ie.float().cpu())[0]
        if s[0].item()>s[1].item(): off_win[k]+=1
    n+=1
print(f'official score_multi_vector, gold-beats-random over {n} qs:')
for k,off in enumerate((-1,0,1)):
    print(f'   page offset {off:+d}: {off_win[k]}/{n}')
