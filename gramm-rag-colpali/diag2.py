import json, ast, random, torch
from pathlib import Path
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
MID='vidore/colqwen2.5-v0.2'
m,info=ColQwen2_5.from_pretrained(MID,torch_dtype=torch.bfloat16,device_map='cuda:0',output_loading_info=True)
print('missing:',len(info.get('missing_keys',[])),'unexpected:',len(info.get('unexpected_keys',[])))
m=m.eval(); p=ColQwen2_5_Processor.from_pretrained(MID)
src=json.loads(Path('/work/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json').read_text())
qm=json.loads(Path('/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json').read_text())['rows']
IMG=Path('/work/gramm-rag-colpali/page_images_full/mmlongbench')

@torch.no_grad()
def emb(paths,bs=8):
    out=[]
    for i in range(0,len(paths),bs):
        ims=[Image.open(x).convert('RGB') for x in paths[i:i+bs]]
        b=p.process_images(ims).to('cuda:0'); e=m(**b)
        out += [e[j].float().cpu() for j in range(e.shape[0])]
    return out

rng=random.Random(42)
cands=[i for i,r in enumerate(qm) if 8<=r['doc_pages']<=40 and ast.literal_eval(src[i]['evidence_pages'])]
rng.shuffle(cands)
i=cands[0]; r=qm[i]; doc=r['doc_id']; stem=doc[:-4] if doc.lower().endswith('.pdf') else doc
paths=[IMG/f'{stem}_p{x}.png' for x in range(1,r['doc_pages']+1)]
paths=[x for x in paths if x.exists()]
pe=emb(paths)
print('pages',len(pe),'patch counts', sorted({t.shape[0] for t in pe}))
with torch.no_grad():
    qe=m(**p.process_queries([src[i]['question']]).to('cuda:0')).float().cpu()
gp=ast.literal_eval(src[i]['evidence_pages'])
print('question:',src[i]['question'][:90]); print('gold pages:',gp,'of',r['doc_pages'])
# A: list form (documented API)
sA=p.score_multi_vector(qe, pe)[0]
# B: padded form (what preflight_rank used)
sB=p.score_multi_vector(qe, torch.nn.utils.rnn.pad_sequence(pe,batch_first=True))[0]
for nm,s in (('list',sA),('padded',sB)):
    order=[int(paths[j].stem.rsplit('_p',1)[1]) for j in torch.argsort(s,descending=True).tolist()]
    rank=min([order.index(g)+1 for g in gp if g in order], default=-1)
    print(f'  {nm:7s} top5={order[:5]}  gold rank={rank}  score range {s.min():.2f}..{s.max():.2f}')
