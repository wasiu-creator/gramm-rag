"""Real preflight: rank the gold page against EVERY page of its own document.
That is what retrieval does, and unlike a pairwise test it has a meaningful
random baseline (k/n_pages)."""
import json, ast, random, torch, numpy as np
from pathlib import Path
from PIL import Image
from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
MID='vidore/colqwen2.5-v0.2'
m,info=ColQwen2_5.from_pretrained(MID,torch_dtype=torch.bfloat16,device_map='cuda:0',
                                  output_loading_info=True)
assert not [k for k in info.get('missing_keys',[]) if 'lora' not in k.lower()]
m=m.eval(); p=ColQwen2_5_Processor.from_pretrained(MID)
src=json.loads(Path('/work/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json').read_text())
qm=json.loads(Path('/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json').read_text())['rows']
IMG=Path('/work/gramm-rag-colpali/page_images_full/mmlongbench')

@torch.no_grad()
def emb_imgs(paths,bs=8):
    out=[]
    for i in range(0,len(paths),bs):
        ims=[Image.open(x).convert('RGB') for x in paths[i:i+bs]]
        b=p.process_images(ims).to('cuda:0'); e=m(**b)
        out += [e[j].float().cpu() for j in range(e.shape[0])]
        [im.close() for im in ims]
    return out

rng=random.Random(42)
# modest-length docs so the sample runs fast
cands=[i for i,r in enumerate(qm) if 8<=r['doc_pages']<=60 and ast.literal_eval(src[i]['evidence_pages'])]
rng.shuffle(cands)
by_doc={}
for i in cands:
    by_doc.setdefault(qm[i]['doc_id'],[]).append(i)
docs=list(by_doc)[:6]

hits={1:0,5:0,10:0}; tot=0; rand_exp={1:0.,5:0.,10:0.}
for doc in docs:
    r=qm[by_doc[doc][0]]; stem=doc[:-4] if doc.lower().endswith('.pdf') else doc
    paths=[IMG/f'{stem}_p{x}.png' for x in range(1,r['doc_pages']+1)]
    paths=[x for x in paths if x.exists()]
    if len(paths)<5: continue
    pe=emb_imgs(paths)
    pages=[int(x.stem.rsplit('_p',1)[1]) for x in paths]
    qs=by_doc[doc][:4]
    with torch.no_grad():
        qb=p.process_queries([src[i]['question'] for i in qs]).to('cuda:0')
        qe=m(**qb).float().cpu()
    scores=p.score_multi_vector(qe,torch.nn.utils.rnn.pad_sequence(pe,batch_first=True))
    for row,i in enumerate(qs):
        gp=[g for g in ast.literal_eval(src[i]['evidence_pages']) if 1<=g<=r['doc_pages']]
        if not gp: continue
        order=[pages[j] for j in torch.argsort(scores[row],descending=True).tolist()]
        tot+=1
        for k in hits:
            hits[k]+= any(g in order[:k] for g in gp)
            rand_exp[k]+= min(1.0, k*len(gp)/len(pages))
print(f'docs={len(docs)}  questions={tot}')
for k in (1,5,10):
    print(f'  gold-in-top{k:<3d} {hits[k]}/{tot} = {hits[k]/tot:.2f}   (random baseline {rand_exp[k]/tot:.2f})')
