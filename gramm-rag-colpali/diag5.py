import json, ast, random, torch
from pathlib import Path
from PIL import Image
def _rs(im, maxside=1024):
    w,h=im.size; s=maxside/max(w,h)
    return im.resize((max(1,int(w*s)),max(1,int(h*s))), Image.LANCZOS) if s<1 else im
from colpali_engine.models import ColQwen2 as M, ColQwen2Processor as P
MID='vidore/colqwen2-v1.0'
try:
    import peft; print('peft', peft.__version__)
except ImportError: print('peft NOT INSTALLED')

src=json.loads(Path('/work/gramm-rag-mmlongbench/data/mmlongbench/mmlongbench_doc.json').read_text())
qm=json.loads(Path('/work/gramm-rag-3benchmarks/data/mmlongbench/qmeta_mmlb.json').read_text())['rows']
IMG=Path('/work/gramm-rag-colpali/page_images_full/mmlongbench')
p=P.from_pretrained(MID)
rng=random.Random(42)
cands=[i for i,r in enumerate(qm) if 8<=r['doc_pages']<=40 and ast.literal_eval(src[i]['evidence_pages'])]
rng.shuffle(cands)

def run(model,label):
    model=model.eval()
    ranks=[]
    for i in cands[:8]:
        r=qm[i]; doc=r['doc_id']; stem=doc[:-4] if doc.lower().endswith('.pdf') else doc
        paths=[IMG/f'{stem}_p{x}.png' for x in range(1,r['doc_pages']+1)]
        paths=[x for x in paths if x.exists()]
        if len(paths)<5: continue
        pe=[]
        with torch.no_grad():
            for j in range(0,len(paths),2):
                ims=[_rs(Image.open(x).convert('RGB')) for x in paths[j:j+2]]
                e=model(**p.process_images(ims).to('cuda:0'))
                pe += [e[t].float().cpu() for t in range(e.shape[0])]
            qe=model(**p.process_queries([src[i]['question']]).to('cuda:0')).float().cpu()
        s=p.score_multi_vector(qe,pe)[0]
        order=[int(paths[j].stem.rsplit('_p',1)[1]) for j in torch.argsort(s,descending=True).tolist()]
        gp=[g for g in ast.literal_eval(src[i]['evidence_pages']) if g in order]
        if not gp: continue
        ranks.append((min(order.index(g)+1 for g in gp), len(order), float(s.max()-s.min())))
    top1=sum(1 for a,_,_ in ranks if a==1); top5=sum(1 for a,_,_ in ranks if a<=5)
    spread=sum(z for _,_,z in ranks)/len(ranks)
    print(f'{label:26s} n={len(ranks)} top1={top1} top5={top5} '
          f'meanrank={sum(a for a,_,_ in ranks)/len(ranks):.1f} '
          f'mean_pages={sum(b for _,b,_ in ranks)/len(ranks):.0f} score_spread={spread:.2f}')

m1=M.from_pretrained(MID,torch_dtype=torch.bfloat16,device_map='cuda:0')
run(m1,'WITHOUT loading_info'); del m1; torch.cuda.empty_cache()
m2,_=M.from_pretrained(MID,torch_dtype=torch.bfloat16,device_map='cuda:0',output_loading_info=True)
run(m2,'WITH output_loading_info')
