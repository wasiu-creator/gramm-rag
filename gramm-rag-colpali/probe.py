import torch, sys, traceback
from colpali_engine.models import (ColQwen2, ColQwen2Processor,
                                   ColQwen2_5, ColQwen2_5_Processor,
                                   ColIdefics3, ColIdefics3Processor)
CANDS = [
    ('vidore/colqwen2.5-v0.2', ColQwen2_5, ColQwen2_5_Processor),
    ('vidore/colqwen2-v1.0-hf', ColQwen2, ColQwen2Processor),
    ('vidore/colSmol-256M',     ColIdefics3, ColIdefics3Processor),
]
for mid, cls, proc in CANDS:
    try:
        m, info = cls.from_pretrained(mid, torch_dtype=torch.bfloat16,
                                      device_map='cpu', output_loading_info=True)
        miss = [k for k in info.get('missing_keys', []) if 'lora' not in k.lower()]
        unexp = info.get('unexpected_keys', [])
        status = 'CLEAN' if not miss else f'MISSING {len(miss)}'
        print(f'{mid:32s} {status:14s} unexpected={len(unexp)}')
        if miss: print('     e.g.', miss[:3])
        del m; torch.cuda.empty_cache()
    except Exception as e:
        print(f'{mid:32s} FAILED  {type(e).__name__}: {str(e)[:100]}')
