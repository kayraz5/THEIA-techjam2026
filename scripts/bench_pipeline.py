import time, torch, sys
from torch.utils.data import DataLoader
from src.common import load_config, get_device, build_backbone
from src.data import build_train
from src.degradation import RandomDegradation
from src.features import ImageDS
if __name__ == "__main__":
    cfg = load_config("configs/frozen_siglip2_giant.yaml"); dev = get_device()
    s = build_train(cfg)[:192]
    ds = ImageDS(s, 384, (0.5,)*3, (0.5,)*3, degrade=RandomDegradation(seed=0), seed=0)
    t=time.time(); n=0
    for x,y,i,p in DataLoader(ds, batch_size=32, num_workers=6): n+=len(y)
    print(f"dataloader only, workers=6: {n/(time.time()-t):.1f} img/s", flush=True)
    bb = build_backbone(cfg, dev)
    x,y,i,p = next(iter(DataLoader(ds, batch_size=32)))
    with torch.no_grad():
        bb(x.to(dev, torch.bfloat16)); torch.mps.synchronize(); t=time.time()
        for _ in range(3): bb(x.to(dev, torch.bfloat16))
        torch.mps.synchronize()
    print(f"GPU only, real batch bs32: {96/(time.time()-t):.1f} img/s", flush=True)
    for w in (6, 2):
        t=time.time(); n=0
        with torch.no_grad():
            for x,y,i,p in DataLoader(ds, batch_size=32, num_workers=w):
                f=bb(x.to(dev, torch.bfloat16)).float().cpu(); n+=len(y)
        print(f"combined workers={w}: {n/(time.time()-t):.1f} img/s", flush=True)
