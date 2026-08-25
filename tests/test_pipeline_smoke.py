"""End-to-end smoke test of train (frozen) + evaluate + predict on a tiny synthetic dataset with a stub backbone,
so pipeline bugs surface in seconds instead of after hours of feature extraction."""
import json, os, subprocess, sys
import numpy as np, torch, torch.nn as nn, yaml
from PIL import Image
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class StubBackbone(nn.Module):
    """Mean colour + std as an 8-d feature: fakes are generated bluish, reals reddish -> separable."""
    def __init__(self):
        super().__init__(); self.dummy = nn.Parameter(torch.zeros(1))
        from src.models.backbone import BackboneInfo
        self.info = BackboneInfo("stub", "stub", 1, 8, 64, (0.5,) * 3, (0.5,) * 3, 0)
    @property
    def feature_dim(self): return 8
    def forward(self, x):
        return torch.cat([x.mean((2, 3)), x.std((2, 3)), x[:, :2].amax((2, 3))], 1)


def _make(tmp, n_real=30, n_fake=50):
    rng = np.random.default_rng(0)
    def img(p, fake):
        base = np.array([60, 60, 200] if fake else [200, 80, 60]) + rng.integers(-40, 40, 3)
        a = np.clip(base + rng.integers(0, 50, (96, 128, 3)), 0, 255).astype(np.uint8)
        os.makedirs(os.path.dirname(p), exist_ok=True); Image.fromarray(a).save(p)
    wf = os.path.join(tmp, "wildfake"); os.makedirs(os.path.join(wf, "csv"))
    rows = {"real_coco": [], "dalle3": [], "real_imagenet": [], "real_laion5b": [], "dalle2": []}
    for i in range(n_real):
        p = f"./Real/coco/coco2017/val2017/img{i}.jpg"; img(os.path.join(wf, "images", p[2:]), False); rows["real_coco"].append((p, 0, 0))
        p = f"./Real/imagenet/train/n0/img{i}.jpg"; img(os.path.join(wf, "images", p[2:]), False); rows["real_imagenet"].append((p, 0, 0))
        p = f"./Real/laion5b/imgs/{i}.jpg"; img(os.path.join(wf, "images", p[2:]), False); rows["real_laion5b"].append((p, 0, 0))
    for i in range(n_fake):
        p = f"./Diffusion_based/DALLE/Advanced/DALLE3/dalle3/x/{i}.jpg"; img(os.path.join(wf, "images", p[2:]), True); rows["dalle3"].append((p, 1, 1))
        p = f"./Diffusion_based/DALLE/Typical/DALLE2/dalle/{i}.png"; img(os.path.join(wf, "images", p[2:]), True); rows["dalle2"].append((p, 0, 1))
    for k, v in rows.items():
        with open(os.path.join(wf, "csv", f"{k}.csv"), "w") as f:
            f.write("Generator,Architecture,Weight,Category,IsAdvanced,IsFake,Image_path,Num\n")
            for j, (p, adv, fk) in enumerate(v): f.write(f"g,a,w,c,{adv},{fk},{p},{j}\n")
    cfg = yaml.safe_load(open(os.path.join(ROOT, "configs/frozen_siglip2_giant.yaml")))
    cfg["data"]["sources"] = ["wildfake"]; cfg["data"]["wildfake"]["root"] = wf; cfg["data"]["wildfake"]["max_per_subset"] = 0
    cfg["data"]["validation"]["exclusion_list"] = os.path.join(tmp, "excl.txt"); cfg["data"]["alt_real"]["n"] = 20
    cfg["features"]["cache_dir"] = os.path.join(tmp, "feat"); cfg["output_dir"] = os.path.join(tmp, "out")
    cfg["train"].update(epochs=40, lr=0.05, batch_size=16, feature_epochs=1); cfg["eval"].update(batch_size=16, n_eval_max=40)
    return cfg, wf


def test_end_to_end(tmp_path, monkeypatch):
    cfg, wf = _make(str(tmp_path))
    import src.common, src.train, src.evaluate
    monkeypatch.setattr(src.common, "build_backbone", lambda c, d: StubBackbone().to(d))
    monkeypatch.setattr(src.train, "build_backbone", lambda c, d: StubBackbone().to(d))
    monkeypatch.setattr(src.evaluate, "build_backbone", lambda c, d: StubBackbone().to(d))
    monkeypatch.setattr(src.common, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(src.train, "get_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(src.evaluate, "get_device", lambda: torch.device("cpu"))
    monkeypatch.chdir(tmp_path); os.makedirs("data/exclusion", exist_ok=True)
    cfgp = str(tmp_path / "cfg.yaml"); yaml.safe_dump(cfg, open(cfgp, "w"))
    # build_val
    from src.data.registry import wildfake_validation, wildfake_alt_real
    from src.data.exclusion import build_exclusion_list, ExclusionList
    val = wildfake_validation(wf); alt = wildfake_alt_real(wf, n=20)
    assert len(val) == 80 and len(alt) == 20
    build_exclusion_list([s.path for s in val + alt], cfg["data"]["validation"]["exclusion_list"])
    # leakage check must fire on a validation image
    with pytest.raises(AssertionError):
        ExclusionList(cfg["data"]["validation"]["exclusion_list"]).assert_disjoint([val[0].path], "leak")
    monkeypatch.setattr(sys, "argv", ["train", "--config", cfgp]); src.train.main()
    summ = json.load(open(os.path.join(cfg["output_dir"], "train_summary.json")))
    assert summ["clean_auc_coco_real"] > 0.9, summ
    monkeypatch.setattr(sys, "argv", ["evaluate", "--config", cfgp]); src.evaluate.main()
    ev = json.load(open(os.path.join(cfg["output_dir"], "eval", "summary.json")))
    assert ev["clean_auc_full_set"] > 0.9 and os.path.exists(os.path.join(cfg["output_dir"], "eval", "auc_grid.png"))
    assert os.path.exists(os.path.join(cfg["output_dir"], "eval", "errors_fp.png"))
    # predict.py
    import predict
    monkeypatch.setattr(predict, "build_backbone", lambda c, d: StubBackbone().to(d)); monkeypatch.setattr(predict, "get_device", lambda: torch.device("cpu"))
    open(os.path.join(wf, "images", "Real", "corrupt.jpg"), "wb").write(b"not an image")
    monkeypatch.setattr(sys, "argv", ["predict", "--image_dir", os.path.join(wf, "images", "Real"), "--output", str(tmp_path / "p.json"), "--config", cfgp, "--workers", "0"])
    predict.main()
    out = json.load(open(tmp_path / "p.json")); assert len(out) == 90 and all(0 <= r["pred"] <= 1 for r in out)
