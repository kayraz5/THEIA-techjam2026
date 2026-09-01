# Data and model licenses

The hackathon brief requires "public / properly licensed datasets only". Every external asset this project
touches is listed below with its license as published by the source at the time of checking (2026-08-29),
what we used it for, and whether it is redistributed. **Nothing under `data/` is committed to git**; all
data is pulled by the scripts in `scripts/` from the original hosts.

## Datasets

| Asset | Host | License | Our use | Redistributed? |
|---|---|---|---|---|
| **SID_Set** | [Hugging Face `saberzl/SID_Set`](https://huggingface.co/datasets/saberzl/SID_Set) | **CC BY 4.0** | training (real + full-synthetic; "tampered" class excluded) | No |
| **WildFake** | [ModelScope `hy2628982280/WildFake`](https://modelscope.cn/datasets/hy2628982280/WildFake) | **Apache 2.0** | training slices (LAION reals, COCO test2017 reals, MJv5, six GAN families; ablation arms also used DALL-E 2, MJv4, SDXL, ADM/DDIM/DDPM, VQDM, Imagen, MAGE/MAE/VQGAN/VQVAE, FFHQ/CelebA-HQ/AFHQ/Church) and the organisers' designated **validation** subset (COCO val2017 reals + DALL-E 3 fakes) and the ImageNet alt-real set | No |
| **Community Forensics-Eval** | [Hugging Face `OwensLab/CommunityForensics-Eval`](https://huggingface.co/datasets/OwensLab/CommunityForensics-Eval) | **CC BY-NC-SA 4.0** — *"non-commercial research and educational purposes only"* | **evaluation only** — external generalization test. Never trained on; no derivative model or weights depend on it | No |
| **RRDataset** | [Zenodo record 14963880](https://zenodo.org/records/14963880) | **CC BY 4.0** | **evaluation only** — second external generalization test | No |
| CIFAKE | Kaggle | MIT (per dataset card) | disabled (`data.cifake.enabled: false`); smoke-test only | No |

Notes:

- **Community Forensics is non-commercial.** It is used purely as a held-out scorer, which is squarely
  within "research and educational purposes". Nothing shipped — no checkpoint, no training data, no
  feature cache that feeds training — is derived from it. If this project were ever commercialised, the
  Community Forensics numbers would need to be re-run on a differently-licensed external set (RRDataset
  is CC BY 4.0 and already fills that role); the shipped model itself would be unaffected.
- WildFake and Community Forensics are themselves **aggregations** of upstream sources (COCO, LAION,
  ImageNet, FFHQ, RAISE, and the generators' own outputs), each with its own terms. We rely on the
  aggregators' published licenses and cite the upstream sources they cite. COCO val2017 in particular is
  CC BY 4.0; ImageNet is research-only under its own terms and is used here solely as a held-out real set.
- SID_Set's card states it incorporates CC BY 4.0 material and commits to attribution; we attribute both
  SID_Set and its upstream sources in the README.

## Models

| Asset | License | Our use |
|---|---|---|
| `google/siglip2-giant-opt-patch16-384` (vision tower, 1.164 B params) | **Apache 2.0** | frozen backbone; weights downloaded at runtime, never committed |
| `google/siglip2-so400m-patch14-384` | Apache 2.0 | declared fallback, never triggered |
| `openai/clip-vit-large-patch14` | MIT (per OpenAI CLIP repository) | baseline comparison arm only |
| `facebook/dinov3-vitl16-pretrain-lvd1689m` | gated — access not obtained | **not used**; config exists, arm never ran |

## Demo media (`project_demo/`)

The live feed demo plays ten short clips. **None of this media is committed or redistributed.**
`project_demo/videos.json` records only where each clip comes from, and
`project_demo/fetch_videos.py` pulls from the original hosts at run time into
`project_demo/videos*/`, which are gitignored. The clips are re-encoded locally for playback
(720x1280, audio stripped) and that derived copy never leaves the machine.

| Asset | License | Our use | Redistributed? |
|---|---|---|---|
| DeepAction v1 — AI-generated clips (Veo, RunwayML, VideoPoet, CogVideoX-5B, Stable Diffusion video) | **CC BY 4.0** | the AI half of the demo feed; attribution shown on each card | No |
| DeepAction v1 — `Pexels/` real clips | **Pexels License** | the two "stock" real clips, deliberately included as the hard false-positive case | No |
| Wikimedia Commons — *Chinese New Year parade, Chinatown* | **CC BY 3.0** (China News Service) | handheld real clip | No |
| Wikimedia Commons — *Mani Rimdu festival short* | **CC BY-SA 4.0** (Rohit14400) | handheld real clip | No |
| Wikimedia Commons — *Train enters platform 1, Bialik station* | **CC BY-SA 4.0** (Assaf Yekuel) | handheld real clip | No |
| Wikimedia Commons — *Santa Apolonia CP 2320, Lisbon* | **CC BY 2.0** (_morgado) | candidate pool only, not in the shipped feed | No |

DeepAction v1 (`https://huggingface.co/datasets/faridlab/deepaction_v1`) is ungated and splits its
licensing: the six generator folders are CC BY 4.0, the `Pexels/` folder carries the Pexels
License. Both are recorded separately above because only the first is a Creative Commons grant.
Pexels and Wikimedia attribution strings are rendered on each card in the demo UI.

Two deliberate exclusions:

- **No generator showcase clips.** Marketing reels from commercial video products are not licensed
  for this use, and a visible generator watermark would make the demo solvable by OCR rather than
  by the detector. `fetch_videos.py --verify` fails a clip if `cropdetect` finds residual bars, and
  the manifest carries a `watermark_crop` field for any clip that would need one.
- **No Community Forensics imagery** (CC BY-NC-SA 4.0), consistent with the rule already stated in
  `docs/DEMO_SCRIPT.md`.

## Code

| Asset | License | Notes |
|---|---|---|
| This repository | MIT (see `LICENSE`) | |
| `third_party/aug_utils_train/` — NTIRE 2026 official distortion pipeline | original authors' terms (distributed via Codabench 12761) | vendored; one `LOCAL PATCH` for numpy ≥ 2. Not the default backend; used for the #24 ablation only |

## What we ship

The only trained artifact is `results/frozen_siglip2_giant_ship/head_best.pt` — a 16 KB linear layer
trained on SID_Set (CC BY 4.0) and WildFake (Apache 2.0) slices. Both permit derivative works. The
backbone is Apache 2.0. There is no license conflict in the shipped model.
