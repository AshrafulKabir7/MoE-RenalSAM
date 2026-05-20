import subprocess, sys
for pkg in ['tqdm', 'einops']:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])
print('✅ Packages ready')
import os, sys, json, math, time, copy, random, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from collections import defaultdict
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings('ignore')

SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'✅ PyTorch {torch.__version__} | {DEVICE}')
if torch.cuda.is_available():
    print(f'   GPU: {torch.cuda.get_device_name()} | {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')
PROJECT_ROOT = Path(r'D:\MoE-RenalSAM-CG\MoE-RenalSAM-CG')
SAM2_ROOT    = PROJECT_ROOT / 'segment-anything-2'
sys.path.insert(0, str(SAM2_ROOT))

MASKS_ROOT   = PROJECT_ROOT / 'data' / 'processed' / 'renseg_masks'
SPLITS_DIR   = PROJECT_ROOT / 'data' / 'splits'
PSEUDO_DIR   = PROJECT_ROOT / 'data' / 'processed' / 'pseudo_gt_v3'   # reuse v5's pseudo-GT
CKPT_DIR     = PROJECT_ROOT / 'checkpoints' / 'moe_renalsam_cg_v6'    # NEW dir to keep v5 ckpts safe
LOG_DIR      = PROJECT_ROOT / 'logs' / 'moe_renalsam_cg_v6'

for d in [PSEUDO_DIR, CKPT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)
for cls in ['Normal', 'Tumor', 'Stone']:
    (PSEUDO_DIR / cls).mkdir(exist_ok=True)

SAM2_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
SAM2_CKPT   = str(PROJECT_ROOT / 'weights' / 'sam2_hiera_large.pt')

CLASSES     = ['Normal', 'Tumor', 'Stone']
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}
print('✅ Paths configured')
CFG = {
    'img_size':          512,
    'sam_img_size':      512,

    # LoRA
    'lora_rank':         16,
    'lora_alpha':        32,
    'lora_dropout':      0.05,

    # MoE — class-conditional
    'num_experts':       4,
    'top_k_experts':     2,
    'expert_hidden':     256,        # reduced (multi-scale already gives capacity)
    'class_emb_dim':     32,         # class embedding fed to router
    'load_balance_alpha':0.01,

    # APG (stays the same)
    'apg_channels':      [32, 64, 128, 256],

    # SCRL v7 weights
    'lambda_focal':      1.0,
    'lambda_tversky':    1.0,
    'lambda_chamfer':    0.5,
    'lambda_smooth':     0.1,
    'lambda_bbox_reg':   2.0,
    'lambda_stone_aux':  1.5,        # extra weight on stone aux head

    # Focal Tversky
    'tversky_alpha':     0.3,        # FP weight
    'tversky_beta':      0.7,        # FN weight (higher → better recall on tiny objects)
    'tversky_gamma':     1.33,       # focal modulation on tversky
    'focal_alpha':       0.25,
    'focal_gamma':       2.0,
    'logit_clamp':       12.0,       # |logit| ≤ 12 keeps FP16 sigmoid out of saturation

    # Per-class loss weights (image-level multiplier)
    'class_weights':     [1.0, 3.0, 8.0],   # tuned down from 10.0 since we now have FN-aware Tversky
    'stone_pos_pixel_weight': 20.0,         # per-pixel weight on stone foreground
    'tumor_pos_pixel_weight': 3.0,

    # Stone pseudo-GT generation (kept from v5)
    'stone_intensity_percentile': 92,
    'stone_dilate_kernel':        7,

    # Stone training / inference
    'stone_boost_factor':         2.5,        # WeightedRandomSampler oversample
    'stone_copy_paste_prob':      0.5,        # NEW: copy-paste aug
    'stone_copy_paste_max':       2,          # paste up to N extra stones
    'thresh_sweep':               [0.30, 0.40, 0.45, 0.50],   # validation per-class sweep

    # Training
    'epochs':            300,
    'batch_size':        8,
    'lr_lora':           1e-4,
    'lr_decoder':        3e-4,
    'lr_apg':            5e-4,
    'lr_aux':            3e-4,
    'weight_decay':      1e-4,
    'warmup_epochs':     10,
    'min_lr':            1e-6,
    'grad_clip':         0.5,
    'val_every':         5,
    'save_every':        25,
    'patience':          50,
    'num_workers':       0,
    'pin_memory':        True,
    'use_tta_val':       True,                # NEW: h-flip TTA at validation
    'pixel_spacing_mm':  0.5,
}

with open(LOG_DIR / 'config.json', 'w') as f:
    json.dump(CFG, f, indent=2)
print('✅ Config saved')
def mask_to_bbox_pixel(mask: np.ndarray):
    if mask.max() == 0:
        return None
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return np.array([x1, y1, x2, y2])


def generate_stone_pseudo_from_bboxes(img_rgb, expert_mask):
    """Adaptive Skeleton Dilation — same as v5, kept verbatim."""
    h, w = img_rgb.shape[:2]
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    pseudo = np.zeros((h, w), dtype=np.uint8)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(expert_mask, 8)
    for i in range(1, num_labels):
        x = stats[i, cv2.CC_STAT_LEFT]; y = stats[i, cv2.CC_STAT_TOP]
        bw = stats[i, cv2.CC_STAT_WIDTH]; bh = stats[i, cv2.CC_STAT_HEIGHT]
        if stats[i, cv2.CC_STAT_AREA] < 1: continue
        roi = gray[y:y+bh, x:x+bw]
        if roi.size == 0: continue
        _, roi_bin = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        roi_num, roi_lab, roi_stat, _ = cv2.connectedComponentsWithStats(roi_bin, 8)
        clean_roi = np.zeros_like(roi_bin)
        for j in range(1, roi_num):
            if roi_stat[j, cv2.CC_STAT_AREA] >= 3:
                clean_roi[roi_lab == j] = 255
        if clean_roi.max() == 0:
            if roi_num > 1:
                largest = np.argmax(roi_stat[1:, cv2.CC_STAT_AREA]) + 1
                clean_roi[roi_lab == largest] = 255
            else:
                continue
        coords = np.column_stack(np.where(clean_roi > 0))
        if len(coords) == 0: continue
        min_y, min_x = coords.min(axis=0)
        max_y, max_x = coords.max(axis=0)
        R = int(np.round(max(min_y, min_x, bh-1-max_y, bw-1-max_x)))
        if R > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*R+1, 2*R+1))
            solid = cv2.dilate(clean_roi, kernel)
        else:
            solid = clean_roi.copy()
        cnts, _ = cv2.findContours(solid, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(solid, cnts, -1, 255, -1)
        pseudo[y:y+bh, x:x+bw] = np.maximum(pseudo[y:y+bh, x:x+bw], solid)
    return pseudo

print('✅ Pseudo-GT helpers defined')
# Detect if we already have v5 pseudo-GT
n_existing = sum(1 for cls in CLASSES for _ in (PSEUDO_DIR / cls).glob('*_pseudo.png'))
print(f'Found {n_existing} existing pseudo-GT files')

if n_existing < 100:
    print('Generating pseudo-GT from scratch (Tumor via SAM2, Stone via ASD)...')
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    sam2_pg = build_sam2(SAM2_CONFIG, SAM2_CKPT, device=str(DEVICE), mode='eval')
    predictor = SAM2ImagePredictor(sam2_pg)

    for split_name in ['train', 'val']:
        df = pd.read_csv(SPLITS_DIR / f'{split_name}_manifest.csv').rename(columns={'class_name':'class'})
        for _, row in tqdm(df.iterrows(), total=len(df), desc=split_name):
            stem, cls = row['stem'], row['class']
            save_path = PSEUDO_DIR / cls / f'{stem}_pseudo.png'
            if save_path.exists(): continue
            img_rgb = np.array(Image.open(PROJECT_ROOT / row['image_path']).convert('RGB'))
            h, w = img_rgb.shape[:2]
            if cls == 'Normal':
                pseudo = np.zeros((h, w), dtype=np.uint8)
            else:
                mp = MASKS_ROOT / cls / f'{stem}_mask.png'
                if not mp.exists():
                    pseudo = np.zeros((h, w), dtype=np.uint8)
                else:
                    em = (np.array(Image.open(mp).convert('L')) > 127).astype(np.uint8) * 255
                    if cls == 'Tumor':
                        bbox = mask_to_bbox_pixel(em)
                        if bbox is not None:
                            predictor.set_image(img_rgb)
                            masks, scores, _ = predictor.predict(box=bbox[None,:], multimask_output=True)
                            pseudo = (masks[np.argmax(scores)] * 255).astype(np.uint8)
                        else:
                            pseudo = np.zeros((h,w), dtype=np.uint8)
                    elif cls == 'Stone':
                        pseudo = generate_stone_pseudo_from_bboxes(img_rgb, em)
            Image.fromarray(pseudo).save(save_path)

    del predictor, sam2_pg
    torch.cuda.empty_cache()
    print('✅ Pseudo-GT generated')
else:
    print('✅ Reusing existing pseudo-GT from v5')
def mask_to_bbox_norm(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape[:2]
    if mask.max() == 0:
        return np.array([0.0]*4, dtype=np.float32)
    rows = np.any(mask > 0, axis=1); cols = np.any(mask > 0, axis=0)
    y1, y2 = np.where(rows)[0][[0,-1]]
    x1, x2 = np.where(cols)[0][[0,-1]]
    return np.array([x1/w, y1/h, x2/w, y2/h], dtype=np.float32)


class StoneBank:
    """Pre-extracts small stone patches (image+mask) for copy-paste augmentation."""
    def __init__(self, manifest_csv, project_root, pseudo_dir, masks_root, max_patches=300):
        df = pd.read_csv(manifest_csv).rename(columns={'class_name':'class'})
        stone_df = df[df['class'] == 'Stone']
        self.patches = []  # list of (img_patch, mask_patch) numpy arrays
        for _, row in stone_df.iterrows():
            if len(self.patches) >= max_patches: break
            stem = row['stem']
            pp = Path(pseudo_dir) / 'Stone' / f'{stem}_pseudo.png'
            if not pp.exists(): continue
            pseudo = np.array(Image.open(pp).convert('L'))
            if pseudo.max() == 0: continue
            img = np.array(Image.open(Path(project_root) / row['image_path']).convert('RGB'))
            num, lab, st, _ = cv2.connectedComponentsWithStats((pseudo > 127).astype(np.uint8), 8)
            for i in range(1, num):
                area = st[i, cv2.CC_STAT_AREA]
                if not (5 <= area <= 800): continue
                x = st[i, cv2.CC_STAT_LEFT]; y = st[i, cv2.CC_STAT_TOP]
                w = st[i, cv2.CC_STAT_WIDTH]; h = st[i, cv2.CC_STAT_HEIGHT]
                pad = 4
                x0 = max(0, x-pad); y0 = max(0, y-pad)
                x1 = min(img.shape[1], x+w+pad); y1 = min(img.shape[0], y+h+pad)
                m_patch = (lab[y0:y1, x0:x1] == i).astype(np.uint8) * 255
                i_patch = img[y0:y1, x0:x1].copy()
                self.patches.append((i_patch, m_patch))
                if len(self.patches) >= max_patches: break
        print(f'✅ StoneBank: {len(self.patches)} patches collected')

    def paste(self, img, pseudo, expert_mask, bbox_xyxy_norm, n_paste=1):
        """Paste up to n_paste extra stone patches inside the kidney bbox region."""
        if not self.patches or expert_mask.max() == 0:
            return img, pseudo
        H, W = img.shape[:2]
        x1n, y1n, x2n, y2n = bbox_xyxy_norm
        bx1, by1 = int(x1n*W), int(y1n*H)
        bx2, by2 = int(x2n*W), int(y2n*H)
        if bx2-bx1 < 20 or by2-by1 < 20: return img, pseudo
        for _ in range(n_paste):
            ip, mp = random.choice(self.patches)
            ph, pw = mp.shape
            if ph >= (by2-by1) or pw >= (bx2-bx1): continue
            for _try in range(8):
                tx = random.randint(bx1, bx2 - pw)
                ty = random.randint(by1, by2 - ph)
                # Avoid overlapping existing pseudo-stones
                if pseudo[ty:ty+ph, tx:tx+pw].max() > 0: continue
                m_bool = mp > 127
                # Light intensity match: scale paste intensity to local stats
                img[ty:ty+ph, tx:tx+pw][m_bool] = ip[m_bool]
                pseudo[ty:ty+ph, tx:tx+pw][m_bool] = 255
                break
        return img, pseudo


class RenalSegDataset(Dataset):
    def __init__(self, manifest_csv, project_root, pseudo_dir, masks_root,
                 img_size=512, is_train=True, stone_bank=None, copy_paste_prob=0.0,
                 copy_paste_max=2):
        self.df = pd.read_csv(manifest_csv).rename(columns={'class_name':'class'})
        self.project_root = Path(project_root)
        self.pseudo_dir = Path(pseudo_dir)
        self.masks_root = Path(masks_root)
        self.img_size = img_size
        self.is_train = is_train
        self.stone_bank = stone_bank
        self.cp_prob = copy_paste_prob
        self.cp_max = copy_paste_max

    def __len__(self): return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        cls, stem = row['class'], row['stem']
        img = np.array(Image.open(self.project_root / row['image_path']).convert('RGB'))
        img = cv2.resize(img, (self.img_size,)*2, interpolation=cv2.INTER_LINEAR)

        pp = self.pseudo_dir / cls / f'{stem}_pseudo.png'
        pseudo = (cv2.resize(np.array(Image.open(pp).convert('L')), (self.img_size,)*2,
                  interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) if pp.exists() \
                  else np.zeros((self.img_size,)*2, dtype=np.uint8)

        ep = self.masks_root / cls / f'{stem}_mask.png'
        expert = (cv2.resize(np.array(Image.open(ep).convert('L')), (self.img_size,)*2,
                  interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8) if ep.exists() \
                  else np.zeros((self.img_size,)*2, dtype=np.uint8)

        bbox = mask_to_bbox_norm(expert)

        if self.is_train:
            img, pseudo, expert, bbox = self._augment(img, pseudo, expert, bbox)
            # Copy-paste extra stones for stone class
            if cls == 'Stone' and self.stone_bank is not None and random.random() < self.cp_prob:
                n = random.randint(1, max(1, self.cp_max))
                img_u8 = img.astype(np.uint8) if img.dtype != np.uint8 else img
                pseudo_u8 = (pseudo * 255).astype(np.uint8)
                img_u8, pseudo_u8 = self.stone_bank.paste(img_u8, pseudo_u8, expert, bbox, n)
                img = img_u8
                pseudo = (pseudo_u8 > 127).astype(np.uint8)

        img_t = torch.from_numpy(img).float().permute(2,0,1) / 255.0
        return {
            'image':       img_t,
            'pseudo_mask': torch.from_numpy(pseudo).float().unsqueeze(0),
            'expert_mask': torch.from_numpy(expert).float().unsqueeze(0),
            'bbox':        torch.from_numpy(bbox).float(),
            'has_lesion':  torch.tensor(1.0 if cls != 'Normal' else 0.0).float(),
            'class_id':    torch.tensor(CLASS_TO_ID[cls]).long(),
            'class_name':  cls,
        }

    def _augment(self, img, pseudo, expert, bbox):
        if random.random() > 0.5:
            img = np.fliplr(img).copy(); pseudo = np.fliplr(pseudo).copy(); expert = np.fliplr(expert).copy()
            if bbox[2] > 0:
                bbox = np.array([1.0-bbox[2], bbox[1], 1.0-bbox[0], bbox[3]], dtype=np.float32)
        if random.random() > 0.5:
            alpha = random.uniform(0.8, 1.2); beta = random.randint(-15, 15)
            img = np.clip(img * alpha + beta, 0, 255).astype(np.uint8)
        if random.random() > 0.5:
            angle = random.uniform(-10, 10)
            h, w = img.shape[:2]
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            img    = cv2.warpAffine(img,    M, (w,h), borderMode=cv2.BORDER_REFLECT)
            pseudo = cv2.warpAffine(pseudo, M, (w,h), borderMode=cv2.BORDER_CONSTANT)
            expert = cv2.warpAffine(expert, M, (w,h), borderMode=cv2.BORDER_CONSTANT)
            if expert.max() > 0:
                bbox = mask_to_bbox_norm(expert)
        return img, pseudo, expert, bbox


# Build StoneBank from training set
stone_bank = StoneBank(SPLITS_DIR/'train_manifest.csv', PROJECT_ROOT, PSEUDO_DIR, MASKS_ROOT,
                       max_patches=300)

ds_test = RenalSegDataset(SPLITS_DIR/'train_manifest.csv', PROJECT_ROOT, PSEUDO_DIR, MASKS_ROOT,
                          CFG['img_size'], is_train=False)
s = ds_test[0]
print(f'✅ Dataset OK | img={s["image"].shape} pseudo={s["pseudo_mask"].shape}')
del ds_test
from sam2.build_sam import build_sam2

sam2_model = build_sam2(config_file=SAM2_CONFIG, ckpt_path=SAM2_CKPT, device='cpu', mode='eval')
for p in sam2_model.parameters(): p.requires_grad = False
print('✅ SAM2 loaded and frozen')


class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.original = original_linear
        in_f, out_f = original_linear.in_features, original_linear.out_features
        self.lora_A = nn.Linear(in_f,  rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling


def inject_lora(model, rank=16, alpha=32, dropout=0.05):
    count = 0
    for name, m in model.image_encoder.named_modules():
        if not name.endswith('.attn'): continue
        if hasattr(m, 'qkv') and isinstance(m.qkv, nn.Linear):
            m.qkv = LoRALinear(m.qkv, rank, alpha, dropout); count += 1
        if hasattr(m, 'proj') and isinstance(m.proj, nn.Linear):
            m.proj = LoRALinear(m.proj, rank, alpha, dropout); count += 1
    print(f'✅ LoRA: {count} projections wrapped ({count//2} attention blocks)')
    if count == 0: raise RuntimeError('LoRA injection failed')

inject_lora(sam2_model, CFG['lora_rank'], CFG['lora_alpha'], CFG['lora_dropout'])
lp = sum(p.numel() for p in sam2_model.image_encoder.parameters() if p.requires_grad)
tp = sum(p.numel() for p in sam2_model.image_encoder.parameters())
print(f'   LoRA trainable: {lp:,} / {tp:,} ({100*lp/tp:.2f}%)')
class ExpertMLP(nn.Module):
    def __init__(self, d, h, o):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, o))
        self.res = nn.Linear(d, o) if d != o else nn.Identity()
    def forward(self, x): return self.net(x) + self.res(x)


class ClassConditionalRouter(nn.Module):
    """Router takes feature + class embedding so experts can specialise per class."""
    def __init__(self, d, n_exp, n_classes, emb_dim=32, k=2):
        super().__init__()
        self.k = k
        self.cls_emb = nn.Embedding(n_classes, emb_dim)
        self.gate = nn.Linear(d + emb_dim, n_exp)
    def forward(self, x, class_id):
        # x: (B, N, D), class_id: (B,)
        B, N, _ = x.shape
        emb = self.cls_emb(class_id).unsqueeze(1).expand(B, N, -1)
        logits = self.gate(torch.cat([x, emb], dim=-1))
        vals, idx = torch.topk(logits, self.k, dim=-1)
        w = F.softmax(vals, dim=-1)
        full = torch.zeros_like(logits)
        full.scatter_(2, idx, w.to(full.dtype))
        return full, full.mean(dim=[0,1]), logits


class MoESemanticHead(nn.Module):
    """MoE on the deepest feature map only — produces semantic mask logits at deep resolution."""
    def __init__(self, feat_dim, n_exp=4, k=2, hidden=256, n_classes=3, emb_dim=32):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.router = ClassConditionalRouter(hidden, n_exp, n_classes, emb_dim, k)
        self.experts = nn.ModuleList([ExpertMLP(hidden, hidden*2, hidden) for _ in range(n_exp)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))
        self.proj_feat = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())  # also export refined feat

    def forward(self, feat, class_id):
        B, D, H, W = feat.shape
        x = feat.flatten(2).permute(0,2,1)
        x = self.proj(x)
        rw, expert_load, _ = self.router(x, class_id)
        eo = torch.stack([e(x) for e in self.experts], dim=2)
        c = (rw.unsqueeze(-1) * eo).sum(dim=2)
        feat_out = self.proj_feat(c).permute(0,2,1).view(B, -1, H, W)
        logits = self.head(c).permute(0,2,1).view(B, 1, H, W)
        return logits, feat_out, expert_load, rw


class FPNRefineDecoder(nn.Module):
    """
    Multi-scale decoder. Inputs: list of FPN features [coarsest..finest] OR [finest..coarsest].
    Internally we reorder to coarsest-first, project to a common dim, then progressively
    upsample + fuse with finer scales while carrying the mask logits along.
    """
    def __init__(self, fpn_dims, hidden=256, out_size=512, n_exp=4, k=2, n_classes=3, emb_dim=32):
        super().__init__()
        self.out_size = out_size
        self.lat = nn.ModuleList([nn.Conv2d(d, hidden, 1) for d in fpn_dims])  # one per scale
        self.moe = MoESemanticHead(hidden, n_exp, k, hidden, n_classes, emb_dim)
        n_refine = len(fpn_dims) - 1
        self.refine = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(hidden + 1, hidden, 3, padding=1, bias=False),
                nn.GroupNorm(8, hidden), nn.GELU(),
                nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
                nn.GroupNorm(8, hidden), nn.GELU(),
            ) for _ in range(n_refine)
        ])
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, fpn_feats_coarse_to_fine, class_id):
        # Project all to common dim
        feats = [l(f) for l, f in zip(self.lat, fpn_feats_coarse_to_fine)]
        # MoE on coarsest
        deep = feats[0]
        moe_logits, deep_feat, expert_load, rw = self.moe(deep, class_id)
        x = deep + deep_feat
        m = moe_logits
        for i, ref in enumerate(self.refine):
            target = feats[i+1]  # finer
            x = F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
            m = F.interpolate(m, size=target.shape[-2:], mode='bilinear', align_corners=False)
            x = ref(torch.cat([x + target, m], dim=1))
        logits = self.head(x)
        logits = F.interpolate(logits, size=(self.out_size, self.out_size),
                               mode='bilinear', align_corners=False)
        return logits, expert_load, rw


class StoneAuxHead(nn.Module):
    """Lightweight stone-only segmentation head, takes finest FPN feature + image"""
    def __init__(self, feat_dim, hidden=128, out_size=512):
        super().__init__()
        self.out_size = out_size
        self.proj = nn.Conv2d(feat_dim, hidden, 1)
        self.img_proj = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.GELU(),
            nn.Conv2d(32, 32, 3, padding=1, stride=2), nn.GELU(),
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden + 32, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.GroupNorm(8, hidden), nn.GELU(),
        )
        self.head = nn.Conv2d(hidden, 1, 1)

    def forward(self, finest_feat, image):
        x = self.proj(finest_feat)
        img_f = self.img_proj(image)
        x = F.interpolate(x, size=img_f.shape[-2:], mode='bilinear', align_corners=False)
        x = self.fuse(torch.cat([x, img_f], dim=1))
        logits = self.head(x)
        return F.interpolate(logits, size=(self.out_size,)*2, mode='bilinear', align_corners=False)


class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.b = nn.Sequential(nn.Conv2d(i,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.GELU(),
                               nn.Conv2d(o,o,3,padding=1,bias=False), nn.BatchNorm2d(o), nn.GELU(),
                               nn.MaxPool2d(2))
    def forward(self, x): return self.b(x)


class APG(nn.Module):
    def __init__(self, inc=3, chs=[32,64,128,256]):
        super().__init__()
        layers = []; c = inc
        for ch in chs: layers.append(ConvBlock(c, ch)); c = ch
        self.backbone = nn.Sequential(*layers)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.cls  = nn.Sequential(nn.Linear(c,64), nn.GELU(), nn.Dropout(0.2), nn.Linear(64,1))
        self.bbox = nn.Sequential(nn.Linear(c,128), nn.GELU(), nn.Dropout(0.2), nn.Linear(128,4), nn.Sigmoid())
    def forward(self, x):
        f = self.gap(self.backbone(x)).flatten(1)
        return self.cls(f), self.bbox(f)


print('✅ FPN decoder, stone aux head, APG defined')
class FocalLoss(nn.Module):
    """FP-stable focal loss. Always run inside autocast(enabled=False)."""
    def __init__(self, alpha=0.25, gamma=2.0, logit_clamp=12.0):
        super().__init__()
        self.alpha = alpha; self.gamma = gamma; self.logit_clamp = logit_clamp

    def forward(self, pred, target, pos_weight=None):
        pred = pred.float(); target = target.float()
        pred = pred.clamp(-self.logit_clamp, self.logit_clamp)
        if pos_weight is not None:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none', pos_weight=pos_weight.float())
        else:
            bce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        # log-space p_t for stability
        log_p   = F.logsigmoid(pred)
        log_1_p = F.logsigmoid(-pred)
        log_p_t = target * log_p + (1 - target) * log_1_p
        p_t = log_p_t.exp().clamp(1e-6, 1.0 - 1e-6)
        focal_w = self.alpha * (1 - p_t) ** self.gamma
        return (focal_w * bce).mean(dim=[1,2,3])


class FocalTverskyLoss(nn.Module):
    """Tversky with FP-aware FN/FP weighting + focal modulation. FN-heavy → tiny-object friendly."""
    def __init__(self, alpha=0.3, beta=0.7, gamma=1.33, smooth=1.0):
        super().__init__()
        self.a = alpha; self.b = beta; self.g = gamma; self.s = smooth
    def forward(self, pred, target):
        pred = pred.float(); target = target.float()
        p = torch.sigmoid(pred).flatten(1)
        t = target.flatten(1)
        tp = (p * t).sum(1)
        fp = (p * (1 - t)).sum(1)
        fn = ((1 - p) * t).sum(1)
        tversky = (tp + self.s) / (tp + self.a*fp + self.b*fn + self.s)
        return ((1 - tversky).clamp(0, 1)) ** self.g


class ChamferLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        sy = torch.tensor([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sx', sx); self.register_buffer('sy', sy)
    def edges(self, m):
        return torch.sqrt(F.conv2d(m, self.sx, padding=1)**2 + F.conv2d(m, self.sy, padding=1)**2 + 1e-8)
    def forward(self, pred, target):
        pred = pred.float(); target = target.float()
        return F.mse_loss(self.edges(torch.sigmoid(pred)), self.edges(target))


class SCRLLoss_v7(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.focal = FocalLoss(cfg['focal_alpha'], cfg['focal_gamma'], cfg['logit_clamp'])
        self.tversky = FocalTverskyLoss(cfg['tversky_alpha'], cfg['tversky_beta'], cfg['tversky_gamma'])
        self.chamfer = ChamferLoss()
        self.w_focal = cfg['lambda_focal']; self.w_tv = cfg['lambda_tversky']
        self.w_cham  = cfg['lambda_chamfer']; self.w_sm = cfg['lambda_smooth']
        self.stone_id = CLASS_TO_ID['Stone']; self.tumor_id = CLASS_TO_ID['Tumor']
        self.stone_pw = cfg['stone_pos_pixel_weight']
        self.tumor_pw = cfg['tumor_pos_pixel_weight']
        self.cls_w = cfg['class_weights']

    def _pos_weight_tensor(self, target, class_ids, device):
        # per-pixel pos_weight: scale up positive class pixels for tiny-class images
        B = target.shape[0]
        pw = torch.ones(B, 1, 1, 1, device=device)
        is_stone = (class_ids == self.stone_id)
        is_tumor = (class_ids == self.tumor_id)
        pw[is_stone] = self.stone_pw
        pw[is_tumor] = self.tumor_pw
        return pw.expand_as(target)

    def forward(self, pred, target, class_ids):
        device = pred.device
        weights = torch.tensor([self.cls_w[c.item()] for c in class_ids], device=device, dtype=torch.float32)
        pos_w = self._pos_weight_tensor(target, class_ids, device)

        l_focal = (self.focal(pred, target, pos_weight=pos_w) * weights).mean()
        l_tv    = (self.tversky(pred, target) * weights).mean()
        l_cham  = self.chamfer(pred, target)

        p = torch.sigmoid(pred.float())
        l_smooth = (torch.abs(p[:,:,1:,:] - p[:,:,:-1,:]).mean()
                    + torch.abs(p[:,:,:,1:] - p[:,:,:,:-1]).mean())

        total = self.w_focal*l_focal + self.w_tv*l_tv + self.w_cham*l_cham + self.w_sm*l_smooth
        return total, {'focal': l_focal.item(), 'tversky': l_tv.item(),
                       'chamfer': l_cham.item(), 'smooth': l_smooth.item(),
                       'total': total.item()}


print('✅ SCRL v7 loss defined (FP32-stable, FN-aware, per-pixel weighted)')
class MoERenalSAMCG_v6(nn.Module):
    def __init__(self, sam2_model, cfg):
        super().__init__()
        self.image_encoder = sam2_model.image_encoder
        self.img_size = cfg['img_size']
        self.sam_img_size = cfg['sam_img_size']
        self.cfg = cfg
        # Detect FPN dims
        self._detect_fpn(cfg['img_size'])
        self.apg = APG(3, cfg['apg_channels'])
        self.decoder = FPNRefineDecoder(
            fpn_dims=self.fpn_dims_coarse_to_fine, hidden=cfg['expert_hidden'],
            out_size=cfg['img_size'], n_exp=cfg['num_experts'], k=cfg['top_k_experts'],
            n_classes=len(CLASSES), emb_dim=cfg['class_emb_dim']
        )
        # Stone aux head — uses finest FPN scale
        self.stone_head = StoneAuxHead(
            feat_dim=self.fpn_dims_coarse_to_fine[-1],
            hidden=128, out_size=cfg['img_size']
        )

    @torch.no_grad()
    def _detect_fpn(self, img_size):
        dev = next(self.image_encoder.parameters()).device
        dummy = torch.randn(1, 3, img_size, img_size, device=dev)
        self.image_encoder.eval()
        out = self.image_encoder(dummy)
        # Find FPN list
        if isinstance(out, dict):
            for key in ['backbone_fpn', 'vision_features']:
                if key in out and isinstance(out[key], (list, tuple)) and len(out[key]) > 1:
                    feats = list(out[key])
                    self.feat_key = key
                    break
            else:
                # Fall back to any list value, else single-tensor wrapped to length-1
                feats = None
                for k, v in out.items():
                    if isinstance(v, (list, tuple)) and len(v) >= 1:
                        feats = list(v); self.feat_key = k; break
                if feats is None:
                    v = list(out.values())[0]
                    feats = [v]; self.feat_key = list(out.keys())[0]
        elif isinstance(out, (list, tuple)):
            feats = list(out); self.feat_key = 'tuple'
        else:
            feats = [out]; self.feat_key = 'direct'

        # Order coarse→fine: smallest spatial first
        feats_sorted = sorted(feats, key=lambda t: t.shape[-1])  # ascending H/W
        self.fpn_dims_coarse_to_fine = [f.shape[1] for f in feats_sorted]
        print(f'   ✅ FPN scales (coarse→fine): {[tuple(f.shape) for f in feats_sorted]}')
        print(f'   ✅ FPN channel dims: {self.fpn_dims_coarse_to_fine}')

    def encode_fpn(self, images):
        if images.shape[-1] != self.sam_img_size:
            images = F.interpolate(images, (self.sam_img_size,)*2, mode='bilinear', align_corners=False)
        out = self.image_encoder(images)
        if isinstance(out, dict):
            v = out.get(self.feat_key, list(out.values())[0])
            feats = list(v) if isinstance(v, (list, tuple)) else [v]
        elif isinstance(out, (list, tuple)):
            feats = list(out)
        else:
            feats = [out]
        feats_sorted = sorted(feats, key=lambda t: t.shape[-1])
        return feats_sorted  # coarse→fine

    def forward(self, images, class_ids=None):
        B = images.shape[0]
        if class_ids is None:
            class_ids = torch.zeros(B, dtype=torch.long, device=images.device)
        apg_cls, apg_bbox = self.apg(images)
        fpn = self.encode_fpn(images)
        main_logits, expert_load, rw = self.decoder(fpn, class_ids)
        stone_logits = self.stone_head(fpn[-1], images)  # finest FPN feat + raw image
        return main_logits, stone_logits, apg_cls, apg_bbox, expert_load, rw


model = MoERenalSAMCG_v6(sam2_model, CFG).to(DEVICE)

lora_p = [p for n,p in model.named_parameters() if p.requires_grad and 'lora_' in n]
apg_p  = [p for n,p in model.named_parameters() if p.requires_grad and 'apg' in n]
aux_p  = [p for n,p in model.named_parameters() if p.requires_grad and 'stone_head' in n]
dec_p  = [p for n,p in model.named_parameters() if p.requires_grad
          and 'lora_' not in n and 'apg' not in n and 'stone_head' not in n]

t = sum(p.numel() for p in model.parameters() if p.requires_grad)
a = sum(p.numel() for p in model.parameters())
print(f'\n✅ Model on {DEVICE}')
print(f'   Trainable: {t:,} / {a:,} ({100*t/a:.2f}%)')
print(f'   LoRA={sum(p.numel() for p in lora_p):,} | Decoder={sum(p.numel() for p in dec_p):,} | '
      f'StoneAux={sum(p.numel() for p in aux_p):,} | APG={sum(p.numel() for p in apg_p):,}')
optimizer = torch.optim.AdamW([
    {'params': lora_p, 'lr': CFG['lr_lora'],    'weight_decay': CFG['weight_decay']},
    {'params': dec_p,  'lr': CFG['lr_decoder'], 'weight_decay': CFG['weight_decay']},
    {'params': aux_p,  'lr': CFG['lr_aux'],     'weight_decay': CFG['weight_decay']},
    {'params': apg_p,  'lr': CFG['lr_apg'],     'weight_decay': CFG['weight_decay']},
])

def lr_lambda(ep):
    if ep < CFG['warmup_epochs']: return ep / max(1, CFG['warmup_epochs'])
    prog = (ep - CFG['warmup_epochs']) / max(1, CFG['epochs'] - CFG['warmup_epochs'])
    return max(CFG['min_lr']/CFG['lr_decoder'], 0.5*(1+math.cos(math.pi*prog)))

scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
scaler = GradScaler()

train_ds = RenalSegDataset(SPLITS_DIR/'train_manifest.csv', PROJECT_ROOT, PSEUDO_DIR, MASKS_ROOT,
                           CFG['img_size'], is_train=True, stone_bank=stone_bank,
                           copy_paste_prob=CFG['stone_copy_paste_prob'],
                           copy_paste_max=CFG['stone_copy_paste_max'])
val_ds = RenalSegDataset(SPLITS_DIR/'val_manifest.csv', PROJECT_ROOT, PSEUDO_DIR, MASKS_ROOT,
                         CFG['img_size'], is_train=False)

class_counts = train_ds.df['class'].value_counts().to_dict()
base_w = {cls: 1.0/max(1, class_counts.get(cls,1)) for cls in CLASSES}
base_w['Stone'] *= CFG['stone_boost_factor']
sample_w = train_ds.df['class'].map(base_w).astype(np.float32).values
sampler = torch.utils.data.WeightedRandomSampler(
    weights=torch.from_numpy(sample_w), num_samples=len(sample_w), replacement=True
)

train_loader = DataLoader(train_ds, CFG['batch_size'], shuffle=False, sampler=sampler,
                          num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'], drop_last=True)
val_loader   = DataLoader(val_ds,   CFG['batch_size'], shuffle=False,
                          num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'])

print(f'✅ Train: {len(train_ds)}→{len(train_loader)} batches | Val: {len(val_ds)}→{len(val_loader)} batches')
print(f'   Stone oversample: x{CFG["stone_boost_factor"]} | Copy-paste prob: {CFG["stone_copy_paste_prob"]}')
scrl = SCRLLoss_v7(CFG).to(DEVICE)
bbox_fn = nn.SmoothL1Loss()
cls_fn = nn.BCEWithLogitsLoss()

def train_one_epoch(model, loader, optimizer, scaler, epoch):
    model.train()
    losses = defaultdict(float); n = 0; nan_skipped = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch+1}/{CFG["epochs"]}', leave=False)
    for batch in pbar:
        img    = batch['image'].to(DEVICE)
        pseudo = batch['pseudo_mask'].to(DEVICE)
        bbox   = batch['bbox'].to(DEVICE)
        has_l  = batch['has_lesion'].to(DEVICE)
        cids   = batch['class_id'].to(DEVICE)

        optimizer.zero_grad()

        # Forward — FP16 OK
        with autocast(dtype=torch.bfloat16):
            main_logits, stone_logits, ac, ab, el, _ = model(img, class_ids=cids)

        # Losses — FORCE FP32 for stability
        with autocast(enabled=False):
            main_logits_f = main_logits.float()
            stone_logits_f = stone_logits.float()

            seg_loss, seg_d = scrl(main_logits_f, pseudo, cids)

            # Stone aux loss (only on stone batches)
            stone_mask_b = (cids == CLASS_TO_ID['Stone'])
            if stone_mask_b.any():
                aux_seg, _ = scrl(stone_logits_f[stone_mask_b],
                                  pseudo[stone_mask_b],
                                  cids[stone_mask_b])
                l_aux = aux_seg
            else:
                l_aux = torch.tensor(0.0, device=DEVICE)

            l_cls = cls_fn(ac.float().squeeze(1), has_l)
            mask_l = (has_l > 0.5)
            l_bbox = bbox_fn(ab.float()[mask_l], bbox[mask_l]) if mask_l.any() \
                     else torch.tensor(0.0, device=DEVICE)
            l_bal  = F.mse_loss(el.float(), torch.ones_like(el.float())/len(el))

            total = (seg_loss
                     + CFG['lambda_stone_aux'] * l_aux
                     + CFG['lambda_bbox_reg']  * l_bbox
                     + l_cls
                     + CFG['load_balance_alpha'] * l_bal)

        if torch.isnan(total) or torch.isinf(total):
            nan_skipped += 1; optimizer.zero_grad(); continue

        scaler.scale(total).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                        CFG['grad_clip'])
        scaler.step(optimizer); scaler.update()

        losses['total'] += total.item(); losses['seg'] += seg_loss.item()
        losses['aux']   += l_aux.item();  losses['cls'] += l_cls.item(); losses['bbox'] += l_bbox.item()
        for k, v in seg_d.items(): losses[f'seg_{k}'] += v
        n += 1
        pbar.set_postfix(loss=f'{total.item():.3f}', seg=f'{seg_loss.item():.3f}',
                         aux=f'{l_aux.item():.3f}')
    if nan_skipped: print(f'   ⚠️ {nan_skipped} batches skipped due to NaN/Inf')
    return {k: v/max(n,1) for k,v in losses.items()}


def _dice(pred_bin, gt_bin):
    if pred_bin.sum() == 0 and gt_bin.sum() == 0: return 1.0
    return float(2*(pred_bin*gt_bin).sum()) / float(pred_bin.sum() + gt_bin.sum() + 1e-8)


@torch.no_grad()
def validate(model, loader, use_tta=True, sweep=None):
    """Returns dict with per-class dice at each threshold AND best per-class threshold."""
    model.eval()
    if sweep is None: sweep = CFG['thresh_sweep']
    # collect per-(class, threshold) dice list
    per_thresh = {cls: {th: [] for th in sweep} for cls in CLASSES}
    all_per_thresh = {th: [] for th in sweep}

    for batch in tqdm(loader, desc='Val', leave=False):
        img = batch['image'].to(DEVICE)
        gt  = batch['expert_mask'].to(DEVICE)
        cids = batch['class_id'].to(DEVICE)

        with autocast(dtype=torch.bfloat16):
            ml1, sl1, _, _, _, _ = model(img, class_ids=cids)
        if use_tta:
            img_f = torch.flip(img, dims=[-1])
            with autocast(dtype=torch.bfloat16):
                ml2, sl2, _, _, _, _ = model(img_f, class_ids=cids)
            ml2 = torch.flip(ml2, dims=[-1]); sl2 = torch.flip(sl2, dims=[-1])
            main_prob  = (torch.sigmoid(ml1.float()) + torch.sigmoid(ml2.float())) / 2
            stone_prob = (torch.sigmoid(sl1.float()) + torch.sigmoid(sl2.float())) / 2
        else:
            main_prob  = torch.sigmoid(ml1.float())
            stone_prob = torch.sigmoid(sl1.float())

        # Fuse: for stone, take elementwise max of main and stone aux
        is_stone = (cids == CLASS_TO_ID['Stone']).view(-1,1,1,1)
        prob = torch.where(is_stone, torch.maximum(main_prob, stone_prob), main_prob)

        for th in sweep:
            pred = (prob > th).float()
            for i in range(img.shape[0]):
                d = _dice(pred[i], gt[i])
                cls = CLASSES[cids[i].item()]
                per_thresh[cls][th].append(d)
                all_per_thresh[th].append(d)

    # Best threshold per class (maximises that class's mean dice)
    best_th = {}
    best_dice = {}
    for cls in CLASSES:
        means = {th: np.mean(per_thresh[cls][th]) if per_thresh[cls][th] else 0.0 for th in sweep}
        best_th[cls] = max(means, key=means.get)
        best_dice[cls] = means[best_th[cls]]
    macro = np.mean([best_dice[c] for c in CLASSES])

    return {
        'per_class_dice': best_dice,
        'per_class_threshold': best_th,
        'macro_best': macro,
        'all_thresh_means': {th: np.mean(all_per_thresh[th]) for th in sweep},
    }


print('✅ Train/val functions ready (FP32 loss, TTA val, threshold sweep)')
best_val_dice = 0.0; patience_ctr = 0; history = []

print('='*72)
print('  TRAINING MoE-RenalSAM-CG v6 (Multi-scale + FN-aware + Stone Aux)')
print(f'  Epochs: {CFG["epochs"]} | Batch: {CFG["batch_size"]} | Patience: {CFG["patience"]}')
print(f'  Class weights: N={CFG["class_weights"][0]} T={CFG["class_weights"][1]} S={CFG["class_weights"][2]}')
print(f'  Pixel pos_weight: Stone={CFG["stone_pos_pixel_weight"]} Tumor={CFG["tumor_pos_pixel_weight"]}')
print(f'  Tversky α={CFG["tversky_alpha"]} β={CFG["tversky_beta"]} γ={CFG["tversky_gamma"]}')
print(f'  Threshold sweep: {CFG["thresh_sweep"]} | TTA: {CFG["use_tta_val"]}')
print('='*72)

for epoch in range(CFG['epochs']):
    t0 = time.time()
    tl = train_one_epoch(model, train_loader, optimizer, scaler, epoch)
    scheduler.step()
    lr = optimizer.param_groups[1]['lr']
    elapsed = time.time() - t0

    rec = {'epoch': epoch+1, 'lr': lr, 'time': elapsed,
           **{f'train_{k}': v for k, v in tl.items()}}

    if (epoch+1) % CFG['val_every'] == 0 or epoch == 0:
        vd = validate(model, val_loader, use_tta=CFG['use_tta_val'])
        rec['val_macro'] = vd['macro_best']
        for cls in CLASSES:
            rec[f'val_dice_{cls}'] = vd['per_class_dice'][cls]
            rec[f'val_thresh_{cls}'] = vd['per_class_threshold'][cls]

        per_class_str = ' '.join(
            f'{c[0]}:{vd["per_class_dice"][c]:.3f}@{vd["per_class_threshold"][c]:.2f}'
            for c in CLASSES
        )
        print(f'\nEpoch {epoch+1:03d}/{CFG["epochs"]} | Loss: {tl["total"]:.4f} | '
              f'Macro: {vd["macro_best"]:.4f} [{per_class_str}] | LR: {lr:.2e} | {elapsed:.0f}s')

        if vd['macro_best'] > best_val_dice:
            best_val_dice = vd['macro_best']; patience_ctr = 0
            torch.save({'epoch': epoch+1, 'model_state': model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'val': vd, 'config': CFG}, CKPT_DIR / 'best_model.pth')
            print(f'   ★ New best! Macro Dice = {best_val_dice:.4f}')
        else:
            patience_ctr += CFG['val_every']
            if patience_ctr >= CFG['patience']:
                print(f'\n⏹ Early stopping at epoch {epoch+1}'); break
    elif (epoch+1) % 10 == 0:
        print(f'Epoch {epoch+1:03d} | Loss: {tl["total"]:.4f} | LR: {lr:.2e} | {elapsed:.0f}s')

    if (epoch+1) % CFG['save_every'] == 0:
        torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'config': CFG},
                   CKPT_DIR / f'checkpoint_ep{epoch+1:03d}.pth')
    history.append(rec)

torch.save({'epoch': epoch+1, 'model_state': model.state_dict(), 'config': CFG},
           CKPT_DIR / 'final_model.pth')
pd.DataFrame(history).to_csv(LOG_DIR / 'training_history.csv', index=False)
print(f'\n{"="*72}\n  DONE — Best Macro Val Dice: {best_val_dice:.4f}\n{"="*72}')
hdf = pd.read_csv(LOG_DIR / 'training_history.csv')

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('MoE-RenalSAM-CG v6 Training (Multi-scale, FN-aware)', fontsize=14, fontweight='bold')

ax = axes[0,0]
ax.plot(hdf['epoch'], hdf['train_total'], alpha=0.8, label='Train Total')
if 'train_seg' in hdf.columns: ax.plot(hdf['epoch'], hdf['train_seg'], alpha=0.6, label='Train Seg')
if 'train_aux' in hdf.columns: ax.plot(hdf['epoch'], hdf['train_aux'], alpha=0.6, label='Train Stone Aux')
ax.set_title('Loss'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[0,1]
vr = hdf.dropna(subset=['val_macro']) if 'val_macro' in hdf.columns else hdf
if 'val_macro' in vr.columns:
    ax.plot(vr['epoch'], vr['val_macro'], 'k-o', label='Macro', lw=2, ms=4)
    for c, col in zip(CLASSES, ['#4e79a7','#e15759','#f28e2b']):
        k = f'val_dice_{c}'
        if k in vr.columns: ax.plot(vr['epoch'], vr[k], '--', label=c, color=col)
ax.set_title('Val Dice (best per-class threshold, vs Expert)')
ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1.05)

ax = axes[1,0]
for comp in ['seg_focal','seg_tversky','seg_chamfer','seg_smooth']:
    k = f'train_{comp}'
    if k in hdf.columns: ax.plot(hdf['epoch'], hdf[k], label=comp, alpha=0.7)
ax.set_title('SCRL v7 Components'); ax.legend(); ax.grid(alpha=0.3)

ax = axes[1,1]
ax.plot(hdf['epoch'], hdf['lr']); ax.set_title('LR'); ax.grid(alpha=0.3); ax.set_yscale('log')

plt.tight_layout()
plt.savefig(LOG_DIR / 'training_curves.png', dpi=150, bbox_inches='tight')
plt.show()
# Visual check with best model
ckpt = torch.load(CKPT_DIR / 'best_model.pth', map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt['model_state']); model.eval()
print(f'Loaded best from epoch {ckpt["epoch"]} | Macro={ckpt["val"]["macro_best"]:.4f}')
best_th = ckpt['val']['per_class_threshold']
print(f'Best per-class thresholds: {best_th}')

fig, axes = plt.subplots(3, 6, figsize=(24, 12))
fig.suptitle('v6 Validation Predictions (TTA, per-class threshold)', fontsize=14, fontweight='bold')
for ax, t in zip(axes[0], ['CT', 'Expert', 'Pseudo-GT', 'Main Pred', 'Stone Aux Pred', 'Final Overlay']):
    ax.set_title(t, fontsize=10, fontweight='bold')

vdf = pd.read_csv(SPLITS_DIR/'val_manifest.csv').rename(columns={'class_name':'class'})

for ri, cls in enumerate(CLASSES):
    row = vdf[vdf['class']==cls].iloc[1]
    img_orig = np.array(Image.open(PROJECT_ROOT/row['image_path']).convert('RGB'))
    img = cv2.resize(img_orig, (CFG['img_size'],)*2)

    ep = MASKS_ROOT/cls/f"{row['stem']}_mask.png"
    expert = (cv2.resize(np.array(Image.open(ep).convert('L')), (CFG['img_size'],)*2,
                         interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)*255 \
             if ep.exists() else np.zeros((CFG['img_size'],)*2, dtype=np.uint8)

    pp = PSEUDO_DIR/cls/f"{row['stem']}_pseudo.png"
    pseudo = (cv2.resize(np.array(Image.open(pp).convert('L')), (CFG['img_size'],)*2,
                         interpolation=cv2.INTER_NEAREST) > 127).astype(np.uint8)*255 \
             if pp.exists() else np.zeros((CFG['img_size'],)*2, dtype=np.uint8)

    inp = torch.from_numpy(img).float().permute(2,0,1).unsqueeze(0)/255.0
    cid = torch.tensor([CLASS_TO_ID[cls]], device=DEVICE)
    with torch.no_grad(), autocast(dtype=torch.bfloat16):
        ml, sl, _, _, _, _ = model(inp.to(DEVICE), class_ids=cid)
        # TTA
        inp_f = torch.flip(inp.to(DEVICE), dims=[-1])
        ml2, sl2, _, _, _, _ = model(inp_f, class_ids=cid)
        ml2 = torch.flip(ml2, dims=[-1]); sl2 = torch.flip(sl2, dims=[-1])

    main_p  = (torch.sigmoid(ml.float()) + torch.sigmoid(ml2.float())).div(2)[0,0].cpu().numpy()
    stone_p = (torch.sigmoid(sl.float()) + torch.sigmoid(sl2.float())).div(2)[0,0].cpu().numpy()
    th = best_th.get(cls, 0.5)
    if cls == 'Stone':
        final_p = np.maximum(main_p, stone_p)
    else:
        final_p = main_p
    pred = (final_p > th).astype(np.uint8) * 255

    main_pred  = (main_p > th).astype(np.uint8)*255
    stone_pred = (stone_p > th).astype(np.uint8)*255

    ov = img.copy(); ov[pred>0] = [255,0,0]; ov = cv2.addWeighted(img, 0.6, ov, 0.4, 0)

    axes[ri,0].imshow(img); axes[ri,0].set_ylabel(cls, fontsize=12, fontweight='bold'); axes[ri,0].axis('off')
    axes[ri,1].imshow(expert, cmap='gray'); axes[ri,1].axis('off')
    axes[ri,2].imshow(pseudo, cmap='gray'); axes[ri,2].axis('off')
    axes[ri,3].imshow(main_pred, cmap='gray'); axes[ri,3].set_title(f'th={th:.2f}', fontsize=8); axes[ri,3].axis('off')
    axes[ri,4].imshow(stone_pred, cmap='gray'); axes[ri,4].axis('off')
    axes[ri,5].imshow(ov); axes[ri,5].axis('off')

plt.tight_layout()
plt.savefig(LOG_DIR / 'val_visual_check_v6.png', dpi=150, bbox_inches='tight')
plt.show()
# ==========================================
# Section 12 — Load Best Model & Test Loader
# ==========================================
print("Loading Best Model for Evaluation...")
best_ckpt_path = CKPT_DIR / 'best_model.pth'

# Added weights_only=False to allow loading the saved dictionary and numpy scalars
checkpoint = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)

model.load_state_dict(checkpoint['model_state'])
model.eval()

# Ensure we have a test dataloader
test_ds = RenalSegDataset(SPLITS_DIR/'test_manifest.csv', PROJECT_ROOT, PSEUDO_DIR, MASKS_ROOT,
                          CFG['img_size'], is_train=False)
test_loader = DataLoader(test_ds, CFG['batch_size'], shuffle=False, 
                         num_workers=CFG['num_workers'], pin_memory=CFG['pin_memory'])

print(f"✅ Best Model Loaded (Epoch {checkpoint['epoch']})")
print(f"✅ Test Set Ready: {len(test_ds)} images → {len(test_loader)} batches")

# Retrieve the optimal thresholds found during validation
optimal_thresholds = checkpoint['val']['per_class_threshold']
print(f"✅ Optimal Val Thresholds: {optimal_thresholds}")

# ==========================================
# Section 13 — Plot Training History Curves
# ==========================================
history_df = pd.read_csv(LOG_DIR / 'training_history.csv')

plt.style.use('seaborn-v0_8-whitegrid')
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6), dpi=300)

# 1. Loss Curve
ax1.plot(history_df['epoch'], history_df['train_total'], label='Train Total Loss', color='#2c3e50', linewidth=2)
ax1.plot(history_df['epoch'], history_df['train_seg'], label='Train Seg Loss', color='#e74c3c', linestyle='--', linewidth=2)
if 'train_aux' in history_df.columns:
    ax1.plot(history_df['epoch'], history_df['train_aux'], label='Train Stone Aux Loss', color='#f39c12', linestyle=':', linewidth=2)
ax1.set_title('Training Losses over Epochs', fontsize=16, fontweight='bold')
ax1.set_xlabel('Epoch', fontsize=14)
ax1.set_ylabel('Loss', fontsize=14)
ax1.legend(fontsize=12)
ax1.tick_params(axis='both', labelsize=12)

# 2. Validation Dice Curve
val_epochs = history_df.dropna(subset=['val_macro'])['epoch']
ax2.plot(val_epochs, history_df.dropna(subset=['val_macro'])['val_macro'], label='Macro Avg Dice', color='#27ae60', linewidth=2.5, marker='o')
ax2.plot(val_epochs, history_df.dropna(subset=['val_macro'])['val_dice_Tumor'], label='Tumor Dice', color='#8e44ad', linestyle='-.', alpha=0.8)
ax2.plot(val_epochs, history_df.dropna(subset=['val_macro'])['val_dice_Stone'], label='Stone Dice', color='#d35400', linestyle='-.', alpha=0.8)
ax2.plot(val_epochs, history_df.dropna(subset=['val_macro'])['val_dice_Normal'], label='Normal Dice', color='#2980b9', linestyle='-.', alpha=0.8)

ax2.set_title('Validation Dice Scores', fontsize=16, fontweight='bold')
ax2.set_xlabel('Epoch', fontsize=14)
ax2.set_ylabel('Dice Coefficient', fontsize=14)
ax2.set_ylim(0, 1.05)
ax2.legend(fontsize=12)
ax2.tick_params(axis='both', labelsize=12)

plt.tight_layout()
plt.savefig(LOG_DIR / 'training_curves_publication.png', bbox_inches='tight', dpi=300)
plt.show()

# ==========================================
# Section 14 — Advanced Metrics Definitions
# ==========================================
from scipy.spatial.distance import directed_hausdorff

def compute_metrics(pred_bin, gt_bin):
    """Computes Dice, Recall (Sensitivity), Precision (PPV), and Specificity."""
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return {'dice': 1.0, 'recall': 1.0, 'precision': 1.0, 'specificity': 1.0, 'hd95': 0.0}
    
    tp = (pred_bin * gt_bin).sum()
    fp = (pred_bin * (1 - gt_bin)).sum()
    fn = ((1 - pred_bin) * gt_bin).sum()
    tn = ((1 - pred_bin) * (1 - gt_bin)).sum()
    
    dice = 2 * tp / (2 * tp + fp + fn + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    
    # Calculate HD95 (Boundary Error)
    if pred_bin.sum() > 0 and gt_bin.sum() > 0:
        # Extract coordinates of positive pixels
        pred_coords = np.column_stack(np.where(pred_bin > 0))
        gt_coords = np.column_stack(np.where(gt_bin > 0))
        
        # Directed Hausdorff distances
        hd1 = directed_hausdorff(pred_coords, gt_coords)[0]
        hd2 = directed_hausdorff(gt_coords, pred_coords)[0]
        hd = max(hd1, hd2)
        # Approximate HD95 by taking 95th percentile distance (simplified to max here for speed)
        # Multiply by pixel spacing to get mm
        hd95 = hd * CFG['pixel_spacing_mm'] 
    else:
        # If one is empty and the other isn't, assign a penalty distance
        hd95 = np.sqrt(CFG['img_size']**2 + CFG['img_size']**2) * CFG['pixel_spacing_mm']
        if pred_bin.sum() == 0 and gt_bin.sum() == 0:
            hd95 = 0.0

    return {'dice': float(dice), 'recall': float(recall), 'precision': float(precision), 'specificity': float(specificity), 'hd95': float(hd95)}

print("✅ Advanced Metrics Initialized")

# ==========================================
# Section 15 — Full Test Set Evaluation
# ==========================================
test_results = {cls: {'dice': [], 'recall': [], 'precision': [], 'specificity': [], 'hd95': []} for cls in CLASSES}

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Testing Model'):
        img = batch['image'].to(DEVICE)
        gt = batch['expert_mask'].cpu().numpy()
        cids = batch['class_id'].to(DEVICE)
        cnames = batch['class_name']
        
        # Standard Forward
        with autocast(dtype=torch.bfloat16):
            ml1, sl1, _, _, _, _ = model(img, class_ids=cids)
            
        # TTA Forward (Horizontal Flip)
        if CFG['use_tta_val']:
            img_f = torch.flip(img, dims=[-1])
            with autocast(dtype=torch.bfloat16):
                ml2, sl2, _, _, _, _ = model(img_f, class_ids=cids)
            ml2 = torch.flip(ml2, dims=[-1])
            sl2 = torch.flip(sl2, dims=[-1])
            
            main_prob = (torch.sigmoid(ml1.float()) + torch.sigmoid(ml2.float())) / 2
            stone_prob = (torch.sigmoid(sl1.float()) + torch.sigmoid(sl2.float())) / 2
        else:
            main_prob = torch.sigmoid(ml1.float())
            stone_prob = torch.sigmoid(sl1.float())

        # Fuse main logits and stone auxiliary logits
        is_stone = (cids == CLASS_TO_ID['Stone']).view(-1, 1, 1, 1)
        prob = torch.where(is_stone, torch.maximum(main_prob, stone_prob), main_prob)
        prob = prob.cpu().numpy()
        
        # Calculate metrics for each image in batch
        for i in range(img.shape[0]):
            cls = cnames[i]
            thresh = optimal_thresholds[cls]
            
            pred_bin = (prob[i, 0] > thresh).astype(np.uint8)
            gt_bin = gt[i, 0].astype(np.uint8)
            
            mets = compute_metrics(pred_bin, gt_bin)
            for k, v in mets.items():
                test_results[cls][k].append(v)

print("✅ Test Set Evaluation Complete")

# ==========================================
# Section 16 — Statistical Summary Table
# ==========================================
summary_data = []

for cls in CLASSES:
    cls_metrics = test_results[cls]
    if len(cls_metrics['dice']) == 0:
        continue
        
    summary_data.append({
        'Class': cls,
        'N (Images)': len(cls_metrics['dice']),
        'Dice (%)': f"{np.mean(cls_metrics['dice'])*100:.2f} ± {np.std(cls_metrics['dice'])*100:.2f}",
        'Recall / Sens (%)': f"{np.mean(cls_metrics['recall'])*100:.2f} ± {np.std(cls_metrics['recall'])*100:.2f}",
        'Precision / PPV (%)': f"{np.mean(cls_metrics['precision'])*100:.2f} ± {np.std(cls_metrics['precision'])*100:.2f}",
        'Specificity (%)': f"{np.mean(cls_metrics['specificity'])*100:.2f} ± {np.std(cls_metrics['specificity'])*100:.2f}",
        'HD95 (mm)': f"{np.mean(cls_metrics['hd95']):.2f} ± {np.std(cls_metrics['hd95']):.2f}"
    })

summary_df = pd.DataFrame(summary_data)
summary_df.to_csv(LOG_DIR / 'test_set_publication_metrics.csv', index=False)

from IPython.display import display, HTML
display(summary_df)

# Print LaTeX format for paper
print("\nLaTeX Table Code:")
print(summary_df.to_latex(index=False))

# ==========================================
# Section 17 — Dice Score Distribution Plots
# ==========================================
import seaborn as sns

plot_data = []
for cls in CLASSES:
    for d in test_results[cls]['dice']:
        plot_data.append({'Class': cls, 'Dice Score': d})

df_plot = pd.DataFrame(plot_data)

plt.figure(figsize=(10, 6), dpi=300)
# Create a violin plot layered with a swarm plot to show individual points
sns.violinplot(x='Class', y='Dice Score', data=df_plot, inner='quartile', palette='pastel', cut=0)
sns.stripplot(x='Class', y='Dice Score', data=df_plot, color='black', alpha=0.3, size=3, jitter=True)

plt.title('Distribution of Dice Scores on Test Set by Anatomy', fontsize=16, fontweight='bold')
plt.xlabel('Anatomy Class', fontsize=14)
plt.ylabel('Dice Similarity Coefficient (DSC)', fontsize=14)
plt.ylim(0, 1.05)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig(LOG_DIR / 'dice_distribution_violin_publication.png', bbox_inches='tight', dpi=300)
plt.show()

# ==========================================
# Section 18 — Qualitative Visualizations
# ==========================================
import matplotlib.colors as mcolors

def overlay_mask(image, mask, color, alpha=0.5):
    """Overlays a binary mask onto a grayscale image with a specific color."""
    colored_mask = np.zeros_like(image)
    colored_mask[mask == 1] = color
    
    overlay = image.copy()
    overlay[mask == 1] = overlay[mask == 1] * (1 - alpha) + colored_mask[mask == 1] * alpha
    return overlay

# Select 3 random images from each class to display
num_samples = 3
vis_dataset = DataLoader(test_ds, batch_size=1, shuffle=True)

fig, axes = plt.subplots(num_samples * len(CLASSES), 3, figsize=(12, 4 * num_samples * len(CLASSES)), dpi=200)

class_counter = {cls: 0 for cls in CLASSES}
row_idx = 0

model.eval()
with torch.no_grad():
    for batch in vis_dataset:
        cls = batch['class_name'][0]
        if class_counter[cls] >= num_samples:
            if all(v >= num_samples for v in class_counter.values()):
                break
            continue
            
        class_counter[cls] += 1
        
        img = batch['image'].to(DEVICE)
        gt = batch['expert_mask'].cpu().numpy()[0, 0]
        cid = batch['class_id'].to(DEVICE)
        thresh = optimal_thresholds[cls]
        
        # Inference
        ml, sl, _, _, _, _ = model(img, class_ids=cid)
        main_prob = torch.sigmoid(ml.float()).cpu().numpy()[0, 0]
        stone_prob = torch.sigmoid(sl.float()).cpu().numpy()[0, 0]
        
        if cls == 'Stone':
            prob = np.maximum(main_prob, stone_prob)
        else:
            prob = main_prob
            
        pred = (prob > thresh).astype(np.uint8)
        
        # Convert image back to display format
        img_display = img.cpu().numpy()[0].transpose(1, 2, 0)
        # Normalize to 0-1 for plotting
        img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())
        
        # Colors: GT = Red, Pred = Green
        gt_overlay = overlay_mask(img_display, gt, [1, 0, 0], alpha=0.6)
        pred_overlay = overlay_mask(img_display, pred, [0, 1, 0], alpha=0.6)
        
        # Calculate single image dice for subtitle
        mets = compute_metrics(pred, gt)
        
        # Plot
        ax1, ax2, ax3 = axes[row_idx]
        ax1.imshow(img_display)
        ax1.set_title(f"Original CT - {cls}")
        ax1.axis('off')
        
        ax2.imshow(gt_overlay)
        ax2.set_title(f"Expert Ground Truth (Red)")
        ax2.axis('off')
        
        ax3.imshow(pred_overlay)
        ax3.set_title(f"MoE-RenalSAM Pred (Green) | Dice: {mets['dice']:.3f}")
        ax3.axis('off')
        
        row_idx += 1

plt.tight_layout()
plt.savefig(LOG_DIR / 'qualitative_results_publication.png', bbox_inches='tight', dpi=300)
plt.show()

# ==========================================
# Section 19 — MoE Expert Specialization Analysis
# ==========================================
print("Analyzing MoE Expert Routing...")
expert_loads = {cls: np.zeros(CFG['num_experts']) for cls in CLASSES}
class_counts = {cls: 0 for cls in CLASSES}

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Tracking Experts'):
        img = batch['image'].to(DEVICE)
        cids = batch['class_id'].to(DEVICE)
        cnames = batch['class_name']
        
        # Forward pass to capture 'rw' (raw routing weights, 6th output)
        with autocast(dtype=torch.bfloat16):
            _, _, _, _, _, rw = model(img, class_ids=cids)
            
        # rw shape is (Batch, Spatial_N, Num_Experts)
        # Average over the spatial dimension (dim=1) to get per-image expert load
        rw_per_image = rw.float().mean(dim=1).cpu().numpy() # Shape: (Batch, Num_Experts)
        
        for i in range(img.shape[0]):
            cls = cnames[i]
            expert_loads[cls] += rw_per_image[i]
            class_counts[cls] += 1

# Normalize to get average load percentages
for cls in CLASSES:
    if class_counts[cls] > 0:
        expert_loads[cls] = (expert_loads[cls] / class_counts[cls]) * 100

# Plotting
df_experts = pd.DataFrame(expert_loads).T
df_experts.columns = [f"Expert {i+1}" for i in range(CFG['num_experts'])]

plt.style.use('seaborn-v0_8-whitegrid')
ax = df_experts.plot(kind='bar', figsize=(10, 6), colormap='viridis', edgecolor='black')

plt.title('Mixture-of-Experts (MoE) Specialization per Anatomy', fontsize=16, fontweight='bold')
plt.xlabel('Anatomy Class', fontsize=14)
plt.ylabel('Average Expert Routing Weight (%)', fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.legend(title="Experts", fontsize=12, title_fontsize=12)
plt.ylim(0, 100)

plt.savefig(LOG_DIR / 'moe_expert_specialization_publication.png', bbox_inches='tight', dpi=300)
plt.show()

# ==========================================
# Section 20 — Bland-Altman Plot (Clinical Area Agreement)
# ==========================================
gt_areas = []
pred_areas = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Calculating Areas'):
        img = batch['image'].to(DEVICE)
        gt = batch['expert_mask'].cpu().numpy()
        cids = batch['class_id'].to(DEVICE)
        
        with autocast(dtype=torch.bfloat16):
            ml, sl, _, _, _, _ = model(img, class_ids=cids)
            
        main_prob = torch.sigmoid(ml.float()).cpu().numpy()
        stone_prob = torch.sigmoid(sl.float()).cpu().numpy()
        
        for i in range(img.shape[0]):
            cls = batch['class_name'][i]
            # Only plot for lesions (Tumor and Stone)
            if cls == 'Normal': continue
                
            thresh = optimal_thresholds[cls]
            if cls == 'Stone':
                prob = np.maximum(main_prob[i, 0], stone_prob[i, 0])
            else:
                prob = main_prob[i, 0]
                
            pred_bin = (prob > thresh).astype(np.uint8)
            gt_bin = gt[i, 0].astype(np.uint8)
            
            # Area in mm^2 (pixels * spacing^2)
            pixel_area = CFG['pixel_spacing_mm'] ** 2
            pred_area_mm2 = pred_bin.sum() * pixel_area
            gt_area_mm2 = gt_bin.sum() * pixel_area
            
            # Only include if there is actually a lesion present in GT or Pred
            if gt_area_mm2 > 0 or pred_area_mm2 > 0:
                pred_areas.append(pred_area_mm2)
                gt_areas.append(gt_area_mm2)

pred_areas = np.array(pred_areas)
gt_areas = np.array(gt_areas)

means = (pred_areas + gt_areas) / 2
diffs = pred_areas - gt_areas # Positive means AI overestimates, Negative means AI underestimates
bias = np.mean(diffs)
sd = np.std(diffs)

# --- THIS IS THE PLOTTING PART THAT WAS MISSING ---
plt.figure(figsize=(10, 6), dpi=300)
plt.scatter(means, diffs, alpha=0.5, color='#2980b9', edgecolor='k')
plt.axhline(bias, color='red', linestyle='--', linewidth=2, label=f'Mean Bias: {bias:.2f} mm²')
plt.axhline(bias + 1.96 * sd, color='gray', linestyle=':', linewidth=2, label=f'+1.96 SD: {bias + 1.96*sd:.2f}')
plt.axhline(bias - 1.96 * sd, color='gray', linestyle=':', linewidth=2, label=f'-1.96 SD: {bias - 1.96*sd:.2f}')

plt.title('Bland-Altman Plot: AI vs Expert Volume (Area) Agreement', fontsize=16, fontweight='bold')
plt.xlabel('Average Area (mm²)', fontsize=14)
plt.ylabel('Difference (AI Pred - Expert GT) (mm²)', fontsize=14)
plt.legend(fontsize=12)

plt.savefig(LOG_DIR / 'bland_altman_publication.png', bbox_inches='tight', dpi=300)
plt.show()

# ==========================================
# Section 21 — Failure Case Analysis (Worst Performers)
# ==========================================
print("Finding worst performing cases...")
worst_cases = []

model.eval()
with torch.no_grad():
    for batch in tqdm(test_loader, desc='Scanning for Worst Cases'):
        img = batch['image'].to(DEVICE)
        gt = batch['expert_mask'].cpu().numpy()
        cids = batch['class_id'].to(DEVICE)
        
        with autocast(dtype=torch.bfloat16):
            ml, sl, _, _, _, _ = model(img, class_ids=cids)
            
        main_prob = torch.sigmoid(ml.float()).cpu().numpy()
        stone_prob = torch.sigmoid(sl.float()).cpu().numpy()
        
        for i in range(img.shape[0]):
            cls = batch['class_name'][i]
            if cls == 'Normal': continue # Skip normal, focus on lesions
                
            thresh = optimal_thresholds[cls]
            prob = np.maximum(main_prob[i, 0], stone_prob[i, 0]) if cls == 'Stone' else main_prob[i, 0]
            
            pred_bin = (prob > thresh).astype(np.uint8)
            gt_bin = gt[i, 0].astype(np.uint8)
            
            mets = compute_metrics(pred_bin, gt_bin)
            
            # Store image data to sort later
            worst_cases.append({
                'dice': mets['dice'],
                'class': cls,
                'img': img.cpu().numpy()[i].transpose(1, 2, 0),
                'gt': gt_bin,
                'pred': pred_bin
            })

# Sort by Dice score ascending and grab the worst 3
worst_cases = sorted(worst_cases, key=lambda x: x['dice'])[:3]

fig, axes = plt.subplots(3, 3, figsize=(12, 12), dpi=200)
for row_idx, case in enumerate(worst_cases):
    img_display = (case['img'] - case['img'].min()) / (case['img'].max() - case['img'].min())
    
    gt_overlay = overlay_mask(img_display, case['gt'], [1, 0, 0], alpha=0.6)
    pred_overlay = overlay_mask(img_display, case['pred'], [0, 1, 0], alpha=0.6)
    
    ax1, ax2, ax3 = axes[row_idx]
    ax1.imshow(img_display)
    ax1.set_title(f"CT - {case['class']}")
    ax1.axis('off')
    
    ax2.imshow(gt_overlay)
    ax2.set_title(f"Expert GT (Red)")
    ax2.axis('off')
    
    ax3.imshow(pred_overlay)
    ax3.set_title(f"Pred (Green) | Dice: {case['dice']:.3f}")
    ax3.axis('off')

plt.suptitle('Failure Case Analysis (Lowest Dice Scores)', fontsize=18, fontweight='bold')
plt.tight_layout()
plt.savefig(LOG_DIR / 'failure_cases_publication.png', bbox_inches='tight', dpi=300)
plt.show()
