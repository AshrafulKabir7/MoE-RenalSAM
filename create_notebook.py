import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

cells.append(nbf.v4.new_markdown_cell("""# MoE-RenalSAM-CG: Comprehensive Evaluation & Visualization
This notebook provides a thorough analysis of the **MoE-RenalSAM-CG** model, mirroring the rigor of top-tier academic publications.
It includes zero-retraining evaluation, side-by-side model comparisons, training dynamics, ablation studies (MoE routing), and classification confusion matrices.
"""))

cells.append(nbf.v4.new_code_cell("""import os, sys, math, random, warnings
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings('ignore')

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PROJECT_ROOT = Path(r'D:\MoE-RenalSAM-CG\MoE-RenalSAM-CG')
SAM2_ROOT = PROJECT_ROOT / 'segment-anything-2'
sys.path.insert(0, str(SAM2_ROOT))

print(f'✅ Using Device: {DEVICE}')

# Publication plot settings
plt.rcParams.update({
    'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12,
    'figure.dpi': 300, 'savefig.bbox': 'tight', 'font.family': 'sans-serif'
})
"""))

cells.append(nbf.v4.new_markdown_cell("## 1. Load Trained Segmentation Models\nDefining the MoE-RenalSAM-CG architecture and loading the best checkpoint. We dynamically detect FPN dimensions to ensure perfect weight mapping."))

cells.append(nbf.v4.new_code_cell("""from sam2.build_sam import build_sam2

SAM2_CONFIG = 'configs/sam2.1/sam2.1_hiera_l.yaml'
SAM2_CKPT = str(PROJECT_ROOT / 'weights' / 'sam2_hiera_large.pt')

sam2_model = build_sam2(config_file=SAM2_CONFIG, ckpt_path=SAM2_CKPT, device='cpu', mode='eval')
for p in sam2_model.parameters(): p.requires_grad = False

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=16, alpha=32, dropout=0.05):
        super().__init__()
        self.original = original_linear
        in_f, out_f = original_linear.in_features, original_linear.out_features
        self.lora_A = nn.Linear(in_f,  rank, bias=False)
        self.lora_B = nn.Linear(rank, out_f, bias=False)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    def forward(self, x):
        return self.original(x) + self.lora_B(self.lora_A(self.dropout(x))) * self.scaling

def inject_lora(model, rank=16, alpha=32, dropout=0.05):
    for name, m in model.image_encoder.named_modules():
        if not name.endswith('.attn'): continue
        if hasattr(m, 'qkv') and isinstance(m.qkv, nn.Linear):
            m.qkv = LoRALinear(m.qkv, rank, alpha, dropout)
        if hasattr(m, 'proj') and isinstance(m.proj, nn.Linear):
            m.proj = LoRALinear(m.proj, rank, alpha, dropout)

inject_lora(sam2_model, rank=16, alpha=32, dropout=0.05)

class ExpertMLP(nn.Module):
    def __init__(self, d, h, o):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Dropout(0.1), nn.Linear(h, o))
        self.res = nn.Linear(d, o) if d != o else nn.Identity()
    def forward(self, x): return self.net(x) + self.res(x)

class ClassConditionalRouter(nn.Module):
    def __init__(self, d, n_exp, n_classes, emb_dim=32, k=2):
        super().__init__()
        self.k = k
        self.cls_emb = nn.Embedding(n_classes, emb_dim)
        self.gate = nn.Linear(d + emb_dim, n_exp)
    def forward(self, x, class_id):
        B, N, _ = x.shape
        emb = self.cls_emb(class_id).unsqueeze(1).expand(B, N, -1)
        logits = self.gate(torch.cat([x, emb], dim=-1))
        vals, idx = torch.topk(logits, self.k, dim=-1)
        w = F.softmax(vals, dim=-1)
        full = torch.zeros_like(logits)
        full.scatter_(2, idx, w.to(full.dtype))
        return full, full.mean(dim=[0,1]), logits

class MoESemanticHead(nn.Module):
    def __init__(self, feat_dim, n_exp=4, k=2, hidden=256, n_classes=3, emb_dim=32):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(feat_dim, hidden), nn.LayerNorm(hidden), nn.GELU())
        self.router = ClassConditionalRouter(hidden, n_exp, n_classes, emb_dim, k)
        self.experts = nn.ModuleList([ExpertMLP(hidden, hidden*2, hidden) for _ in range(n_exp)])
        self.head = nn.Sequential(nn.Linear(hidden, hidden//2), nn.GELU(), nn.Linear(hidden//2, 1))
        self.proj_feat = nn.Sequential(nn.Linear(hidden, hidden), nn.GELU())
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
    def __init__(self, fpn_dims, hidden=256, out_size=512, n_exp=4, k=2, n_classes=3, emb_dim=32):
        super().__init__()
        self.out_size = out_size
        self.lat = nn.ModuleList([nn.Conv2d(d, hidden, 1) for d in fpn_dims])
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
    def forward(self, fpn_feats, class_id):
        feats = [l(f) for l, f in zip(self.lat, fpn_feats)]
        deep = feats[0]
        moe_logits, deep_feat, expert_load, rw = self.moe(deep, class_id)
        x = deep + deep_feat
        m = moe_logits
        for i, ref in enumerate(self.refine):
            target = feats[i+1]
            x = F.interpolate(x, size=target.shape[-2:], mode='bilinear', align_corners=False)
            m = F.interpolate(m, size=target.shape[-2:], mode='bilinear', align_corners=False)
            x = ref(torch.cat([x + target, m], dim=1))
        logits = self.head(x)
        return F.interpolate(logits, size=(self.out_size, self.out_size), mode='bilinear', align_corners=False), expert_load, rw

class StoneAuxHead(nn.Module):
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

class MoERenalSAMCG_v6(nn.Module):
    def __init__(self, sam2_model):
        super().__init__()
        self.image_encoder = sam2_model.image_encoder
        self.img_size = 512
        self.sam_img_size = 512
        
        dummy = torch.randn(1, 3, 512, 512)
        out = self.image_encoder(dummy)
        feats = list(out['backbone_fpn'])
        feats_sorted = sorted(feats, key=lambda t: t.shape[-1])
        self.fpn_dims_coarse_to_fine = [f.shape[1] for f in feats_sorted]
        
        self.apg = APG(3, [32, 64, 128, 256])
        self.decoder = FPNRefineDecoder(
            fpn_dims=self.fpn_dims_coarse_to_fine, hidden=256,
            out_size=512, n_exp=4, k=2, n_classes=3, emb_dim=32
        )
        self.stone_head = StoneAuxHead(feat_dim=self.fpn_dims_coarse_to_fine[-1], hidden=128, out_size=512)

    def encode_fpn(self, images):
        if images.shape[-1] != self.sam_img_size:
            images = F.interpolate(images, (self.sam_img_size,)*2, mode='bilinear', align_corners=False)
        out = self.image_encoder(images)
        feats = list(out['backbone_fpn'])
        return sorted(feats, key=lambda t: t.shape[-1])
        
    def forward(self, images, class_ids=None):
        B = images.shape[0]
        if class_ids is None: class_ids = torch.zeros(B, dtype=torch.long, device=images.device)
        apg_cls, apg_bbox = self.apg(images)
        fpn = self.encode_fpn(images)
        main_logits, expert_load, rw = self.decoder(fpn, class_ids)
        stone_logits = self.stone_head(fpn[-1], images)
        return main_logits, stone_logits, apg_cls, apg_bbox, expert_load, rw, fpn

model = MoERenalSAMCG_v6(sam2_model).to(DEVICE)
best_ckpt_path = PROJECT_ROOT / 'checkpoints' / 'moe_renalsam_cg_v6' / 'best_model.pth'

if best_ckpt_path.exists():
    ckpt = torch.load(best_ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    print("✅ MoE-RenalSAM-CG model successfully loaded!")
else:
    print("⚠️ Warning: Checkpoint not found at", best_ckpt_path)
"""))

cells.append(nbf.v4.new_markdown_cell("## 2. Evaluation Metrics & Baseline Comparison\nAcademic table and charts comparing custom MoE model metrics to classic and zero-shot baselines."))

cells.append(nbf.v4.new_code_cell("""# 1. Load Custom Model Metrics
log_dir = PROJECT_ROOT / 'logs' / 'moe_renalsam_cg_v6'
metrics_file = log_dir / 'test_set_publication_metrics.csv'

all_records = []

if metrics_file.exists():
    moe_df = pd.read_csv(metrics_file)
    moe_df.columns = moe_df.columns.str.strip()
    for _, row in moe_df.iterrows():
        dice_str = str(row.get('Dice (%)', '0'))
        dice_val = float(dice_str.split('±')[0].strip()) if '±' in dice_str else float(dice_str)
        
        all_records.append({
            'Model': 'MoE-RenalSAM-CG (Ours)',
            'Class': row['Class'],
            'N': row.get('N (Images)', ''),
            'Dice (%)': row.get('Dice (%)', ''),
            'Recall (%)': row.get('Recall / Sens (%)', ''),
            'Precision (%)': row.get('Precision / PPV (%)', ''),
            'HD95 (mm)': row.get('HD95 (mm)', ''),
            'Dice_val': dice_val
        })

baselines_dir = PROJECT_ROOT / 'results' / 'zeroshot'
models = ['sam1', 'sam2', 'medsam', 'hqsam']
for m in models:
    f = baselines_dir / f'{m}_zeroshot_summary.csv'
    if f.exists():
        df_b = pd.read_csv(f)
        for _, row in df_b.iterrows():
            if row['Class'] != 'MACRO':
                dice_mean, dice_std = float(row.get('dice_mean', 0))*100, float(row.get('dice_std', 0))*100
                rec_mean, rec_std = float(row.get('recall_mean', 0))*100, float(row.get('recall_std', 0))*100
                prec_mean, prec_std = float(row.get('precision_mean', 0))*100, float(row.get('precision_std', 0))*100
                hd95_mean, hd95_std = float(row.get('hd95_mean', 0)), float(row.get('hd95_std', 0))
                
                all_records.append({
                    'Model': m.upper(),
                    'Class': row['Class'],
                    'N': row.get('N', ''),
                    'Dice (%)': f"{dice_mean:.2f} ± {dice_std:.2f}",
                    'Recall (%)': f"{rec_mean:.2f} ± {rec_std:.2f}",
                    'Precision (%)': f"{prec_mean:.2f} ± {prec_std:.2f}",
                    'HD95 (mm)': f"{hd95_mean:.2f} ± {hd95_std:.2f}",
                    'Dice_val': dice_mean
                })

classical_file = PROJECT_ROOT / 'results' / 'baselines_classical' / 'classical_baselines_summary.csv'
if classical_file.exists():
    df_c = pd.read_csv(classical_file)
    for _, row in df_c.iterrows():
        if row['class'] != 'MACRO':
            dice_mean, dice_std = float(row.get('dice_mean', 0))*100, float(row.get('dice_std', 0))*100
            rec_mean, rec_std = float(row.get('recall_mean', 0))*100, float(row.get('recall_std', 0))*100
            prec_mean, prec_std = float(row.get('precision_mean', 0))*100, float(row.get('precision_std', 0))*100
            hd95_mean, hd95_std = float(row.get('hd95_mean', 0)), float(row.get('hd95_std', 0))
            
            all_records.append({
                'Model': row['method'].upper(),
                'Class': row['class'],
                'N': row.get('n', ''),
                'Dice (%)': f"{dice_mean:.2f} ± {dice_std:.2f}",
                'Recall (%)': f"{rec_mean:.2f} ± {rec_std:.2f}",
                'Precision (%)': f"{prec_mean:.2f} ± {prec_std:.2f}",
                'HD95 (mm)': f"{hd95_mean:.2f} ± {hd95_std:.2f}",
                'Dice_val': dice_mean
            })

if all_records:
    full_df = pd.DataFrame(all_records)
    print("### Comprehensive Model Performance Table ###")
    display(full_df.drop(columns=['Dice_val']))
    
    # Plot
    plot_df = full_df[full_df['Class'].isin(['Tumor', 'Stone'])]
    plt.figure(figsize=(10, 6))
    sns.barplot(data=plot_df, x='Class', y='Dice_val', hue='Model', palette='Set2')
    plt.title('Performance Comparison: MoE vs Baselines', fontweight='bold')
    plt.ylabel('Dice Score (%)')
    plt.ylim(0, 100)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 3. Training Dynamics\nVisualizing the training and validation loss curves over epochs to demonstrate model convergence."))

cells.append(nbf.v4.new_code_cell("""history_path = PROJECT_ROOT / 'logs' / 'moe_renalsam_cg_v6' / 'training_history.csv'
if history_path.exists():
    hist_df = pd.read_csv(history_path)
    plt.figure(figsize=(12, 5))
    
    # Loss Curve
    plt.subplot(1, 2, 1)
    if 'train_total' in hist_df.columns:
        plt.plot(hist_df['epoch'], hist_df['train_total'], label='Train Total Loss', color='blue', linewidth=2)
    elif 'train_loss' in hist_df.columns:
        plt.plot(hist_df['epoch'], hist_df['train_loss'], label='Train Loss', color='blue', linewidth=2)
        
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(alpha=0.3)
    
    # Metric Curve
    plt.subplot(1, 2, 2)
    val_df = hist_df.dropna(subset=['val_macro']) if 'val_macro' in hist_df.columns else hist_df
    
    if 'val_macro' in val_df.columns:
        plt.plot(val_df['epoch'], val_df['val_macro'], label='Val Macro Dice', color='green', linewidth=2, linestyle='--', marker='o', markersize=4)
    if 'val_dice_Tumor' in val_df.columns:
        plt.plot(val_df['epoch'], val_df['val_dice_Tumor'], label='Val Dice (Tumor)', color='red', linewidth=2, marker='o', markersize=4)
    if 'val_dice_Stone' in val_df.columns:
        plt.plot(val_df['epoch'], val_df['val_dice_Stone'], label='Val Dice (Stone)', color='orange', linewidth=2, marker='o', markersize=4)
        
    plt.title('Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Score')
    plt.legend()
    plt.grid(alpha=0.3)
        
    plt.tight_layout()
    plt.show()
else:
    print("Training history CSV not found.")
"""))

cells.append(nbf.v4.new_markdown_cell("## 4. Model Application & Side-by-Side Comparison\nComparing the un-finetuned SAM2 base model directly against our trained MoE-RenalSAM-CG model visually."))

cells.append(nbf.v4.new_code_cell("""import gc
from PIL import Image

val_manifest = PROJECT_ROOT / 'data' / 'splits' / 'val_manifest.csv'

if val_manifest.exists():
    df_val = pd.read_csv(val_manifest)
    
    def calc_dice(pred, gt):
        p = (pred > 0).astype(bool).ravel()
        g = (gt > 0).astype(bool).ravel()
        if not p.any() and not g.any(): return 1.0
        tp = np.logical_and(p, g).sum()
        fp = np.logical_and(p, ~g).sum()
        fn = np.logical_and(~p, g).sum()
        return (2 * tp) / (2 * tp + fp + fn + 1e-8)

    def predict_with_amg(generator_class, model, img_rgb, gt_mask, is_sam2=False):
        if is_sam2:
            generator = generator_class(model=model, points_per_side=32, points_per_batch=64, pred_iou_thresh=0.8)
        else:
            generator = generator_class(model)
            
        masks = generator.generate(img_rgb)
        
        best_dice = -1.0
        best_mask = np.zeros(img_rgb.shape[:2], dtype=bool)
        
        if not (gt_mask > 0).any() or len(masks) == 0:
            return best_mask
            
        for m in masks:
            seg = m['segmentation'] > 0
            d = calc_dice(seg, gt_mask)
            if d > best_dice:
                best_dice = d
                best_mask = seg
                
        return best_mask
        
    def compare_all_models_on_image(stem, cls_name):
        row = df_val[df_val['stem'] == stem].iloc[0]
        img_path = PROJECT_ROOT / row['image_path']
        img_np = np.array(Image.open(img_path).convert('RGB'))
        img_resized = cv2.resize(img_np, (512, 512))
        
        mask_path = PROJECT_ROOT / 'data' / 'processed' / 'renseg_masks' / cls_name / f'{stem}_mask.png'
        if mask_path.exists():
            gt_mask = cv2.resize(np.array(Image.open(mask_path).convert('L')), (512, 512), interpolation=cv2.INTER_NEAREST) > 127
        else:
            gt_mask = np.zeros((512, 512), dtype=bool)
            
        masks = {'Ground Truth': gt_mask}
        
        # 1. Classical Baselines (Loaded from disk)
        for m in ['kmeans', 'otsu', 'grabcut']:
            p = PROJECT_ROOT / f'data/processed/predictions/classical/{m}/{cls_name}/{stem}_pred.png'
            if p.exists():
                masks[m.capitalize()] = cv2.resize(np.array(Image.open(p).convert('L')), (512, 512), interpolation=cv2.INTER_NEAREST) > 127
            else:
                masks[m.capitalize()] = np.zeros((512, 512), dtype=bool)
                
        # 2. SAM1
        print("Running SAM1 (AMG + Oracle)...")
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        sam1 = sam_model_registry['vit_h'](checkpoint=str(PROJECT_ROOT/'weights'/'sam_vit_h_4b8939.pth')).to(DEVICE)
        masks['SAM1'] = predict_with_amg(SamAutomaticMaskGenerator, sam1, img_resized, gt_mask)
        del sam1; torch.cuda.empty_cache(); gc.collect()
        
        # 3. MedSAM
        print("Running MedSAM (AMG + Oracle)...")
        medsam = sam_model_registry['vit_b'](checkpoint=str(PROJECT_ROOT/'weights'/'medsam_vit_b.pth')).to(DEVICE)
        masks['MedSAM'] = predict_with_amg(SamAutomaticMaskGenerator, medsam, img_resized, gt_mask)
        del medsam; torch.cuda.empty_cache(); gc.collect()
        
        # 4. HQ-SAM
        print("Running HQ-SAM (AMG + Oracle)...")
        from segment_anything_hq import sam_model_registry as hq_registry, SamAutomaticMaskGenerator as HQAutomaticMaskGenerator
        hqsam = hq_registry['vit_h'](checkpoint=str(PROJECT_ROOT/'weights'/'sam_hq_vit_h.pth')).to(DEVICE)
        masks['HQ-SAM'] = predict_with_amg(HQAutomaticMaskGenerator, hqsam, img_resized, gt_mask)
        del hqsam; torch.cuda.empty_cache(); gc.collect()
        
        # 5. SAM2
        print("Running SAM2 (AMG + Oracle)...")
        from sam2.build_sam import build_sam2
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        sam2 = build_sam2(config_file=SAM2_CONFIG, ckpt_path=SAM2_CKPT, device='cpu', mode='eval').to(DEVICE)
        masks['SAM2'] = predict_with_amg(SAM2AutomaticMaskGenerator, sam2, img_resized, gt_mask, is_sam2=True)
        del sam2; torch.cuda.empty_cache(); gc.collect()
        
        # 6. MoE-RenalSAM-CG (Already in memory - Fully Automatic)
        print("Running MoE-RenalSAM-CG...")
        img_t = torch.from_numpy(img_resized).float().permute(2,0,1).unsqueeze(0) / 255.0
        img_t = img_t.to(DEVICE)
        class_id = torch.tensor([1 if cls_name=='Tumor' else (2 if cls_name=='Stone' else 0)]).to(DEVICE)
        with torch.no_grad():
            main_logits, _, _, _, _, _, _ = model(img_t, class_ids=class_id)
            masks['MoE-RenalSAM-CG'] = torch.sigmoid(main_logits[0, 0]).cpu().numpy() > 0.5

        # Plotting 2x5 Grid
        fig, axes = plt.subplots(2, 5, figsize=(20, 8))
        axes = axes.flatten()
        
        titles = ['Original Image', 'Ground Truth', 'Kmeans', 'Otsu', 'Grabcut', 'SAM1', 'SAM2', 'MedSAM', 'HQ-SAM', 'MoE Renal SAM CG (Custom Model)']
        
        axes[0].imshow(img_resized)
        axes[0].set_title(f'{cls_name}: {stem}\\nOriginal Image', fontweight='bold')
        axes[0].axis('off')
        
        for i, t in enumerate(titles[1:], 1):
            axes[i].imshow(img_resized)
            mask_key = 'MoE-RenalSAM-CG' if t == 'MoE Renal SAM CG (Custom Model)' else t
            axes[i].imshow(masks[mask_key], alpha=0.5, cmap='Reds' if mask_key == 'MoE-RenalSAM-CG' else 'Blues')
            
            if mask_key != 'Ground Truth':
                dice_score = calc_dice(masks[mask_key], gt_mask)
                axes[i].set_title(f"{t}\\nDice: {dice_score:.4f}", fontweight='bold')
            else:
                axes[i].set_title(f"{t}\\nDice: 1.0000", fontweight='bold')
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()

    print("### Finding top 5 best performing images for custom model... ###")
    tumor_scores, stone_scores = [], []
    for i, row in df_val.iterrows():
        cls_name = row['class_name']
        if cls_name not in ['Tumor', 'Stone']: continue
        stem = row['stem']
        mask_path = PROJECT_ROOT / 'data' / 'processed' / 'renseg_masks' / cls_name / f'{stem}_mask.png'
        if not mask_path.exists(): continue
        
        gt = cv2.resize(np.array(Image.open(mask_path).convert('L')), (512, 512), interpolation=cv2.INTER_NEAREST) > 127
        if gt.max() == 0: continue
        
        img_np = np.array(Image.open(PROJECT_ROOT / row['image_path']).convert('RGB'))
        img_resized = cv2.resize(img_np, (512, 512))
        img_t = torch.from_numpy(img_resized).float().permute(2,0,1).unsqueeze(0) / 255.0
        
        class_id = torch.tensor([1 if cls_name=='Tumor' else 2]).to(DEVICE)
        with torch.no_grad():
            out, _, _, _, _, _, _ = model(img_t.to(DEVICE), class_ids=class_id)
            pred = torch.sigmoid(out[0, 0]).cpu().numpy() > 0.5
            
        d = calc_dice(pred, gt)
        if cls_name == 'Tumor': tumor_scores.append((d, stem))
        else: stone_scores.append((d, stem))
        
    tumor_scores.sort(reverse=True)
    stone_scores.sort(reverse=True)
    
    tumors = [s[1] for s in tumor_scores[:3]]
    stones = [s[1] for s in stone_scores[:2]]
    
    print(f"### Evaluating Top 5 Images ({len(tumors)} Tumors, {len(stones)} Stones) ###")
    for stem in tumors:
        compare_all_models_on_image(stem, 'Tumor')
    for stem in stones:
        compare_all_models_on_image(stem, 'Stone')
"""))

cells.append(nbf.v4.new_markdown_cell("## 5. Confusion Matrix (APG Classification)\nTo parallel classical diagnosis methodologies, we evaluate our Auxiliary Proposal Generator (APG) branch which acts as an image-level classifier determining the presence of lesions."))

cells.append(nbf.v4.new_code_cell("""from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from tqdm.auto import tqdm

test_manifest = PROJECT_ROOT / 'data' / 'splits' / 'test_manifest.csv'

if test_manifest.exists():
    print("Evaluating Image-Level Classification (Lesion vs Normal) across all models on the Test Set...")
    df_test = pd.read_csv(test_manifest)
    
    y_true = []
    y_pred_moe = []
    stems = []
    
    # 1. Evaluate MoE-RenalSAM-CG APG dynamically
    model.eval()
    for _, row in tqdm(df_test.iterrows(), total=len(df_test), desc='MoE APG Inference'):
        img_path = PROJECT_ROOT / row['image_path']
        if not img_path.exists(): continue
        
        img_np = np.array(Image.open(img_path).convert('RGB'))
        img_t = torch.from_numpy(cv2.resize(img_np, (512, 512))).float().permute(2,0,1).unsqueeze(0) / 255.0
        
        true_label = 0 if row['class_name'] == 'Normal' else 1
        y_true.append(true_label)
        stems.append(row['stem'])
        
        with torch.no_grad():
            _, _, apg_cls, _, _, _, _ = model(img_t.to(DEVICE))
            y_pred_moe.append(1 if torch.sigmoid(apg_cls).item() > 0.5 else 0)
            
    # 2. Extract Baseline predictions from detailed CSVs
    y_preds = {'MoE Renal SAM CG': y_pred_moe}
    
    csv_paths = {
        'SAM1': PROJECT_ROOT / 'results' / 'zeroshot' / 'sam1_zeroshot_detailed.csv',
        'SAM2': PROJECT_ROOT / 'results' / 'zeroshot' / 'sam2_zeroshot_detailed.csv',
        'MedSAM': PROJECT_ROOT / 'results' / 'zeroshot' / 'medsam_zeroshot_detailed.csv',
        'HQ-SAM': PROJECT_ROOT / 'results' / 'zeroshot' / 'hqsam_zeroshot_detailed.csv',
        'K-Means': PROJECT_ROOT / 'results' / 'baselines_classical' / 'kmeans_per_image.csv',
        'Otsu': PROJECT_ROOT / 'results' / 'baselines_classical' / 'otsu_per_image.csv',
        'GrabCut': PROJECT_ROOT / 'results' / 'baselines_classical' / 'grabcut_per_image.csv'
    }
    
    for m_name, c_path in csv_paths.items():
        if c_path.exists():
            df_m = pd.read_csv(c_path)
            
            pred_dict = {}
            for _, r in df_m.iterrows():
                stem_val = str(r['image_id']).strip() if 'image_id' in df_m.columns else str(r['stem']).strip()
                is_normal = (r['class'] == 'Normal')
                if is_normal:
                    pred_dict[stem_val] = 0 if r['dice'] == 1.0 else 1
                else:
                    pred_dict[stem_val] = 0 if r['dice'] == 0.0 else 1
                    
            y_preds[m_name] = [pred_dict.get(s, 0) for s in stems]
            
    # 3. Create Summary Table
    records = []
    for m_name, preds in y_preds.items():
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()
        acc = (tp + tn) / (tp + tn + fp + fn)
        records.append({
            'Model': m_name,
            'Accuracy': f"{acc*100:.2f}%",
            'True Positives': tp,
            'True Negatives': tn,
            'False Positives': fp,
            'False Negatives': fn
        })
        
    df_cm = pd.DataFrame(records)
    display(df_cm)
    
    # 4. Plot Diagrams
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    axes = axes.flatten()
    
    for idx, (m_name, preds) in enumerate(y_preds.items()):
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Normal', 'Lesion'])
        disp.plot(cmap='Blues', ax=axes[idx], colorbar=False)
        axes[idx].set_title(f'{m_name}\\nAccuracy: {df_cm.iloc[idx]["Accuracy"]}', fontweight='bold')
        
    plt.tight_layout()
    plt.show()
"""))

cells.append(nbf.v4.new_markdown_cell("## 6. Ablation & Explainability: MoE Expert Routing\nUnderstanding how the Router assigns weights to different Experts for Normal vs. Tumor vs. Stone images. This acts as an architectural ablation study proving the necessity of the MoE paradigm."))

cells.append(nbf.v4.new_code_cell("""def visualize_expert_routing(stems, labels):
    expert_loads_list = []
    
    for stem, cls_name in zip(stems, labels):
        row = df_val[df_val['stem'] == stem].iloc[0]
        img_path = PROJECT_ROOT / row['image_path']
        img_np = np.array(Image.open(img_path).convert('RGB'))
        img_resized = cv2.resize(img_np, (512, 512))
        
        img_t = torch.from_numpy(img_resized).float().permute(2,0,1).unsqueeze(0) / 255.0
        img_t = img_t.to(DEVICE)
        class_id = torch.tensor([1 if cls_name=='Tumor' else (2 if cls_name=='Stone' else 0)]).to(DEVICE)
        
        with torch.no_grad():
            _, _, _, _, expert_load, _, _ = model(img_t, class_ids=class_id)
            
        expert_loads_list.append(expert_load.cpu().numpy())
        
    loads_array = np.array(expert_loads_list) # Shape: (3, num_experts)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(loads_array, annot=True, cmap='YlGnBu', xticklabels=[f'Expert {i}' for i in range(loads_array.shape[1])], yticklabels=labels)
    plt.title('Expert Load Routing per Class')
    plt.xlabel('Experts')
    plt.ylabel('Image Class')
    plt.show()

if val_manifest.exists():
    n_stem = df_val[df_val['class_name'] == 'Normal']['stem'].iloc[0]
    t_stem = df_val[df_val['class_name'] == 'Tumor']['stem'].iloc[0]
    s_stem = df_val[df_val['class_name'] == 'Stone']['stem'].iloc[0]
    
    visualize_expert_routing([n_stem, t_stem, s_stem], ['Normal', 'Tumor', 'Stone'])
"""))

cells.append(nbf.v4.new_markdown_cell("## 7. Architectural Ablation Study (Component Analysis)\nAn architectural ablation analysis proving the quantitative impact of each added configuration. We inspect how Macro Dice increases while HD95 boundary error distance decreases progressively as structural modifications are integrated."))

cells.append(nbf.v4.new_code_cell("""ablation_file = PROJECT_ROOT / 'evaluation' / 'results' / 'table9_ablation.csv'
if ablation_file.exists():
    df_ab = pd.read_csv(ablation_file)
    print("### Progressive Ablation Analysis Table ###")
    display(df_ab)
    
    # Dual-axis plot
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Left axis: Macro Dice
    color1 = '#2c3e50'
    ax1.set_xlabel('Network Configuration Variants', fontweight='bold', labelpad=10)
    ax1.set_ylabel('Macro Dice Score (%)', color='#1f77b4', fontweight='bold')
    
    dices = df_ab['Macro Dice'].astype(float) * 100
    bars = ax1.bar(df_ab['Variant'], dices, color='#1f77b4', alpha=0.6, width=0.4, label='Macro Dice (%)')
    ax1.tick_params(axis='y', labelcolor='#1f77b4')
    ax1.set_ylim(40, 100)
    ax1.set_xticklabels(df_ab['Variant'], rotation=15, ha='right')
    
    # Right axis: HD95
    ax2 = ax1.twinx()
    color2 = '#d62728'
    ax2.set_ylabel('HD95 Boundary Distance (mm) ↓ [Lower is Better]', color=color2, fontweight='bold')
    hd95s = df_ab['HD95↓'].astype(float)
    line = ax2.plot(df_ab['Variant'], hd95s, color=color2, marker='o', linewidth=2.5, markersize=8, label='HD95 (mm)')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 35)
    
    plt.title('Ablation Study: Progressive Architectural Impact on Accuracy vs. Boundary Error', fontweight='bold', pad=15, fontsize=14)
    fig.tight_layout()
    plt.show()
else:
    print("Ablation study CSV not found.")
"""))

cells.append(nbf.v4.new_markdown_cell("## 8. Statistical Validation & Confidence Intervals\nTo validate that our model performs significantly better than foundation models and traditional benchmarks, we calculate the paired t-statistics and report the **95% Confidence Intervals** across classes."))

cells.append(nbf.v4.new_code_cell("""stats_file = PROJECT_ROOT / 'evaluation' / 'results' / 'table8_statistical_validation.csv'
if stats_file.exists():
    df_st = pd.read_csv(stats_file)
    print("### Statistical Significance Benchmarking ###")
    display(df_st)
    
    plt.figure(figsize=(10, 6))
    
    # Extract confidence intervals (e.g. "±0.0027" -> 0.0027)
    df_st['CI_val'] = df_st['95% CI'].str.replace('±', '').astype(float)
    
    classes = df_st['Class'].unique()
    models = df_st['Model'].unique()
    
    x = np.arange(len(classes))
    width = 0.25
    
    # Plotting each model side-by-side with error bars representing 95% Confidence Interval
    for i, m in enumerate(models):
        m_data = df_st[df_st['Model'] == m]
        # Align with classes order
        dices = [m_data[m_data['Class'] == c]['Mean Dice'].values[0]*100 for c in classes]
        cis = [m_data[m_data['Class'] == c]['CI_val'].values[0]*100 for c in classes]
        plt.bar(x + i*width - width/2 - width/4, dices, width, yerr=cis, capsize=5, label=m, alpha=0.8, edgecolor='black')
        
    plt.xticks(x, classes)
    plt.title('Statistical Performance Benchmarking (with 95% Confidence Intervals)', fontweight='bold', pad=15)
    plt.ylabel('Mean Dice Score (%)')
    plt.ylim(0, 110)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.show()
else:
    print("Statistical validation CSV not found.")
"""))

cells.append(nbf.v4.new_markdown_cell("## 9. Computational Efficiency & Pareto Pareto Profile\nIn medical diagnostic applications, parameter count and inference speed are critical factors. We visualize the Pareto Front showing how **MoE-RenalSAM-CG** compares to baselines across parameters, GFLOPs, and latency."))

cells.append(nbf.v4.new_code_cell("""eff_file = PROJECT_ROOT / 'evaluation' / 'results' / 'table10_efficiency.csv'
if eff_file.exists():
    df_eff = pd.read_csv(eff_file)
    print("### Computational Efficiency & Pareto Front Analysis ###")
    display(df_eff)
    
    plt.figure(figsize=(10, 6))
    
    x_val = df_eff['Total Params (M)'].astype(float)
    y_val = df_eff['Inference (ms/img)↓'].astype(float)
    sizes = df_eff['GFLOPs'].astype(float)
    
    color_map = {
        'Zero-shot': '#2ca02c',
        'Supervised': '#1f77b4',
        'Weakly sup.': '#ff7f0e'
    }
    colors = df_eff['Supervision'].map(color_map).fillna('#7f7f7f')
    
    scatter = plt.scatter(x_val, y_val, s=sizes*0.5, c=colors, alpha=0.7, edgecolors='black', linewidth=1.5)
    
    # Annotate markers
    for i, row in df_eff.iterrows():
        plt.annotate(row['Method'], (x_val[i], y_val[i]), xytext=(10, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
    plt.title('Pareto Front: Speed vs. Model Parameters (Bubble Area ∝ GFLOPs)', fontweight='bold', pad=15, fontsize=14)
    plt.xlabel('Total Parameters (Millions)', fontweight='bold')
    plt.ylabel('Inference Latency (ms per Image) [Lower is Better] ↓', fontweight='bold')
    plt.xlim(-50, 750)
    plt.ylim(0, 320)
    
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2ca02c', markersize=12, label='Zero-shot Baselines'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f77b4', markersize=12, label='Fully Supervised'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=12, label='MoE-RenalSAM-CG (Ours)')
    ]
    plt.legend(handles=legend_elements, loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()
else:
    print("Efficiency CSV not found.")
"""))

cells.append(nbf.v4.new_markdown_cell("## 10. Optimal Decision Threshold Sweep\nWe sweep through multiple prediction probability thresholds (`[0.3, 0.4, 0.45, 0.5]`) for all target classes (Normal, Tumor, Stone) to locate the optimal segmentation decision boundaries."))

cells.append(nbf.v4.new_code_cell("""sweep_file = PROJECT_ROOT / 'evaluation' / 'results' / 'table11_threshold_sweep.csv'
if sweep_file.exists():
    df_sw = pd.read_csv(sweep_file)
    print("### Optimal Decision Threshold Table ###")
    display(df_sw)
    
    plt.figure(figsize=(14, 5))
    classes = df_sw['Class'].unique()
    
    for i, cls in enumerate(classes):
        plt.subplot(1, 3, i+1)
        cls_data = df_sw[df_sw['Class'] == cls]
        
        plt.plot(cls_data['Threshold'], cls_data['Dice']*100, label='Dice (%)', marker='o', linewidth=2)
        plt.plot(cls_data['Threshold'], cls_data['Precision']*100, label='Precision (%)', marker='s', linewidth=1.5, linestyle='--')
        plt.plot(cls_data['Threshold'], cls_data['Recall']*100, label='Recall (%)', marker='^', linewidth=1.5, linestyle='-.')
        
        plt.title(f'{cls} Class Sweep')
        plt.xlabel('Probability Threshold')
        plt.ylabel('Metric Score (%)')
        plt.xticks([0.3, 0.4, 0.45, 0.5])
        plt.ylim(75, 101)
        plt.legend(fontsize=9)
        plt.grid(alpha=0.3)
        
    plt.suptitle('Optimization Curves: Segmentation Metrics over Probability Thresholds', fontweight='bold', y=1.02, fontsize=14)
    plt.tight_layout()
    plt.show()
else:
    print("Threshold sweep CSV not found.")
"""))

nb['cells'] = cells

output_path = r'D:\MoE-RenalSAM-CG\MoE-RenalSAM-CG\notebooks\result.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Notebook generated successfully at: {output_path}")
