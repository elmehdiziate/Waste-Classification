"""
El Mehdi Ziate
SAM 2.1 for WaRP-D detection and WaRP-S segmentation.
Supports:
    - Zero-shot inference (no training) (4/28/2026)
    - LoRA fine-tuning on WaRP-S masks (1/5/2026)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from PIL import Image

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("[SAM2WaRP] Install: pip install sam2")


# LoRA linear layer 

class LoRALinear(nn.Module):
    """
    Drop-in replacement for nn.Linear with LoRA adaptation.

    W_new = W_frozen + (B @ A) * scaling
    A : (rank, in_features): random init * 0.01
    B : (out_features, rank): zero init  (LoRA starts as identity)

    Scaling = alpha / rank follows the original LoRA paper recommendation.
    """
    def __init__(self, linear: nn.Linear, rank: int, scaling: float):
        super().__init__()
        self.linear  = linear
        self.scaling = scaling
        self.A = nn.Parameter(torch.randn(rank, linear.in_features)  * 0.01)
        self.B = nn.Parameter(torch.zeros(linear.out_features, rank))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + (x @ self.A.T @ self.B.T) * self.scaling


# Main class 

class SAM2WaRP:
    """
    SAM 2.1 for WaRP-D detection and WaRP-S segmentation.

    Supports:
      - Zero-shot inference (no training)
      - LoRA fine-tuning on WaRP-S masks
    """

    def __init__(
        self,
        model_id: str = "facebook/sam2.1-hiera-large",
        device:   str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        if not SAM2_AVAILABLE:
            raise ImportError("pip install sam2")

        self.device   = device
        self.model_id = model_id
        self._lora_applied = False

        print(f"[SAM2WaRP] Loading {model_id}")
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id)
        self.predictor.model.to(device)
        total = sum(p.numel() for p in self.predictor.model.parameters())
        print(f"  Device     : {device}")
        print(f"  Parameters : {total:,}")

    # Image encoding 

    def _set_image(self, image: np.ndarray) -> None:
        """Encode image: must be called before any predict call."""
        if self._lora_applied:
            self.predictor.set_image(image)
        else:
            with torch.inference_mode():
                self.predictor.set_image(image)

    # Zero-shot segmentation

    def predict_masks_from_boxes(
        self,
        image: np.ndarray,
        boxes: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        WaRP-S segmentation: given bounding boxes -> pixel masks.
        """
        self._set_image(image)
        boxes_t = torch.tensor(boxes, dtype=torch.float32, device=self.device)

        ctx = torch.inference_mode() if not self._lora_applied \
              else torch.enable_grad()
        with ctx:
            masks, scores, _ = self.predictor.predict(
                point_coords     = None,
                point_labels     = None,
                box              = boxes_t,
                multimask_output = False,
            )

        if masks.ndim == 4:
            masks = masks[:, 0]
        if hasattr(scores, 'detach'):
            scores = scores.detach().cpu().numpy()
        return masks.astype(bool), np.array(scores).squeeze()

    # Zero-shot detection

    def predict_boxes_from_points(
        self,
        image:  np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        WaRP-D detection: given points -> bounding boxes (via mask extent).
        """
        self._set_image(image)
        all_boxes, all_masks, all_scores = [], [], []

        for point in points:
            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords     = np.array([[point]],  dtype=np.float32)[0],
                    point_labels     = np.array([[1]],      dtype=np.int32)[0],
                    multimask_output = False,
                )

            mask  = masks[0].astype(bool)
            rows  = np.where(mask.any(axis=1))[0]
            cols  = np.where(mask.any(axis=0))[0]
            if len(rows) == 0:
                continue
            all_boxes.append([float(cols[0]), float(rows[0]),
                               float(cols[-1]), float(rows[-1])])
            all_masks.append(mask)
            all_scores.append(float(scores[0]))

        if not all_boxes:
            H, W = image.shape[:2]
            return (np.zeros((0, 4)), np.zeros((0, H, W), bool), np.zeros(0))
        return (np.stack(all_boxes), np.stack(all_masks), np.array(all_scores))

    # Automatic segmentation 

    def auto_segment(
        self,
        image:           np.ndarray,
        points_per_side: int   = 32,
        score_thresh:    float = 0.7,
    ) -> list[dict]:
        """
        Fully automatic mask generation: no prompts.
        """
        from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        gen = SAM2AutomaticMaskGenerator(
            model                  = self.predictor.model,
            points_per_side        = points_per_side,
            pred_iou_thresh        = score_thresh,
            stability_score_thresh = 0.9,
        )
        with torch.inference_mode():
            return gen.generate(image)

    # Evaluation

    def evaluate_segmentation(
        self,
        warp_s_root: Path,
        split_file:  Path,
        n_samples:   int = 100,
    ) -> dict:
        """
        Compute mIoU on WaRP-S using GT box prompts.
        """
        warp_s_root = Path(warp_s_root)
        img_dir     = warp_s_root / "JPEGImages"
        seg_dir     = warp_s_root / "SegmentationObject"
        names       = split_file.read_text().strip().split("\n")[:n_samples]

        ious = []
        for name in names:
            name = name.strip()
            if not name:
                continue
            img_path = img_dir / f"{name}.jpg"
            seg_path = seg_dir / f"{name}.png"
            if not img_path.exists() or not seg_path.exists():
                continue

            image_np  = np.array(Image.open(img_path).convert("RGB"))
            inst_mask = np.array(Image.open(seg_path).convert("L"))
            ids = np.unique(inst_mask)
            ids = ids[ids != 0]
            if len(ids) == 0:
                continue

            gt_boxes, gt_masks = [], []
            for iid in ids:
                m    = inst_mask == iid
                rows = np.where(m.any(axis=1))[0]
                cols = np.where(m.any(axis=0))[0]
                if len(rows) == 0:
                    continue
                gt_boxes.append([float(cols[0]), float(rows[0]),
                                  float(cols[-1]), float(rows[-1])])
                gt_masks.append(m)

            if not gt_boxes:
                continue

            pred_masks, _ = self.predict_masks_from_boxes(
                image_np, np.array(gt_boxes)
            )
            for pred_m, gt_m in zip(pred_masks, gt_masks):
                inter = (pred_m & gt_m).sum()
                union = (pred_m | gt_m).sum()
                if union > 0:
                    ious.append(inter / union)

        mean_iou = float(np.mean(ious)) if ious else 0.0
        print(f"[SAM2WaRP] mIoU={mean_iou:.4f}  ({len(ious)} instances)")
        return {"mean_iou": mean_iou, "per_instance_iou": ious, "total": len(ious)}

    # LoRA fine-tuning 

    def apply_lora(self, rank: int = 16, alpha: int = 32) -> None:
        """
        Inject LoRA adapters into SAM 2.1 image encoder attention layers.

        Freezes all 224M original parameters. Adds trainable LoRA matrices
        A and B to every Q/K/V projection in the Hiera ViT image encoder.
        Also unfreezes the mask decoder (~4M) for full task adaptation.

        Only ~1-2M parameters become trainable (<1% of total).
        """
        # Freeze everything
        for param in self.predictor.model.parameters():
            param.requires_grad = False

        scaling = alpha / rank
        n_lora  = 0

        # Inject LoRA into image encoder Q/K/V attention projections
        for name, module in self.predictor.model.image_encoder.named_modules():
            if not isinstance(module, nn.Linear):
                continue
            if not any(x in name for x in ['q_proj', 'k_proj', 'v_proj']):
                continue

            lora   = LoRALinear(module, rank=rank, scaling=scaling).to(self.device)
            parts  = name.split('.')
            parent = self.predictor.model.image_encoder
            for p in parts[:-1]:
                parent = getattr(parent, p)
            setattr(parent, parts[-1], lora)
            n_lora += 1

        # Unfreeze mask decoder: small and task-critical
        for param in self.predictor.model.sam_mask_decoder.parameters():
            param.requires_grad = True

        trainable = sum(p.numel() for p in self.predictor.model.parameters()
                        if p.requires_grad)
        total     = sum(p.numel() for p in self.predictor.model.parameters())

        self._lora_applied = True
        print(f"[SAM2WaRP] LoRA applied — rank={rank}  alpha={alpha}")
        print(f"  LoRA layers : {n_lora}")
        print(f"  Trainable   : {trainable:,} / {total:,} "
              f"({trainable/total*100:.2f}%)")

    def train_lora(self, warp_s_root, split_file, val_file=None,
               epochs=10, lr=1e-4, save_path=None):
        if not self._lora_applied:
            raise RuntimeError("Call apply_lora() first")

        from torchvision.transforms.functional import resize as tv_resize

        warp_s_root = Path(warp_s_root)
        img_dir     = warp_s_root / "JPEGImages"
        seg_dir     = warp_s_root / "SegmentationObject"
        names       = [n.strip() for n in split_file.read_text().split('\n')
                    if n.strip()]

        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad,
                self.predictor.model.parameters()),
            lr=lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs
        )
        best_iou = 0.0

        # SAM 2.1 pixel mean/std used internally by the predictor
        PIXEL_MEAN = torch.tensor([0.485, 0.456, 0.406],
                                device=self.device).view(3,1,1)
        PIXEL_STD  = torch.tensor([0.229, 0.224, 0.225],
                                device=self.device).view(3,1,1)

        for epoch in range(1, epochs + 1):
            self.predictor.model.train()
            total_loss = 0.0
            n_batches  = 0

            for name in names:
                img_path = img_dir / f"{name}.jpg"
                seg_path = seg_dir / f"{name}.png"
                if not img_path.exists() or not seg_path.exists():
                    continue

                image_np  = np.array(Image.open(img_path).convert("RGB"))
                inst_mask = np.array(Image.open(seg_path).convert("L"))
                H, W      = image_np.shape[:2]
                ids       = np.unique(inst_mask)
                ids       = ids[ids != 0]
                if len(ids) == 0:
                    continue

                gt_boxes, gt_masks_list = [], []
                for iid in ids:
                    m    = inst_mask == iid
                    rows = np.where(m.any(axis=1))[0]
                    cols = np.where(m.any(axis=0))[0]
                    if len(rows) == 0:
                        continue
                    gt_boxes.append([float(cols[0]), float(rows[0]),
                                    float(cols[-1]), float(rows[-1])])
                    gt_masks_list.append(m)

                if not gt_boxes:
                    continue

                try:
                    # Normalise image exactly as SAM does internally
                    img_t = torch.from_numpy(
                        image_np.transpose(2,0,1).astype(np.float32) / 255.0
                    ).to(self.device)
                    img_t = (img_t - PIXEL_MEAN) / PIXEL_STD

                    # Resize to 1024 keeping aspect ratio with padding
                    img_t = F.interpolate(
                        img_t.unsqueeze(0), size=(1024,1024),
                        mode='bilinear', align_corners=False
                    )  

                    # Scale boxes to 1024 space
                    sx, sy = 1024/W, 1024/H
                    boxes_1024 = torch.tensor(
                        [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy]
                        for b in gt_boxes],
                        dtype=torch.float32, device=self.device
                    )

                    features    = self.predictor.model.image_encoder(img_t)
                    image_embed = features["vision_features"]     # (1,256,64,64)
                    # backbone_fpn[0]=(1,256,256,256)  [1]=(1,256,128,128)
                    # mask decoder upsampling expects finest resolution first
                    high_res = [
                        self.predictor.model.sam_mask_decoder.conv_s0(features["backbone_fpn"][0]),
                        self.predictor.model.sam_mask_decoder.conv_s1(features["backbone_fpn"][1]),
                    ]

                    # prompt encoder
                    with torch.no_grad():
                        sparse, dense = \
                            self.predictor.model.sam_prompt_encoder(
                                points=None, boxes=boxes_1024, masks=None
                            )

                    # mask decoder
                    masks_low, _, _, _ = \
                        self.predictor.model.sam_mask_decoder(
                            image_embeddings         = image_embed,
                            image_pe                 = self.predictor.model
                                                        .sam_prompt_encoder
                                                        .get_dense_pe(),
                            sparse_prompt_embeddings = sparse,
                            dense_prompt_embeddings  = dense,
                            multimask_output         = False,
                            repeat_image             = len(gt_boxes) > 1,
                            high_res_features        = high_res,
                        )

                    # Upsample to original size
                    pred_t = F.interpolate(
                        masks_low, size=(H, W),
                        mode='bilinear', align_corners=False
                    )

                    gt_t = torch.zeros(len(gt_masks_list), 1, H, W,
                                    device=self.device)
                    for i, m in enumerate(gt_masks_list):
                        gt_t[i,0] = torch.from_numpy(m.astype(np.float32))

                    bce   = F.binary_cross_entropy_with_logits(pred_t, gt_t)
                    p     = pred_t.sigmoid()
                    inter = (p * gt_t).sum()
                    dice  = 1 - (2*inter+1) / (p.sum() + gt_t.sum() + 1)
                    loss  = bce + dice

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad,
                            self.predictor.model.parameters()),
                        max_norm=1.0
                    )
                    optimizer.step()
                    scheduler.step()
                    total_loss += loss.item()
                    n_batches  += 1

                except Exception as e:
                    print(f"  [LoRA] Skipped: {type(e).__name__}: {e}")
                    continue

            avg_loss = total_loss / max(n_batches, 1)

            if val_file and Path(val_file).exists():
                self.predictor.model.eval()
                res     = self.evaluate_segmentation(
                    warp_s_root, Path(val_file), n_samples=50
                )
                val_iou = res["mean_iou"]
                print(f"Epoch {epoch:2d}/{epochs}  "
                    f"loss={avg_loss:.4f}  val_mIoU={val_iou:.4f}")
                if val_iou > best_iou and save_path:
                    best_iou = val_iou
                    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                    torch.save(self.predictor.model.state_dict(), save_path)
                    print(f"  Saved (mIoU={best_iou:.4f})")
            else:
                print(f"Epoch {epoch:2d}/{epochs}  loss={avg_loss:.4f}")

        print(f"Best mIoU: {best_iou:.4f}")
        return best_iou
    def load_lora_weights(self, checkpoint_path: Path) -> None:
        """Load saved LoRA + decoder weights. Call apply_lora() first."""
        if not self._lora_applied:
            raise RuntimeError("Call apply_lora() before load_lora_weights()")
        state = torch.load(checkpoint_path, map_location=self.device,
                           weights_only=True)
        self.predictor.model.load_state_dict(state)
        print(f"[SAM2WaRP] Loaded weights from {checkpoint_path}")