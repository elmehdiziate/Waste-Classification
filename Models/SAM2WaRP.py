"""
El Mehdi Ziate

WHAT IS SAM 2.1:
SAM 2 (Segment Anything Model 2, Ravi et al. 2024) is a foundation model
from Meta AI that segments any object in images and videos given a prompt.


HOW SAM 2.1 WORKS:
  A prompt (box, point, or mask) is passed alongside the image.
  SAM 2.1 encodes the image with a Hiera ViT backbone, encodes the
  prompt with a lightweight prompt encoder, then a mask decoder
  predicts a binary mask for the prompted object.

  No training on WaRP is needed: SAM 2.1 is zero-shot.

TWO MODES FOR WARP:

  Detection (WaRP-D):
    Input:  image + one point per object
    Output: bounding box derived from the predicted mask extent
    This gives zero-shot detection without any WaRP-D labels.

  Segmentation (WaRP-S):
    Input:  image + bounding boxes (from Faster R-CNN or ground truth)
    Output: pixel-level binary mask per box
    This gives zero-shot instance segmentation without WaRP-S labels.



Reference SAM 2:
  Ravi et al. (2024) SAM 2: Segment Anything in Images and Videos.
  arXiv:2408.00714. https://arxiv.org/abs/2408.00714
"""

import numpy as np
import torch
from pathlib import Path
from PIL import Image
from typing import Optional

try:
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("[SAM2WaRP] Install with: pip install sam2")


class SAM2WaRP:
    """
    SAM 2.1 wrapper for WaRP-D detection and WaRP-S segmentation.

    Parameters
    ----------
    model_id : HuggingFace model ID for SAM 2.1
    device   : 'cuda' or 'cpu'
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

        print(f"[SAM2WaRP] Loading {model_id}")
        self.predictor = SAM2ImagePredictor.from_pretrained(model_id)
        self.predictor.model.to(device)
        print(f"  Device : {device}")

    def _set_image(self, image: np.ndarray) -> None:
        """Encode image features — called once per image before prompting."""
        with torch.inference_mode():
            self.predictor.set_image(image)

    def predict_masks_from_boxes(
        self,
        image:   np.ndarray,
        boxes:   np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        WaRP-S segmentation: given bounding boxes, predict pixel masks.

        Each box produces one binary mask. Boxes come from either:
          - Faster R-CNN predictions on WaRP-D (full pipeline)
          - Ground truth WaRP-D annotations (evaluation mode)

        Parameters
        ----------
        image : (H, W, 3) uint8 numpy array
        boxes : (N, 4) float array  [x1, y1, x2, y2] in pixel coordinates

        Returns
        -------
        masks  : (N, H, W) bool array — True = object pixel
        scores : (N,) float array — SAM confidence per mask
        """
        self._set_image(image)
        boxes_t = torch.tensor(boxes, dtype=torch.float32, device=self.device)

        with torch.inference_mode():
            masks, scores, _ = self.predictor.predict(
                point_coords  = None,
                point_labels  = None,
                box           = boxes_t,
                multimask_output = False,
            )

        # masks: (N, 1, H, W) → (N, H, W) bool
        if masks.ndim == 4:
            masks = masks[:, 0]
        return masks.astype(bool), scores.squeeze(-1)

    def predict_boxes_from_points(
        self,
        image:  np.ndarray,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        WaRP-D detection: given one point per object, predict bounding boxes.

        SAM 2.1 predicts a mask for each point. We derive the bounding box
        from the tight axis-aligned rectangle around each predicted mask.

        Parameters
        ----------
        image  : (H, W, 3) uint8 numpy array
        points : (N, 2) float array  [x, y] pixel coordinates
                 one point per object of interest

        Returns
        -------
        boxes  : (N, 4) float array  [x1, y1, x2, y2]
        masks  : (N, H, W) bool array
        scores : (N,) float array
        """
        self._set_image(image)

        all_boxes  = []
        all_masks  = []
        all_scores = []

        # Process one point at a time — SAM handles multi-object via
        # separate calls since each point = one independent object prompt
        for point in points:
            pt     = np.array([[point]], dtype=np.float32)   # (1, 1, 2)
            labels = np.array([[1]], dtype=np.int32)          # 1 = foreground

            with torch.inference_mode():
                masks, scores, _ = self.predictor.predict(
                    point_coords     = pt[0],
                    point_labels     = labels[0],
                    multimask_output = False,
                )

            mask = masks[0].astype(bool)         # (H, W)
            score = float(scores[0])

            # Derive bounding box from mask extent
            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            if len(rows) == 0 or len(cols) == 0:
                continue
            box = np.array([
                float(cols[0]), float(rows[0]),
                float(cols[-1]), float(rows[-1])
            ])
            all_boxes.append(box)
            all_masks.append(mask)
            all_scores.append(score)

        if not all_boxes:
            H, W = image.shape[:2]
            return (np.zeros((0, 4)), np.zeros((0, H, W), dtype=bool),
                    np.zeros(0))

        return (np.stack(all_boxes), np.stack(all_masks),
                np.array(all_scores))

    def auto_segment(
        self,
        image:          np.ndarray,
        points_per_side: int = 32,
        score_thresh:    float = 0.7,
    ) -> list[dict]:
        """
        Automatic mask generation — no prompts needed.

        SAM 2.1 generates a grid of points, predicts a mask for each,
        then filters and deduplicates. Returns all objects found.

        Parameters
        ----------
        image            : (H, W, 3) uint8 numpy array
        points_per_side  : density of the point grid (32 = ~1024 points)
        score_thresh     : minimum SAM score to keep a mask

        Returns
        -------
        list of dicts with keys: 'mask', 'score', 'bbox'
        """
        try:
            from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
        except ImportError:
            raise ImportError("Automatic mask generation requires sam2 >= 1.0")

        generator = SAM2AutomaticMaskGenerator(
            model                  = self.predictor.model,
            points_per_side        = points_per_side,
            pred_iou_thresh        = score_thresh,
            stability_score_thresh = 0.9,
        )
        with torch.inference_mode():
            results = generator.generate(image)
        return results

    def evaluate_segmentation(
        self,
        warp_s_root: Path,
        split_file:  Path,
        boxes_source: str = "gt",
        n_samples:   int  = 100,
    ) -> dict:
        """
        Evaluate SAM 2.1 segmentation on WaRP-S.

        Loads images and ground truth masks from the WaRP-S directory
        structure, runs SAM with box prompts, and computes mIoU.

        Parameters
        ----------
        warp_s_root  : path to WaRP-S root (contains JPEGImages/, etc.)
        split_file   : path to ImageSets/val.txt or test.txt
        boxes_source : 'gt' = use ground truth boxes from annotations
                       'faster_rcnn' = use Faster R-CNN predictions
        n_samples    : number of images to evaluate

        Returns
        -------
        dict with keys: mean_iou, per_image_iou, total
        """
        warp_s_root = Path(warp_s_root)
        img_dir     = warp_s_root / "JPEGImages"
        seg_dir     = warp_s_root / "SegmentationObject"
        cls_dir     = warp_s_root / "SegmentationClass"

        names = split_file.read_text().strip().split("\n")[:n_samples]

        ious   = []
        for name in names:
            name = name.strip()
            if not name:
                continue

            img_path  = img_dir / f"{name}.jpg"
            seg_path  = seg_dir / f"{name}.png"
            if not img_path.exists() or not seg_path.exists():
                continue

            image_np  = np.array(Image.open(img_path).convert("RGB"))
            inst_mask = np.array(Image.open(seg_path).convert("L"))

            # Get instance ids (each unique value = one object)
            ids = np.unique(inst_mask)
            ids = ids[ids != 0]
            if len(ids) == 0:
                continue

            # Build ground truth boxes from instance masks
            gt_boxes = []
            gt_masks = []
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

            gt_boxes = np.array(gt_boxes)

            # Predict masks from ground truth boxes
            pred_masks, _ = self.predict_masks_from_boxes(image_np, gt_boxes)

            # IoU per instance
            for pred_m, gt_m in zip(pred_masks, gt_masks):
                inter = (pred_m & gt_m).sum()
                union = (pred_m | gt_m).sum()
                if union > 0:
                    ious.append(inter / union)

        mean_iou = float(np.mean(ious)) if ious else 0.0
        print(f"[SAM2WaRP] Segmentation mIoU: {mean_iou:.4f}  ({len(ious)} instances)")
        return {"mean_iou": mean_iou, "per_instance_iou": ious, "total": len(ious)}
