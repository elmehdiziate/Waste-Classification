"""
El Mehdi Ziate
-----------------
OPEN-VOCABULARY RECOGNITION : CLIP + LLaVA + Hybrid Methods: what I wanted to do here is
to implement a range of open-vocabulary methods, from pure CLIP zero-shot to hybrid RAG 
with LLaVA, and compare them all in one place. I want to see if using my own fine-tuned 
Swin features for retrieval can give LLaVA better context than CLIP's often wrong guesses.
-----------------
WHY OPEN VOCABULARY?
--------------------
All fine-tuned classifiers (Swin, ViT, MobileViT) output exactly 28 fixed
labels. Open-vocabulary models break this constraint by grounding vision in
language because they can understand descriptions, not just memorised label indices.

This module implements SEVEN complementary methods:

    Clip:
    1. CLIPClassifier: it uses zero-shot text-image cosine similarity
    2. CLIPPrototypeClassifier: it uses image-image cosine similarity using train set centroids (no text at all : pure visual RAG)
    3. CLIPkNNClassifier: it uses k-nearest-neighbour over all train embeddings with optional label-weighted voting

    LLAVA-BASED
    4. LLaVAClassifier: it uses four prompting modes: zero_shot, one_shot, few_shot, open_vocab
    5. CascadeVLMClassifier: it uses CLIP to filter top-K candidates ranked by probability and LLaVA only handles uncertain cases this is based on a paper(Wei et al. EMNLP Findings 2024)

    HYBRID (Swin/MobileViT + LLaVA)
    6. HybridClassifier        here I have used my trained Swin model (we can use the best model for now swin is the best) to  extracts features
                               the I have used kNN on those features to find nearest train images after that LLaVA sees those images as context
    7. EnsembleClassifier      soft-vote ensemble: CLIP probs + Swin logits weighted sum and the took the best of both worlds

RAG IMPLEMENTATIONS COMPARED
-----------------------------
  CLIPPrototypeClassifier:
    - Compute one mean embedding per class (centroid) from all train images
    - Classify by nearest centroid
    - Cleaner than kNN : one comparison per class, not per image
    - Based on: Snell et al. (2017) Prototypical Networks

  CLIPkNNClassifier:
    - Build (N, D) embedding index over all 8767 train images
    - At query time: cosine similarity → top-K neighbours → majority vote
    - Label-weighted: weight each vote by similarity score
    - Better than prototype for multi-modal distributions

  HybridClassifier :
    - Use the fine-tuned Swin backbone as feature extractor
    - These features ARE trained on WaRP-C implies much better than CLIP features
    - Build kNN index in Swin feature space (768-dim) instead of CLIP space
    - Retrieve closest training images and the pass to LLaVA as context
    - LLaVA now sees genuinely relevant examples, not CLIP's wrong guesses

REFERENCES
----------
Radford et al. (2021) CLIP. ICML 2021. https://arxiv.org/abs/2103.00020
Liu et al. (2023) LLaVA. NeurIPS 2023. https://arxiv.org/abs/2304.10592
Wei et al. (2024) CascadeVLM. EMNLP Findings 2024. arxiv:2405.11301
Snell et al. (2017) Prototypical Networks. NeurIPS 2017. arxiv:1703.05175
"""

import base64
import json
import time
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm

try:
    import open_clip
    OPEN_CLIP_AVAILABLE = True
except ImportError:
    OPEN_CLIP_AVAILABLE = False
    print("[OpenVocabulary] open_clip not found. pip install open-clip-torch")

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# This is used for clip-based methods
# Maximally distinctive : each description emphasises the one feature that
# most separates this class from visually similar ones.

WARP_CLASS_DESCRIPTIONS: dict[str, str] = {
    "bottle-blue":              "a crumpled flattened empty blue plastic bottle",
    "bottle-blue-full":         "a round inflated full blue plastic bottle with air inside",
    "bottle-blue5l":            "a large 5-litre empty blue plastic water jug",
    "bottle-blue5l-full":       "a large 5-litre full blue plastic water jug with liquid",
    "bottle-dark":              "a dark brown or black empty compressed plastic bottle",
    "bottle-dark-full":         "a dark brown or black full rigid plastic bottle",
    "bottle-green":             "a green empty squashed plastic bottle",
    "bottle-green-full":        "a green full upright plastic bottle",
    "bottle-milk":              "a white opaque empty flat plastic milk bottle",
    "bottle-milk-full":         "a white opaque full plastic milk bottle with liquid",
    "bottle-multicolor":        "a colourful plastic bottle with printed brand label and logo",
    "bottle-multicolorv-full":  "a full colourful plastic bottle with brand label still inflated",
    "bottle-oil":               "an empty yellow or clear plastic cooking oil bottle",
    "bottle-oil-full":          "a full heavy plastic cooking oil bottle with yellow liquid",
    "bottle-transp":            "a clear transparent empty crushed crinkled plastic bottle",
    "bottle-transp-full":       "a clear transparent full round bulging plastic bottle",
    "bottle-yogurt":            "a small round white plastic yogurt pot or container",
    "canister":                 "a cylindrical metal or plastic fuel canister with handle",
    "cans":                     "a dented crushed silver aluminium soda drink can",
    "detergent-box":            "a rectangular cardboard laundry detergent box",
    "detergent-color":          "a brightly coloured plastic detergent or soap bottle",
    "detergent-transparent":    "a clear transparent plastic detergent bottle with pump",
    "detergent-white":          "a plain white plastic detergent or cleaning product bottle",
    "glass-dark":               "a dark brown or green glass wine or beer bottle",
    "glass-green":              "a green glass bottle",
    "glass-transp":             "a clear transparent glass bottle",
    "juice-cardboard":          "a rectangular cardboard juice box or tetra pak carton",
    "milk-cardboard":           "a white rectangular cardboard milk carton with cap",
}



#  SHARED INDEX — reused across multiple classifiers
class EmbeddingIndex:
    """
    Shared embedding index: (N, D) matrix + parallel paths + labels.

    Built once, reused by CLIPkNNClassifier, CLIPPrototypeClassifier,
    HybridClassifier. Pass the same index to all of them.

    Parameters
    ----------
    embeddings : (N, D) normalised float32 tensor
    paths      : list of N image paths
    labels     : list of N class name strings
    class_names: sorted list of 28 class names
    """

    def __init__(
        self,
        embeddings:  torch.Tensor,
        paths:       list,
        labels:      list,
        class_names: list[str],
    ):
        self.embeddings  = embeddings   # (N, D) normalised
        self.paths       = paths
        self.labels      = labels
        self.class_names = class_names
        self.label_to_idx = {c: i for i, c in enumerate(class_names)}

    def query(self, feat: torch.Tensor, k: int = 5):
        """
        Find k nearest neighbours by cosine similarity.

        Parameters
        ----------
        feat : (1, D) or (D,) normalised query embedding
        k    : number of neighbours

        Returns
        -------
        indices    : (k,) tensor of row indices into self.embeddings
        sims       : (k,) tensor of cosine similarities
        neighbour_labels : list of k class name strings
        """
        if feat.dim() == 1:
            feat = feat.unsqueeze(0)
        sims     = (feat @ self.embeddings.T).squeeze(0)  # (N,)
        top      = sims.topk(k)
        indices  = top.indices
        neighbour_labels = [self.labels[i.item()] for i in indices]
        return indices, top.values, neighbour_labels

    def get_prototypes(self) -> torch.Tensor:
        """
        Compute per-class mean embedding (prototype).

        Returns
        -------
        prototypes : (C, D) normalised tensor, row i = class i prototype
        """
        C = len(self.class_names)
        D = self.embeddings.shape[1]
        proto = torch.zeros(C, D)
        counts = torch.zeros(C)
        for emb, lbl in zip(self.embeddings, self.labels):
            idx = self.label_to_idx.get(lbl, -1)
            if idx >= 0:
                proto[idx] += emb
                counts[idx] += 1
        mask = counts > 0
        proto[mask] = F.normalize(proto[mask], dim=-1)
        return proto  # (C, D)

    @staticmethod
    def build_clip_index(
        clip_model,
        train_loader,
        save_path: Optional[Path] = None,
        verbose:   bool = True,
    ) -> "EmbeddingIndex":
        """
        Build EmbeddingIndex using CLIP ViT-B/32 visual encoder.

        Parameters
        ----------
        clip_model   : CLIPClassifier instance
        train_loader : DataLoader (dataset.samples used)
        save_path    : optional .pt path to save/reload
        """
        if save_path and Path(save_path).exists():
            print(f"[EmbeddingIndex] Loading CLIP index from {save_path}")
            data = torch.load(save_path, map_location="cpu")
            return EmbeddingIndex(
                data["embeddings"],
                [Path(p) for p in data["paths"]],
                data["labels"],
                data["class_names"],
            )

        dataset = train_loader.dataset
        samples = dataset.samples
        embs, paths, labels = [], [], []

        if verbose:
            print(f"[EmbeddingIndex] Building CLIP index ({len(samples)} images)...")

        for img_path, label_idx in tqdm(samples, disable=not verbose):
            img  = Image.open(img_path).convert("RGB")
            feat = clip_model.encode_image(img).squeeze(0).cpu()
            embs.append(feat)
            paths.append(Path(img_path))
            labels.append(clip_model.class_names[label_idx])

        index = EmbeddingIndex(
            torch.stack(embs),
            paths,
            labels,
            clip_model.class_names,
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "embeddings":  index.embeddings,
                "paths":       [str(p) for p in index.paths],
                "labels":      index.labels,
                "class_names": index.class_names,
            }, save_path)
            print(f"  Saved → {save_path}")

        return index

    @staticmethod
    def build_swin_index(
        swin_model,
        train_loader,
        device:    str = "cuda",
        save_path: Optional[Path] = None,
        verbose:   bool = True,
    ) -> "EmbeddingIndex":
        """
        Build EmbeddingIndex using your fine-tuned Swin backbone.

        WHY THIS IS BETTER THAN CLIP:
        Swin was trained on WaRP-C → its features discriminate exactly
        the 28 waste classes. CLIP features were trained on internet images
        and struggle with industrial conveyor belt shots.

        Extracts 768-dim features from swin_model.backbone (before head).

        Parameters
        ----------
        swin_model   : SwinTransformerWaRP instance (fine-tuned, eval mode)
        train_loader : DataLoader
        """
        import torchvision.transforms as T
        from Pipeline_.preprocessor import PadToSquare

        if save_path and Path(save_path).exists():
            print(f"[EmbeddingIndex] Loading Swin index from {save_path}")
            data = torch.load(save_path, map_location="cpu")
            return EmbeddingIndex(
                data["embeddings"],
                [Path(p) for p in data["paths"]],
                data["labels"],
                data["class_names"],
            )

        transform = T.Compose([
            PadToSquare("reflect"),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        swin_model.eval()
        dataset = train_loader.dataset
        samples = dataset.samples
        embs, paths, labels = [], [], []
        class_names = dataset.classes

        if verbose:
            print(f"[EmbeddingIndex] Building Swin index ({len(samples)} images)...")

        with torch.no_grad():
            for img_path, label_idx in tqdm(samples, disable=not verbose):
                img  = Image.open(img_path).convert("RGB")
                x    = transform(img).unsqueeze(0).to(device)
                feat = swin_model.backbone(x)          # (1, 768)
                feat = F.normalize(feat, dim=-1).squeeze(0).cpu()
                embs.append(feat)
                paths.append(Path(img_path))
                labels.append(class_names[label_idx])

        index = EmbeddingIndex(
            torch.stack(embs),
            paths,
            labels,
            class_names,
        )

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "embeddings":  index.embeddings,
                "paths":       [str(p) for p in index.paths],
                "labels":      index.labels,
                "class_names": index.class_names,
            }, save_path)
            print(f"  Saved → {save_path}")

        return index


#  METHOD 1: CLIP ZERO-SHOT (text-image cosine similarity)

class CLIPClassifier:
    """
    Method 1: Zero-shot classification via CLIP text-image similarity.

    Encodes 28 text descriptions → computes cosine similarity with query image.
    Also used as the visual encoder for all other methods.

    Reference: Radford et al. (2021) ICML. https://arxiv.org/abs/2103.00020
    """

    def __init__(
        self,
        class_names:  list[str],
        model_name:   str = "ViT-B-32",
        pretrained:   str = "openai",
        device:       str = "cuda" if torch.cuda.is_available() else "cpu",
        descriptions: Optional[dict[str, str]] = None,
    ):
        if not OPEN_CLIP_AVAILABLE:
            raise ImportError("pip install open-clip-torch")

        self.class_names  = class_names
        self.device       = device
        self.descriptions = descriptions or WARP_CLASS_DESCRIPTIONS

        print(f"[CLIPClassifier] Loading {model_name} pretrained='{pretrained}'")
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.model     = self.model.to(device).eval()
        self._text_features = self._encode_text()

        print(f"  Device: {device}  Classes: {len(class_names)}")

    def _encode_text(self) -> torch.Tensor:
        desc = [self.descriptions.get(c, c.replace("-", " ")) for c in self.class_names]
        tokens = self.tokenizer(desc).to(self.device)
        with torch.no_grad():
            f = F.normalize(self.model.encode_text(tokens), dim=-1)
        return f  # (28, D)

    def encode_image(self, img: Image.Image) -> torch.Tensor:
        """Encode PIL image → (1, D) normalised CLIP embedding."""
        x = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            f = F.normalize(self.model.encode_image(x), dim=-1)
        return f

    def predict(self, img, return_all=False):
        feat   = self.encode_image(img)
        logits = 100.0 * feat @ self._text_features.T
        probs  = logits.softmax(dim=-1).squeeze(0)
        idx    = probs.argmax().item()
        return idx, probs[idx].item(), (probs if return_all else None)

    def evaluate(self, test_loader, verbose=True) -> dict:
        correct = correct_5 = total = 0
        per_class = {c: {"c": 0, "t": 0} for c in self.class_names}
        for img_path, true_lbl in tqdm(test_loader.dataset.samples,
                                       disable=not verbose,
                                       desc="CLIP eval"):
            img = Image.open(img_path).convert("RGB")
            pred, _, probs = self.predict(img, return_all=True)
            if pred == true_lbl:
                correct += 1
                per_class[self.class_names[true_lbl]]["c"] += 1
            if true_lbl in probs.topk(5).indices.tolist():
                correct_5 += 1
            per_class[self.class_names[true_lbl]]["t"] += 1
            total += 1
        acc  = correct / total * 100
        acc5 = correct_5 / total * 100
        pca  = {c: (v["c"]/v["t"]*100 if v["t"] > 0 else 0)
                for c, v in per_class.items()}
        if verbose:
            print(f"\n  CLIP  Top-1: {acc:.2f}%  Top-5: {acc5:.2f}%")
        return {"method": "CLIP_zero_shot", "accuracy": acc,
                "top5": acc5, "per_class": pca}

    def compare_descriptions(self, img, n_top=5):
        _, _, p = self.predict(img, return_all=True)
        for i, idx in enumerate(p.topk(n_top).indices.tolist()):
            print(f"  {i+1}. {self.class_names[idx]:<32} {p[idx]:.1%}"
                  f"  — {self.descriptions.get(self.class_names[idx],'')}")


#  METHOD 2: CLIP PROTOTYPE (image-image, class centroid)

class CLIPPrototypeClassifier:
    """
    Method 2: Nearest-class-centroid in CLIP embedding space.

    Compute one mean embedding per class (prototype) from all training images.
    Classify by cosine similarity to nearest prototype.

    WHY BETTER THAN TEXT-BASED CLIP:
    Uses actual training image distributions, not text descriptions.
    Avoids the domain gap between internet captions and WaRP-C imagery.

    Reference: Snell et al. (2017) Prototypical Networks for Few-shot
    Learning. NeurIPS 2017. https://arxiv.org/abs/1703.05175
    """

    def __init__(self, clip_model: CLIPClassifier, index: EmbeddingIndex):
        self.clip       = clip_model
        self.index      = index
        self.class_names = clip_model.class_names
        self.prototypes = index.get_prototypes()  # (C, D)
        print(f"[CLIPPrototype] Prototypes: {self.prototypes.shape}")

    def predict(self, img: Image.Image):
        feat = self.clip.encode_image(img).squeeze(0).cpu()  # (D,)
        sims = feat @ self.prototypes.T                       # (C,)
        idx  = sims.argmax().item()
        return idx, sims[idx].item()

    def evaluate(self, test_loader, verbose=True) -> dict:
        correct = total = 0
        per_class = {c: {"c": 0, "t": 0} for c in self.class_names}
        for img_path, true_lbl in tqdm(test_loader.dataset.samples,
                                       disable=not verbose,
                                       desc="Prototype eval"):
            img = Image.open(img_path).convert("RGB")
            pred, _ = self.predict(img)
            if pred == true_lbl:
                correct += 1
                per_class[self.class_names[true_lbl]]["c"] += 1
            per_class[self.class_names[true_lbl]]["t"] += 1
            total += 1
        acc = correct / total * 100
        pca = {c: (v["c"]/v["t"]*100 if v["t"] > 0 else 0)
               for c, v in per_class.items()}
        if verbose:
            print(f"\n  CLIPPrototype  Top-1: {acc:.2f}%")
        return {"method": "CLIP_prototype", "accuracy": acc, "per_class": pca}



#  METHOD 3: CLIP kNN (image-image, k-nearest-neighbour)

class CLIPkNNClassifier:
    """
    Method 3: k-Nearest-Neighbour classification in CLIP embedding space.

    Build (N, D) index of all 8767 training images with CLIP embeddings.
    Classify query by majority vote of k nearest neighbours, weighted by
    cosine similarity score.

    WHY WEIGHTED VOTE:
    A neighbour with similarity 0.95 should outweigh one with 0.60.
    Simple majority vote treats all neighbours equally — worse performance.

    Parameters
    ----------
    k : number of neighbours (default 7 — odd to avoid ties)
    """

    def __init__(self, clip_model: CLIPClassifier, index: EmbeddingIndex,
                 k: int = 7):
        self.clip        = clip_model
        self.index       = index
        self.class_names = clip_model.class_names
        self.k           = k
        print(f"[CLIPkNN] k={k}  index size: {index.embeddings.shape}")

    def predict(self, img: Image.Image):
        feat    = self.clip.encode_image(img).squeeze(0).cpu()
        _, sims, nbr_labels = self.index.query(feat, k=self.k)

        # Similarity-weighted vote
        votes = {}
        for lbl, sim in zip(nbr_labels, sims.tolist()):
            votes[lbl] = votes.get(lbl, 0.0) + sim
        pred_cls = max(votes, key=votes.get)
        pred_idx = self.class_names.index(pred_cls)
        return pred_idx, votes[pred_cls] / sum(votes.values())

    def evaluate(self, test_loader, verbose=True) -> dict:
        correct = total = 0
        per_class = {c: {"c": 0, "t": 0} for c in self.class_names}
        for img_path, true_lbl in tqdm(test_loader.dataset.samples,
                                       disable=not verbose,
                                       desc=f"kNN(k={self.k}) eval"):
            img = Image.open(img_path).convert("RGB")
            pred, _ = self.predict(img)
            if pred == true_lbl:
                correct += 1
                per_class[self.class_names[true_lbl]]["c"] += 1
            per_class[self.class_names[true_lbl]]["t"] += 1
            total += 1
        acc = correct / total * 100
        pca = {c: (v["c"]/v["t"]*100 if v["t"] > 0 else 0)
               for c, v in per_class.items()}
        if verbose:
            print(f"\n  CLIPkNN(k={self.k})  Top-1: {acc:.2f}%")
        return {"method": f"CLIP_kNN_k{self.k}", "accuracy": acc,
                "per_class": pca}


#  LLAVA SHARED UTILITIES
LLAVA_SYSTEM = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the "
    "user's questions."
)

def _encode_image_b64(img: Image.Image, size: int = 336) -> str:
    """Pad to square → resize → base64 JPEG for Ollama API."""
    w, h   = img.size
    side   = max(w, h)
    canvas = Image.new("RGB", (side, side), (128, 128, 128))
    canvas.paste(img, ((side - w) // 2, (side - h) // 2))
    canvas = canvas.resize((size, size), Image.LANCZOS)
    buf    = BytesIO()
    canvas.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def _call_ollama(
    system: str, user: str, images: list[str],
    ollama_url: str = "http://localhost:11434",
    model: str = "llava:13b",
    max_tok: int = 60,
    temp: float = 0.0,
    timeout: int = 120,
) -> str:
    payload = {
        "model":    model,
        "stream":   False,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user, "images": images},
        ],
        "options": {"temperature": temp, "num_predict": max_tok},
    }
    r = requests.post(f"{ollama_url}/api/chat", json=payload, timeout=timeout)
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def _parse_to_class(response: str, class_names: list[str]) -> Optional[int]:
    """Parse LLaVA response → class index. Handles ANSWER:/CLASSIFICATION: tags."""
    clean = response.lower().strip()
    for tag in ["answer:", "classification:"]:
        if tag in clean:
            after = clean.split(tag)[-1].strip()
            first = after.split()[0].strip(".,;:\n") if after.split() else ""
            for i, c in enumerate(class_names):
                if c.lower() == first:
                    return i
            for i, c in enumerate(class_names):
                if c.lower() in after:
                    return i
    for i, c in enumerate(class_names):
        if c.lower() == clean:
            return i
    for i, c in enumerate(class_names):
        if c.lower() in clean:
            return i
    scores = [sum(ch in clean for ch in c.lower()) / max(len(c), 1)
              for c in class_names]
    best = int(np.argmax(scores))
    return best if scores[best] > 0.5 else None


#  METHOD 4: LLaVA (four prompting modes)

class LLaVAClassifier:
    """
    Method 4: LLaVA-13b with four prompting strategies.

    Uses the correct LLaVA-1.5 system prompt format from HuggingFace docs.
    All four modes use the /api/chat endpoint with system + user roles.

    Modes: zero_shot | one_shot | few_shot | open_vocab (CoT)
    """

    ZERO_SHOT_USER = (
        "You are analysing a waste item from an industrial recycling plant.\n\n"
        "Step 1: Describe what you see — colour, material, shape, full or empty.\n"
        "Step 2: Choose EXACTLY ONE from: {class_list}\n\n"
        "Format:\nI see: <description>\nANSWER: <category>"
    )
    ONE_SHOT_USER = (
        "You are classifying recycling plant waste.\n\n"
        "FIRST image: confirmed '{cls}' from our database.\n"
        "SECOND image: item to classify.\n\n"
        "Categories: {class_list}\n\n"
        "I see: <description of second image>\nANSWER: <category>"
    )
    FEW_SHOT_USER = (
        "You are classifying recycling plant waste.\n\n"
        "First THREE images are confirmed examples:\n{labels}\n"
        "LAST image: item to classify.\n\n"
        "Categories: {class_list}\n\n"
        "I see: <description of last image>\nANSWER: <category>"
    )
    OPEN_VOCAB_USER = (
        "Analyse this waste item step by step:\n\n"
        "<SUMMARY>One sentence: what is the main object?</SUMMARY>\n\n"
        "<CAPTION>Detail: colour, transparency, material, shape, "
        "full/empty/crushed, labels, damage.</CAPTION>\n\n"
        "<REASONING>Which waste category and why?</REASONING>\n\n"
        "<CONCLUSION>Choose exactly one:\n{class_list}\n"
        "Write: CLASSIFICATION: <name></CONCLUSION>"
    )

    def __init__(self, class_names, train_root=None, clip_index=None,
                 model="llava:13b", ollama_url="http://localhost:11434",
                 temperature=0.0, timeout=120):
        self.class_names  = class_names
        self.train_root   = Path(train_root) if train_root else None
        self.clip_index   = clip_index   # EmbeddingIndex for shot retrieval
        self.model        = model
        self.ollama_url   = ollama_url
        self.temperature  = temperature
        self.timeout      = timeout
        self._cls_str     = ", ".join(class_names)
        self._shot_cache: dict = {}
        self._check_ollama()
        print(f"[LLaVA] {model}  modes: zero_shot|one_shot|few_shot|open_vocab")

    def _check_ollama(self):
        try:
            r = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            models = [m["name"] for m in r.json().get("models", [])]
            ok = any(self.model in m for m in models)
            print(f"[LLaVA] Ollama {'OK' if ok else 'WARNING: model not found'}")
        except Exception as e:
            print(f"[LLaVA] WARNING: {e}")

    def _get_shots(self, class_name, n=3):
        """Load n reference images from training folder."""
        if self.train_root is None:
            return []
        if class_name not in self._shot_cache:
            d = self.train_root / class_name
            if not d.exists():
                return []
            paths = sorted(p for p in d.iterdir()
                          if p.suffix.lower() in {".jpg",".jpeg",".png"})
            self._shot_cache[class_name] = paths[:n]
        return [_encode_image_b64(Image.open(p).convert("RGB"))
                for p in self._shot_cache[class_name]]

    def predict(self, img, mode="zero_shot", true_label=None, verbose=False):
        b64 = _encode_image_b64(img)

        if mode == "zero_shot":
            user   = self.ZERO_SHOT_USER.format(class_list=self._cls_str)
            images = [b64]
            tok, temp = 60, self.temperature

        elif mode in ("one_shot", "few_shot"):
            n = 1 if mode == "one_shot" else 3
            # Use CLIP index if available, else first n from train folder
            if self.clip_index is not None:
                feat = None  # will be passed in by caller if needed
                # Fallback: use alphabetically first class
                shot_cls = self.class_names[0]
            else:
                shot_cls = self.class_names[0]
            shots = self._get_shots(shot_cls, n=n)
            if mode == "one_shot":
                user = self.ONE_SHOT_USER.format(
                    cls=shot_cls, class_list=self._cls_str)
            else:
                lbl_str = "\n".join(f"Image {i+1}: {shot_cls}"
                                    for i in range(len(shots)))
                user = self.FEW_SHOT_USER.format(
                    labels=lbl_str, class_list=self._cls_str)
            images = shots + [b64]
            tok, temp = 60, self.temperature

        elif mode == "open_vocab":
            user   = self.OPEN_VOCAB_USER.format(class_list=self._cls_str)
            images = [b64]
            tok, temp = 350, max(self.temperature, 0.1)

        else:
            raise ValueError(f"Unknown mode: {mode}")

        try:
            resp = _call_ollama(LLAVA_SYSTEM, user, images,
                                self.ollama_url, self.model, tok, temp,
                                self.timeout)
        except Exception as e:
            return None, ""

        pred = _parse_to_class(resp, self.class_names)
        if verbose:
            pname = self.class_names[pred] if pred is not None else "FAIL"
            m = (" ✓" if pname==true_label else f" ✗ (true:{true_label})") \
                if true_label else ""
            if mode == "open_vocab":
                print(f"\n  [open_vocab]\n  {resp}\n  → {pname}{m}")
            else:
                print(f"\n  [{mode}] \"{resp}\" → {pname}{m}")
        return pred, resp


#  METHOD 5: CascadeVLM (CLIP entropy → LLaVA on top-K)

class CascadeVLMClassifier:
    """
    Method 5: CascadeVLM — CLIP filters candidates, LLaVA handles hard cases.

    Algorithm:
      1. CLIP classifies → top-K candidates sorted by probability
      2. Compute Shannon entropy of CLIP's distribution
         - H ≤ threshold → CLIP is confident → return CLIP answer (no LLaVA)
         - H > threshold → CLIP is uncertain → ask LLaVA to pick from top-K
      3. LLaVA sees only K candidates IN CLIP'S RANKED ORDER
         (ordering is critical: +33.9% accuracy per paper vs random order)

    Key insight: never show LLaVA all 28 classes — it anchors on early ones.
    Show only CLIP's top-K, already ranked. LLaVA's job is to refine CLIP.

    Reference: Wei et al. (2024) CascadeVLM. EMNLP Findings 2024.
    https://arxiv.org/abs/2405.11301
    """

    CASCADE_USER = (
        "You are classifying a waste item on a recycling conveyor.\n\n"
        "Visual analysis suggests it is one of these {k} categories "
        "(ranked most to least likely):\n{candidates}\n\n"
        "Look at the image carefully. Choose EXACTLY ONE.\n"
        "I see: <one sentence description>\nANSWER: <category name>"
    )

    def __init__(self, clip_model: CLIPClassifier, llava: LLaVAClassifier,
                 top_k: int = 5, entropy_threshold: float = 0.25):
        self.clip      = clip_model
        self.llava     = llava
        self.top_k     = top_k
        self.threshold = entropy_threshold
        print(f"[CascadeVLM] top_k={top_k}  H_threshold={entropy_threshold}")

    def _entropy(self, probs: torch.Tensor) -> float:
        p = probs.clamp(min=1e-9)
        return float(-(p * p.log()).sum())

    def predict(self, img, verbose=False):
        pred, conf, probs = self.clip.predict(img, return_all=True)
        H = self._entropy(probs)

        if H <= self.threshold:
            return pred, conf, "clip"

        # Top-K ranked by CLIP probability
        top_k_idx   = probs.topk(self.top_k).indices.tolist()
        candidates  = "\n".join(
            f"  {i+1}. {self.clip.class_names[j]}"
            for i, j in enumerate(top_k_idx)
        )
        user = self.CASCADE_USER.format(k=self.top_k, candidates=candidates)
        b64  = _encode_image_b64(img)
        try:
            resp = _call_ollama(LLAVA_SYSTEM, user, [b64],
                                self.llava.ollama_url, self.llava.model,
                                40, 0.0, self.llava.timeout)
            llava_pred = _parse_to_class(resp, self.clip.class_names)
            # Must be in top-K — if not, fall back to CLIP
            if llava_pred not in top_k_idx:
                llava_pred = pred
        except Exception:
            llava_pred = pred

        return llava_pred, probs[llava_pred].item() if llava_pred is not None \
               else conf, "llava"

    def evaluate(self, test_loader, n_samples=200, save_path=None,
                 verbose=True) -> dict:
        dataset = test_loader.dataset
        rng     = np.random.default_rng(42)
        indices = rng.choice(len(dataset.samples),
                             size=min(n_samples, len(dataset.samples)),
                             replace=False)
        correct = clip_used = llava_used = total = 0
        per_class = {c: {"c": 0, "t": 0} for c in self.clip.class_names}
        results = []

        for i, idx in enumerate(tqdm(indices, disable=not verbose,
                                     desc="CascadeVLM")):
            img_path, true_lbl = dataset.samples[idx]
            img = Image.open(img_path).convert("RGB")
            pred, conf, src = self.predict(img, verbose=(i < 2 and verbose))
            ok = (pred == true_lbl)
            if ok:
                correct += 1
                per_class[self.clip.class_names[true_lbl]]["c"] += 1
            per_class[self.clip.class_names[true_lbl]]["t"] += 1
            if src == "clip": clip_used += 1
            else: llava_used += 1
            total += 1
            results.append({
                "true": self.clip.class_names[true_lbl],
                "pred": self.clip.class_names[pred] if pred is not None else None,
                "correct": ok, "source": src,
            })

        acc = correct / total * 100
        pca = {c: (v["c"]/v["t"]*100 if v["t"] > 0 else 0)
               for c, v in per_class.items()}
        if verbose:
            print(f"\n  CascadeVLM  Top-1: {acc:.2f}%"
                  f"  CLIP: {clip_used}  LLaVA: {llava_used}")
        if save_path:
            with open(save_path, "w") as f:
                json.dump({"method":"CascadeVLM","accuracy":acc,
                           "per_class":pca,"results":results}, f, indent=2)
        return {"method": "CascadeVLM", "accuracy": acc,
                "clip_used": clip_used, "llava_used": llava_used,
                "per_class": pca, "results": results}


#  METHOD 6: HYBRID: Swin features + LLaVA

class HybridClassifier:
    """
    Method 6: Fine-tuned Swin/MobileViT features + LLaVA context.

    WHY THIS IS THE KEY CONTRIBUTION:
    Previous RAG methods used CLIP features (weak on WaRP-C, 19.6% accuracy).
    This method uses YOUR fine-tuned Swin backbone as the feature extractor.
    Swin was trained on WaRP-C → its 768-dim features perfectly discriminate
    the 28 waste classes. kNN in Swin space finds genuinely similar images.
    LLaVA then gets REAL examples that look like the query.

    Pipeline:
      1. Extract 768-dim Swin features from query image
      2. kNN in Swin feature space → top-K most similar training images
      3. Pass those K images to LLaVA with their ground-truth labels
      4. LLaVA classifies query given these high-quality visual references

    Modes:
      'swin_knn'    — Swin kNN classification only (no LLaVA, fast)
      'swin_llava'  — Swin kNN retrieval → LLaVA classification
    """

    HYBRID_USER = (
        "You are classifying a waste item on a recycling conveyor.\n\n"
        "I found the {k} most visually similar items in our database:\n"
        "{retrieved}\n\n"
        "The LAST image is the new item to classify.\n"
        "Based on the reference images, choose EXACTLY ONE category:\n"
        "{class_list}\n\n"
        "I see: <one sentence description of last image>\n"
        "ANSWER: <category name>"
    )

    def __init__(
        self,
        swin_model,           # SwinTransformerWaRP (fine-tuned, eval mode)
        swin_index: EmbeddingIndex,
        llava: Optional[LLaVAClassifier] = None,
        k: int = 3,
        device: str = "cuda",
    ):
        import torchvision.transforms as T
        from Pipeline_.preprocessor import PadToSquare

        self.swin        = swin_model
        self.index       = swin_index
        self.llava       = llava
        self.k           = k
        self.device      = device
        self.class_names = swin_index.class_names

        self.transform = T.Compose([
            PadToSquare("reflect"),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.swin.eval()
        print(f"[HybridClassifier] k={k}  "
              f"LLaVA: {'yes' if llava else 'no (kNN only)'}")

    def _extract_feat(self, img: Image.Image) -> torch.Tensor:
        """Extract 768-dim Swin features from PIL image."""
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.swin.backbone(x)     # (1, 768)
            feat = F.normalize(feat, dim=-1).squeeze(0).cpu()
        return feat   # (768,)

    def predict_knn(self, img: Image.Image) -> tuple[int, float]:
        """Swin-kNN classification — no LLaVA."""
        feat = self._extract_feat(img)
        _, sims, nbr_labels = self.index.query(feat, k=self.k)
        votes = {}
        for lbl, sim in zip(nbr_labels, sims.tolist()):
            votes[lbl] = votes.get(lbl, 0.0) + sim
        best = max(votes, key=votes.get)
        idx  = self.class_names.index(best)
        return idx, votes[best] / sum(votes.values())

    def predict_hybrid(self, img: Image.Image,
                       verbose: bool = False) -> tuple[Optional[int], str]:
        """Swin-kNN retrieval → LLaVA classification."""
        if self.llava is None:
            raise RuntimeError("Pass llava= to use hybrid mode")

        feat = self._extract_feat(img)
        idx_list, sims, nbr_labels = self.index.query(feat, k=self.k)

        # Build reference images for LLaVA
        ref_images = [
            _encode_image_b64(Image.open(self.index.paths[i.item()]).convert("RGB"))
            for i in idx_list
        ]
        retrieved_str = "\n".join(
            f"  Image {i+1}: {lbl} (similarity {sims[i]:.3f})"
            for i, lbl in enumerate(nbr_labels)
        )
        cls_str = ", ".join(self.class_names)
        user    = self.HYBRID_USER.format(
            k=self.k,
            retrieved=retrieved_str,
            class_list=cls_str,
        )
        query_b64 = _encode_image_b64(img)
        images    = ref_images + [query_b64]  # references first, query last

        try:
            resp = _call_ollama(LLAVA_SYSTEM, user, images,
                                self.llava.ollama_url, self.llava.model,
                                60, 0.0, self.llava.timeout)
            pred = _parse_to_class(resp, self.class_names)
            if verbose:
                pname = self.class_names[pred] if pred else "FAIL"
                print(f"\n  [Hybrid] \"{resp}\" → {pname}")
            return pred, resp
        except Exception as e:
            if verbose:
                print(f"\n  [Hybrid] Error: {e}")
            knn_pred, _ = self.predict_knn(img)
            return knn_pred, ""

    def evaluate(self, test_loader, mode="swin_knn", n_samples=200,
                 save_path=None, verbose=True) -> dict:
        """
        Evaluate Hybrid classifier.

        Parameters
        ----------
        mode : 'swin_knn' (fast, no LLaVA) or 'swin_llava' (full hybrid)
        """
        dataset = test_loader.dataset
        rng     = np.random.default_rng(42)
        indices = rng.choice(len(dataset.samples),
                             size=min(n_samples, len(dataset.samples)),
                             replace=False)
        correct = total = 0
        per_class = {c: {"c": 0, "t": 0} for c in self.class_names}
        results   = []

        for i, idx in enumerate(tqdm(indices, disable=not verbose,
                                     desc=f"Hybrid({mode})")):
            img_path, true_lbl = dataset.samples[idx]
            img = Image.open(img_path).convert("RGB")

            if mode == "swin_knn":
                pred, conf = self.predict_knn(img)
            else:
                pred, _ = self.predict_hybrid(img, verbose=(i < 2 and verbose))
                conf = 0.0

            ok = (pred == true_lbl)
            if ok:
                correct += 1
                per_class[self.class_names[true_lbl]]["c"] += 1
            per_class[self.class_names[true_lbl]]["t"] += 1
            total += 1
            results.append({
                "true": self.class_names[true_lbl],
                "pred": self.class_names[pred] if pred is not None else None,
                "correct": ok,
            })

        acc = correct / total * 100
        pca = {c: (v["c"]/v["t"]*100 if v["t"] > 0 else 0)
               for c, v in per_class.items()}
        if verbose:
            print(f"\n  Hybrid({mode})  Top-1: {acc:.2f}%")
        if save_path:
            with open(save_path, "w") as f:
                json.dump({"method": f"Hybrid_{mode}", "accuracy": acc,
                           "per_class": pca, "results": results}, f, indent=2)
        return {"method": f"Hybrid_{mode}", "accuracy": acc,
                "per_class": pca, "results": results}


#  METHOD 7: ENSEMBLE: Swin logits + CLIP probs

class EnsembleClassifier:
    """
    Method 7: Soft-vote ensemble of fine-tuned Swin + CLIP zero-shot.

    Combines the strengths of both:
    - Swin: trained on WaRP-C → excellent at fine-grained discrimination
    - CLIP: trained on 400M pairs → better generalisation and open-vocab

    Soft vote: final_score = α × swin_softmax + (1-α) × clip_softmax

    α=0.8 means trust Swin more (it was trained on WaRP-C).
    α=0.5 means equal weight.

    This is interesting for the report because it shows how much CLIP
    adds on top of Swin — likely the ensemble beats Swin alone on
    classes where CLIP's text understanding helps (cardboard, glass).
    """

    def __init__(self, swin_model, clip_model: CLIPClassifier,
                 alpha: float = 0.8, device: str = "cuda"):
        import torchvision.transforms as T
        from Pipeline_.preprocessor import PadToSquare

        self.swin        = swin_model
        self.clip        = clip_model
        self.alpha       = alpha
        self.device      = device
        self.class_names = clip_model.class_names

        self.transform = T.Compose([
            PadToSquare("reflect"),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.swin.eval()
        print(f"[Ensemble] Swin(α={alpha}) + CLIP(α={1-alpha:.1f})")

    def predict(self, img: Image.Image) -> tuple[int, float]:
        # Swin softmax
        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            swin_logits = self.swin(x)
        swin_probs = swin_logits.softmax(dim=-1).squeeze(0).cpu()  # ← force CPU

        # CLIP softmax (always on CPU already)
        _, _, clip_probs = self.clip.predict(img, return_all=True)

        # Soft vote — both on CPU now
        combined = self.alpha * swin_probs + (1 - self.alpha) * clip_probs
        idx      = combined.argmax().item()
        return idx, combined[idx].item()
    def evaluate(self, test_loader, verbose=True) -> dict:
        correct = total = 0
        per_class = {c: {"c": 0, "t": 0} for c in self.class_names}
        for img_path, true_lbl in tqdm(test_loader.dataset.samples,
                                       disable=not verbose,
                                       desc=f"Ensemble(α={self.alpha})"):
            img = Image.open(img_path).convert("RGB")
            pred, _ = self.predict(img)
            if pred == true_lbl:
                correct += 1
                per_class[self.class_names[true_lbl]]["c"] += 1
            per_class[self.class_names[true_lbl]]["t"] += 1
            total += 1
        acc = correct / total * 100
        pca = {c: (v["c"]/v["t"]*100 if v["t"] > 0 else 0)
               for c, v in per_class.items()}
        if verbose:
            print(f"\n  Ensemble(α={self.alpha})  Top-1: {acc:.2f}%")
        return {"method": f"Ensemble_a{self.alpha}", "accuracy": acc,
                "per_class": pca}
