# Week 8 Log — MSc AI Project

# 1. Summary of Activities This Week
This week focused on consolidating the project structure, improving maintainability, and preparing the training pipeline for Phase 1 and Phase 2 experimentation. I refactored the codebase to remove duplication, created a clean `utils/` module, aligned YAML configurations with the updated pipeline, and finalised the augmentation and dataloader logic. This work ensures the project is now stable, modular, and ready for large‑scale runs.

---

# 2. Technical Progress
# 2.1 Refactored Codebase into a Modular `utils/` Package
- Extracted shared functions from outdated `train.py` and notebooks.
- Created a structured module containing:
  - `config.py` (config loading, merging, saving)
  - `data.py` (transforms, dataloaders, class mapping)
  - `models.py` (model construction)
  - `optim.py` (optimisers and schedulers)
  - `plotting.py` (loss curves, confusion matrices, metrics)
  - `repro.py` (seed control)
- Updated both `train.ipynb` and `test.ipynb` to import from this unified source.

# 2.2 Cleaned and Standardised YAML Configuration Files
- Rebuilt Phase 1 YAMLs to ensure fairness and reproducibility.
- Removed fine‑tuning keys from Phase 1.
- Added missing `output.` keys required by the notebook.
- Updated Phase 2 YAMLs to match the new augmentation and fine‑tuning pipeline.
- Ensured consistent structure across all experiment files.

# 2.3 Updated and Finalised Data Augmentation Pipeline
- Integrated:
  - `RandomResizedCrop`
  - Horizontal + vertical flips
  - Rotation
  - Colour jitter
  - RandomErasing (Phase 2 only)
- Ensured augmentation is fully YAML‑driven and phase‑aware.

# 2.4 Improved Dataloader Logic
- Added optional weighted sampler for Phase 2.
- Ensured deterministic behaviour with seeds and generators.
- Cleaned up worker/pin‑memory logic for cross‑platform compatibility.

# 2.5 Removed Legacy Code
- Fully retired `train.py` after extracting all relevant functionality.
- Eliminated duplicated functions across notebooks.
- Ensured the notebook is now the single source of truth for training logic.

---

# 3. Challenges Encountered
- The original project had drift between `.py` files and notebook versions, making it unclear which functions were authoritative.
- YAML files were inconsistent across phases, leading to missing keys and potential runtime errors.
- Augmentation keys in YAML no longer matched the updated transform pipeline.
- The dataloader and sampler logic needed to be aligned with the new Phase 1/Phase 2 methodology.

---

# 4. Solutions / How Challenges Were Addressed
- Performed a full audit of `train.py` and both notebooks to identify the canonical versions of each function.
- Created a unified `utils/` module to eliminate duplication and centralise logic.
- Rewrote Phase 1 and Phase 2 YAML templates to match the updated codebase.
- Removed unused augmentation keys and ensured the transform builder only uses valid parameters.
- Added clear documentation and import structure to improve readability for collaborators.

---

# 5. Plans for Next Week
- Run the full Phase 1 grid search across backbones and hyperparameters.
- Analyse Phase 1 results to select the best LR/BS combination.
- Begin Phase 2 fine‑tuning using the selected hyperparameters.
- Generate Grad‑CAM visualisations for interpretability.
- Start drafting the Methodology chapter, especially:
  - data pipeline
  - augmentation strategy
  - training phases
  - evaluation metrics

---

# 6. Reflections
This week was a turning point in terms of project organisation. Cleaning the codebase and unifying the pipeline has made the entire workflow more robust and easier to reason about. I now feel confident that Phase 1 and Phase 2 experiments will run consistently and reproducibly. The process also clarified the importance of modular design, especially when working with notebooks and multiple collaborators.

