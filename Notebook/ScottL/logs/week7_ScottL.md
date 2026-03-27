Weekly Training Log — EEEM068  
Week: 7 — 20 Mar – 26 Mar 2026  
Member: Scott Lewis  

---

1. Work Completed This Week
Most of the week went into setting up the full project infrastructure: preparing the WaRP‑C dataset, building the YAML config system, implementing the training and testing pipelines, and validating everything with a complete smoke‑test run. The repository is now fully structured, documented, and ready for large‑scale experiments.

---

2. Experiments Run
A short smoke test using ResNet50 confirmed that the entire pipeline works end‑to‑end on GPU, including data loading, training, checkpointing, evaluation, and plotting. Performance was low as expected for a 2‑epoch, non‑pretrained run, but the model showed clear learning and the GPU speedup was substantial.

---

3. Issues Encountered
A series of setup issues surfaced around dataset structure, PyTorch installation, Windows dataloader behaviour, and missing config keys, but each was resolved and the pipeline is now stable across platforms.

---

4. Key Observations & Analysis
Local GPU training is fast enough to run the full Phase 1 grid within a reasonable timeframe. The dataset required restructuring to work with ImageFolder, and the new config system now guarantees reproducible experiments. Early signs of class imbalance reinforce the need for weighted losses or sampling strategies.

---

5. Plan for Next Week
The focus shifts to team coordination, assigning backbones, starting the EDA notebook, and running the first proper Phase 1 reference experiment with pretrained weights. Shared output storage will also be set up for collaborative results.

---

6. Commits Made This Week
Initial project infrastructure commit covering configs, scripts, data preparation, and documentation.
