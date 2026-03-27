 Week 8 Log — MSc AI Project

 1. Summary of Activities This Week
The focus this week was on consolidating the entire codebase into a clean, modular structure. The training pipeline, YAML configs, and augmentation logic were fully aligned, leaving the project stable, maintainable, and ready for Phase 1 and Phase 2 experimentation.

---

 2. Technical Progress

Refactoring into a unified `utils/` package
The old notebook‑driven and script‑driven functions were merged into a single authoritative module, removing duplication and making the pipeline easier to maintain.

Standardising YAML configurations
Phase 1 and Phase 2 configs were rebuilt for consistency, removing outdated keys and ensuring all experiments follow the same structure.

Finalising augmentation and dataloaders
The transform pipeline was cleaned and made fully YAML‑driven, and the dataloader logic was updated for determinism, imbalance handling, and cross‑platform stability.

Removing legacy code
The outdated `train.py` and duplicated notebook functions were retired, leaving the notebooks as clean front‑ends that rely entirely on the new utilities.

---

 3. Challenges Encountered
There was significant drift between notebooks, scripts, and YAML files, which caused inconsistencies in augmentation, dataloaders, and experiment configs. The project also lacked a single source of truth for core functions.

---

 4. Solutions
A full audit of the codebase led to the creation of a unified utilities module, rewritten YAML templates, and a consistent import structure. This eliminated duplication and ensured that all training logic now lives in one place.

---

 5. Plans for Next Week
The next step is to run the full Phase 1 grid, analyse the results, and begin Phase 2 fine‑tuning. Work will also start on Grad‑CAM visualisations and drafting the Methodology chapter.

---

 6. Reflections
This week significantly improved the project’s organisation and reliability. With the pipeline unified and reproducible, the upcoming experimental phases should run smoothly and consistently.
