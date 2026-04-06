# Plan: Synthetic Overlapping Dataset for Stage 3

## 1. Objective
Create a new script (e.g., `src/stage3_sep/overlapping/synthetic_dataset_overlapping.py`) that generates synthetic overlapping chromosome images based on `synthetic_chromosome_generator.py`, but structures the output exactly like `create_dataset_overlapping.py`.

## 2. Output Structure
The script will output the following separate image channels for each synthetic sample:
- `images/`: The RGB synthetic canvas
- `foregrounds/`: The union of all instance masks
- `overlaps/`: The intersection mask of the instances
- `boundaries/`: The edge masks of each instance (dilated)
- `debug/`: Optional combined visualization (RGB side-by-side with masks)

These should be separated into `train/` and `val/` subdirectories (or parameterized by input args).

## 3. Core Logic Breakdown
1. **Data Loading:** 
   - Load single chromosome images (and their masks if necessary, or compute masks via thresholding like in `synthetic_chromosome_generator.py`).
2. **Synthetic Canvas Generation:**
   - Define a target canvas size (e.g., `(345, 345)`).
   - Sample 2 (or more) single chromosomes.
   - Extract bounding box, create masks, apply random rotation.
   - Paste `chromosome 1` onto canvas to get `mask 1` and `rgb 1`.
   - Paste `chromosome 2` onto canvas to get `mask 2` and `rgb 2`.
   - Enforce an overlapping condition (similar to `generate_overlapping` logic, maintaining a certain overlap ratio).
3. **Multi-Task Target Generation:**
   - **Foreground:** `mask 1 | mask 2` (Logic OR)
   - **Overlap:** `mask 1 & mask 2` (Logic AND)
   - **Boundary:** Compute edge of `mask 1` + edge of `mask 2`, dilate them by 1-2px, then combine.
4. **Saving Outputs:**
   - Save the synthetic results into the respective `images`, `foregrounds`, `overlaps`, `boundaries`, `debug` folders.
   - Optionally generate a `.csv` manifest to track the source images and coordinates for traceability.

## 4. Phase Breakdown
- **Phase 1:** Setup CLI arguments (`--single_train_dir`, `--single_val_dir`, `--output_dir` default to something like `data/stage3_synthetic/overlapping`, `--num_samples`, etc.).
- **Phase 2:** Refactor core synthetic generation functions to return independent instance masks (instead of just a single canvas mask) so overlaps and boundaries can be computed properly.
- **Phase 3:** Port boundary, overlap map, and save_debug functions from `create_dataset_overlapping.py`.
- **Phase 4:** Assembly, loop, and save.

## 5. Agent Assignments
- `backend-specialist` (or Python Developer): Implement the logic.
- `orchestrator`: Coordinate any necessary testing of the generated maps.

## 6. Verification Criteria
- [ ] Output directories (`train/images/`, `train/overlaps/`, etc.) map exactly to Stage 3 expectations.
- [ ] Overlap masks are accurate intersections of instances.
- [ ] Boundaries accurately reflect individual instance boundaries.
- [ ] Script successfully runs without crash.
