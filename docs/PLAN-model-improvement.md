# Plan: Model Improvement for Chromosome Classification

This plan outlines the steps to upgrade the PoolFormer classification model with advanced architectural and training techniques to further improve F1-Scores, especially for the "touching" class.

## 🛠 PHASE 1: Architectural Improvements
- [ ] **StarReLU & MLN Integration:**
    - Verify if `mmpretrain` 1.2.0 supports `StarReLU` in `act_cfg` and `MLN` in `norm_cfg`.
    - If not natively supported, implement a Custom Plugin for these modules as they are critical for PoolFormer V2 performance.
- [ ] **Pretrained Weights Recovery:**
    - Provide instructions/link to download `poolformer_m36_3rdparty_in1k_20220321-61f05dac.pth`.
    - Configure `init_cfg` to load these weights.

## 📈 PHASE 2: Training Strategy Optimization
- [ ] **Gradient Accumulation:**
    - Configure `OptimWrapper` with `accumulative_counts=4` to reach an Effective Batch Size of 32 (8 * 4). This avoids OOM while gaining benefits of larger batches.
- [ ] **Warmup Schedule:**
    - Update `param_scheduler` to use `LinearLR` warmup for 5-10 epochs before switching to `CosineAnnealingLR`.
- [ ] **Augmentation Tuning:**
    - Disable `Mixup` and `CutMix` entirely in the model's `data_preprocessor`. This will help the model focus on the actual visual boundaries of "Touching" vs "Overlapping" without synthetic interpolation between samples.

## ⚖️ PHASE 3: Loss Function Refinement
- [ ] **Focal Loss Implementation:**
    - Swap `CrossEntropyLoss` for `FocalLoss`.
    - **Touching Specialization:** Use `class_weight: [1.0, 1.0, 2.0]` and potentially a higher `gamma` specifically for Touching if the implementation allows, or a higher global `gamma=2.5`.

## 🚀 PHASE 4: Implementation & Verification
- [ ] **Code Implementation:** Update `src/stage2_cls/train_cluster.py` and `configs/stage2_cls.yaml`.
- [ ] **Dry Run:** Verify if `StarReLU` and `MLN` load correctly.
- [ ] **Monitor:** Check Confusion Matrix at Epoch 20 to compare with previous baseline.

---

## 🚦 Verification Checklist
- [ ] Does it crash with OOM on Batch Size 32 effective?
- [ ] Is the Warmup visible in the LR curve in `scalars.json`?
- [ ] Is the "Touching" class performance improving in validation?
