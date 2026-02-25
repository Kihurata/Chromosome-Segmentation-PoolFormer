import argparse
import os
import json
import torch
import numpy as np
import pandas as pd
from mmengine.config import Config
from mmengine.runner import Runner
from mmpretrain.apis import get_model, init_model
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix, classification_report
try:
    from tqdm import tqdm
except ImportError:
    # Fallback if tqdm not available
    def tqdm(x, **kwargs): return x

import mmengine
import yaml

# Decoupled build_mm_config to avoid import issues
def build_mm_config(yaml_cfg, work_dir, ann_train, ann_val):
    # Mapping YAML values to MM dict structure
    DATA_ROOT = yaml_cfg['training']['data_root']
    WORK_DIR = work_dir
    
    IMG_SIZE = yaml_cfg['model']['input_size']
    EPOCHS = yaml_cfg['training']['epochs']
    BATCH_SIZE = yaml_cfg['training']['batch_size']
    NUM_WORKERS = yaml_cfg['training']['num_workers']
    BASE_LR = float(yaml_cfg['training']['base_lr'])
    WD = float(yaml_cfg['training']['weight_decay'])
    CLASSES = yaml_cfg['model']['classes']
    NUM_CLASSES = len(CLASSES)
    
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='RandomResizedCrop', scale=IMG_SIZE, crop_ratio_range=(0.9, 1.0)),
        dict(type='RandomFlip', prob=0.5, direction='horizontal'),
        dict(type='RandomFlip', prob=0.5, direction='vertical'),
        dict(type='RandomRotate', prob=0.5, angle=tuple(yaml_cfg['augmentation']['random_rotate_angle'])),
        dict(type='ColorJitter', brightness=0.15, contrast=0.15, saturation=0.15, hue=0.02),
        dict(type='RandomErasing', erase_prob=yaml_cfg['augmentation']['random_erase_prob'], mode='const'),
        dict(type='PackInputs'),
    ]

    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='ResizeEdge', scale=IMG_SIZE, edge='short'),
        dict(type='CenterCrop', crop_size=IMG_SIZE),
        dict(type='PackInputs'),
    ]

    # Datasets
    train_dataset = dict(
        type='mmpretrain.ImageNet',
        data_root=DATA_ROOT,
        data_prefix='',
        classes=CLASSES,
        pipeline=train_pipeline,
        ann_file=ann_train
    )
    val_dataset = dict(
        type='mmpretrain.ImageNet',
        data_root=DATA_ROOT,
        data_prefix='',
        classes=CLASSES,
        pipeline=test_pipeline,
        ann_file=ann_val
    )

    train_dataloader = dict(
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        prefetch_factor=2,
        pin_memory=True,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=train_dataset,
    )
    val_dataloader = dict(
        batch_size=64,
        num_workers=NUM_WORKERS,
        persistent_workers=True,
        pin_memory=True,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=val_dataset,
    )
    test_dataloader = val_dataloader

    # Evaluators
    val_evaluator = [
        dict(type='mmpretrain.Accuracy', topk=(1,)),
        dict(type='mmpretrain.SingleLabelMetric', items=('precision', 'recall', 'f1-score'), average='macro'),
        dict(type='mmpretrain.ConfusionMatrix'),
    ]
    test_evaluator = val_evaluator

    # Model - Safe Pretrained Loading
    PRETRAIN = yaml_cfg['model']['pretrain_weights']
    if PRETRAIN and os.path.isfile(PRETRAIN):
        init_cfg = dict(type='Pretrained', checkpoint=PRETRAIN)
    else:
        init_cfg = None

    model = dict(
        type=yaml_cfg['model']['type'],
        data_preprocessor=dict(
            num_classes=NUM_CLASSES,
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375],
            to_rgb=True,
            batch_augments=dict(
                augments=[
                    dict(type='Mixup', alpha=yaml_cfg['augmentation']['mixup_alpha']),
                    dict(type='CutMix', alpha=yaml_cfg['augmentation']['cutmix_alpha']),
                ]
            )
        ),
        backbone=dict(type='mmpretrain.PoolFormer', arch=yaml_cfg['model']['arch']),
        neck=dict(type='GlobalAveragePooling'),
        head=dict(
            type='mmpretrain.LinearClsHead',
            num_classes=NUM_CLASSES,
            in_channels=768, 
            loss=dict(type='mmpretrain.FocalLoss', gamma=2.0, alpha=0.25, loss_weight=1.0),
            cal_acc=False
        ),
        init_cfg=init_cfg
    )

    # Optimizer
    common_optim_cfg = dict(
        optimizer=dict(type='AdamW', lr=BASE_LR, weight_decay=WD),
        clip_grad=dict(max_norm=1.0, norm_type=2),
        paramwise_cfg=dict(custom_keys={'norm': dict(decay_mult=0.0), 'pos_block': dict(decay_mult=0.0), 'head': dict(lr_mult=1.0)})
    )
    
    if torch.cuda.is_available():
        optim_wrapper = dict(type='AmpOptimWrapper', dtype='float16', **common_optim_cfg)
    else:
        optim_wrapper = dict(type='OptimWrapper', **common_optim_cfg)

    param_scheduler = [
        dict(type='LinearLR', start_factor=1e-6, by_epoch=False, begin=0, end=1500),
        dict(type='CosineAnnealingLR', T_max=EPOCHS, by_epoch=True),
    ]

    # Hooks
    default_hooks = dict(
        checkpoint=dict(type='CheckpointHook', interval=yaml_cfg['training']['ckpt_interval'], by_epoch=True, max_keep_ckpts=1, save_best='single-label/f1-score', rule='greater'),
        logger=dict(type='LoggerHook', interval=50),
        param_scheduler=dict(type='ParamSchedulerHook'),
        early_stopping=dict(type='EarlyStoppingHook', monitor='single-label/f1-score', rule='greater', patience=8)
    )
    
    # Custom Hooks
    custom_hooks = [
        dict(type='EMAHook', momentum=0.0002, update_buffers=True),
    ]

    cfg = Config(dict(
        default_scope='mmpretrain',
        model=model,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        test_dataloader=test_dataloader,
        train_cfg=dict(type='EpochBasedTrainLoop', max_epochs=EPOCHS, val_interval=yaml_cfg['training']['val_interval']),
        val_cfg=dict(type='ValLoop'),
        test_cfg=dict(type='TestLoop'),
        val_evaluator=val_evaluator,
        test_evaluator=test_evaluator,
        optim_wrapper=optim_wrapper,
        param_scheduler=param_scheduler,
        default_hooks=default_hooks,
        custom_hooks=custom_hooks,
        work_dir=WORK_DIR,
        randomness=dict(seed=yaml_cfg['training']['seed'], deterministic=False),
        log_level='INFO',
        launcher='none',
        resume=yaml_cfg['training']['resume'],
        env_cfg=dict(cudnn_benchmark=True),
    ))
    return cfg

def parse_args():
    parser = argparse.ArgumentParser(description='Analyze Stage 2 Classification Results')
    parser.add_argument('--config', required=True, help='Path to config file')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint file')
    parser.add_argument('--out-dir', default=None, help='Output directory for reports (default: same as checkpoint)')
    parser.add_argument('--ann-val', default=None, help='Explicit path to validation annotation file (val.txt)')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 1. Setup Paths
    if args.out_dir:
        out_dir = args.out_dir
    else:
        out_dir = os.path.dirname(args.checkpoint)
    
    os.makedirs(out_dir, exist_ok=True)
    print(f"[INFO] Output directory: {out_dir}")

    # 2. Load YAML Config
    print(f"[INFO] Loading YAML config: {args.config}")
    with open(args.config, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    # Resolve ann_val
    if args.ann_val:
        ann_val = args.ann_val
    else:
        # Try finding it in checkpoint directory
        possible_ann_val = os.path.join(os.path.dirname(args.checkpoint), 'val.txt')
        if os.path.exists(possible_ann_val):
            ann_val = possible_ann_val
        else:
            print(f"[WARNING] val.txt not found in {os.path.dirname(args.checkpoint)}. \nPlease provide --ann-val if validation fails.")
            ann_val = 'dummy_val.txt' # Will fail if used

    print(f"[INFO] Using ann_val: {ann_val}")
    ann_val = os.path.abspath(ann_val) 
    
    # Build MM Config
    ann_train = 'dummy_train.txt' 
    cfg = build_mm_config(yaml_cfg, out_dir, ann_train, ann_val)

    # Ensure correct device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] Device: {device}")

    print(f"[INFO] Loading model from {args.checkpoint}...")
    model = init_model(cfg, args.checkpoint, device=device)
    model.eval()

    # 3. Build Dataloader
    # Note: We do NOT rely on mmengine.registry.DATALOADERS explicitly to avoid ImportError.
    # We use Runner.build_dataloader which internally uses the registry.
    print("[INFO] Building validation dataset...")
    val_loader = Runner.build_dataloader(cfg.val_dataloader) 

    # 4. Inference Loop
    print("[INFO] Starting Inference...")
    all_preds = []
    all_labels = []
    all_paths = []

    # Iterate
    with torch.no_grad():
        for i, batch in tqdm(enumerate(val_loader), total=len(val_loader)):
            # outputs is a list of DataSample
            outputs = model.val_step(batch) 
            
            for data_sample in outputs:
                try:
                    # Logic to safely extract Pred Label
                    if hasattr(data_sample, 'pred_label'):
                        pred_struct = data_sample.pred_label
                        
                        # Case A: it has .label attribute
                        if hasattr(pred_struct, 'label'):
                            pred_val = pred_struct.label
                            if isinstance(pred_val, torch.Tensor):
                                pred_label = pred_val.item()
                            else:
                                pred_label = pred_val
                                
                        # Case B: it has .score attribute (maybe argmax needed)
                        elif hasattr(pred_struct, 'score'):
                             pred_label = pred_struct.score.argmax().item()
                        
                        # Case C: It IS a tensor (AttributeError was 'Tensor' object has no attribute 'label')
                        elif isinstance(pred_struct, torch.Tensor):
                             pred_label = pred_struct.item()
                        else:
                             print(f"DEBUG: Unknown structure for pred_label: {type(pred_struct)}")
                             pred_label = -1
                    else:
                        print("DEBUG: data_sample has no pred_label")
                        pred_label = -1


                    # Logic to safely extract GT Label
                    if hasattr(data_sample, 'gt_label'):
                         gt_struct = data_sample.gt_label
                         if hasattr(gt_struct, 'label'):
                             gt_val = gt_struct.label
                             if isinstance(gt_val, torch.Tensor):
                                 gt_label = gt_val.item()
                             else:
                                 gt_label = gt_val
                         elif isinstance(gt_struct, torch.Tensor):
                             gt_label = gt_struct.item()
                         else:
                             gt_label = -1
                    else:
                         gt_label = -1

                    # Image Path
                    img_path = data_sample.img_path if hasattr(data_sample, 'img_path') else "unknown"
                    
                    all_preds.append(pred_label)
                    all_labels.append(gt_label)
                    all_paths.append(img_path)

                except Exception as e:
                    print(f"[ERROR] processing sample: {e}")
                    # Skip or break?
                    continue

    # 5. Generate Reports
    # Get classes from dataset metainfo
    # We can access it via val_loader.dataset if accessible, or rebuild prompt logic
    # Since we built the loader via Runner.build_dataloader, it is partially opaque.
    # But init_model returns a model with .dataset_meta usually if loaded from checkpoint properly?
    # Actually mmpretrain models usually have dataset_meta in the checkpoint meta...
    # But easier: we have classes in yaml!
    classes = yaml_cfg['model']['classes'] 
    print(f"[INFO] Classes: {classes}")

    if not all_preds:
        print("[ERROR] No predictions collected!")
        return

    # --- A. Confusion Matrix ---
    # --- A. Confusion Matrix ---
    cm = confusion_matrix(all_labels, all_preds)
    
    # Custom Labels as requested
    # Rows: Class (Thật)
    # Cols: Class (Đoán)
    row_labels = [f"{c} (Thật)" for c in classes]
    col_labels = [f"{c} (Đoán)" for c in classes]
    
    cm_df = pd.DataFrame(cm, index=row_labels, columns=col_labels)
    
    # Set the index name to get the top-left corner label "(Nhãn thật \ Dự đoán)"
    cm_df.index.name = "(Nhãn thật \\ Dự đoán)"
    
    cm_path = os.path.join(out_dir, 'best_confusion_matrix.csv')
    cm_df.to_csv(cm_path)
    print(f"[INFO] Saved Confusion Matrix to {cm_path}")

    # --- B. Per-class Scores (Precision, Recall, F1) ---
    report_dict = classification_report(all_labels, all_preds, target_names=classes, output_dict=True)
    json_path = os.path.join(out_dir, 'per_class_score.json')
    with open(json_path, 'w') as f:
        json.dump(report_dict, f, indent=4)
    print(f"[INFO] Saved Per-class Scores to {json_path}")
    
    # --- C. Misclassified Images ---
    misclassified = []
    for path, pred, gt in zip(all_paths, all_preds, all_labels):
        if pred != gt and gt != -1 and pred != -1:
            if 0 <= gt < len(classes) and 0 <= pred < len(classes):
                misclassified.append({
                    'image_path': path,
                    'ground_truth': classes[gt],
                    'predicted': classes[pred],
                    'ground_truth_id': gt,
                    'predicted_id': pred
                })
            
    error_path = os.path.join(out_dir, 'misclassified_images.txt')
    with open(error_path, 'w', encoding='utf-8') as f:
        f.write(f"Total Misclassified: {len(misclassified)}\n")
        f.write("Format: [Image Path] | GT: [Ground Truth] | Pred: [Prediction]\n")
        f.write("-" * 80 + "\n")
        for item in misclassified:
            f.write(f"{item['image_path']} | GT: {item['ground_truth']} | Pred: {item['predicted']}\n")
    
    print(f"[INFO] Saved Misclassified Images list to {error_path}")
    print("[SUCCESS] Analysis Complete.")

if __name__ == '__main__':
    main()
