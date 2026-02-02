import os
import sys
import argparse
import yaml
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine.registry import init_default_scope
from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import torch

# Add src to path to import utils
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import ExperimentLogger

try:
    from random_rotate import RandomRotate
except ImportError:
    sys.path.append(os.path.dirname(__file__))
    from random_rotate import RandomRotate

# ================== WRAPPER HOOK ==================
@HOOKS.register_module()
class AdapterLoggerHook(Hook):
    """
    Hook adaptor to use ExperimentLogger within mmengine Runner.
    """
    def __init__(self, experiment_logger):
        self.logger = experiment_logger

    def after_train_epoch(self, runner):
        # Collect metrics
        metrics = runner.message_hub.get_scalar_metrics()
        # Filter for current epoch metrics or just dump all scalars
        # mmengine stores scalars. We want 'loss', 'acc', etc.
        log_dict = {k: v.current_value(mode='mean') for k, v in metrics.items()}
        
        # We can also add mode='train' prefix if needed
        self.logger.log_epoch(runner.epoch + 1, log_dict)

    def after_val_epoch(self, runner):
        metrics = runner.message_hub.get_scalar_metrics()
        # get validation metrics (usually prefixed with 'single-label/')
        val_metrics = {k: v.current_value(mode='mean') for k, v in metrics.items() if 'single-label' in k}
        
        # Log to csv (update the row for this epoch)
        # Note: log_epoch might have been called in after_train_epoch. 
        # ExperimentLogger appends a row. We might be appending twice per epoch?
        # User's log_epoch appends to a list. 
        # Better strategy: Call log_epoch ONCE per epoch with merged metrics?
        # Or just let it log multiple rows (one for train, one for val)?
        # For simplicity, we'll log val separately or together.
        # Let's try to merge if possible, but hooks are separate.
        # We will just log validation metrics.
        if val_metrics:
            self.logger.log_epoch(runner.epoch + 1, val_metrics)
            
            # Check for best model
            # Assuming 'single-label/accuracy_top1' or 'single-label/f1-score' as metric
            target_metric = 'single-label/f1-score' # Default to f1
            if target_metric in val_metrics:
                score = val_metrics[target_metric]
                self.logger.save_model(runner.model, score, metric_name='f1', mode='max')

def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage 2 Classification')
    parser.add_argument('--config', default='configs/stage2_cls.yaml', help='path to config file')
    return parser.parse_args()

# ================== MAIN ==================
if __name__ == '__main__':
    args = parse_args()
    
    # 1. Load YAML Config
    # Assuming run from root NCKH/
    project_root = os.getcwd()
    config_path = os.path.join(project_root, args.config)
    
    with open(config_path, 'r') as f:
        yaml_cfg = yaml.safe_load(f)

    # 2. Setup ExperimentLogger
    logger = ExperimentLogger(yaml_cfg)
    print(f"[INFO] Experiment Directory: {logger.exp_dir}")

    # 3. Construct MM Configuration
    # We map YAML values to the MM dict structure
    
    DATA_ROOT = yaml_cfg['training']['data_root']
    WORK_DIR = logger.exp_dir # Use logger's dir
    
    IMG_SIZE = yaml_cfg['model']['input_size']
    EPOCHS = yaml_cfg['training']['epochs']
    BATCH_SIZE = yaml_cfg['training']['batch_size']
    NUM_WORKERS = yaml_cfg['training']['num_workers']
    BASE_LR = float(yaml_cfg['training']['base_lr'])
    WD = float(yaml_cfg['training']['weight_decay'])
    CLASSES = yaml_cfg['model']['classes']
    NUM_CLASSES = len(CLASSES)
    
    # ... [Copying pipelines and model definitions from original file] ...
    # To keep this clean, I will inline the definitions but use variables
    
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
        data_root=DATA_ROOT, # logic for split updated below
        data_prefix='',
        classes=CLASSES,
        pipeline=train_pipeline,
    )
    val_dataset = dict(
        type='mmpretrain.ImageNet',
        data_root=DATA_ROOT,
        data_prefix='',
        classes=CLASSES,
        pipeline=test_pipeline,
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

    # Model
    PRETRAIN = yaml_cfg['model']['pretrain_weights']
    init_cfg = dict(type='Pretrained', checkpoint=PRETRAIN) if os.path.isfile(PRETRAIN) else None

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
    
    # Add our Adapter Hook
    custom_hooks = [
        dict(type='EMAHook', momentum=0.0002, update_buffers=True),
        AdapterLoggerHook(experiment_logger=logger)
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

    # === Pre-Split Logic (Copied from original) ===
    import glob
    from sklearn.model_selection import train_test_split
    
    def list_items(root_dir, classes):
        items = []
        # Check if train/val split exists or flat structure
        # Original code checked 'train' and 'val' subdirs for each class
        # We'll assume the same structure or adapt
        for cls_idx, cls in enumerate(classes):
            for split in ['train', 'val']: # Check both just in case
                folder = os.path.join(root_dir, split, cls)
                if not os.path.isdir(folder):
                    # Try flat structure (root/class)
                    folder = os.path.join(root_dir, cls)
                    if not os.path.isdir(folder):
                        continue
                
                for p in glob.glob(os.path.join(folder, '*')):
                    if os.path.isfile(p):
                        rel = os.path.relpath(p, root_dir)
                        items.append((rel.replace('\\', '/'), cls_idx))
        items = list(set(items)) # dedupe
        items.sort(key=lambda x: x[0])
        return items

    all_items = list_items(DATA_ROOT, CLASSES)
    if not all_items:
        print(f"[WARNING] No data found in {DATA_ROOT}. Please check paths.")
    else:
        labels = [y for _, y in all_items]
        train_items, val_items = train_test_split(all_items, test_size=0.2, random_state=42, stratify=labels)

        # Generate annotation files in EXP_DIR
        ann_train = os.path.join(WORK_DIR, 'train_80.txt')
        ann_val   = os.path.join(WORK_DIR, 'val_20.txt')

        with open(ann_train, 'w', encoding='utf-8') as f:
            for rel, y in train_items:
                f.write(f'{rel} {y}\n')
        with open(ann_val, 'w', encoding='utf-8') as f:
            for rel, y in val_items:
                f.write(f'{rel} {y}\n')

        cfg.train_dataloader['dataset']['ann_file'] = ann_train
        cfg.val_dataloader['dataset']['ann_file']   = ann_val
        
        # Calculate Class Weights (Optional)
        from collections import Counter
        cnt_train = Counter([y for _, y in train_items])
        print(f"[INFO] Train Samples: {len(train_items)} | Val Samples: {len(val_items)}")
        print(f"[INFO] Class Counts: {cnt_train}")

        # Run
        init_default_scope('mmpretrain')
        runner = Runner.from_cfg(cfg)
        runner.train()
