import os
import sys
import argparse
import yaml
from mmengine.config import Config
from mmengine.runner import Runner
from mmengine import init_default_scope
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

    def after_train_epoch(self, runner, **kwargs):
        # Collect metrics
        if hasattr(runner.message_hub, 'get_scalar_metrics'):
            metrics = runner.message_hub.get_scalar_metrics()
        else:
            # Fallback for older mmengine versions
            metrics = runner.message_hub.log_scalars
        # Filter for current epoch metrics or just dump all scalars
        # mmengine stores scalars. We want 'loss', 'acc', etc.
        log_dict = {}
        for k, v in metrics.items():
            if hasattr(v, 'current_value'):
                val = v.current_value(mode='mean')
            elif hasattr(v, 'mean'):
                val = v.mean()
            else:
                # Last resort: try accessing index -1
                try:
                    val = v.data[-1]
                except:
                    val = 0.0
            log_dict[k] = val
        # Prepare Clean Training Log Dcit
        clean_log_dict = {}
        for k, v in log_dict.items():
            if 'loss' in k:
               clean_log_dict['train_loss'] = v
            elif 'accuracy' in k:
               clean_log_dict['train_acc'] = v
            else:
               clean_log_dict[k] = v # Keep other keys like 'lr'
               
        self.logger.log_epoch(runner.epoch + 1, clean_log_dict)

    def after_val_epoch(self, runner, metrics=None, **kwargs):
        val_metrics = {}
        
        # Case 1: metrics passed directly (new mmengine structure)
        if metrics is not None:
             # These are raw values, usually dict[str, float]
             # Filter validation metrics
             val_metrics = {k: v for k, v in metrics.items() if 'single-label' in k}
             
        # Case 2: metrics not passed, fetch from message_hub (older mmengine)
        else:
            if hasattr(runner.message_hub, 'get_scalar_metrics'):
                hub_metrics = runner.message_hub.get_scalar_metrics()
            else:
                hub_metrics = runner.message_hub.log_scalars
                
            for k, v in hub_metrics.items():
                if 'single-label' not in k:
                    continue
                    
                if hasattr(v, 'current_value'):
                    val = v.current_value(mode='mean')
                elif hasattr(v, 'mean'):
                    val = v.mean()
                else:
                    try:
                        val = v.data[-1]
                    except:
                        val = 0.0
                val_metrics[k] = val
        # Log to csv (update the row for this epoch)
        # Note: log_epoch might have been called in after_train_epoch. 
        # ExperimentLogger appends a row. We might be appending twice per epoch?
        # User's log_epoch appends to a list. 
        # Better strategy: Call log_epoch ONCE per epoch with merged metrics?
        # Or just let it log multiple rows (one for train, one for val)?
        # For simplicity, we'll log val separately or together.
        # Let's try to merge if possible, but hooks are separate.
        # We will just log validation metrics.
        # Prepare Clean Validation Log Dict
        clean_val_dict = {}
        if val_metrics:
            for k, v in val_metrics.items():
                if 'accuracy' in k:
                    clean_val_dict['val_accuracy'] = v
                elif 'f1-score' in k:
                    clean_val_dict['val_f1'] = v
                elif 'precision' in k:
                    clean_val_dict['val_precision'] = v
                elif 'recall' in k:
                    clean_val_dict['val_recall'] = v
                # Note: val_loss is usually not computed by SingleLabelMetric/Accuracy, 
                # unless a separate Loss metric is added to val_evaluator. 
                # If 'loss' appears in val_metrics, map it.
                elif 'loss' in k:
                    clean_val_dict['val_loss'] = v
                else:
                    clean_val_dict[k] = v

            self.logger.log_epoch(runner.epoch + 1, clean_val_dict)
            
            # Check for best model
            # Use the cleaned metric name 'val_f1'
            target_metric = 'val_f1' 
            if target_metric in clean_val_dict:
                score = clean_val_dict[target_metric]
                self.logger.save_model(runner.model, score, metric_name='f1', mode='max')

def parse_args():
    parser = argparse.ArgumentParser(description='Train Stage 2 Classification')
    parser.add_argument('--config', default='configs/stage2_cls.yaml', help='path to config file')
    return parser.parse_args()


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
            loss=dict(type='mmpretrain.CrossEntropyLoss', loss_weight=1.0, use_soft=True),
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
        load_from=yaml_cfg['training'].get('load_from', None),
        env_cfg=dict(cudnn_benchmark=True),
    ))
    return cfg

def get_latest_checkpoint(base_dir):
    """Tìm file .pth mới nhất trong tất cả các thư mục con của base_dir."""
    import glob
    # Tìm tất cả file .pth trong các thư mục con
    checkpoint_files = glob.glob(os.path.join(base_dir, "**", "*.pth"), recursive=True)
    if not checkpoint_files:
        return None
    # Trả về file có thời gian chỉnh sửa mới nhất
    return max(checkpoint_files, key=os.path.getmtime)

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


    # Config building block removed from here as it is moved outside
    DATA_ROOT = yaml_cfg['training']['data_root']
    WORK_DIR = logger.exp_dir # Use logger's dir
    CLASSES = yaml_cfg['model']['classes'] # Needed for get_file_list below


    # === Data Split Logic ===
    import glob
    from sklearn.model_selection import train_test_split

    def get_file_list(folder, classes):
        items = []
        for cls_idx, cls in enumerate(classes):
            cls_folder = os.path.join(folder, cls)
            if not os.path.isdir(cls_folder):
                continue
            
            for p in glob.glob(os.path.join(cls_folder, '*')):
                if os.path.isfile(p):
                    rel = os.path.relpath(p, DATA_ROOT) # relative to DATA_ROOT
                    items.append((rel.replace('\\', '/'), cls_idx))
        items.sort()
        return items

    # Check for train/val structure
    train_dir = os.path.join(DATA_ROOT, 'train')
    val_dir = os.path.join(DATA_ROOT, 'val')
    
    has_explicit_split = os.path.isdir(train_dir) and os.path.isdir(val_dir)

    if has_explicit_split:
        print(f"[INFO] Found explicit train/val split in {DATA_ROOT}")
        train_items = get_file_list(train_dir, CLASSES)
        val_items = get_file_list(val_dir, CLASSES)
    else:
        print(f"[INFO] No explicit split found. Checking flat structure in {DATA_ROOT}")
        # Legacy flat structure support
        all_items = get_file_list(DATA_ROOT, CLASSES)
        if not all_items:
             print(f"[WARNING] No data found in {DATA_ROOT}. Please check paths.")
             train_items, val_items = [], []
        else:
            labels = [y for _, y in all_items]
            train_items, val_items = train_test_split(all_items, test_size=0.2, random_state=42, stratify=labels)

    if not train_items:
        print("[ERROR] No training items found!")
        sys.exit(1)

    # Generate annotation files in EXP_DIR
    # Make absolute so mmpretrain doesn't join with data_root
    ann_train = os.path.abspath(os.path.join(WORK_DIR, 'train.txt'))
    ann_val   = os.path.abspath(os.path.join(WORK_DIR, 'val.txt'))

    with open(ann_train, 'w', encoding='utf-8') as f:
        for rel, y in train_items:
            f.write(f'{rel} {y}\n')
    with open(ann_val, 'w', encoding='utf-8') as f:
        for rel, y in val_items:
            f.write(f'{rel} {y}\n')

    # Logic tự động tìm checkpoint nếu resume được bật
    if yaml_cfg['training'].get('resume'):
        # Đường dẫn gốc chứa các folder timestamp
        base_stage_dir = os.path.join(project_root, "experiments", "stage2_cls")
        latest_ckpt = get_latest_checkpoint(base_stage_dir)
        
        if latest_ckpt:
            print(f"[INFO] Auto-detected latest checkpoint: {latest_ckpt}")
            yaml_cfg['training']['load_from'] = latest_ckpt
        else:
            print("[WARNING] Resume is True but no checkpoint found. Training from scratch.")
            yaml_cfg['training']['resume'] = False

    # Build Config
    cfg = build_mm_config(
        yaml_cfg=yaml_cfg,
        work_dir=WORK_DIR,
        ann_train=ann_train,
        ann_val=ann_val
    )
    
    # Calculate Class Weights (Optional logging)
    from collections import Counter
    cnt_train = Counter([y for _, y in train_items])
    print(f"[INFO] Train Samples: {len(train_items)} | Val Samples: {len(val_items)}")
    print(f"[INFO] Class Counts: {cnt_train}")

    # Run
    print("--> Initializing Default Scope")
    init_default_scope('mmpretrain')
    print("--> Building Runner")
    runner = Runner.from_cfg(cfg)
    # Register AdapterLoggerHook manually to avoid config serialization errors
    runner.register_hook(AdapterLoggerHook(experiment_logger=logger))
    print("--> Starting Training")
    runner.train()
