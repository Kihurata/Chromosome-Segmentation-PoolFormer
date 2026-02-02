import os
import yaml
import pandas as pd
import torch
from datetime import datetime

class ExperimentLogger:
    def __init__(self, config):
        self.config = config
        
        # 1. Create unique path based on timestamp
        # Example: experiments/stage1_seg/20231027_0900_UNet_Resnet34_Baseline
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
        save_dir = config.get('training', {}).get('save_dir', 'experiments')
        stage = config.get('stage', 'unknown_stage')
        exp_name = config.get('experiment_name', 'default_experiment')
        
        # Avoid double nesting if user included stage in save_dir
        if os.path.basename(os.path.normpath(save_dir)) == stage:
            metrics_root = save_dir
        else:
            metrics_root = os.path.join(save_dir, stage)
            
        self.exp_dir = os.path.join(
            metrics_root, 
            f"{timestamp}_{exp_name}"
        )
        os.makedirs(self.exp_dir, exist_ok=True)
        
        # 2. Save config file immediately
        with open(os.path.join(self.exp_dir, 'config.yaml'), 'w') as f:
            yaml.dump(config, f)
            
        # Initialize DataFrame for logs
        self.metrics_log = []
        # Support both 'min' (loss) and 'max' (accuracy/iou) for best metric
        self.best_metric = -float('inf') 
        self.best_loss = float('inf')

    def log_epoch(self, epoch, metrics_dict):
        """
        Log metrics for current epoch
        metrics_dict example: {'loss': 0.5, 'accuracy': 0.9, 'iou': 0.85}
        """
        metrics_dict['epoch'] = epoch
        self.metrics_log.append(metrics_dict)
        
        # Save to CSV every epoch
        df = pd.DataFrame(self.metrics_log)
        df.to_csv(os.path.join(self.exp_dir, 'metrics.csv'), index=False)

    def save_model(self, model, metric_val, metric_name='val_acc', mode='max'):
        """
        Save model if it improves over previous best
        """
        is_best = False
        
        if mode == 'max':
            if metric_val > self.best_metric:
                is_best = True
                self.best_metric = metric_val
        elif mode == 'min':
            if metric_val < self.best_loss:
                is_best = True
                self.best_loss = metric_val
        
        # Save Model
        torch.save(model.state_dict(), os.path.join(self.exp_dir, 'last_model.pth'))
        
        # Save Best
        if is_best:
            # Construct new best model name
            new_best_name = f'best_model_{metric_name}_{metric_val:.4f}.pth'
            save_path = os.path.join(self.exp_dir, new_best_name)
            torch.save(model.state_dict(), save_path)
            print(f"--> Saved New Best Model: {new_best_name}")
            
            # Optional: Remove previous best model to keep directory clean (if filename changed)
            # Find files starting with 'best_model_' but not the new one
            for f in os.listdir(self.exp_dir):
                if f.startswith('best_model_') and f != new_best_name and f.endswith('.pth'):
                    try:
                        os.remove(os.path.join(self.exp_dir, f))
                    except OSError:
                        pass
            
        return is_best
