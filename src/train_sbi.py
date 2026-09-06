import os
import sys
import random
import argparse
from datetime import datetime
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from utils.esbi import ESBI_Dataset
from utils.scheduler import LinearDecayLR
from utils.logs import log
from utils.funcs import load_json
from model import Detector

def compute_accuracy(pred, true):
    """Computes categorical accuracy across target labels."""
    pred_idx = pred.argmax(dim=1).cpu().data.numpy()
    true_idx = true.cpu().numpy()
    return (pred_idx == true_idx).mean()

def main(args):
    cfg = load_json(args.config)

    # Reproducibility Seeds
    seed = 5
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # SPEED FIX: deterministic=True disables cuDNN's fastest kernels and largely
    # cancels out the benefit of benchmark=True. Turn determinism off unless you
    # specifically need bit-exact reproducibility between runs.
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    # SPEED FIX: RTX 4060 (Ada Lovelace) supports TF32 matmuls/convs, which are
    # faster than full FP32 with negligible accuracy impact. Free throughput.
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Force 380x380 to match paper benchmark if omitted in config
    image_size = cfg.get('image_size', 380) 
    batch_size = cfg['batch_size']

    # Initialize Datasets
    train_dataset_esbi = ESBI_Dataset(phase='train', image_size=image_size, wavelet=args.wavelet, mode=args.mode)
    val_dataset_esbi = ESBI_Dataset(phase='val', image_size=image_size, wavelet=args.wavelet, mode=args.mode)

    # SPEED FIX: Ryzen 5 9600X has 6 cores / 12 threads. 4 workers was leaving
    # capacity unused if the dataloader (image decode + wavelet transform) is
    # the bottleneck. Bumped to 6 — watch nvidia-smi; if GPU util is still <90%,
    # try 8.
    num_workers = 6

    train_loader = torch.utils.data.DataLoader(
        train_dataset_esbi, 
        batch_size=batch_size // 2, 
        shuffle=True, 
        collate_fn=train_dataset_esbi.collate_fn, 
        num_workers=num_workers, 
        pin_memory=True, 
        drop_last=True, 
        persistent_workers=True,
        worker_init_fn=train_dataset_esbi.worker_init_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset_esbi, 
        batch_size=batch_size, 
        shuffle=False, 
        collate_fn=val_dataset_esbi.collate_fn, 
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True,
        worker_init_fn=val_dataset_esbi.worker_init_fn
    )
    
    # Initialize Model & Optimizer
    model = Detector()

    # SPEED FIX: channels_last memory format gives EfficientNet a real speedup
    # on modern NVIDIA GPUs (uses tensor cores more effectively) with no
    # accuracy cost.
    model = model.to(device, memory_format=torch.channels_last)

    # NOTE: torch.compile was tried here and REMOVED. It only intercepts
    # model(x)/model.forward(), not model.training_step(). Since SAM's
    # training_step calls self(x) on the *original* uncompiled module, wrapping
    # with torch.compile gave zero benefit while triggering graph-break
    # recompilation on every batch (contains loss.backward() + optimizer.first_step()/
    # second_step() inside the traced method) — this is what caused the 4x slowdown
    # (1.53s/it instead of the expected ~0.3-0.4s/it). Do not re-add it without
    # restructuring training_step to only compile the inner self.net(x) call.

    n_epoch = cfg['epoch']
    lr_scheduler = LinearDecayLR(model.optimizer, n_epoch, int(n_epoch * 0.75))

    # Checkpoint Resuming Logic
    start_epoch = 0
    if args.resume_path:
        if os.path.exists(args.resume_path):
            print(f"\n[INFO] Loading checkpoint from: {args.resume_path}")
            checkpoint = torch.load(args.resume_path, map_location=device)
            
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
                
            if 'optimizer' in checkpoint:
                model.optimizer.load_state_dict(checkpoint['optimizer'])
                
            if 'epoch' in checkpoint:
                start_epoch = checkpoint['epoch'] + 1
                
            print(f"[INFO] Resuming training from Epoch {start_epoch + 1} / {n_epoch}\n")
            
            for _ in range(start_epoch):
                lr_scheduler.step()
        else:
            print(f"[WARNING] Checkpoint path '{args.resume_path}' not found! Starting from scratch.\n")

    # Output Directory Setup
    now = datetime.now()
    save_path = 'output/{}_'.format(args.session_name) + os.path.splitext(os.path.basename(args.config))[0] + '_' + now.strftime("%m_%d_%H_%M_%S") + '/'
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(os.path.join(save_path, 'weights/'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'logs/'), exist_ok=True)
    logger = log(path=save_path + "logs/", file="losses.logs")

    criterion = nn.CrossEntropyLoss()

    weight_dict = {}
    n_weight = 5
    last_val_auc = 0.0

    # Main Training Loop
    for epoch in range(start_epoch, n_epoch):
        np.random.seed(seed + epoch)
        train_loss = 0.0
        train_acc = 0.0
        model.train()
        
        # Training Phase
        for step, data in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Train]")):
            # SPEED FIX: keep tensors in channels_last on the way to the GPU too.
            img = data['img'].to(device, non_blocking=True).float().to(memory_format=torch.channels_last)
            target = data['label'].to(device, non_blocking=True).long()
            
            model.optimizer.zero_grad()

            # NOTE: model.training_step() runs SAM's two required forward/backward
            # passes AND both optimizer.first_step()/second_step() calls internally
            # (see model.py). It already updates the weights. We only recompute the
            # loss here for logging — we must NOT call .backward()/optimizer.step()
            # again, since that graph has already been freed by the internal
            # backward() calls (this was causing the "backward through the graph a
            # second time" crash).
            try:
                with torch.amp.autocast('cuda'):
                    output = model.training_step(img, target)
                    loss = criterion(output, target)
            except torch.cuda.OutOfMemoryError:
                print(f"\n[OOM] Ran out of VRAM at batch_size in config. Lower it in base.json and restart.")
                torch.cuda.empty_cache()
                raise

            train_loss += loss.item()
            acc = compute_accuracy(F.log_softmax(output, dim=1), target)
            train_acc += acc
            
        train_losses_avg = train_loss / len(train_loader)
        train_accs_avg = train_acc / len(train_loader)

        log_text = "Epoch {}/{} | train loss: {:.4f}, train acc: {:.4f}, ".format(
            epoch + 1, n_epoch, train_losses_avg, train_accs_avg
        )
        lr_scheduler.step()

        # Validation Phase
        model.eval()
        val_acc, val_loss = 0.0, 0.0
        output_dict, target_dict = [], []
        
        for step, data in enumerate(tqdm(val_loader, desc=f"Epoch {epoch+1}/{n_epoch} [Val]")):
            img = data['img'].to(device, non_blocking=True).float().to(memory_format=torch.channels_last)
            target = data['label'].to(device, non_blocking=True).long()
            
            with torch.no_grad():
                with torch.amp.autocast('cuda'):
                    output = model(img)
                    loss = criterion(output, target)
            
            val_loss += loss.item()
            val_acc += compute_accuracy(F.log_softmax(output, dim=1), target)
            
            output_dict.extend(output.softmax(1).cpu().data.numpy().tolist())
            target_dict.extend(target.cpu().data.numpy().tolist())
            
        val_losses_avg = val_loss / len(val_loader)
        val_accs_avg = val_acc / len(val_loader)
        
        # Safe Multi-Class ROC AUC Calculation
        try:
            val_auc = float(roc_auc_score(target_dict, output_dict, multi_class='ovr', labels=[0, 1, 2]))
        except Exception as e:
            print(f"\n[WARNING] ROC AUC computation error on epoch {epoch+1}: {e}")
            val_auc = float(val_accs_avg)
        
        log_text += "val loss: {:.4f}, val acc: {:.4f}, val auc: {:.4f}".format(
            val_losses_avg, val_accs_avg, val_auc
        )

        # Weight Checkpoint Saving
        save_model_path = os.path.join(save_path + 'weights/', "{}_{:.4f}_val.tar".format(epoch + 1, val_auc))
        if len(weight_dict) < n_weight:
            weight_dict[save_model_path] = val_auc
            torch.save({"model": model.state_dict(), "optimizer": model.optimizer.state_dict(), "epoch": epoch}, save_model_path)
            last_val_auc = min([weight_dict[k] for k in weight_dict])
        elif val_auc >= last_val_auc:
            for k in list(weight_dict.keys()):
                if weight_dict[k] == last_val_auc:
                    del weight_dict[k]
                    if os.path.exists(k):
                        try:
                            os.remove(k)
                        except OSError:
                            pass
                    break
            weight_dict[save_model_path] = val_auc
            torch.save({"model": model.state_dict(), "optimizer": model.optimizer.state_dict(), "epoch": epoch}, save_model_path)
            last_val_auc = min([weight_dict[k] for k in weight_dict])
        
        logger.info(log_text)
        sys.stdout.flush()
        
        current_lr = model.optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1} Complete | Current LR: {current_lr:.6f}\n")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train / Resume ESBI Deepfake Detector")
    parser.add_argument('config', help="Path to config JSON file")
    parser.add_argument('-n', dest='session_name', required=True, help="Session identifier name")
    parser.add_argument('-w', dest='wavelet', default='sym2', help="Wavelet transform type (default: sym2)")
    parser.add_argument('-m', dest='mode', default='reflect', help="Padding mode (default: reflect)")
    parser.add_argument('-r', '--resume', dest='resume_path', default=None, help="Path to checkpoint .tar file to resume training from")
    args = parser.parse_args()
    
    main(args)