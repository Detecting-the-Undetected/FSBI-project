import os
import sys
import time
import argparse
import traceback
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

from bilstm_model import VideoEmbeddingDataset, BiLSTMClassifier


def save_checkpoint_atomic(state, path):
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)


def compute_accuracy(pred, true):
    pred_idx = pred.argmax(dim=1).cpu().numpy()
    true_idx = true.cpu().numpy()
    return (pred_idx == true_idx).mean()


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_dataset = VideoEmbeddingDataset(args.embedding_dir, phase='train', test_list=args.test_list)
    val_dataset = VideoEmbeddingDataset(args.embedding_dir, phase='val', test_list=args.test_list)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                               num_workers=min(4, os.cpu_count() or 2), pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=min(4, os.cpu_count() or 2), pin_memory=True)

    # Infer embed_dim from the first sample rather than hardcoding it —
    # EfficientNet-B5's pooled feature size is 2048, not the 1280 you'd get
    # from B0; let the data tell us instead of assuming.
    sample_emb, _ = train_dataset[0]
    embed_dim = sample_emb.shape[-1]
    print(f"[INFO] Detected embedding dim: {embed_dim}")

    model = BiLSTMClassifier(embed_dim=embed_dim, hidden_dim=args.hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    now = datetime.now()
    save_path = f'output/bilstm_{args.session_name}_' + now.strftime("%m_%d_%H_%M_%S") + '/'
    os.makedirs(os.path.join(save_path, 'weights'), exist_ok=True)
    latest_checkpoint_path = os.path.join(save_path, 'weights', 'latest.tar')

    weight_dict = {}
    n_weight = 5
    last_val_auc = 0.0
    run_start_time = time.time()

    for epoch in range(args.epochs):
        if args.max_hours is not None:
            elapsed_hours = (time.time() - run_start_time) / 3600.0
            if elapsed_hours >= args.max_hours:
                print(f"\n[TIME BUDGET] {elapsed_hours:.2f}h elapsed. Stopping before epoch {epoch+1}.")
                break

        try:
            model.train()
            train_loss, train_acc = 0.0, 0.0
            for emb, label in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]"):
                emb, label = emb.to(device), label.to(device)
                optimizer.zero_grad()
                output = model(emb)
                loss = criterion(output, label)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                train_acc += compute_accuracy(output, label)
            train_loss /= len(train_loader)
            train_acc /= len(train_loader)

            model.eval()
            val_loss, val_acc = 0.0, 0.0
            output_dict, target_dict = [], []
            with torch.no_grad():
                for emb, label in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                    emb, label = emb.to(device), label.to(device)
                    output = model(emb)
                    loss = criterion(output, label)
                    val_loss += loss.item()
                    val_acc += compute_accuracy(output, label)
                    output_dict.extend(output.float().softmax(1).cpu().numpy().tolist())
                    target_dict.extend(label.cpu().numpy().tolist())
            val_loss /= max(1, len(val_loader))
            val_acc /= max(1, len(val_loader))

            try:
                val_auc = float(roc_auc_score(target_dict, output_dict, multi_class='ovr', labels=[0, 1, 2]))
            except Exception as e:
                print(f"[WARNING] ROC AUC error: {e}")
                val_auc = float(val_acc)

            print(f"Epoch {epoch+1}/{args.epochs} | train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f} auc {val_auc:.4f}")

            save_model_path = os.path.join(save_path, 'weights', f"{epoch+1}_{val_auc:.4f}_val.tar")
            if len(weight_dict) < n_weight:
                weight_dict[save_model_path] = val_auc
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, save_model_path)
                last_val_auc = min(weight_dict.values())
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
                torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch}, save_model_path)
                last_val_auc = min(weight_dict.values())

            save_checkpoint_atomic(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch},
                latest_checkpoint_path
            )
            sys.stdout.flush()

        except Exception as e:
            print(f"\n[ERROR] Epoch {epoch+1} failed with: {e}")
            traceback.print_exc()
            print(f"[INFO] Latest checkpoint preserved at {latest_checkpoint_path}. Stopping.\n")
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train the Bi-LSTM temporal classifier on FSBI frame embeddings.")
    parser.add_argument('--embedding-dir', required=True, help="Folder of .npy files from extract_embeddings.py.")
    parser.add_argument('-n', dest='session_name', required=True, help="Session identifier name.")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--max-hours', type=float, default=None)
    parser.add_argument('--test-list', default=None, help="Celeb-DF List_of_testing_videos.txt (omit for FF++).")
    args = parser.parse_args()
    main(args)
