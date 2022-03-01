import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import numpy as np

import torch
from tqdm import trange, tqdm
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset import SeqClsDataset
from utils import Vocab
from model import SeqClassifier

TRAIN = "train"
DEV = "eval"
SPLITS = [TRAIN, DEV]


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data_paths = {split: args.data_dir / f"{split}.json" for split in SPLITS}
    data = {split: json.loads(path.read_text()) for split, path in data_paths.items()}
    datasets: Dict[str, SeqClsDataset] = {
        split: SeqClsDataset(split_data, vocab, intent2idx, args.max_len)
        for split, split_data in data.items()
    }
    # TODO: crecate DataLoader for train / dev datasets
    same_seed(827)
    train_loader = torch.utils.data.DataLoader(datasets[TRAIN], batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=datasets[TRAIN].collate_fn)
    dev_loader = torch.utils.data.DataLoader(datasets[DEV], batch_size=args.batch_size, shuffle=True, pin_memory=True, collate_fn=datasets[DEV].collate_fn)
    
    embeddings = torch.load(args.cache_dir / "embeddings.pt")
    model_path = args.ckpt_dir / "model.ckpt"
    # TODO: init model and move model to target device(cpu / gpu)
    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        datasets[TRAIN].num_classes,
    )
    model = model.to(args.device)
    writer = SummaryWriter()
    # TODO: init optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay = 1e-6)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[6,10], gamma=0.1)

    epoch_pbar = trange(args.num_epoch, desc="Epoch")
    best_loss, step = np.inf, 0
    for epoch in epoch_pbar:
        model.train()
        loss_record = []
        
        train_pbar = tqdm(train_loader, position=0, leave=True)
        # TODO: Training loop - iterate over train dataloader and update model weights
        for x_train, y_train in train_pbar:
            optimizer.zero_grad()  
            x_train, y_train = x_train.to(args.device), y_train.to(args.device)
            pred = model(x_train)
            loss = F.nll_loss(pred, y_train)
            loss.backward()
            optimizer.step()
            step+=1
            loss_record.append(loss.detach().item())
            
            train_pbar.set_description(f'Epoch [{epoch+1}/{args.num_epoch}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})
        
        mean_train_loss = sum(loss_record)/len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        model.eval()
        loss_record = []
        
        for x_val, y_val in dev_loader:
            x_val, y_val = x_val.to(args.device), y_val.to(args.device)
            with torch.no_grad():
                pred = model(x_val)
                loss = F.nll_loss(pred, y_val)
                
            loss_record.append(loss.item())
            
        mean_valid_loss = sum(loss_record)/len(loss_record)
        print(f'Epoch [{epoch+1}/{args.num_epoch}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)
        scheduler.step()
        
        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), model_path) # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else: 
            early_stop_count += 1

        if early_stop_count >= 5:
            print('\nModel is not improving, so we halt the training session.')
            return
    # TODO: Inference on test set


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--data_dir",
        type=Path,
        help="Directory to the dataset.",
        default="./drive/MyDrive/ColabNotebooks/ADL21-HW1/data/intent/",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./drive/MyDrive/ColabNotebooks/ADL21-HW1/cache/intent/",
    )
    parser.add_argument(
        "--ckpt_dir",
        type=Path,
        help="Directory to save the model file.",
        default="./drive/MyDrive/ColabNotebooks/ADL21-HW1/ckpt/intent",
    )

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # optimizer
    parser.add_argument("--lr", type=float, default=1e-3)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    # training
    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    parser.add_argument("--num_epoch", type=int, default=15)

    args = parser.parse_args()
    return args

def same_seed(seed): 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

if __name__ == "__main__":
    args = parse_args()
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    main(args)
