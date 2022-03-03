import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import csv

import torch
from tqdm import tqdm

from dataset import SeqClsDataset
from model import SeqClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    intent_idx_path = args.cache_dir / "intent2idx.json"
    intent2idx: Dict[str, int] = json.loads(intent_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqClsDataset(data, vocab, intent2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=dataset.collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqClassifier(
        embeddings,
        args.hidden_size,
        args.num_layers,
        args.dropout,
        args.bidirectional,
        dataset.num_classes,
    )
    model = model.to(args.device)
    model.eval()

    ckpt = torch.load(args.ckpt_path)
    # load weights into model
    model.load_state_dict(ckpt)
    # TODO: predict dataset
    prediction = []
    prediction_id = []
    for x_test, test_id in tqdm(test_loader):
        x_test = x_test.to(args.device)
        with torch.no_grad():
            # out.size = (batch_size, num_class)
            out = model(x_test)
            pred = torch.exp(out)
            # pred.size = [batch_size]
            pred = torch.max(out,dim=1)[1]
            # pred.size = [1, batch_size]
            pred = torch.unsqueeze(pred, 0)
            # pred.size = [batch_size, 1]
            pred = torch.transpose(pred, 0, 1)
            prediction.append(pred.detach().cpu())
            prediction_id.extend(test_id)
    # prediction.shape = [len(test_data),1]
    prediction = torch.cat(prediction, dim=0).numpy()
    # reshape prediction to [len(test_data)]
    prediction = prediction[:,0]
    prediction = [dataset.idx2label(pred) for pred in prediction]
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'intent'])
        for id, intent in zip(prediction_id, prediction):
            writer.writerow([id, intent])

def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument(
        "--test_file",
        type=Path,
        help="Path to the test file.",
        required=True
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        help="Directory to the preprocessed caches.",
        default="./drive/MyDrive/ColabNotebooks/ADL21-HW1/cache/intent/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.intent.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--bidirectional", type=bool, default=True)

    # data loader
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument(
        "--device", type=torch.device, help="cpu, cuda, cuda:0, cuda:1", default="cpu"
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)