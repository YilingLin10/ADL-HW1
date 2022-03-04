import json
import pickle
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict
import csv

import torch
from tqdm import tqdm

from dataset import SeqTagDataset
from model import SeqTagClassifier
from utils import Vocab


def main(args):
    with open(args.cache_dir / "vocab.pkl", "rb") as f:
        vocab: Vocab = pickle.load(f)

    tag_idx_path = args.cache_dir / "tag2idx.json"
    tag2idx: Dict[str, int] = json.loads(tag_idx_path.read_text())

    data = json.loads(args.test_file.read_text())
    dataset = SeqTagDataset(data, vocab, tag2idx, args.max_len)
    # TODO: crecate DataLoader for test dataset
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, collate_fn=dataset.collate_fn)
    embeddings = torch.load(args.cache_dir / "embeddings.pt")

    model = SeqTagClassifier(
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
            batch_size = x_test.size(0)
            # out.size = (batch_size * max_len, num_class)
            out = model(x_test)
            pred = torch.exp(out)
            # pred.size = [batch_size * max_len]
            pred = torch.max(out,dim=1)[1]
            # pred.size = [batch_size, max_len]
            pred = pred.view(batch_size, -1)
            batch_prediction = []
            for sentence, tags in zip(x_test, pred):
                single_prediction = []
                for i, word in enumerate(sentence):
                    if word.item() != 0:
                        single_prediction.append(dataset.idx2tag(tags[i].item()))
                
                batch_prediction.append(' '.join(single_prediction))
                        
            prediction.extend(batch_prediction)
            prediction_id.extend(test_id)
    # TODO: write prediction to file (args.pred_file)
    with open(args.pred_file, 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['id', 'tags'])
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
        default="./drive/MyDrive/ColabNotebooks/ADL21-HW1/cache/slot/",
    )
    parser.add_argument(
        "--ckpt_path",
        type=Path,
        help="Path to model checkpoint.",
        required=True
    )
    parser.add_argument("--pred_file", type=Path, default="pred.slot.csv")

    # data
    parser.add_argument("--max_len", type=int, default=128)

    # model
    parser.add_argument("--hidden_size", type=int, default=256)
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