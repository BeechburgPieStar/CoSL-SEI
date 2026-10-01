import os
import copy
import time
import random
import argparse

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils.load_data import load_single_dataset
from utils.model import build_model

EPS = 1e-12
PROTOCOLS = {"a": ([1], [2, 3, 4]), "b": ([1, 2], [3, 4]), "c": ([1, 2, 3], [4])}


def get_args():
    p = argparse.ArgumentParser(
        "Large-Scale Cross-Day SEI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Data
    p.add_argument("--dataset_root", type=str, default="dataset")
    p.add_argument("--dataset", type=str, default="ManyTx")
    p.add_argument("--tx_num", type=int, default=150)
    p.add_argument("--is_eq", type=str, default="equalized",
                   choices=["equalized", "non_equalized"])
    p.add_argument("--rx_indices", type=int, nargs="+", default=list(range(18)))
    p.add_argument("--protocol", type=str, default=None, choices=list(PROTOCOLS),
                   help="a: D1->D2-4, b: D1-2->D3-4, c: D1-3->D4 "
                        "(overrides --date_pretrain / --date_test)")
    p.add_argument("--date_pretrain", type=int, nargs="+", default=[1])
    p.add_argument("--date_test", type=int, nargs="+", default=[2, 3, 4])
    p.add_argument("--sig_len", type=int, default=256)

    # Model
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--depth", type=int, default=1)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--patch", type=int, default=16)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--use_filter", type=int, default=1)
    p.add_argument("--n_filters", type=int, default=4)

    # Training
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--val_ratio", type=float, default=0.3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--min_delta", type=float, default=0.0)
    p.add_argument("--warmup_epochs", type=int, default=5)
    p.add_argument("--min_lr", type=float, default=1e-6)
    p.add_argument("--seed", type=int, default=0)

    # Fusion
    p.add_argument("--min_collab_rx", type=int, default=2)
    p.add_argument("--collab_trials", type=int, default=50)

    # Runtime
    p.add_argument("--mode", type=str, default="test_only",
                   choices=["train_test", "test_only"])
    p.add_argument("--weights_root", type=str, default="weights")
    p.add_argument("--device", type=str,
                   default="cuda:0" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()
    if args.protocol is not None:
        args.date_pretrain, args.date_test = PROTOCOLS[args.protocol]
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g


def seed_worker(worker_id):
    s = torch.initial_seed() % 2 ** 32
    np.random.seed(s)
    random.seed(s)


def make_tag(args):
    flt = f"flt{args.n_filters}" if args.use_filter else "fltOff"
    return (f"specformer_{flt}_sd{args.seed}"
            f"_dim{args.dim}_dep{args.depth}_h{args.heads}_p{args.patch}"
            f"_do{args.dropout}_tx{args.tx_num}_{args.is_eq}")


def make_ckpt_path(args):
    # e.g. weights/dep1_p16_f4_d1_sd0/specformer_flt4_sd0_..._equalized.pt
    flt = f"f{args.n_filters}" if args.use_filter else "fOff"
    days = "".join(str(d) for d in args.date_pretrain)
    subdir = f"dep{args.depth}_p{args.patch}_{flt}_d{days}_sd{args.seed}"
    return os.path.join(args.weights_root, subdir, f"{make_tag(args)}.pt")


def build_dataset(args, date_indices):
    xs, ys, rs, loaded, skipped = [], [], [], [], []
    for rx in args.rx_indices:
        for d in date_indices:
            try:
                x, y = load_single_dataset(
                    dataset=args.dataset, rx_index=rx, date_index=d,
                    tx_num=args.tx_num, is_eq=args.is_eq, dataset_root=args.dataset_root)
            except FileNotFoundError:
                skipped.append((rx, d))
                continue
            if y is None or len(y) == 0:
                skipped.append((rx, d))
                continue
            xs.append(x.astype(np.float32))
            ys.append(y.astype(np.int64))
            rs.append(np.full(len(y), rx, dtype=np.int64))
            loaded.append((rx, d))
    print(f"[data] days={list(date_indices)} loaded {len(loaded)} (rx, day) pairs, "
          f"skipped {len(skipped)}")
    if not xs:
        raise RuntimeError("No usable data; check path / is_eq / rx_indices / dates.")
    return np.concatenate(xs, 0), np.concatenate(ys, 0), np.concatenate(rs, 0)


def random_split(n, val_ratio, seed):
    idx = np.random.default_rng(seed).permutation(n)
    k = int(n * (1 - val_ratio))
    return idx[:k], idx[k:]


def make_loader(X, Y, args, shuffle, drop_last=False):
    ds = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(Y))
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, drop_last=drop_last,
                      generator=make_generator(args.seed), worker_init_fn=seed_worker)


class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.best_state = None
        self.counter = 0

    def step(self, metric, model):
        if self.best is None or metric > self.best + self.min_delta:
            self.best = metric
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


def build_optim_sched(model, args):
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup = max(0, args.warmup_epochs)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=max(1, args.epochs - warmup), eta_min=args.min_lr)
    if warmup == 0:
        return opt, cosine
    warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.01, end_factor=1.0,
                                             total_iters=warmup)
    return opt, torch.optim.lr_scheduler.SequentialLR(opt, [warm, cosine], milestones=[warmup])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for xb, yb in loader:
        pred = model(xb.to(device)).argmax(1).cpu()
        correct += (pred == yb).sum().item()
        total += yb.numel()
    return correct / max(total, 1)


@torch.no_grad()
def predict_probs(model, X, args):
    model.eval()
    loader = DataLoader(TensorDataset(torch.from_numpy(X).float()),
                        batch_size=args.batch_size, shuffle=False)
    out = [torch.softmax(model(xb.to(args.device)), 1).cpu().numpy() for (xb,) in loader]
    return np.concatenate(out, 0)


def run_train(args, model, ckpt):
    X, Y, _ = build_dataset(args, args.date_pretrain)
    tr, va = random_split(len(X), args.val_ratio, args.seed)
    train_loader = make_loader(X[tr], Y[tr], args, shuffle=True, drop_last=True)
    val_loader = make_loader(X[va], Y[va], args, shuffle=False) if len(va) > 0 else None

    opt, sched = build_optim_sched(model, args)
    crit = nn.CrossEntropyLoss()
    stopper = EarlyStopping(args.patience, args.min_delta) if val_loader is not None else None

    for ep in range(1, args.epochs + 1):
        model.train()
        running, t0 = 0.0, time.time()
        for xb, yb in train_loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
            running += loss.item()
        sched.step()
        msg = (f"[train] epoch {ep:02d} | loss {running / len(train_loader):.4f} "
               f"| lr {opt.param_groups[0]['lr']:.2e} | {time.time() - t0:.1f}s")
        if val_loader is not None:
            acc = evaluate(model, val_loader, args.device)
            print(msg + f" | val acc {acc:.4f}")
            if stopper.step(acc, model):
                print(f"[early-stop] epoch {ep}, best val acc {stopper.best:.4f}")
                break
        else:
            print(msg)

    if stopper is not None:
        stopper.restore(model)
    os.makedirs(os.path.dirname(ckpt), exist_ok=True)
    torch.save(model.state_dict(), ckpt)
    print(f"[save] {ckpt}")


# hard: majority vote, soft: arithmetic mean, geo: geometric mean (mean log-probability)
FUSION_MODES = ("hard", "soft", "geo")


def fuse_predict(P, mode):
    if mode == "hard":
        return int(np.bincount(P.argmax(1), minlength=P.shape[1]).argmax())
    if mode == "soft":
        return int(P.mean(0).argmax())
    if mode == "geo":
        return int(np.log(P + EPS).mean(0).argmax())
    raise ValueError(mode)


def collab_groups(Y, R, args):
    # Re-seeded on every call, so all fusion modes are scored on identical groups.
    rng = np.random.default_rng(args.seed)
    rxs = np.unique(R)
    for c in np.unique(Y):
        per_rx = [np.where((Y == c) & (R == r))[0] for r in rxs]
        per_rx = [idx for idx in per_rx if len(idx) > 0]
        if len(per_rx) < args.min_collab_rx:
            continue
        for _ in range(args.collab_trials):
            yield c, np.array([int(rng.choice(idx)) for idx in per_rx])


def collaborative_eval(probs, Y, R, args, mode):
    correct = total = 0
    for c, grp in collab_groups(Y, R, args):
        correct += int(fuse_predict(probs[grp], mode) == c)
        total += 1
    return correct / max(total, 1)


def run_eval(args, model):
    X, Y, R = build_dataset(args, args.date_test)
    probs = predict_probs(model, X, args)
    hit = probs.argmax(1) == Y
    acc_single = np.mean([hit[R == r].mean() for r in np.unique(R)])

    print(f"\n[result] source days {args.date_pretrain} -> target days {args.date_test}")
    print(f"    single-Rx (avg) : {acc_single * 100:.2f}")
    for mode in FUSION_MODES:
        print(f"    fusion {mode:<4}     : {collaborative_eval(probs, Y, R, args, mode) * 100:.2f}")


def main():
    args = get_args()
    set_seed(args.seed)
    ckpt = make_ckpt_path(args)
    print(f"[ckpt] {ckpt}\n[mode] {args.mode}\n[device] {args.device}")

    model = build_model(args, args.tx_num).to(args.device)

    if args.mode == "test_only":
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(f"Weights not found at {ckpt}")
        model.load_state_dict(torch.load(ckpt, map_location=args.device))
    else:
        run_train(args, model, ckpt)
    run_eval(args, model)


if __name__ == "__main__":
    main()