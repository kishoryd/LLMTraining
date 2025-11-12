#!/usr/bin/env python3
"""
train_pp_gpt_final.py

Pipeline-parallel training using torch.distributed.pipelining.pipeline + ScheduleGPipe.
Example:
  torchrun --nproc_per_node=2 --standalone train_pp_gpt_final.py --epochs 2 --batch_size 8 --chunks 4 --data_path sample_paragraphs_200000.txt
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
from transformers import GPT2TokenizerFast

# -------------------------
# Dataset (keeps your original logic)
# -------------------------
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, add_special_tokens=True)
        if len(token_ids) < max_length + 1:
            token_ids = token_ids + [tokenizer.eos_token_id] * (max_length + 1 - len(token_ids))
        for i in range(0, len(token_ids) - max_length, stride):
            inp = token_ids[i:i + max_length]
            tgt = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(inp, dtype=torch.long))
            self.target_ids.append(torch.tensor(tgt, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, tokenizer, batch_size=8, max_length=512, stride=512,
                         shuffle=True, drop_last=True, num_workers=2, distributed=False):
    ds = GPTDatasetV1(txt, tokenizer, max_length, stride)
    if distributed:
        sampler = DistributedSampler(ds, shuffle=shuffle)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          drop_last=drop_last, num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          drop_last=drop_last, num_workers=num_workers, pin_memory=True)


# -------------------------
# Model definition (your GPT-like model)
# -------------------------
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return self.scale * (x - mean) / torch.sqrt(var + self.eps) + self.shift

class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x ** 3)))

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, _ = x.shape
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
        attn_scores.masked_fill_(mask_bool, float('-inf'))
        attn_weights = torch.softmax(attn_scores / (self.head_dim ** 0.5), dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = (attn_weights @ values).transpose(1, 2).reshape(b, num_tokens, self.d_out)
        return self.out_proj(context_vec)

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"])
        )
    def forward(self, x):
        return self.layers(x)

class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(cfg["emb_dim"], cfg["emb_dim"], cfg["context_length"],
                                      cfg["drop_rate"], cfg["n_heads"], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])
    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        return x + shortcut

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)
    def forward(self, in_idx):
        b, t = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(t, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))


# -------------------------
# distributed helpers
# -------------------------
def setup_dist():
    if not dist.is_available():
        raise RuntimeError("torch.distributed not available")
    if not dist.is_initialized():
        # torchrun provides env://
        dist.init_process_group(backend="nccl", init_method="env://")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    return rank, world_size, device

def build_split_spec(cfg, world_size):
    n_layer = cfg["n_layers"]
    per_rank = (n_layer + world_size - 1) // world_size
    split_spec = {}
    for i in range(1, world_size):
        idx = i * per_rank
        if idx >= n_layer:
            idx = n_layer - 1
        split_spec[f"trf_blocks.{idx}"] = SplitPoint.BEGINNING
    return split_spec

# -------------------------
# main training with ScheduleGPipe
# -------------------------
def run(args, cfg):
    rank, world_size, device = setup_dist()
    last_rank = world_size - 1
    if rank == 0:
        print(f"[pipeline] world_size={world_size}, chunks={args.chunks}, device={device}")

    # load tokenizer & data
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    with open(args.data_path, "r", encoding="utf-8") as f:
        text = f.read()
    split_idx = int(0.75 * len(text))
    train_loader = create_dataloader_v1(text[:split_idx], tokenizer,
                                        batch_size=args.batch_size, max_length=cfg["context_length"],
                                        stride=cfg["context_length"], distributed=True, num_workers=2)
    val_loader = create_dataloader_v1(text[split_idx:], tokenizer,
                                      batch_size=args.batch_size, max_length=cfg["context_length"],
                                      stride=cfg["context_length"], distributed=False, num_workers=2)

    # build model on CPU for tracing/export
    model_cpu = GPTModel(cfg).to("cpu")
    if rank == 0:
        total_params = sum(p.numel() for p in model_cpu.parameters()) / 1e6
        print(f"Total params ~ {total_params:.2f}M")

    # automatic split points
    split_spec = build_split_spec(cfg, world_size)
    if rank == 0:
        print("split_spec:", split_spec)

    # microbatch for tracing (CPU). name 'in_idx' must match full_kwargs we pass later.
    micro_bs = max(1, args.batch_size // args.chunks)
    mb_example = torch.zeros((micro_bs, cfg["context_length"]), dtype=torch.long, device="cpu")

    # create pipeline IR (do NOT pass chunks here)
    pipe = pipeline(model_cpu, mb_args=(), mb_kwargs={"in_idx": mb_example}, split_spec=split_spec)

    # runtime stage / schedule
    stage = pipe.build_stage(rank, device=device)
    schedule = ScheduleGPipe(stage, args.chunks)   # pass chunks here

    # get per-stage local module & optimizer (local params only)
    local_module = pipe.get_stage_module(rank)
    local_module.to(device)
    optimizer = optim.AdamW(local_module.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()

    if rank == 0:
        print(f"[Rank {rank}] stage params ~ {sum(p.numel() for p in local_module.parameters())/1e6:.2f}M")

    # training loop
    for epoch in range(args.epochs):
        # distributed sampler epoch set
        if isinstance(getattr(train_loader, "sampler", None), DistributedSampler):
            train_loader.sampler.set_epoch(epoch)

        if rank == 0:
            print(f"\nEpoch {epoch+1}/{args.epochs}")

        total_loss = 0.0
        iters = 0

        for batch_idx, (inp, tgt) in enumerate(train_loader):
            inp = inp.to(device)
            tgt = tgt.to(device)

            full_kwargs = {"in_idx": inp}   # must match mb_kwargs name

            # zero grads for local optimizer
            optimizer.zero_grad(set_to_none=True)

            # run pipeline: source rank supplies data kwargs; other ranks call schedule.step()
            if rank == 0:
                out = schedule.step(**full_kwargs)
            else:
                out = schedule.step()

            # only last rank gets logits back (out != None)
            if rank == last_rank and out is not None:
                logits = out if not isinstance(out, (tuple, list)) else out[0]   # (B, T, V)
                loss = criterion(logits.view(-1, logits.size(-1)), tgt.view(-1))
                # backward only on last rank (autograd propagates across pipeline)
                loss.backward()
                # optional: accumulate metrics only on last rank
                total_loss += loss.item()
                iters += 1

            # synchronize before optimizer.step to ensure backward finished on last rank
            # NOTE: this barrier is conservative but prevents races between grads and optimizer.step.
            # If you want more performance, experiment with removing it once correctness is verified.
            if args.sync_per_batch:
                dist.barrier()

            # step local optimizer on EACH rank (grads for local params will be populated by autograd)
            optimizer.step()

            # printing / evaluation trigger
            if rank == last_rank and (batch_idx % args.print_every == 0):
                avg = (total_loss / max(1, iters))
                print(f"[Epoch {epoch+1}] batch {batch_idx} avg train loss (last stage): {avg:.4f}")

            # run validation occasionally (only on last rank we collect logits)
            if rank == last_rank and args.eval_every > 0 and (batch_idx % args.eval_every == 0) and batch_idx != 0:
                # lightweight validation loop
                local_val_loss = 0.0
                vsteps = 0
                schedule_eval = schedule  # schedule is used for forward
                # put underlying modules to eval mode
                for r in range(world_size):
                    # best-effort: set the mapped submodule to eval for all ranks
                    try:
                        pipe.get_stage_module(r).eval()
                    except Exception:
                        pass
                with torch.no_grad():
                    for val_inp, val_tgt in val_loader:
                        val_inp = val_inp.to(device)
                        val_tgt = val_tgt.to(device)
                        if rank == 0:
                            val_out = schedule_eval.step(**{"in_idx": val_inp})
                        else:
                            val_out = schedule_eval.step()
                        if val_out is not None:
                            val_logits = val_out if not isinstance(val_out, (tuple, list)) else val_out[0]
                            loss_v = criterion(val_logits.view(-1, val_logits.size(-1)), val_tgt.view(-1))
                            local_val_loss += loss_v.item()
                            vsteps += 1
                        # small break to avoid long val loops; adjust as needed
                        if vsteps >= 8:
                            break
                if vsteps > 0:
                    print(f"[Epoch {epoch+1}] quick val loss (last stage): {local_val_loss / vsteps:.4f}")
                # restore train mode for all stages
                for r in range(world_size):
                    try:
                        pipe.get_stage_module(r).train()
                    except Exception:
                        pass

        # epoch end summary (last rank)
        if rank == last_rank and iters > 0:
            print(f"[Epoch {epoch+1}] avg train loss (last stage): {total_loss/iters:.4f}")

        # barrier at epoch end to keep ranks in sync
        dist.barrier()

    # save CPU copy of traced model weights (rank 0)
    if rank == 0:
        cpu_state = {k: v.cpu() for k, v in model_cpu.state_dict().items()}
        torch.save(cpu_state, "gpt_pipeline_final.pth")
        print("Saved model to gpt_pipeline_final.pth")

    dist.barrier()
    dist.destroy_process_group()


# -------------------------
# CLI and entrypoint
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--chunks", type=int, default=4, help="number of GPipe microbatches")
    p.add_argument("--data_path", type=str, default="sample_paragraphs_2000.txt")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--print_every", type=int, default=1000)
    p.add_argument("--eval_every", type=int, default=0, help="run quick validation every N batches (0 disables)")
    p.add_argument("--sync_per_batch", action="store_true",
                   help="use dist.barrier() each batch to avoid potential optimizer/backward races (safe, slower)")
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()

    GPT_CFG = {
        "vocab_size": 50257,
        "context_length": 128,   # reduce to fit memory if debugging
        "emb_dim": 384,
        "n_heads": 6,
        "n_layers": 6,
        "drop_rate": 0.0,
        "qkv_bias": False
    }

    run(args, GPT_CFG)

