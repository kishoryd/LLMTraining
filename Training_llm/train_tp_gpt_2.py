import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt

import torch.distributed as dist
from torch.distributed.tensor.parallel import parallelize_module, RowwiseParallel, ColwiseParallel
#from torch.distributed.tensor.parallel import parallelize_module, SequenceParallel
from torch.distributed.device_mesh import init_device_mesh

# ---------------- Dataset ----------------
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []
        token_ids = tokenizer.encode(txt, add_special_tokens=True)
        # safety: if too small, still allow one sample
        if len(token_ids) < max_length + 1:
            token_ids = token_ids + [tokenizer.eos_token_id] * (max_length + 1 - len(token_ids))
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
            self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]

def create_dataloader_v1(txt, tokenizer, batch_size=8, max_length=512, stride=512,
                         shuffle=True, drop_last=True, num_workers=2, distributed=False):
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)
    if distributed:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                          drop_last=drop_last, num_workers=num_workers, pin_memory=True)
    else:
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                          drop_last=drop_last, num_workers=num_workers, pin_memory=True)

# -------------------------------
# Transformer Blocks
# -------------------------------
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
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) *
                                         (x + 0.044715 * x ** 3)))


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
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(tok_embeds + pos_embeds)
        x = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))



# -------------------------------
# Training (with Early Stopping)
# -------------------------------
def calc_loss_batch(input_batch, target_batch, model, device, scaler=None):
    input_batch = input_batch.to(device, non_blocking=True).long()
    target_batch = target_batch.to(device, non_blocking=True).long()
    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
        logits = model(input_batch)  # logits shape (B, T, V)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
    return loss

def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, scaler=None, patience=5, rank=0):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        if dist.is_initialized():
            train_loader.sampler.set_epoch(epoch)
        for step, (input_batch, target_batch) in enumerate(train_loader):
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device, scaler)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            if step % eval_freq == 0:
                train_losses.append(loss.item())
                # Validation
                model.eval()
                val_loss_batch = []
                with torch.no_grad():
                    for val_input, val_target in val_loader:
                        val_loss = calc_loss_batch(val_input, val_target, model, device, scaler=None)
                        val_loss_batch.append(val_loss.item())
                mean_val_loss = sum(val_loss_batch) / max(1, len(val_loss_batch))
                val_losses.append(mean_val_loss)
                model.train()

                if rank == 0:
                    print(f"[E{epoch+1}] Step {step} Train {loss.item():.4f} Val {mean_val_loss:.4f}")

                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    epochs_no_improve = 0
                    if rank == 0:
                        torch.save(model.state_dict(), "best_model.pth")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        if rank == 0:
                            print(f"Early stopping at epoch {epoch+1}, step {step}")
                        # load best
                        map_location = {"cuda:%d" % 0: "cpu"} if rank == 0 else None
                        # only rank 0 needs to load for printing; others can continue
                        if rank == 0:
                            model.load_state_dict(torch.load("best_model.pth", map_location="cpu"))
                        return

        torch.cuda.empty_cache()
    if rank == 0:
        plot_losses(train_losses, val_losses)
        print("Training finished")

def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(8,4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.close()

# ---------------- Main with sequence parallel ----------------
def main(gpt_config, settings):
    # MUST be launched with torchrun
    assert "LOCAL_RANK" in os.environ and "WORLD_SIZE" in os.environ, "Run this with torchrun"
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    # set device
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # init process group (env:// expects torchrun to set addresses)
    dist.init_process_group(backend="nccl", init_method="env://")

    rank = dist.get_rank()
    if rank == 0:
        print("Distributed initialized:", dist.get_world_size(), "ranks")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    with open("sample_paragraphs_2000.txt", "r") as f:
        text_data = f.read()

    split_idx = int(0.75 * len(text_data))
    train_loader = create_dataloader_v1(text_data[:split_idx], tokenizer,
                                        batch_size=settings["batch_size"],
                                        max_length=gpt_config["context_length"],
                                        stride=gpt_config["context_length"],
                                        distributed=True, num_workers=2)
    val_loader = create_dataloader_v1(text_data[split_idx:], tokenizer,
                                      batch_size=settings["batch_size"],
                                      max_length=gpt_config["context_length"],
                                      stride=gpt_config["context_length"],
                                      distributed=False, num_workers=2)

    # build model and move to device
    model = GPTModel(gpt_config).to(device)
    tp_mesh = init_device_mesh("cuda", (world_size,))
    print(world_size)
# Only parallelize inner feedforward layers
    parallelize_plan = {
    "transformer.h.*.mlp.w1": ColwiseParallel(),
    "transformer.h.*.mlp.w2": RowwiseParallel(),
}

    sharded_model = parallelize_module(model, tp_mesh, parallelize_plan)
    # device mesh for tensor parallelism: a 1D mesh across world_size GPUs
    #tp_mesh = init_device_mesh("cuda", (world_size,))

    # Apply sequence parallel to final_norm (example). You can add other mappings:
    #sharded_model = parallelize_module(model, tp_mesh, {"final_norm": SequenceParallel()})
    # note: parallelize_module returns the module (sharded). Use this for optimizer/training.
    #sharded_model = sharded_model.to(device)

    # optimizer and scaler
    optimizer = torch.optim.AdamW(sharded_model.parameters(),
                                  lr=settings["learning_rate"],
                                  weight_decay=settings["weight_decay"])
    scaler = torch.cuda.amp.GradScaler()

    # train
    train_model_simple(sharded_model, train_loader, val_loader, optimizer, device,
                       num_epochs=settings["num_epochs"], eval_freq=5000,
                       scaler=scaler, patience=settings.get("patience", 5),
                       rank=rank)

    # save only from rank 0 (convert to cpu state dict for safety)
    if rank == 0:
        cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
        torch.save(cpu_state, "gpt_model_weights_200000.pth")
        print("Saved final model")

    dist.barrier()
    dist.destroy_process_group()

if __name__ == "__main__":
    GPT_CONFIG_124M = {
        "vocab_size": 50257,
        "context_length": 512,
        "emb_dim": 768,
        "n_heads": 8,
        "n_layers": 6,
        "drop_rate": 0.2,
        "qkv_bias": False
    }
    OTHER_SETTINGS = {
        "learning_rate": 5e-4,
        "num_epochs": 20,
        "batch_size": 4,
        "weight_decay": 0.1,
        "patience":10 
    }
    main(GPT_CONFIG_124M, OTHER_SETTINGS)    
