# pipeline.py  (revised)
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # avoid tokenizer parallelism warning

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler, SequentialSampler
from transformers import GPT2TokenizerFast
import matplotlib.pyplot as plt
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# -------------------------------
# Dataset
# -------------------------------
class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        # Tokenize and keep simple list of token ids
        token_ids = tokenizer.encode(txt, add_special_tokens=True)

        # Build chunks
        # if len(token_ids) < max_length: this will produce empty dataset; ensure file large enough
        for i in range(0, max(1, len(token_ids) - max_length), max(1, stride)):
            input_chunk = token_ids[i:i + max_length]
            target_chunk = token_ids[i + 1:i + max_length + 1]
            if len(input_chunk) == max_length and len(target_chunk) == max_length:
                self.input_ids.append(torch.tensor(input_chunk, dtype=torch.long))
                self.target_ids.append(torch.tensor(target_chunk, dtype=torch.long))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def make_dataloaders_from_text(txt, tokenizer, gpt_config, settings, local_rank, world_size):
    # Build datasets and dataloaders with DistributedSampler for training
    max_length = gpt_config["context_length"]
    stride = gpt_config["context_length"]

    # split text
    split_idx = int(0.75 * len(txt))
    train_txt = txt[:split_idx]
    val_txt = txt[split_idx:]

    train_dataset = GPTDatasetV1(train_txt, tokenizer, max_length=max_length, stride=stride)
    val_dataset = GPTDatasetV1(val_txt, tokenizer, max_length=max_length, stride=stride)

    # samplers
    if world_size > 1:
        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=local_rank, shuffle=True)
    else:
        train_sampler = None

    # dataloaders
    train_loader = DataLoader(train_dataset,
                              batch_size=settings["batch_size"],
                              sampler=train_sampler if train_sampler is not None else None,
                              shuffle=(train_sampler is None),
                              drop_last=True,
                              num_workers=2,
                              pin_memory=True)

    # validation should not be shuffled, use SequentialSampler if distributed
    val_sampler = SequentialSampler(val_dataset) if world_size > 1 else None
    val_loader = DataLoader(val_dataset,
                            batch_size=settings["batch_size"],
                            sampler=val_sampler,
                            shuffle=False,
                            drop_last=False,
                            num_workers=2,
                            pin_memory=True)

    return train_loader, val_loader, train_sampler


# -------------------------------
# Transformer Blocks (kept your implementations)
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
        # mask_bool has shape (num_tokens, num_tokens) -> broadcast across heads and batch
        attn_scores = attn_scores.masked_fill(mask_bool.unsqueeze(0).unsqueeze(0), float('-inf'))
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
        tok_embeds = self.tok_emb(in_idx)                              # (B, T, C)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # (T, C)
        x = self.drop_emb(tok_embeds + pos_embeds)  # broadcasting (T,C) -> (B,T,C)
        x = self.trf_blocks(x)
        return self.out_head(self.final_norm(x))


# -------------------------------
# Training (with Early Stopping)
# -------------------------------
def calc_loss_batch(input_batch, target_batch, model, device, scaler=None):
    input_batch = input_batch.to(device, non_blocking=True)
    target_batch = target_batch.to(device, non_blocking=True)
    with torch.cuda.amp.autocast(enabled=(scaler is not None)):
        logits = model(input_batch)  # shape (B, T, V)
        # compute CE over flattened tokens
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), target_batch.view(-1))
    return loss


def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, scaler=None, patience=5, train_sampler=None):
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
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

                print(f"Epoch {epoch + 1}, Step {step}, Train Loss: {loss.item():.3f}, Val Loss: {mean_val_loss:.3f}")

                # ---- EARLY STOPPING ----
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    epochs_no_improve = 0
                    # save underlying module if wrapped in DDP
                    to_save = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                    torch.save(to_save, "best_model.pth")
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= patience:
                        print(f"\n⏹ Early stopping at Epoch {epoch + 1}, Step {step}. No improvement for {patience} evals.")
                        # load best weights
                        if isinstance(model, DDP):
                            model.module.load_state_dict(torch.load("best_model.pth"))
                        else:
                            model.load_state_dict(torch.load("best_model.pth"))
                        plot_losses(train_losses, val_losses)
                        return

        torch.cuda.empty_cache()

    plot_losses(train_losses, val_losses)
    print("✅ Loss plot saved as loss_plot_200000_papers.png")


def plot_losses(train_losses, val_losses):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Evaluation Step")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss (with Early Stopping)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("loss_plot_200000_papers.png")
    plt.close()


def main(gpt_config, settings):
    # Standard DDP device selection
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_cuda = torch.cuda.is_available()

    device = torch.device(f"cuda:{local_rank}" if use_cuda else "cpu")

    # tokenizer & text
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    with open("sample_paragraphs_2000.txt", "r") as f:
        text_data = f.read()

    # dataloaders (with distributed sampler if world_size>1)
    train_loader, val_loader, train_sampler = make_dataloaders_from_text(
        text_data, tokenizer, gpt_config, settings, local_rank, world_size
    )

    # init process group for DDP (env:// expects torchrun to set env vars)
    backend = "nccl" if use_cuda else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")

    # build model on correct device and wrap with DDP
    model = GPTModel(gpt_config).to(device)
    ddp_model = DDP(model, device_ids=[local_rank] if use_cuda else None)

    optimizer = torch.optim.AdamW(ddp_model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"])
    scaler = torch.cuda.amp.GradScaler() if use_cuda else None

    # Train using ddp_model
    train_model_simple(ddp_model, train_loader, val_loader, optimizer, device,
                       num_epochs=settings["num_epochs"], eval_freq=5000,
                       scaler=scaler, patience=settings.get("patience", 5),
                       train_sampler=train_sampler)

    # save final weights (underlying module)
    to_save = ddp_model.module.state_dict() if isinstance(ddp_model, DDP) else ddp_model.state_dict()
    if dist.get_rank() == 0:
        torch.save(to_save, "gpt_model_weights_200000.pth")
        print(" Training complete. Weights saved as gpt_model_weights_200000.pth")

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
        "batch_size": 8,
        "weight_decay": 0.1,
        "patience": 5   # <-- early stopping patience
    }
    main(GPT_CONFIG_124M, OTHER_SETTINGS)

