import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import math
from torch.nn.utils.rnn import pad_sequence

from evodiff.pretrained import OA_DM_38M
from evodiff.generate import generate_oaardm

# ---------------------------
# 配置和超参数
# ---------------------------
CONFIG = {
    "beta": 1.0,
    "learning_rate": 1e-6,
    "rl_epochs": 30,
    "steps_per_epoch": 100,
    "batch_size": 32,
    "seq_len": 100,
    "adam_betas": (0.9, 0.98),
    "epsilon": 1e-8,
    "weight_decay": 0.01,
}

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义碱性氨基酸
BASIC_AAS = ['R', 'K', 'H']

# ---------------------------
# 核心功能函数
# ---------------------------

def get_reward(sequences):
    scores = []
    for seq in sequences:
        length = len(seq)
        if length == 0:
            scores.append(0.0)
            continue
        count = sum(seq.count(aa) for aa in BASIC_AAS)
        percentage = (count / length) * 100
        scores.append(percentage)
    return torch.tensor(scores, dtype=torch.float, device=device)

def calculate_nll(model, tokenizer, sequences, device):
    """
    (已再次修正) 
    此函数现在只负责计算 NLL，不管理模型的模式或梯度。
    """
    loss_fn = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=tokenizer.pad_id)

    tokenized_list = [torch.tensor(tokenizer.tokenizeMSA(s)) for s in sequences]
    tokenized_batch = pad_sequence(tokenized_list, batch_first=True, padding_value=tokenizer.pad_id).to(device)
    masked_input = torch.full_like(tokenized_batch, tokenizer.mask_id)
    lengths = torch.tensor([len(s) for s in sequences], device=device)
    
    logits = model(masked_input, lengths)
    
    loss_per_token = loss_fn(logits.permute(0, 2, 1), tokenized_batch)
    mask = (tokenized_batch != tokenizer.pad_id).float()
    nll_per_sequence = (loss_per_token * mask).sum(dim=1)
    
    return nll_per_sequence

def dpo_weighted_loss(policy_log_likelihood, ref_log_likelihood, rewards, beta):
    rewards = rewards.view(-1)
    log_ratios = policy_log_likelihood - ref_log_likelihood
    target_probs = F.softmax(rewards, dim=-1)
    loss = F.cross_entropy(log_ratios * beta, target_probs)
    return loss

# ---------------------------
# 主训练循环
# ---------------------------
def main():
    print(f"Using device: {device}")

    print("Loading models...")
    policy_model, _, policy_tokenizer, _ = OA_DM_38M()
    ref_model, _, ref_tokenizer, _ = OA_DM_38M()

    policy_model.to(device)
    ref_model.to(device)

    assert policy_tokenizer.alphabet == ref_tokenizer.alphabet, "Tokenizers must be the same"
    tokenizer = policy_tokenizer

    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    print("Reference model parameters frozen.")

    optimizer = AdamW(
        policy_model.parameters(),
        lr=CONFIG["learning_rate"],
        betas=CONFIG["adam_betas"],
        eps=CONFIG["epsilon"],
        weight_decay=CONFIG["weight_decay"],
    )

    print("Starting DPO fine-tuning...")
    for epoch in range(CONFIG["rl_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['rl_epochs']} ---")
        
        total_loss = 0
        all_rewards = []

        pbar = tqdm(range(CONFIG["steps_per_epoch"]), desc=f"Epoch {epoch + 1} Loss: N/A, Avg Reward: N/A")

        for step in pbar:
            # 1. 生成序列 (需要 policy_model 在 eval 模式下)
            policy_model.eval()
            with torch.no_grad():
                 _, gen_seqs = generate_oaardm(
                    model=policy_model,
                    tokenizer=tokenizer,
                    seq_len=CONFIG["seq_len"],
                    batch_size=CONFIG["batch_size"],
                    device=device
                )
            
            # 2. 评估奖励
            rewards = get_reward(gen_seqs)
            all_rewards.extend(rewards.cpu().numpy())

            # 3. 计算似然并进行优化 (需要 policy_model 在 train 模式下)
            policy_model.train()
            
            # 为 policy_model 计算 NLL，此时会构建计算图
            policy_nll = calculate_nll(policy_model, tokenizer, gen_seqs, device)
            
            # 使用 `no_grad` 上下文为 ref_model 计算 NLL，不构建计算图
            with torch.no_grad():
                ref_nll = calculate_nll(ref_model, tokenizer, gen_seqs, device)
            
            policy_log_likelihood = -policy_nll
            ref_log_likelihood = -ref_nll
            
            # 4. 优化步骤
            optimizer.zero_grad()
            loss = dpo_weighted_loss(policy_log_likelihood, ref_log_likelihood, rewards, CONFIG["beta"])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if len(all_rewards) > 0:
                avg_reward = sum(all_rewards) / len(all_rewards)
                pbar.set_description(f"Epoch {epoch + 1} Loss: {loss.item():.4f}, Avg Reward: {avg_reward:.2f}%")

        avg_epoch_loss = total_loss / CONFIG["steps_per_epoch"]
        avg_epoch_reward = sum(all_rewards) / len(all_rewards)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, Average Reward: {avg_epoch_reward:.2f}%")

        output_dir = f"evodiff_dpo_checkpoint_epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(policy_model.state_dict(), os.path.join(output_dir, "policy_model.pt"))
        print(f"Policy model checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()