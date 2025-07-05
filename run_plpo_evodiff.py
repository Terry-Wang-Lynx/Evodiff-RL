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
# 配置和超参数 (已更新为带KL散度惩罚的PLPO)
# ---------------------------
CONFIG = {
    "learning_rate": 1e-6,
    "rl_epochs": 30,
    "steps_per_epoch": 100,
    "batch_size": 32,    # 在PLPO中，这代表每一步生成的候选序列数量 (M)
    "seq_len": 100,
    "adam_betas": (0.9, 0.98),
    "epsilon": 1e-8,     # 用于奖励归一化时防止除以零
    "weight_decay": 0.01,
    "top_k": 16,         # PLPO参数：仅使用奖励最高的K个样本进行学习
    "kl_beta": 0.1,      # KL散度惩罚的系数，防止过度优化
}

# 选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 定义碱性氨基酸
BASIC_AAS = ['R', 'K', 'H']

# ---------------------------
# 核心功能函数
# ---------------------------

def get_reward(sequences):
    """
    计算序列的奖励分数。
    奖励定义为序列中碱性氨基酸（R, K, H）的百分比。
    """
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

def calculate_log_likelihood(model, tokenizer, sequences, device):
    """
    计算给定序列的长度归一化对数似然 (Log-Likelihood)。
    此函数不管理模型的模式或梯度。
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
    avg_nll_per_sequence = nll_per_sequence / (lengths + 1e-9)
    
    log_likelihood = -avg_nll_per_sequence
    
    return log_likelihood

def plpo_loss(policy_log_likelihoods, ref_log_likelihoods, rewards, top_k, kl_beta, epsilon):
    """
    计算带KL散度惩罚的 PLPO (Plackett-Luce Preference Optimization) 损失。

    Args:
        policy_log_likelihoods (torch.Tensor): 策略模型对每个序列的长度归一化对数似然。
        ref_log_likelihoods (torch.Tensor): 参考模型对每个序列的长度归一化对数似然。
        rewards (torch.Tensor): 每个序列的奖励分数。
        top_k (int): 选择奖励最高的K个样本进行学习。
        kl_beta (float): KL散度惩罚的系数。
        epsilon (float): 用于归一化时防止除以零的小常数。

    Returns:
        torch.Tensor: 计算出的总损失。
        torch.Tensor: PLPO部分的目标函数值 (用于日志记录)。
        torch.Tensor: KL散度惩罚项的值 (用于日志记录)。
    """
    # 确保奖励为非负数
    if (rewards < 0).any():
        rewards = rewards - rewards.min()

    # 1. 计算PLPO目标
    rewards_norm = rewards / (rewards.sum() + epsilon)
    actual_top_k = min(top_k, len(rewards))
    top_indices = torch.topk(rewards, k=actual_top_k).indices
    
    top_log_likelihoods = policy_log_likelihoods[top_indices]
    top_rewards_norm = rewards_norm[top_indices]
    
    plpo_objective = - (top_rewards_norm * top_log_likelihoods).sum()
    
    # 2. 计算KL散度惩罚
    # KL(policy || ref) 约等于 E[log(ref(x)) - log(policy(x))]
    # 我们希望最大化 policy_log_likelihood - ref_log_likelihood，所以KL惩罚是其负值
    log_ratios = policy_log_likelihoods - ref_log_likelihoods
    kl_penalty = kl_beta * log_ratios.mean() # KL散度是策略和参考之间对数似然差的期望

    # 3. 总损失
    total_loss = plpo_objective + kl_penalty
    
    return total_loss, plpo_objective, kl_penalty

# ---------------------------
# 主训练循环
# ---------------------------
def main():
    print(f"Using device: {device}")

    print("Loading models...")
    # --- 改动: 重新引入参考模型 ---
    policy_model, _, policy_tokenizer, _ = OA_DM_38M()
    ref_model, _, ref_tokenizer, _ = OA_DM_38M()

    policy_model.to(device)
    ref_model.to(device)

    assert policy_tokenizer.alphabet == ref_tokenizer.alphabet, "Tokenizers must be the same"
    tokenizer = policy_tokenizer

    # 冻结参考模型的参数
    for param in ref_model.parameters():
        param.requires_grad = False
    ref_model.eval()
    print("Reference model parameters frozen.")
    # --- 结束改动 ---

    optimizer = AdamW(
        policy_model.parameters(),
        lr=CONFIG["learning_rate"],
        betas=CONFIG["adam_betas"],
        eps=CONFIG["epsilon"],
        weight_decay=CONFIG["weight_decay"],
    )

    print("Starting PLPO fine-tuning with KL penalty...")
    for epoch in range(CONFIG["rl_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['rl_epochs']} ---")
        
        total_loss, total_plpo_obj, total_kl_penalty = 0, 0, 0
        all_rewards = []

        pbar = tqdm(range(CONFIG["steps_per_epoch"]), desc=f"Epoch {epoch + 1} Loss: N/A")

        for step in pbar:
            # 1. 生成序列
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

            # 3. 计算对数似然
            policy_model.train()
            
            # --- 改动: 计算策略和参考模型的对数似然 ---
            policy_log_likelihood = calculate_log_likelihood(policy_model, tokenizer, gen_seqs, device)
            with torch.no_grad():
                ref_log_likelihood = calculate_log_likelihood(ref_model, tokenizer, gen_seqs, device)
            # --- 结束改动 ---

            # 4. 优化步骤
            optimizer.zero_grad()
            loss, plpo_obj, kl_term = plpo_loss(
                policy_log_likelihood, 
                ref_log_likelihood, 
                rewards, 
                CONFIG["top_k"], 
                CONFIG["kl_beta"], 
                CONFIG["epsilon"]
            )
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_plpo_obj += plpo_obj.item()
            total_kl_penalty += kl_term.item()
            
            if len(all_rewards) > 0:
                avg_reward = sum(all_rewards) / len(all_rewards)
                pbar.set_description(f"Epoch {epoch+1} Loss: {loss.item():.4f} (Obj: {plpo_obj.item():.4f}, KL: {kl_term.item():.4f}), Avg Reward: {avg_reward:.2f}%")

        avg_epoch_loss = total_loss / CONFIG["steps_per_epoch"]
        avg_epoch_reward = sum(all_rewards) / len(all_rewards)
        print(f"Epoch {epoch + 1} finished. Avg Loss: {avg_epoch_loss:.4f}, Avg Reward: {avg_epoch_reward:.2f}%")

        output_dir = f"evodiff_plpo_kl_checkpoint_epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(policy_model.state_dict(), os.path.join(output_dir, "policy_model.pt"))
        print(f"Policy model checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
