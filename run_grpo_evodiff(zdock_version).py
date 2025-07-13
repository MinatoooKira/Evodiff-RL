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
    "kl_beta": 0.1,  # KL散度惩罚的系数 (原beta参数)
    "learning_rate": 1e-6,
    "rl_epochs": 30,
    "steps_per_epoch": 100,
    "batch_size": 32, # 在GRPO中，每个batch被视为一个“group”
    "seq_len": 100,
    "adam_betas": (0.9, 0.98),
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "epsilon_std": 1e-8, # 用于计算优势时防止除以零
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

def calculate_nll(model, tokenizer, sequences, device):
    """
    计算给定序列的负对数似然 (Negative Log-Likelihood, NLL)。
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
    
    return nll_per_sequence

def grpo_loss(log_ratios, advantages, kl_beta):
    """
    计算 GRPO (Group Relative Policy Optimization) 的损失。
    
    Args:
        log_ratios (torch.Tensor): 策略与参考模型的对数似然比 (policy_log_likelihood - ref_log_likelihood)。
        advantages (torch.Tensor): 每个序列的归一化优势分数。
        kl_beta (float): KL散度惩罚项的系数。

    Returns:
        torch.Tensor: 计算出的总损失。
    """
    # 策略目标：我们希望最大化 (advantages * log_ratios)，所以损失是其负值。
    # 优势为正的样本会促使 log_ratios 增大，反之亦然。
    policy_objective = - (advantages * log_ratios).mean()
    
    # KL惩罚项：防止策略模型偏离参考模型太远。
    # log_ratios 本身就是 KL 散度的一个估计。
    kl_penalty = kl_beta * log_ratios.mean()

    return policy_objective + kl_penalty

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

    # 冻结参考模型的参数，它只用于计算基线，不参与训练
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

    print("Starting GRPO fine-tuning...")
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

            # --- GRPO核心改动 ---
            # 3. 计算分组相对优势 (Group-Relative Advantage)
            # 我们将每个batch视为一个“group”，计算每个样本相对于组内平均奖励的优势。
            # advantages 是一个归一化的分数，表示每个奖励比平均水平好/差多少。
            with torch.no_grad():
                advantages = (rewards - rewards.mean()) / (rewards.std() + CONFIG["epsilon_std"])
            # --- 结束改动 ---
            
            # 4. 计算似然并进行优化 (需要 policy_model 在 train 模式下)
            policy_model.train()
            
            # 为 policy_model 计算 NLL，此时会构建计算图
            policy_nll = calculate_nll(policy_model, tokenizer, gen_seqs, device)
            
            # 使用 `no_grad` 上下文为 ref_model 计算 NLL，不构建计算图
            with torch.no_grad():
                ref_nll = calculate_nll(ref_model, tokenizer, gen_seqs, device)
            
            policy_log_likelihood = -policy_nll
            ref_log_likelihood = -ref_nll
            
            # 计算对数似然比
            log_ratios = policy_log_likelihood - ref_log_likelihood

            # 5. 优化步骤
            optimizer.zero_grad()
            # 使用 GRPO 损失函数
            loss = grpo_loss(log_ratios, advantages, CONFIG["kl_beta"])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if len(all_rewards) > 0:
                avg_reward = sum(all_rewards) / len(all_rewards)
                pbar.set_description(f"Epoch {epoch + 1} Loss: {loss.item():.4f}, Avg Reward: {avg_reward:.2f}%")

        avg_epoch_loss = total_loss / CONFIG["steps_per_epoch"]
        avg_epoch_reward = sum(all_rewards) / len(all_rewards)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, Average Reward: {avg_epoch_reward:.2f}%")

        output_dir = f"evodiff_grpo_checkpoint_epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(policy_model.state_dict(), os.path.join(output_dir, "policy_model.pt"))
        print(f"Policy model checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
