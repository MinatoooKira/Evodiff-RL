import os
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm
import math
from torch.nn.utils.rnn import pad_sequence
import json
import pickle

from evodiff.pretrained import OA_DM_38M
from evodiff.generate import generate_oaardm

# ---------------------------
# 配置和超参数
# ---------------------------
CONFIG = {
    "kl_beta": 0.1,
    "learning_rate": 1e-6,
    "rl_epochs": 30,
    "steps_per_epoch": 100,
    "batch_size": 32,
    "seq_len": 100,
    "adam_betas": (0.9, 0.98),
    "epsilon": 1e-8,
    "weight_decay": 0.01,
    "epsilon_std": 1e-8,
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---------------------------
# AlphaFold结构打分用的函数
# ---------------------------
def get_reward(sequences, target_sequence, af2_predict_fn, output_root):
    rewards = []
    os.makedirs(output_root, exist_ok=True)

    for i, binder_seq in enumerate(sequences):
        try:
            fasta_path = os.path.join(output_root, f"seq_{i}.fasta")
            with open(fasta_path, "w") as f:
                f.write(">target\n" + target_sequence.strip() + "\n")
                f.write(">binder\n" + binder_seq.strip() + "\n")

            job_dir = os.path.join(output_root, f"job_{i}")
            os.makedirs(job_dir, exist_ok=True)
            af2_predict_fn(fasta_path, job_dir)

            with open(os.path.join(job_dir, "ranking_debug.json")) as f:
                rank_data = json.load(f)
                model_name = rank_data["order"][0]
                iptm = rank_data["ranking_confidences"][model_name]

            with open(os.path.join(job_dir, f"result_{model_name}.pkl"), "rb") as f:
                result = pickle.load(f)
                avg_plddt = float(result["plddt"].mean())

            reward = iptm + 0.1 * avg_plddt
            rewards.append(reward)

        except Exception as e:
            print(f"[!] Error in sequence {i}: {e}")
            rewards.append(-100.0)

    return torch.tensor(rewards, dtype=torch.float, device=device)


# ---------------------------
# NLL + GRPO loss
# ---------------------------
def calculate_nll(model, tokenizer, sequences, device):
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
    policy_objective = - (advantages * log_ratios).mean()
    kl_penalty = kl_beta * log_ratios.mean()
    return policy_objective + kl_penalty


# ---------------------------
# Main Training Loop
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

    target_sequence = "IIGGKEVSPHSRPFMASIQYGGHHVCGGVLIDPQWVLTAAHCQYRFTKGQSPTVVLGAHSLSKNEASKQTLEIKKFIPFSRVTSDPQSNDIMLVKLQTAAKLNKHVKMLHIRSKTSLRSGTKCKVTGWGATDPDSLRPSDTLREVTVTVLSRKLCNSQSYYNGDPFITKDMVCAGDAKGQKDSCKGDSGGPLICKGVFHAIVSGGHECGVATKPGIYTLLTKKYQTWIKSNLVPPHTNDYKDDDDK"  # GZMK-Flag 切后
    output_root = "af2_multimer_outputs"

    def af2_predict_fn(fasta_path, output_dir):
        os.system(f"run_af2_multimer.sh {fasta_path} {output_dir}")  # 注意！！！！！！这部分需要051完成。需要创建一个run_af2_multimer.sh来运行af2multimer。

    print("Starting GRPO fine-tuning...")
    for epoch in range(CONFIG["rl_epochs"]):
        print(f"\n--- Epoch {epoch + 1}/{CONFIG['rl_epochs']} ---")
        
        total_loss = 0
        all_rewards = []

        pbar = tqdm(range(CONFIG["steps_per_epoch"]), desc=f"Epoch {epoch + 1} Loss: N/A, Avg Reward: N/A")

        for step in pbar:
            policy_model.eval()
            with torch.no_grad():
                _, gen_seqs = generate_oaardm(
                    model=policy_model,
                    tokenizer=tokenizer,
                    seq_len=CONFIG["seq_len"],
                    batch_size=CONFIG["batch_size"],
                    device=device
                )
            
            rewards = get_reward(gen_seqs, target_sequence, af2_predict_fn, output_root)
            all_rewards.extend(rewards.cpu().numpy())

            with torch.no_grad():
                advantages = (rewards - rewards.mean()) / (rewards.std() + CONFIG["epsilon_std"])

            policy_model.train()
            policy_nll = calculate_nll(policy_model, tokenizer, gen_seqs, device)

            with torch.no_grad():
                ref_nll = calculate_nll(ref_model, tokenizer, gen_seqs, device)

            policy_log_likelihood = -policy_nll
            ref_log_likelihood = -ref_nll
            log_ratios = policy_log_likelihood - ref_log_likelihood

            optimizer.zero_grad()
            loss = grpo_loss(log_ratios, advantages, CONFIG["kl_beta"])
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            avg_reward = sum(all_rewards) / len(all_rewards)
            pbar.set_description(f"Epoch {epoch + 1} Loss: {loss.item():.4f}, Avg Reward: {avg_reward:.2f}")

        avg_epoch_loss = total_loss / CONFIG["steps_per_epoch"]
        avg_epoch_reward = sum(all_rewards) / len(all_rewards)
        print(f"Epoch {epoch + 1} finished. Average Loss: {avg_epoch_loss:.4f}, Average Reward: {avg_epoch_reward:.2f}")

        output_dir = f"evodiff_grpo_checkpoint_epoch_{epoch+1}"
        os.makedirs(output_dir, exist_ok=True)
        torch.save(policy_model.state_dict(), os.path.join(output_dir, "policy_model.pt"))
        print(f"Policy model checkpoint saved to {output_dir}")


if __name__ == "__main__":
    main()
