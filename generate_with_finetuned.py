import torch
from evodiff.pretrained import OA_DM_38M
from evodiff.generate import generate_oaardm
import argparse

# 定义碱性氨基酸
BASIC_AAS = ['R', 'K', 'H']

def score_sequences(sequences):
    """计算并打印序列的碱性氨基酸含量"""
    print("\n--- Scoring Generated Sequences ---")
    total_reward = 0
    for i, seq in enumerate(sequences):
        length = len(seq)
        if length > 0:
            count = sum(seq.count(aa) for aa in BASIC_AAS)
            percentage = (count / length) * 100
            total_reward += percentage
            print(f"Sequence {i+1} | Length: {length} | Basic AA: {percentage:.2f}%")
        else:
            print(f"Sequence {i+1} is empty.")
    if sequences:
        print(f"\nAverage Basic AA content: {total_reward/len(sequences):.2f}%")

def main(args):
    # 确定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. 加载模型基础架构 ---
    # 这会创建一个与您训练时相同结构的模型实例
    print("Loading base model architecture...")
    model, _, tokenizer, _ = OA_DM_38M()

    # --- 2. 加载您优化后的权重 ---
    print(f"Loading fine-tuned weights from: {args.checkpoint_path}")
    # 注意：如果您在非 GPU 环境下加载模型，请使用 map_location
    try:
        model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        print("This might be due to a mismatch in model keys, which can happen if the model was saved within a DDP wrapper. This script assumes a raw state_dict.")
        return

    # --- 3. 准备模型用于生成 ---
    model.to(device)
    model.eval() # 必须设置为评估模式！

    # --- 4. 使用优化后的模型进行生成 ---
    print(f"\nGenerating {args.num_samples} sequences with the fine-tuned model...")
    with torch.no_grad():
        _, generated_sequences = generate_oaardm(
            model=model,
            tokenizer=tokenizer,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            device=device
        )
    
    # 我们只取所需的样本数量
    final_sequences = generated_sequences[:args.num_samples]

    print("\n--- Generated Sequences ---")
    for i, seq in enumerate(final_sequences):
        print(f"{i+1}: {seq}")

    # --- 5. (可选但推荐) 立刻评估生成序列的质量 ---
    score_sequences(final_sequences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sequences with a fine-tuned EvoDiff model.")
    parser.add_argument(
        '-c', '--checkpoint_path',
        type=str,
        default='evodiff_dpo_checkpoint_epoch_10/policy_model.pt',
        help='Path to the fine-tuned model checkpoint (.pt file).'
    )
    parser.add_argument(
        '-n', '--num_samples',
        type=int,
        default=10,
        help='Number of sequences to generate.'
    )
    parser.add_argument(
        '-l', '--seq_len',
        type=int,
        default=120,
        help='Length of the sequences to generate.'
    )
    parser.add_argument(
        '-b', '--batch_size',
        type=int,
        default=5,
        help='Batch size for generation.'
    )
    args = parser.parse_args()
    main(args)