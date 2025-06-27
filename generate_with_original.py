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

    # --- 1. 加载模型基础架构与原始预训练权重 ---
    # OA_DM_38M() 函数会自动下载并加载原始的、未经优化的模型参数
    print("Loading ORIGINAL pre-trained model...")
    model, _, tokenizer, _ = OA_DM_38M()

    # --- (已移除) 不再加载优化后的检查点 ---
    # 这部分代码已被删除，因此模型将保持其原始状态

    # --- 2. 准备模型用于生成 ---
    model.to(device)
    model.eval() # 必须设置为评估模式！

    # --- 3. 使用原始模型进行生成 ---
    print(f"\nGenerating {args.num_samples} sequences with the ORIGINAL model...")
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

    print("\n--- Generated Sequences (from Original Model) ---")
    for i, seq in enumerate(final_sequences):
        print(f"{i+1}: {seq}")

    # --- 4. 评估原始模型生成序列的质量 ---
    score_sequences(final_sequences)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate sequences with the ORIGINAL pre-trained EvoDiff model.")
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