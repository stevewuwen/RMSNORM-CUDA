import torch
import math
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch.nn as nn
from build import rmsnorm_cuda

class CustomTLPRMSNorm(nn.Module):
    """
    用来替换 Qwen 官方 RMSNorm 的自定义 Wrapper 层
    """
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        y = torch.empty_like(hidden_states)
        stream = torch.cuda.current_stream().cuda_stream
        rmsnorm_cuda.launch_rmsnorm(5, hidden_states, self.weight, y, self.variance_epsilon, stream)
        return y

def replace_rmsnorm_with_custom(model):
    """
    递归遍历模型，将官方的 Qwen3RMSNorm 替换为你的 CustomTLPRMSNorm
    """
    replaced_count = 0
    for name, module in model.named_children():
        # Qwen2 系列的 LayerNorm 类名为 Qwen3RMSNorm
        if type(module).__name__ == "Qwen3RMSNorm":
            # 获取原层参数
            hidden_size = module.weight.shape[0]
            eps = module.variance_epsilon
            
            # 创建新层并拷贝权重
            new_layer = CustomTLPRMSNorm(hidden_size, eps)
            new_layer.weight.data.copy_(module.weight.data)
            new_layer = new_layer.to(module.weight.device).to(module.weight.dtype)
            
            # 替换！
            setattr(model, name, new_layer)
            replaced_count += 1
        else:
            # 递归
            replaced_count += replace_rmsnorm_with_custom(module)
    return replaced_count

def evaluate_wikitext2_ppl(model, tokenizer, device="cuda:0", stride=2048):
    """
    在 WikiText-2 测试集上计算模型的 Perplexity (困惑度)
    """
    print("Loading WikiText-2 dataset...")
    test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

    max_length = model.config.sliding_window if hasattr(model.config, "sliding_window") and model.config.sliding_window is not None else 2048
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    
    print(f"Evaluating PPL (Total tokens: {seq_len})...")
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # 预测步长
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # 我们只计算这一个 block 新生成 token 的 loss

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # NLL is loss * trg_len
            neg_log_likelihood = outputs.loss

        nlls.append(neg_log_likelihood.float())
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    ppl = torch.exp(torch.stack(nlls).mean().float())
    return ppl.item()

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_id = "Qwen/Qwen3-0.6B" # 你可以使用 0.5B 或更小的模型快速验证
    
    print(f"Loading official model {model_id}...")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # 使用 float16 匹配你的 CUDA 混合精度环境
    model_baseline = AutoModelForCausalLM.from_pretrained(
        model_id, dtype=torch.float16, device_map=None
    ).to('cuda:0')
    
    # 1. 跑官方基线 PPL
    print("\n--- Running Baseline (Official PyTorch/Triton RMSNorm) ---")
    baseline_ppl = evaluate_wikitext2_ppl(model_baseline, tokenizer, device=device)
    print(f"✅ Baseline WikiText-2 PPL: {baseline_ppl:.4f}")
    
    # 2. 替换算子并跑你的模型 PPL
    print("\n--- Replacing RMSNorm with Custom CUDA_TLP ---")
    replaced_num = replace_rmsnorm_with_custom(model_baseline)
    print(f"Replaced {replaced_num} RMSNorm layers.")
    
    print("\n--- Running Custom Op (CUDA_TLP) ---")
    custom_ppl = evaluate_wikitext2_ppl(model_baseline, tokenizer, device=device)
    print(f"✅ Custom Op WikiText-2 PPL: {custom_ppl:.4f}")
    
    # 3. 结论
    print("\n================= SUMMARY =================")
    print(f"Baseline PPL : {baseline_ppl:.4f}")
    print(f"Custom PPL   : {custom_ppl:.4f}")
    print(f"Absolute Diff: {abs(baseline_ppl - custom_ppl):.6f}")
    if abs(baseline_ppl - custom_ppl) < 0.1:
        print("🚀 Success! The mathematical deviation does not affect end-to-end model performance.")
    else:
        print("⚠️ Warning: Significant PPL degradation detected. Check your kernel math.")