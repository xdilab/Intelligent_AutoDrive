from pathlib import Path

EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent

# Paths
CSV_PATH = "/data/datasets/BDD-X/BDD-X-Annotations_v1.csv"
IMG_DIR = "/data/datasets/bdd100k-yolopx/images/train"
SPLIT_DIR = "/data/datasets/BDD-X"
CKPT_DIR = str(EXP_DIR / "checkpoints")
LOG_DIR = str(EXP_DIR / "logs")

# Model
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"

# LoRA — applied to LLM layers only (q_proj/k_proj/v_proj/o_proj use _ separator,
# distinguishing them from ViT's qkv/proj — so target_modules is ViT-safe)
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",   # LLM attention
    "gate_proj", "up_proj", "down_proj",        # LLM MLP
]

# Image (Qwen dynamic resolution)
MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 1280 * 28 * 28

# Training
BATCH_SIZE = 1
MAX_EPOCHS = 3
GRAD_ACCUM = 4        # effective batch = 4
LR = 2e-4
WARMUP_STEPS = 100
GRAD_CLIP = 1.0
WEIGHT_DECAY = 0.01

# Generation (eval.py only — not used during training)
MAX_NEW_TOKENS = 80
NUM_BEAMS = 1         # greedy; set to 4 for final numbers if needed
