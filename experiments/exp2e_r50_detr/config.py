from pathlib import Path


EXP_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXP_DIR.parent.parent

# Paths
ANNO_FILE = "/data/datasets/ROAD_plusplus/road_waymo_trainval_v1.1.json"
FRAMES_DIR = "/data/datasets/ROAD_plusplus/rgb-images"
CKPT_DIR = str(EXP_DIR / "checkpoints")
LOG_DIR = str(EXP_DIR / "logs")
DINO_COCO_CKPT = str(REPO_ROOT / "pretrained" / "dino_4scale_r50_1x_coco_checkpoint0011.pth")

# ---- CLIP ViT-L/14 (frozen semantic branch) ----
CLIP_MODEL = "ViT-L/14@336px"       # auto-downloads from OpenAI
CLIP_CACHE_DIR = str(Path.home() / ".cache" / "clip")
CLIP_PATCH_DIM = 1024               # ViT-L internal width (patch tokens)
CLIP_CLS_DIM = 768                  # ViT-L embed dim (CLS token after projection)
CLIP_INPUT_SIZE = 336
CLIP_GRID = 24                      # 336 / 14

# ---- ResNet-50 backbone (spatial branch) ----
# Source: Frozen-DETR/MS-DETR/models/backbone.py
BACKBONE = "resnet50"
FPN_IN_CHANNELS = [512, 1024, 2048]  # R50 layer2/3/4 channel dims
FPN_OUT = 256
FPN_LEVELS = 3                       # P3, P4, P5 (decoder sees these)

# ---- Resolution: variable aspect ratio (paper config) ----
# Source: Frozen-DETR/MS-DETR/datasets/coco.py:139-156
VAL_SHORT_SIDE = 800
VAL_MAX_SIZE = 1333
# Training uses multi-scale [480..800] from augmentations.py

# ---- Deformable DETR encoder ----
NUM_ENCODER_LAYERS = 6
ENCODER_N_LEVELS = 4                 # P3 + P4 + P5 + CLIP patches
ENCODER_D_FFN = 2048                 # match paper train script

# ---- Deformable DETR decoder ----
D_MODEL = 256
NUM_QUERIES = 300
NUM_DECODER_LAYERS = 6
NHEAD = 8
DIM_FFN = 2048                       # paper train script: --dim_feedforward 2048
N_DEFORM_POINTS = 4                  # sampling points per head per level
DROPOUT = 0.0                        # paper train script: --dropout 0.0

# ---- Data ----
CLIP_LEN = 8
CLIP_STRIDE = 16

# ---- Labels ----
N_AGENTS = 10
N_ACTIONS = 22
N_LOCS = 16
N_DUPLEXES = 49
N_TRIPLETS = 86

HEAD_SIZES = {
    "agent": N_AGENTS,
    "action": N_ACTIONS,
    "loc": N_LOCS,
    "duplex": N_DUPLEXES,
    "triplet": N_TRIPLETS,
}

# ---- Hungarian cost ----
COST_CLASS = 2.0
COST_BBOX = 5.0
COST_GIOU = 2.0

# ---- Loss ----
LAMBDA_CLS = 2.0
LAMBDA_BBOX = 5.0
LAMBDA_GIOU = 2.0
LAMBDA_TNORM = 1.0
EOS_COEF = 0.1
FOCAL_GAMMA = 2.0
NOOBJ_GAMMA = 2.0
TUBE_LINK_IOU = 0.3

# ---- Training (paper: main.py defaults + train script) ----
BATCH_SIZE = 1
MAX_EPOCHS = 30
GRAD_ACCUM = 4                       # effective batch 4 (paper: 16 across 4 GPUs)
LR_BACKBONE = 2e-5                   # paper main.py:34
LR_ENCODER_DECODER = 1e-4
LR_HEADS = 1e-4
WARMUP_STEPS = 500
GRAD_CLIP = 0.1                      # paper main.py default
WEIGHT_DECAY = 1e-4                  # paper main.py default

# ---- Inference ----
CONFIDENCE_THRESHOLD = 0.3
CLASS_THRESHOLD = 0.05
