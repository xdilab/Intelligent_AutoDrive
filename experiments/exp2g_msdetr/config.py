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
BACKBONE = "resnet50"
FPN_IN_CHANNELS = [512, 1024, 2048]  # R50 layer2/3/4 channel dims
FPN_OUT = 256
FPN_LEVELS = 3                       # P3, P4, P5 (decoder sees these)

# ---- Resolution: variable aspect ratio (paper config) ----
VAL_SHORT_SIDE = 800
VAL_MAX_SIZE = 1333

# ---- Deformable DETR encoder ----
NUM_ENCODER_LAYERS = 6
ENCODER_N_LEVELS = 4                 # P3 + P4 + P5 + CLIP patches
ENCODER_D_FFN = 2048

# ---- Deformable DETR decoder ----
D_MODEL = 256
NUM_QUERIES = 900                    # top-k proposals from encoder (paper: 900)
NUM_DECODER_LAYERS = 6
NHEAD = 8
DIM_FFN = 2048
N_DEFORM_POINTS = 4
DROPOUT = 0.0

# ---- Two-stage query init (paper: --two_stage --mixed_selection) ----
TWO_STAGE = True
MIXED_SELECTION = True               # learnable content + proposal position queries

# ---- O2M loss (paper: --use_ms_detr --use_aux_ffn) ----
USE_MS_DETR = True
USE_AUX_FFN = True
O2M_K = 6                           # top-k predictions per GT
O2M_THRESHOLD = 0.4                 # IoU threshold for positive assignment
LAMBDA_O2M_CLS = 2.0                # paper: --o2m_cls_loss_coef 2
LAMBDA_O2M_BBOX = 5.0
LAMBDA_O2M_GIOU = 2.0

# ---- Encoder loss (paper: two-stage encoder supervision) ----
LAMBDA_ENC_CLS = 2.0                # paper: --enc_cls_loss_coef 2
LAMBDA_ENC_BBOX = 5.0               # paper: --enc_bbox_loss_coef 5
LAMBDA_ENC_GIOU = 2.0               # paper: --enc_giou_loss_coef 2

# ---- Softmax agent head ----
USE_SOFTMAX_AGENT = True             # softmax CE on agent (single-label)

# ---- Data ----
CLIP_LEN = 8
CLIP_STRIDE = 16

# ---- Labels (flat 184-dim vector, matching 3D-RetinaNet baseline) ----
N_AGENTS = 10
N_ACTIONS = 22
N_LOCS = 16
N_DUPLEXES = 49
N_TRIPLETS = 86

NUM_CLASSES = 1 + N_AGENTS + N_ACTIONS + N_LOCS + N_DUPLEXES + N_TRIPLETS  # 184
NUM_CLASSES_LIST = [1, N_AGENTS, N_ACTIONS, N_LOCS, N_DUPLEXES, N_TRIPLETS]

# Offsets into the flat 184-dim vector
CLS_OFFSETS = {
    "agentness": 0,
    "agent": 1,
    "action": 1 + N_AGENTS,                          # 11
    "loc": 1 + N_AGENTS + N_ACTIONS,                  # 33
    "duplex": 1 + N_AGENTS + N_ACTIONS + N_LOCS,      # 49
    "triplet": 1 + N_AGENTS + N_ACTIONS + N_LOCS + N_DUPLEXES,  # 98
}

# Per-head sizes (used by eval to unpack flat vector)
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
LAMBDA_CLS = 1.0                    # paper: --cls_loss_coef 1
LAMBDA_BBOX = 5.0
LAMBDA_GIOU = 2.0
LAMBDA_TNORM = 1.0
FOCAL_GAMMA = 2.0
TUBE_LINK_IOU = 0.3

# ---- Training ----
BATCH_SIZE = 1
MAX_EPOCHS = 30                      # paper uses 12 but we have smaller effective batch
GRAD_ACCUM = 4
LR_BACKBONE = 2e-5
LR_ENCODER_DECODER = 2e-4           # paper: --lr 2e-4
LR_HEADS = 2e-4
LR_FRESH = 1e-3                     # 5x for fresh-init components (pos_trans, O2M heads, enc_box_head)
LR_DEFORM_MULT = 0.1                # paper: reference_points + sampling_offsets get 0.1x
LR_DROP_EPOCH = 20                  # paper: --lr_drop 11 (of 12); proportionally ~20 of 30
LR_DROP_FACTOR = 0.1                # 10x decay at lr_drop epoch
WARMUP_STEPS = 500
GRAD_CLIP = 0.1
WEIGHT_DECAY = 1e-4

# ---- Inference ----
CONFIDENCE_THRESHOLD = 0.3
CLASS_THRESHOLD = 0.05
