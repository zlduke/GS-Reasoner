# GS-Reasoner Model Hierarchy

## Overview

GS-Reasoner is a multimodal model that combines:
1. **Qwen2** - Language Model (LLM backbone)
2. **SigLIP** - Vision Encoder (2D image features)
3. **Sonata (PointTransformerV3)** - Spatial Encoder (3D point cloud features)

---

## Entry Point Flow

```
eval_vsibench.sh
    │
    │  --model gs_reasoner
    │  --model_args "pretrained=data/models/GS-Reasoner,..."
    │
    ▼
lmms_eval/__main__.py
    │
    │  cli_evaluate() → evaluator.simple_evaluate()
    │
    ▼
lmms_eval/evaluator.py
    │
    │  get_model("gs_reasoner") → GS_Reasoner class
    │  ModelClass.create_from_arg_string(model_args)
    │
    ▼
lmms_eval/models/gs_reasoner.py :: GS_Reasoner
    │
    │  load_pretrained_model("data/models/GS-Reasoner", ...)
    │
    ▼
llava/model/builder.py :: load_pretrained_model()
    │
    │  → LlavaQwenForCausalLM.from_pretrained()
    │
    ▼
[Model Loaded]
```

---

## Model Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           GS_Reasoner (Evaluation Wrapper)                   │
│                     lmms_eval/models/gs_reasoner.py                         │
│                                                                             │
│  - Handles VSI-Bench evaluation                                             │
│  - Loads VideoProcessor for 3D data                                         │
│  - Calls generate_until() / loglikelihood()                                 │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      │ wraps
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LlavaQwenForCausalLM                                 │
│               llava/model/language_model/llava_qwen.py                      │
│                                                                             │
│  Inherits: Qwen2ForCausalLM + LlavaMetaForCausalLM                         │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                         LlavaQwenModel                                │ │
│  │              (LlavaMetaModel + Qwen2Model)                            │ │
│  │                                                                       │ │
│  │  Components initialized in LlavaMetaModel.__init__():                 │ │
│  │                                                                       │ │
│  │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────┐   │ │
│  │  │  vision_tower   │  │  mm_projector   │  │  spatial_encoder    │   │ │
│  │  │    (SigLIP)     │  │   (MLP 2x)      │  │     (Sonata)        │   │ │
│  │  └────────┬────────┘  └────────┬────────┘  └──────────┬──────────┘   │ │
│  │           │                    │                      │              │ │
│  │           ▼                    ▼                      ▼              │ │
│  │    2D Image Features     Project to LLM      3D Spatial Features     │ │
│  │    [V, 729, 1152]        Hidden Size         [B, V, 196, 1232]      │ │
│  │                                                                       │ │
│  │  Additional components for 3D:                                        │ │
│  │  ┌─────────────────────────────────────────────────────────────────┐ │ │
│  │  │  • ada_crossattn (CrossAttnFusion) - fuses 2D + 3D features     │ │ │
│  │  │  • mm_spatial_projector (MLP) - projects fused spatial features │ │ │
│  │  │  • world_position_embedding (Sin3D) - 3D position encoding      │ │ │
│  │  │  • pointnet_pooling (PointNet) - pools point features           │ │ │
│  │  └─────────────────────────────────────────────────────────────────┘ │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
│                                                                             │
│  ┌───────────────────────────────────────────────────────────────────────┐ │
│  │                           lm_head                                     │ │
│  │              Linear(3584 → 151653 vocab)                              │ │
│  └───────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Component Details

### 1. Vision Encoder: SigLIP
```
Location: llava/model/multimodal_encoder/siglip_encoder.py
Config:   data/models/GS-Reasoner/config.json → "mm_vision_tower": "google/siglip-so400m-patch14-384"
Weights:  data/models/siglip-so400m-patch14-384/

Architecture:
  SigLipVisionTower
    └── SigLipVisionModel (from_pretrained)
          └── SigLipVisionTransformer
                ├── SigLipVisionEmbeddings (patch_size=14, image_size=384)
                ├── SigLipEncoder (27 transformer layers)
                └── SigLipMultiheadAttentionPoolingHead

Input:  Images [B, V, 3, 384, 384]
Output: Features [B*V, 729, 1152]  (729 = 27×27 patches)
```

### 2. Language Model: Qwen2
```
Location: llava/model/language_model/qwen2/modeling_qwen2.py
Config:   data/models/GS-Reasoner/config.json

Architecture:
  Qwen2ForCausalLM
    └── Qwen2Model
          ├── embed_tokens (151653 vocab → 3584 hidden)
          ├── 28 Qwen2DecoderLayers
          │     ├── self_attn (28 heads, 4 KV heads)
          │     └── mlp (18944 intermediate)
          └── norm (RMSNorm)

Hidden Size: 3584
Context Length: 32768
```

### 3. Spatial Encoder: Sonata (PointTransformerV3)
```
Location: llava/submodules/sonata/model.py
Weights:  data/models/sonata-wo-normal/sonata_wo_normal.pth

Architecture:
  PointTransformerV3
    ├── embedding (projects point coords + colors)
    ├── encoder stages (point transformer blocks with pooling)
    └── decoder stages (upsampling with skip connections)

Input:  Point cloud {coord: [N, 3], color: [N, 3]}
Output: Point features [N, 1232]
```

### 4. Multimodal Projector
```
Location: llava/model/multimodal_projector/builder.py
Type:     mlp2x_gelu

Architecture:
  nn.Sequential(
    Linear(1152 → 3584),  # mm_hidden_size → hidden_size
    GELU(),
    Linear(3584 → 3584)
  )

Purpose: Project SigLIP features to LLM hidden dimension
```

---

## Data Flow in Forward Pass

```
                          Input Video + Depth/Poses
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
            ┌──────────────┐               ┌──────────────┐
            │   SigLIP     │               │ Unproject to │
            │  Encoding    │               │ 3D Points    │
            └──────┬───────┘               └──────┬───────┘
                   │                              │
                   │ [V, 729, 1152]               │ [V, H, W, 3]
                   │                              │
                   ▼                              ▼
            ┌──────────────┐               ┌──────────────┐
            │  2D Pooling  │               │   Sonata     │
            │  (27→14)     │               │  Encoding    │
            └──────┬───────┘               └──────┬───────┘
                   │                              │
                   │ [V, 196, 1152]               │ [V, 196, 1232]
                   │                              │
                   └───────────┬──────────────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Cross-Attention    │
                    │  Fusion (ada_cross) │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Spatial Projector   │
                    │ + Position Embed    │
                    └──────────┬──────────┘
                               │
                               │ spatial_feat [B, V, 196, 3584]
                               │
                               ▼
                    ┌─────────────────────┐
                    │  image_feat (2D)    │
                    │       +             │
                    │  spatial_feat (3D)  │
                    │       +             │
                    │  position_feat      │
                    └──────────┬──────────┘
                               │
                               │ fused [B, V, 196, 3584]
                               ▼
                    ┌─────────────────────┐
                    │  Flatten + Merge    │
                    │  with Text Tokens   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      Qwen2 LLM      │
                    │   (28 layers)       │
                    └──────────┬──────────┘
                               │
                               ▼
                         Output Tokens
```

---

## File Dependencies Summary

```
eval_vsibench.sh
 │
 └─► lmms_eval/__main__.py
      └─► lmms_eval/evaluator.py
           └─► lmms_eval/models/__init__.py (registry)
                └─► lmms_eval/models/gs_reasoner.py (GS_Reasoner wrapper)
                     │
                     └─► llava/model/builder.py (load_pretrained_model)
                          │
                          ├─► llava/model/language_model/llava_qwen.py
                          │    ├─► LlavaQwenForCausalLM (main model class)
                          │    └─► LlavaQwenModel
                          │         └─► llava/model/llava_arch.py (LlavaMetaModel mixin)
                          │
                          ├─► llava/model/multimodal_encoder/builder.py
                          │    └─► llava/model/multimodal_encoder/siglip_encoder.py (SigLipVisionTower)
                          │
                          ├─► llava/model/multimodal_projector/builder.py
                          │
                          └─► llava/submodules/sonata/model.py (PointTransformerV3)
```

---

## Required Checkpoints

| Component | Path | Source |
|-----------|------|--------|
| GS-Reasoner (full model) | `data/models/GS-Reasoner/` | [HuggingFace](https://huggingface.co/ymccccc/GS-Reasoner) |
| SigLIP | `data/models/siglip-so400m-patch14-384/` | [HuggingFace](https://huggingface.co/google/siglip-so400m-patch14-384) |
| Sonata | `data/models/sonata-wo-normal/sonata_wo_normal.pth` | [HuggingFace](https://huggingface.co/ymccccc/sonata-wo-normal) |

---

## Key Config Fields (config.json)

```json
{
  "architectures": ["LlavaQwenForCausalLM"],
  "mm_vision_tower": "google/siglip-so400m-patch14-384",
  "mm_projector_type": "mlp2x_gelu",
  "mm_hidden_size": 1152,
  "hidden_size": 3584,
  "world_position_embedding_type": "sonata_sample_64"
}
```
