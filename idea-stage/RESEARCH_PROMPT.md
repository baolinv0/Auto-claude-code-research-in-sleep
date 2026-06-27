# Research Prompt: Learning Scene-Adaptive Rendering Policies for Tone Mapping

> 从 IDEA_REPORT.md (v2) 中的 Idea 1 转化而来的可执行研究 Prompt

---

## Prompt

You are a computer vision researcher working on computational photography. Your task is to design and implement a novel framework called **Rendering Policy Learning** for tone mapping.

### Research Context

**Problem**: End-to-end neural tone mapping (CNN/Transformer) achieves high quality but produces opaque, uncontrollable, non-editable mappings. Professional colorists work in structured, sequential stages — but this process cannot be automated or deployed on-device.

**Core Research Question**: Is tone mapping fundamentally a structured sequential decision process? Can we explicitly learn a scene-adaptive rendering policy (ordered semantic operations with learned parameters) that outperforms unstructured direct pixel prediction in quality, interpretability, controllability, and generalization?

**Hypothesis**: Explicitly learning a rendering policy — an ordered sequence of differentiable semantic operations conditioned on scene context — yields better generalization, interpretability, and controllability than end-to-end pixel prediction, because tone mapping IS inherently sequential (highlight compression must precede contrast shaping; shadow recovery must precede local enhancement; color protection must come last).

### Task Specification

Design a system with the following components:

1. **Operation Vocabulary (K=8 differentiable operations)**:
   - Highlight compression (monotonic curve, knee-point parameterized)
   - Shadow lift (toe region curve)
   - S-curve global contrast shaping
   - Local contrast enhancement (edge-aware, radius-parameterized)
   - Saturation adjustment (gamut-bounded)
   - Hue protection (skin-tone aware)
   - White balance shift
   - Sharpening (unsharp mask, differentiable)

2. **Policy Network**:
   - Input: HDR image + global scene statistics (histogram, dynamic range, mean luminance)
   - Output: (a) operation ordering — a permutation over K operations; (b) per-operation parameters θᵢ
   - Architecture: Lightweight CNN or ViT backbone → dual-head (ordering head + parameter head)

3. **Differentiable Execution Engine**:
   - Apply predicted operations sequentially, each as a differentiable image filter
   - Support gradient flow through the entire operation chain
   - Handle permutation learning (e.g., Sinkhorn operator or Gumbel-Softmax for differentiable ordering)

4. **Training Strategy**:
   - Supervised on professional edits (MIT-Adobe FiveK, HDR+ dataset)
   - Loss: L1 + perceptual (LPIPS) + optional color fidelity term
   - Scene-adaptive routing: condition on global statistics to select operation priority

### Evaluation Plan

| Experiment | Claim Tested | Metric | Success Criterion |
|-----------|-------------|--------|-------------------|
| Policy vs. same-capacity end-to-end | Structure helps | PSNR/SSIM/LPIPS on OOD test | ≥ 0.5dB PSNR or ≥ 0.02 LPIPS improvement |
| Visualize learned orderings | Order is scene-adaptive | Correlation with professional practice | Orderings match colorist heuristics per scene type |
| User controllability study | Policy is controllable | Task completion rate | Users can adjust intent without retraining |
| Policy distillation vs. feature distillation | Policy is more compressible | Quality retention at same compression | ≥ 5% better retention |
| Cross-domain transfer (train indoor, test outdoor) | Policy generalizes | Quality drop | < 50% drop vs. end-to-end baseline |

### Key Ablations

- Fixed order vs. learned order (is adaptivity needed?)
- K=4 vs. K=8 vs. K=16 (operation granularity)
- Supervised vs. RL training (when does RL help?)
- Single-teacher vs. multi-teacher distillation

### Constraints

- **Compute**: 8× RTX 3090, ~100 GPU-hours total
- **Datasets**: MIT-Adobe FiveK (5,000 images), HDR+ (3,640 bursts), Fairchild HDR (105 scenes)
- **Target venue**: CVPR / ICCV (computational photography track)
- **Timeline**: 3 months to submission
- **Deployability**: Final model must run on mobile (< 50ms per 1080p frame)

### What NOT To Do

- Do NOT frame this as "RL for tone mapping" — RL is a training choice, not the contribution
- Do NOT rely on "first X+Y+Z" novelty — the novelty is the scientific hypothesis
- Do NOT pursue a pure speed/efficiency story — the contribution is structural insight
- Do NOT make the paper method-first — lead with the problem and hypothesis

### Expected Deliverables

1. PyTorch implementation of 8 differentiable tone mapping operations
2. Policy network with scene-adaptive ordering
3. Training pipeline on MIT-Adobe FiveK
4. Comparison against end-to-end baselines (CSRNet, HDRNet, 3D-LUT)
5. Ablation on operation count and ordering strategy
6. Visualization of learned policies per scene type
7. Paper draft following CVPR format

---

## How to Use This Prompt

This prompt can be used with:

- `/experiment-bridge` — to implement the experiment code
- `/paper-plan` — to structure the paper
- `/run-experiment` — to execute training runs
- As context for any coding agent implementing the Rendering Policy framework

### Quick Start Command

```
/experiment-bridge "idea-stage/RESEARCH_PROMPT.md"
```

Or for the full pipeline:

```
/research-refine-pipeline "Learning Scene-Adaptive Rendering Policies for Tone Mapping"
```
