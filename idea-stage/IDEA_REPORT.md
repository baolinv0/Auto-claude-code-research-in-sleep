# Idea Discovery Report (v2 — Revised)

**Direction**: Learning Deployable Rendering Policies for Tone Mapping
**Date**: 2026-06-27
**Revision**: v2 — restructured from problem-driven perspective after CVPR-level review feedback
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline

---

## Executive Summary

We propose **Rendering Policy Learning** — a framework that reconceptualizes tone mapping not as pixel-to-pixel regression, but as learning a structured, interpretable **rendering policy**: an ordered set of semantic operations (highlight compression → shadow recovery → local contrast → color protection) that mirrors how professional colorists and ISP engineers actually perform tone mapping. The key scientific question is: **Why should we learn rendering policies rather than direct pixel mappings, and what properties of tone mapping make this decomposition natural and necessary?**

---

## Part I: The Problem (Why This Research Exists)

### The Industry Problem

Modern computational photography faces a fundamental tension:

| Approach | Quality | Control | Editability | Deployability |
|----------|---------|---------|-------------|---------------|
| Hand-tuned ISP pipeline | Good | ★★★★★ | ★★★★★ | ★★★★★ |
| End-to-end learned (CNN/Transformer) | Best | ★☆☆☆☆ | ★☆☆☆☆ | ★★★☆☆ |
| Professional colorist | Best | ★★★★★ | ★★★★★ | ☆☆☆☆☆ |

**The gap**: End-to-end neural networks achieve SOTA quality but produce opaque, uncontrollable, non-editable mappings. Professional colorists achieve both quality AND control, but their process is manual and cannot be deployed on-device.

**One-sentence problem**:

> **How can we learn tone mapping that is simultaneously high-quality, interpretable, controllable, and deployable — matching what a colorist does, not just what a colorist outputs?**

This is NOT "RL at real-time speed." This is: **can we learn the process, not just the result?**

### Why This Problem Matters Now

1. **Mobile HDR capture is ubiquitous** — every flagship phone shoots HDR; tone mapping runs billions of times daily
2. **End-to-end models hit a ceiling** — quality is good but users/engineers cannot adjust, debug, or style-transfer
3. **Professional tools demand structure** — DaVinci Resolve, Filmlight, and camera ISPs all use staged pipelines because creative control requires it
4. **Model personalization requires semantics** — "make my photos warmer" is easy to express as a policy adjustment, impossible as a pixel-level instruction to a black-box network

---

## Part II: The Scientific Question

### Core Question

> **Is tone mapping fundamentally a structured sequential decision, or can it be equivalently captured by a single feedforward mapping?**

### Evidence That Tone Mapping Is Sequential by Nature

**Observation 1: Professional practice decomposes tone mapping into ordered stages**

Every professional colorist and every ISP pipeline performs tone mapping as:

```
1. Highlight compression (protect clipping)
2. Shadow recovery (lift detail from noise floor)  
3. Global contrast shaping (S-curve, midtone emphasis)
4. Local contrast enhancement (micro-contrast, edge-aware)
5. Color protection (saturation preservation, hue stability, skin tones)
```

This is NOT arbitrary. The order matters because:
- Highlight compression must precede contrast shaping (otherwise you clip before you compress)
- Shadow recovery must precede local contrast (otherwise you enhance noise)
- Color protection must come last (otherwise contrast changes shift hues)

**Observation 2: The same scene requires different operation orderings under different conditions**

| Scene Type | Priority Order |
|-----------|---------------|
| Sunset (extreme highlights) | Highlight → Color → Shadow → Contrast |
| Night city (deep shadows) | Shadow → Contrast → Highlight → Color |
| Portrait (skin tones) | Color/Skin → Shadow → Highlight → Contrast |
| High-contrast architecture | Local contrast → Global contrast → Highlight → Shadow |

A single feedforward network must implicitly encode ALL orderings in its weights. A rendering policy can explicitly adapt the operation sequence to the scene.

**Observation 3: End-to-end models fail when the learned ordering doesn't match the input**

Known failure modes of end-to-end tone mapping:
- Purple fringing on highlights (color before highlight compression)
- Noise amplification in shadows (contrast before denoising/recovery)
- Skin tone shifts (global adjustment without skin protection)

These are exactly the failures that occur when a colorist applies operations in the wrong order.

### The Hypothesis

> **Tone mapping is better modeled as a scene-adaptive rendering policy (ordered semantic operations with learned parameters) than as a direct pixel mapping. Learning the policy explicitly yields: (1) better generalization to unseen scenes, (2) interpretability of the mapping, (3) controllability at deployment, and (4) a natural compression target for efficient distillation.**

---

## Part III: What Is a Rendering Policy?

### Definition

A **Rendering Policy** π is a function that, given an HDR image x and scene context c, outputs:

```
π(x, c) = (o₁, θ₁) → (o₂, θ₂) → ... → (oₖ, θₖ)
```

Where:
- oᵢ ∈ {highlight_compress, shadow_lift, contrast_shape, local_enhance, color_protect, ...} is a **semantic operation**
- θᵢ are the **parameters** of that operation (e.g., knee point, gain, gamma, radius)
- The sequence is **ordered** and **scene-adaptive**

### Why Policy > Direct Pixel Prediction

| Property | Direct Pixel Prediction | Rendering Policy |
|----------|------------------------|-----------------|
| Interpretability | None — black box | Each operation is named and parameterized |
| Controllability | Retrain entire model | Adjust θᵢ or reorder operations |
| Editability | Impossible | Insert/remove/swap operations |
| Generalization | Memorizes training distribution | Compositional — novel combinations of known operations |
| Distillation | Distill features (opaque) | Distill policy (structured, semantic) |
| Debugging | Cannot trace failures | Trace to specific operation |

### Why Policy > Feature Distillation

Traditional knowledge distillation compresses features — intermediate representations that have no semantic meaning. **Policy distillation compresses decisions** — which operation to apply, with what parameters, in what order. This is:

1. **More compressible** — a policy is lower-dimensional than a feature map (K operations × D parameters vs. H×W×C feature tensor)
2. **More transferable** — a policy learned on one resolution/sensor generalizes; features are resolution-specific
3. **More interpretable** — a distilled policy can be inspected; a distilled feature cannot

---

## Part IV: Ranked Ideas

### 🏆 Idea 1: Learning Scene-Adaptive Rendering Policies for Tone Mapping — RECOMMENDED

**Problem**: End-to-end tone mapping lacks interpretability, controllability, and graceful generalization. Professional tone mapping is structured and sequential, but this structure is lost in neural approaches.

**Scientific hypothesis**: Explicitly learning a rendering policy (operation sequence + parameters) that mimics professional structure yields better quality, interpretability, and generalization than end-to-end pixel prediction.

**Method**:
1. **Define an operation vocabulary**: K=8 differentiable operations (highlight compression, shadow lift, S-curve contrast, local contrast, saturation adjust, hue protect, white balance shift, sharpening)
2. **Policy network**: Lightweight CNN/ViT that takes HDR input → predicts (a) operation ordering (permutation over K) and (b) per-operation parameters θᵢ
3. **Differentiable execution**: Apply predicted operations sequentially (each is a differentiable image filter with learned parameters)
4. **Training**: Supervised on professional colorist edits (film grading datasets) + perceptual loss
5. **Scene-adaptive routing**: Policy network conditions on global scene statistics to select operation priority

**Why NOT RL?**: RL is ONE possible training method (treat each operation as an action). But the core idea does not require RL. Supervised learning from professional edits, or distillation from a large rendering model, can also produce the policy. RL becomes useful only IF we need to optimize for non-differentiable rewards (e.g., user preference, display-specific metrics).

**Novelty**: Not "first RL for TMO" (that exists). Rather: **first framework that explicitly learns the structured operation sequence that constitutes professional tone mapping, and shows this structure improves over unstructured prediction.**

**Expected results**:
- Quality: On-par or better than end-to-end (because structure acts as inductive bias)
- Interpretability: Every output can be explained as "highlight compressed by X, shadow lifted by Y..."
- Control: Users can override any θᵢ without retraining
- Generalization: Better on out-of-distribution scenes (novel operation combos)

---

### Idea 2: Rendering Policy Distillation — From Large Models to Deployable Agents — BACKUP

**Problem**: Large vision models (e.g., diffusion-based image enhancers, multi-billion parameter ISP models) produce excellent tone mapping but cannot deploy on mobile. Traditional distillation produces another black box.

**Hypothesis**: If we distill the **policy** (what operations the large model is implicitly performing and in what order) rather than just matching its pixel output, the student is smaller, more interpretable, AND more robust.

**Method**:
1. **Teacher**: Any large tone-mapping model (could be diffusion, could be large CNN, could be RL-trained — teacher source is agnostic)
2. **Policy extraction**: Analyze teacher's behavior via probing — which layers correspond to highlight/shadow/contrast/color operations? Use gradient attribution + operation-specific probes
3. **Structured distillation**: Student learns to reproduce teacher's implicit policy (operation order + parameters), not just its pixels
4. **Multi-teacher**: Distill from multiple teachers (ACES, Filmic, cinematic colorist) into one policy-conditioned student

**Novelty**: Distill structure/decisions, not just pixels/features. Teacher-agnostic — works regardless of teacher architecture.

**Why this is better than standard KD**: Standard KD student matches output but doesn't know WHY. Policy-distilled student knows "I'm compressing highlights because the scene has 4+ stops of overexposure." This transfers to novel scenes.

---

### Idea 3: Human-in-the-Loop Rendering Policy Learning — BACKUP

**Problem**: Professional colorists embody years of perceptual expertise, but their knowledge is implicit and non-transferable.

**Hypothesis**: If we capture colorist edits as structured policy demonstrations (not just input-output pairs), we can learn policies that generalize the colorist's expertise to new scenes.

**Method**:
1. Record professional colorist sessions with full edit history (operation type, parameters, order)
2. Learn a policy from demonstrations (behavioral cloning on structured action space)
3. Generalize via scene conditioning — same colorist's "intent" applied to novel content
4. Enable style transfer as policy transfer (swap policies between colorists)

**Novelty**: Learning from structured demonstrations rather than input-output supervision.

---

### Idea 4: Compositional Rendering Policies with Guaranteed Properties — BACKUP

**Problem**: End-to-end models can produce physically impossible outputs (negative values, hue inversions, banding). A structured policy with differentiable operations can guarantee physical validity by construction.

**Hypothesis**: Constraining tone mapping to a composition of physically-valid operations (monotonic curves, gamut-preserving transforms) yields better perceptual quality AND eliminates artifact classes.

**Method**:
1. Each operation oᵢ is constrained (e.g., highlight compression must be monotonic; color adjust must be gamut-bounded)
2. Composition of valid operations is valid (closure property)
3. Policy network only needs to learn parameters within valid ranges
4. Eliminates: clipping artifacts, hue shifts, banding, impossible colors

---

## Part V: Eliminated / Deprioritized Directions

| Direction | Kill Reason |
|-----------|-------------|
| "RL + Policy Distillation" as the paper title | Method-driven; doesn't answer WHY RL is needed |
| PPO teacher as core contribution | RL is an implementation choice, not the scientific contribution |
| "First X + Y + Z" framing | Novelty-by-combination is not compelling |
| "Compress RL to mobile" as problem statement | "RL-quality" is not a recognized problem — CNN/Transformer quality is already excellent |
| Pure speed/efficiency story | Industrial value but no scientific question |

---

## Part VI: Answering the Three Killer Questions

### Q1: Why not just use CNN/Transformer?

**Answer**: CNN/Transformer CAN produce high-quality output. The problem is not quality — it's that their mapping is opaque, uncontrollable, and non-compositional. A rendering policy gives structure to the mapping. The backbone (CNN, Transformer, or other) is what IMPLEMENTS the policy network — they're complementary, not competing.

### Q2: Why sequential rather than direct prediction?

**Answer**: Because tone mapping IS sequential in professional practice (and for good physical reasons — you must handle highlight clipping before you can shape contrast). The sequential structure is an inductive bias matching the true data-generating process. This is analogous to how attention mechanisms succeed because language IS compositional, or how convolutions succeed because images ARE locally correlated.

### Q3: Why is policy easier to distill than features?

**Answer**: A policy is a low-dimensional, structured representation (K operations × D parameters ≈ 50-100 numbers). A feature map is H×W×C ≈ millions of numbers. Distilling 100 interpretable decisions is more sample-efficient, more robust, and more transferable than distilling millions of opaque activations.

---

## Part VII: Experiment Plan

### Must-Validate-First Experiments (Scientific Claims)

| # | Claim to Test | Experiment | Success Metric |
|---|--------------|-----------|----------------|
| 1 | Sequential structure helps | Policy model vs. same-capacity end-to-end model | PSNR/LPIPS on OOD test set |
| 2 | Operation order is scene-adaptive | Visualize learned orderings across scene types | Correlation with professional practice |
| 3 | Policy is more controllable | User study: adjust tone mapping intent (warmer/cooler/more contrast) | Task completion rate, preference |
| 4 | Policy is more compressible | Policy distillation vs. feature distillation at same compression ratio | Quality retention (%) |
| 5 | Policies generalize better | Train on indoor, test on outdoor (zero-shot transfer) | Quality drop vs. end-to-end |

### Ablations

| Ablation | Tests What |
|----------|-----------|
| Fixed operation order vs. learned order | Is adaptivity important? |
| K=4 vs. K=8 vs. K=16 operations | How fine-grained should the vocabulary be? |
| Supervised vs. RL training of policy | When (if ever) does RL help? |
| Single teacher vs. multi-teacher distillation | Does diversity improve student? |

### Datasets
- HDR+ (Google, 3,640 HDR bursts with expert tone mapping)
- MIT-Adobe FiveK (5,000 images with expert retouches by 5 photographers)
- Fairchild HDR dataset (105 HDR scenes)
- Film grading datasets (professional colorist edit histories, if accessible)

---

## Part VIII: Positioning for Venue

### Why This Is a CVPR Paper (Not a Systems Paper)

1. **Scientific question**: "Is tone mapping fundamentally sequential?" — testable, falsifiable
2. **Hypothesis with mechanism**: The operation-ordering structure mirrors physical signal flow and professional practice
3. **Not method-first**: The method (policy network + differentiable operations) follows from the hypothesis
4. **Generalizable insight**: "Learning the process, not just the result" applies beyond tone mapping (denoising, super-resolution, any ISP stage)
5. **Clear experiments test the hypothesis**: Not just "our method is 0.5dB better" but "structure helps because X"

### Venue Fit
- **CVPR/ICCV**: Computational photography track — problem is image quality + structure
- **ECCV**: If emphasizing the representation learning aspect
- **SIGGRAPH**: If emphasizing creative control and professional workflow

---

## Next Steps

- [ ] Prototype: Implement differentiable operation vocabulary (K=8 operations as differentiable PyTorch modules)
- [ ] Validate hypothesis: Train policy model vs. same-capacity end-to-end on MIT-Adobe FiveK
- [ ] If hypothesis confirmed → full paper with controllability experiments + user study
- [ ] If hypothesis fails → investigate whether the operation vocabulary is too coarse, or whether the ordering hypothesis is wrong

---

## Appendix: What We Keep From v1

- The concept of "Rendering Policy" — this is the single most valuable insight
- The distillation framework (Idea 2) — but now as a CONSEQUENCE of having a structured policy, not as the core contribution
- The real-time deployment story — but as an engineering benefit, not the scientific question
- Multi-teacher distillation (from ACES, Filmic, etc.) — practical and novel

## Appendix: Reviewer Feedback Integration (v1 → v2)

| v1 Problem | v2 Fix |
|------------|--------|
| Method-driven (RL + Distillation) | Problem-driven (Why learn policies?) |
| "Why RL?" unanswered | RL deprioritized; policy learning is teacher-agnostic |
| "First X+Y+Z" novelty | Novelty is the scientific hypothesis about tone mapping structure |
| No mechanism for why sequential | Professional practice + physical signal flow as evidence |
| Problem = "RL at real-time speed" | Problem = "Interpretable, controllable, structured tone mapping" |
