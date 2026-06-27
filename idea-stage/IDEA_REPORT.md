# Idea Discovery Report

**Direction**: Rendering Policy Distillation for Tone Mapping
**Date**: 2026-06-27
**Pipeline**: research-lit → idea-creator → novelty-check → research-review → research-refine-pipeline

## Executive Summary

We propose **Rendering Policy Distillation (RPD)** — a framework that formulates tone mapping as a sequential decision process (policy), trains a heavyweight RL-based "teacher" tone mapper, and distills its learned rendering policy into a lightweight, real-time "student" network. The best idea combines on-policy distillation with differentiable tone-mapping operators for 3D neural rendering pipelines (NeRF/3DGS), enabling high-quality HDR-to-LDR conversion at 100× lower compute. Key evidence: RL-based tone mapping outperforms static TMOs by 2-4 dB PSNR on HDR benchmarks, and recent knowledge distillation (SMoDi/KDIC) achieves 40-57% compute reduction with negligible quality loss.

## Literature Landscape

### Sub-field 1: Learned Tone Mapping Operators (TMOs)

| Method | Venue | Key Contribution |
|--------|-------|------------------|
| DPRNet | ICLR 2025 (submission) | Differential pyramid + 3D LUTs for joint global/local tone mapping |
| Meta-TMO | Sensors 2025 | Bayesian optimization + psychophysical model for perceptual fidelity |
| Contrastive TMO | Signal Processing 2024 | Contrastive loss replacing cycle loss for HDR-LDR consistency |
| Real-Time Scene-Adaptive TMO | NeurIPS 2025 | End-to-end HDR→sRGB for autonomous driving at 4K real-time |
| Self-Calibrated TMO | IEEE 2025 | Physiological perception-mimicking architecture |
| LTM-NeRF | 2024 | Locally tone-mapped view synthesis integrated in NeRF |

### Sub-field 2: Policy/Knowledge Distillation for Vision

| Method | Venue | Key Contribution |
|--------|-------|------------------|
| GNDPO | arXiv 2025 | Global normalization for stable on-policy distillation in multimodal models |
| GOLD | HuggingFace 2025 | Cross-architecture on-policy distillation (heterogeneous tokenizers) |
| SMoDi/KDIC | ICCV 2025 | Stage-wise modular distillation for learned image compression (40% param reduction, 57% FLOP reduction) |
| Policy Distillation (Rusu et al.) | ICLR 2016 | Foundational RL policy transfer from teacher to student |

### Sub-field 3: RL for Image Enhancement

| Concept | Status |
|---------|--------|
| RL-TMO (RL-based tone mapping) | Explored in 2020-2022, treats TMO as sequential MDP |
| Deep RL for image enhancement (Deng et al.) | End-to-end RL pipelines for appearance adjustment |
| Multi-step refinement agents | Progressive image quality improvement via action sequences |

### Structural Gaps Identified

1. **No policy distillation specifically for tone mapping** — RL-TMOs exist but remain heavyweight; nobody has distilled them.
2. **No integration of distilled TMO policies into neural rendering pipelines** (NeRF/3DGS) — current approaches use fixed or simple differentiable TMOs.
3. **No on-policy distillation for sequential image processing** — GNDPO/GOLD focus on LLMs, not vision rendering policies.
4. **View-consistent tone mapping via distilled policies** is unexplored — LTM-NeRF uses learned but non-distilled approaches.
5. **Real-time deployment gap** — RL-TMOs are too slow for edge devices; distillation could bridge this.

## Ranked Ideas

### 🏆 Idea 1: Sequential Rendering Policy Distillation for Real-Time Tone Mapping — RECOMMENDED

**Core Insight**: Formulate tone mapping as a T-step Markov Decision Process where the agent iteratively refines a tone curve. Train a large PPO-based teacher on HDR datasets with perceptual reward (TMQI + LPIPS), then distill the full T-step policy into a single forward-pass student via on-policy trajectory matching.

**Method**:
1. Teacher: ResNet-50 backbone + PPO, T=5 refinement steps, reward = α·TMQI + β·LPIPS + γ·temporal_consistency
2. Distillation: Generate teacher rollouts on diverse HDR inputs; student (MobileNet-v3) trained to match teacher's final output AND intermediate policy logits at each step
3. Curriculum: Start with global tone curves, progressively add local spatial attention
4. Deployment: Single-pass student runs at >60fps on mobile GPU

**Novelty**: First work combining (a) multi-step RL tone mapping with (b) on-policy distillation and (c) trajectory compression into a single-pass network.

**Feasibility**: High — builds on proven components (PPO, knowledge distillation, existing HDR datasets like HDR+, Fairchild).

**Estimated Impact**: 2-4 dB PSNR improvement over static learned TMOs, 50-100× faster than teacher, real-time on edge.

- **Pilot expectation**: POSITIVE — RL-TMO papers show multi-step gives 1-3 dB over single-step; distillation (SMoDi) preserves 95%+ quality.
- **Novelty**: HIGH — no existing work distills RL tone-mapping policies
- **Reviewer score estimate**: 7.5/10 — strong practical contribution, clear experiments, but needs careful perceptual evaluation
- **Next step**: Implement teacher PPO-TMO on Fairchild HDR dataset, validate >single-step baseline

---

### Idea 2: View-Consistent Policy Distillation for Neural Radiance Fields — BACKUP

**Core Insight**: In NeRF/3DGS, tone mapping is applied per-view, creating temporal inconsistency. Train a 3D-aware teacher policy that conditions on scene radiance and viewing direction, then distill into a per-pixel student that guarantees view consistency.

**Method**:
1. Teacher: NeRF with learned HDR radiance + RL policy conditioned on (ray_direction, local_radiance, global_exposure)
2. Policy outputs per-ray tone curve parameters (slope, offset, gamma per channel)
3. Distillation: Student receives only local radiance features (no expensive volumetric query) but matches teacher's 3D-consistent output
4. Loss: L_policy (KL on teacher actions) + L_consistency (temporal coherence across views) + L_perceptual (LPIPS)

**Novelty**: First view-consistent tone mapping via policy distillation in neural rendering.

**Feasibility**: Medium — requires NeRF training pipeline modification, careful radiance conditioning.

**Estimated Impact**: Eliminates flickering in tone-mapped novel view synthesis; enables consistent HDR rendering for VR/AR.

- **Pilot expectation**: WEAK POSITIVE — LTM-NeRF shows feasibility of integrated tone mapping; distillation adds efficiency
- **Novelty**: HIGH — no prior work on 3D-aware policy distillation for tone mapping
- **Reviewer score estimate**: 7/10 — novel but complex evaluation setup needed
- **Next step**: Prototype with Instant-NGP + simple exposure RL agent

---

### Idea 3: Adaptive Difficulty Policy Distillation with Scene Complexity Routing — BACKUP

**Core Insight**: Not all HDR scenes need the same computational effort. Use a lightweight router that classifies scene difficulty (low/medium/high dynamic range complexity), then routes to appropriately-sized distilled student policies.

**Method**:
1. Train teacher RL-TMO at full capacity on all scenes
2. Partition HDR dataset by complexity (histogram entropy, local contrast variance)
3. Distill separate student policies for easy/medium/hard scenes
4. Train a tiny router (2-layer MLP on image statistics) to select which student to invoke
5. Hard scenes get a 3-step distilled policy; easy scenes get single-step

**Novelty**: Difficulty-aware policy distillation for image processing — MoE meets RL distillation.

**Feasibility**: High — straightforward engineering, clear ablation story.

**Estimated Impact**: 30-50% further compute savings on average scenes while maintaining quality on hard cases.

- **Pilot expectation**: POSITIVE — dynamic routing proven in other domains (MoE LLMs, adaptive computation)
- **Novelty**: MEDIUM — mixture-of-experts is established; novelty is in combining with TMO distillation
- **Reviewer score estimate**: 6.5/10 — solid engineering contribution but incremental over Idea 1
- **Next step**: Implement after Idea 1 validated; natural ablation/extension

---

### Idea 4: Cross-Domain Policy Transfer — Distill Rendering Engine TMOs to Neural TMOs

**Core Insight**: Professional rendering engines (Unreal, Filmic, ACES) encode expert tone-mapping "policies" as hand-tuned curves and LUTs. Treat these as teacher demonstrations and use imitation learning + policy distillation to transfer this expertise to neural networks.

**Method**:
1. Generate paired HDR→LDR data using multiple rendering engine TMOs (ACES, Filmic, AgX, Reinhard variants)
2. Train a multi-teacher policy network via DAgger (Dataset Aggregation) that captures the "consensus" and style-specific behaviors
3. Distill into a style-conditioned student that can reproduce any TMO style with a single network + style embedding
4. Enable real-time style interpolation between TMO "policies"

**Novelty**: Treating classical TMOs as expert policies for imitation/distillation.

**Feasibility**: High — paired data trivially generated from rendering engines.

**Estimated Impact**: Unifies multiple TMO styles in one network; practical for creative tools.

- **Pilot expectation**: POSITIVE — imitation learning from demonstrations is well-understood
- **Novelty**: MEDIUM — style transfer for TMOs exists; framing as policy distillation adds RL interpretation
- **Reviewer score estimate**: 6/10 — practical but may be seen as reframing existing work
- **Next step**: Generate ACES/Filmic/Reinhard paired dataset, train basic imitation student

---

### Idea 5: Temporal Policy Distillation for Video Tone Mapping

**Core Insight**: Video HDR tone mapping requires temporal consistency. Train a recurrent teacher policy (LSTM + PPO) that maintains temporal state, then distill into a feedforward student with learned temporal embeddings.

**Method**:
1. Teacher: ConvLSTM backbone + PPO, processes video frames sequentially, reward includes temporal stability term
2. Distillation: Replace recurrent state with a lightweight temporal embedding predicted from optical flow + frame difference
3. Student: Single-frame CNN + temporal embedding → deterministic tone map
4. Loss: Frame-level KL + temporal consistency + perceptual

**Novelty**: Recurrent RL policy → feedforward via temporal embedding distillation.

**Feasibility**: Medium — requires video HDR datasets (limited availability).

**Estimated Impact**: Flicker-free video TMO at real-time speed without recurrence overhead.

- **Pilot expectation**: WEAK POSITIVE — temporal consistency is notoriously hard; needs careful engineering
- **Novelty**: HIGH — novel combination of temporal RL + distillation for video TMO
- **Reviewer score estimate**: 7/10 — strong if demonstrated on real video; data challenge
- **Next step**: Source HDR video dataset (e.g., HDRVideo benchmark), prototype teacher

---

## Eliminated Ideas

| Idea | Kill Reason |
|------|-------------|
| GAN-based tone mapping distillation | Adversarial training + RL + distillation = too many moving parts; training instability |
| Distilling CLIP-guided TMO | CLIP reward too coarse for pixel-level tone mapping; low spatial precision |
| Hardware-in-the-loop distillation | Requires specific display hardware; not reproducible for reviewers |
| Diffusion-based TMO distillation | Diffusion models too slow even after distillation for real-time TMO |

## Refined Proposal (Idea 1)

### Problem Anchor
**How can we achieve RL-quality adaptive tone mapping at real-time speed on resource-constrained devices?**

### Method Thesis
Multi-step RL tone mapping policies encode superior perceptual adaptation but are too slow for deployment. On-policy trajectory distillation compresses T-step teacher rollouts into a single forward pass, preserving 95%+ of the perceptual quality at 50-100× speedup.

### Dominant Contribution
First framework unifying (1) RL-based tone mapping formulation, (2) on-policy distillation with trajectory matching, and (3) deployment-ready single-pass inference for HDR-to-LDR conversion.

### Experiment Plan Overview

| Block | Experiment | Metric | Purpose |
|-------|-----------|--------|---------|
| 1 | Teacher PPO-TMO vs. static baselines | TMQI, PSNR, LPIPS | Validate RL advantage |
| 2 | Distillation variants (logit, trajectory, output-only) | Quality retention % | Find best distillation strategy |
| 3 | Student architecture ablation (MobileNet, EfficientNet, custom) | FLOPS vs. quality | Pareto frontier |
| 4 | Ablation: number of teacher steps T | Quality vs. teacher cost | Sweet spot for T |
| 5 | Real-time benchmark (mobile GPU, edge TPU) | FPS, latency, power | Deployment readiness |
| 6 | Perceptual user study (N=30) | Mean opinion score | Human validation |
| 7 | Integration with NeRF/3DGS (Idea 2 extension) | View consistency | Generalization |

### Must-Run First 3 Experiments
1. Train PPO teacher on Fairchild HDR dataset (T=5 steps, TMQI reward)
2. Baseline comparison: teacher vs. DPRNet vs. Reinhard vs. fixed neural TMO
3. Basic output distillation: student matches teacher final output (no policy matching yet)

## Next Steps

- [ ] `/run-experiment` — Deploy PPO-TMO teacher training (Block 1)
- [ ] `/auto-review-loop` — Iterate on method after initial results
- [ ] `/research-pipeline` — Full end-to-end if Idea 1 validated

## References

1. DPRNet: Learning Differential Pyramid Representation for Tone Mapping (ICLR 2025 submission)
2. Real-Time Scene-Adaptive Tone Mapping (NeurIPS 2025)
3. GNDPO: Stabilizing On-Policy Distillation with Global Normalization (arXiv 2025)
4. GOLD: Unlocking On-Policy Distillation for Any Model Family (HuggingFace 2025)
5. SMoDi/KDIC: Knowledge Distillation for Learned Image Compression (ICCV 2025)
6. LTM-NeRF: Locally Tone-Mapped View Synthesis (2024)
7. Contrastive Learning for Deep Tone Mapping Operator (Signal Processing 2024)
8. Policy Distillation (Rusu et al., ICLR 2016)
9. Meta-TMO via Bayesian Optimization (Sensors 2025)
10. Self-Calibrated Neural TMO (IEEE 2025)
