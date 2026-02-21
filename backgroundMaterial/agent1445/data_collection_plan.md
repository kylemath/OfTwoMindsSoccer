# Penalty Kick Video Data Collection Plan

**Agent:** 1445 — Data Collection Research Team
**Date:** 2026-02-20
**Goal:** Collect 200-500 real penalty kick video clips with pose estimation for both goalkeeper and striker, replacing the current synthetic demonstration data.

---

## Executive Summary

The research team investigated data sources, scraping tools, and pipeline requirements. The target of 200-500 clips is easily achievable through a **three-tier sourcing strategy**: (1) request existing academic datasets with pre-annotated pose data, (2) extract penalty events from the SoccerNet broadcast archive, and (3) supplement with YouTube compilation scraping. The current pipeline is ~60% production-ready and needs multi-person pose detection and scalability improvements before processing real data.

---

## Data Sources (Ranked by Impact)

### Tier 1: Academic Datasets (Contact Authors — Highest Value)

| Dataset | Clips | Pose Data? | Camera | Access |
|---------|-------|-----------|--------|--------|
| **arXiv:2509.26088** (Ranasinghe & Ranasinghe) | 755 | Yes — frame-level 2D keypoints + ball placement | Mixed broadcast | Email authors |
| **arXiv:2507.12617** (ULPGC, Spain) | 1,010 | No (direction labels only) | Mixed broadcast | Email authors |
| **Pinheiro et al. (Frontiers, 2022)** | 34 | Yes — OpenPose for GK + striker | Broadcast | Email authors |

**Action:** Email all three groups requesting data sharing. The 755-clip dataset alone exceeds our target and already has pose annotations.

### Tier 2: SoccerNet Archive (Free, NDA Required)

- 550+ full broadcast matches at 720p/25fps
- Penalty events labeled in action spotting annotations
- Estimated 50-150 extractable penalties
- **Access:** Sign NDA at soccernet.org, download via their API
- **Advantage:** Consistent broadcast quality, temporal annotations included

### Tier 3: YouTube Compilations (Largest Volume, Most Effort)

| Source Type | Est. Penalties | Camera | Notes |
|-------------|---------------|--------|-------|
| EPL "Every Penalty" season compilations | 70-90/season × 5 seasons = 350-450 | Wide broadcast | Most consistent angle |
| World Cup/Euro shootout compilations | 50-100 per tournament | Mixed | Behind-goal angles available |
| FIFA "All The Angles" playlist | 30-50 | Behind-goal | Best for GK pose estimation |
| UCL penalty compilations | 50-100 | Mixed | Variable quality |

**Estimated total from YouTube:** 500-700+ clips

### Legal Assessment

Fair use analysis is **favorable**: non-commercial academic purpose, factual broadcast material, minimal portion (5-15s clips from hours), no market substitution. Multiple published papers have used broadcast penalty footage for pose estimation. **Key mitigation:** never redistribute raw video — share only derived keypoint data and extracted features.

---

## Pipeline Upgrades Required

### Critical Gap 1: Multi-Person Detection

The current `pose_estimation.py` only extracts `person_id=0`. For real data, both GK and striker must be tracked.

**Solution:** Add YOLOv8 person detection as a pre-processing step:
1. Run YOLOv8 to detect all persons in each frame
2. Classify each detection as GK or striker by spatial position (GK is within 6-yard box region)
3. Crop each person's bounding box
4. Run MediaPipe Pose on each crop independently
5. Remap coordinates back to full-frame space
6. Track person_id across frames using IoU overlap

### Critical Gap 2: Rate-Limited Downloading

Current `collect_clips.py` has no rate limiting, no retries, no cookie handling.

**Solution:** Add to yt-dlp command:
- `--sleep-interval 5 --max-sleep-interval 15` (random delays)
- `--sleep-requests 2` (metadata request delay)
- `--cookies cookies.txt` (for age-restricted content)
- `--retries 3` (automatic retry on failure)

### Critical Gap 3: Automated Scene Detection

For long compilation videos, individual penalties need to be segmented automatically.

**Solution:** PySceneDetect with AdaptiveDetector:
1. Run scene detection on full compilation
2. Filter scenes by duration (3-15 seconds = penalty duration)
3. Merge adjacent short scenes (replay cuts within one penalty)
4. Human reviewer confirms/adjusts timestamps
5. Auto-populate clip_config.yaml

### Critical Gap 4: Camera Angle Handling

Feature extraction assumes goal is centered at x=0.5. Different camera angles need different coordinate transformations.

**Solution:** Per-clip camera angle classification + coordinate normalization:
- Behind-goal: goal width spans full frame, GK centered
- Wide broadcast: goal on one side, perspective correction needed
- Side angle: significant foreshortening, limited utility

---

## Recommended Implementation Plan

### Phase 1: Low-Hanging Fruit (Week 1)

**Goal:** Get real data flowing through the pipeline

1. **Email academic dataset authors** (arXiv:2509.26088, arXiv:2507.12617, Pinheiro et al.)
2. **Sign SoccerNet NDA** and begin downloading matches with penalty labels
3. **Expand clip_config.yaml** to 50-100 penalties from YouTube EPL compilations
4. **Fix collect_clips.py** — add rate limiting, retries, cookie support
5. **Test download** on 10 clips to validate the pipeline end-to-end

### Phase 2: Multi-Person Pose (Week 2)

**Goal:** Extract both GK and striker poses from real footage

1. **Add YOLOv8 person detection** to pose_estimation.py
2. **Implement GK/striker classification** by spatial position
3. **Add per-person tracking** across frames (IoU-based)
4. **Validate** against Pinheiro et al.'s OpenPose results (if dataset obtained)
5. **Update feature_extraction.py** to handle real coordinate variations

### Phase 3: Scale to 200+ Clips (Week 3)

**Goal:** Full dataset with quality assurance

1. **Semi-automated timestamp annotation**: Use PySceneDetect to detect penalty boundaries in compilation videos, human reviewer confirms
2. **Batch processing**: Parallel downloads (5-10 concurrent), batch pose estimation
3. **Quality scoring**: Per-clip confidence metrics, landmark visibility checks
4. **Camera angle classification**: Train simple ResNet-18 on manually labeled subset
5. **Process full dataset** through upgraded pipeline

### Phase 4: Analysis & Figures (Week 4)

**Goal:** Replace synthetic figures with real data

1. **Run feature extraction** on all 200+ clips
2. **Statistical tests** of Predictions P2, P3, P7 with real kinematic data
3. **Replace manuscript figures**: video_head_body_dissociation.pdf, video_feature_distributions.pdf, video_pose_overlay.pdf, video_pipeline_schematic.pdf
4. **Update manuscript text** to report real results instead of simulated demonstration
5. **Add the two missing figures** (pose_overlay and pipeline_schematic) to main.tex

---

## Multi-Agent Implementation Structure

```
Agent 1445-DC (Data Collection Coordinator)
├── M1: Source Acquisition
│   ├── S1: Draft and send academic dataset request emails
│   ├── S2: Sign SoccerNet NDA, download penalty-labeled matches
│   └── S3: Expand clip_config.yaml to 100+ YouTube penalties
│
├── M2: Pipeline Hardening
│   ├── S1: Add rate limiting + retries to collect_clips.py
│   ├── S2: Add YOLOv8 multi-person detection to pose_estimation.py
│   ├── S3: Add PySceneDetect automated segmentation
│   └── S4: Add camera angle handling to feature_extraction.py
│
├── M3: Data Processing & QA
│   ├── S1: Run collection pipeline on first 50 clips
│   ├── S2: Quality scoring + manual review interface
│   ├── S3: Scale to full 200-500 clip dataset
│   └── S4: Aggregate features + statistical analysis
│
└── M4: Manuscript Integration
    ├── S1: Replace synthetic figures with real data figures
    ├── S2: Update video section text with real results
    ├── S3: Embed pose_overlay and pipeline_schematic in main.tex
    └── S4: Recompile PDF and verify
```

---

## Estimated Resources

| Resource | Estimate |
|----------|----------|
| Storage (500 clips @ 720p, 8s each) | ~5-10 GB raw video, ~500 MB pose CSVs |
| Processing time (MediaPipe, CPU) | ~4 hours for 500 clips |
| Processing time (RTMPose, GPU) | ~1.5 hours for 500 clips |
| Manual annotation/QA | ~4-8 hours |
| Total wall-clock time | ~2-3 weeks |
| New dependencies | ultralytics (YOLOv8), scenedetect, easyocr |

---

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Academic datasets not shared | YouTube + SoccerNet provide sufficient volume independently |
| YouTube videos taken down | Download early; target official FIFA/UEFA channels (more stable) |
| Rate limiting / IP ban | Sleep intervals, cookie rotation, limit to 50 downloads/day |
| Multi-person pose fails on broadcast footage | Fall back to manual GK/striker ROI annotation for problematic clips |
| Camera angle too variable | Filter for wide broadcast angle only (most consistent) |
| Insufficient behind-goal footage | Use FIFA "All The Angles" playlist + accept wide-angle as primary |
