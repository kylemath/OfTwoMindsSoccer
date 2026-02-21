# Task Decomposition — Agent 1445

## Original Task
Revise the manuscript "Of Two Minds at the Penalty Spot" for submission as a theory and review paper. Three streams: (1) restructure the manuscript addressing reviewer feedback, (2) harden the analysis pipeline with robustness checks and a shootout replication, (3) build and run a video pose-estimation pilot analysis. Then assemble into a cohesive paper.

## Atomic Subtasks

1. M2-S1: Add time-decay, kick-n-2 ghost, cluster-robust SEs, 3-level disaggregation to prediction2_sequential.py
2. M2-S2: Find public shootout dataset, create prediction2_shootout.py
3. M2-S3: Run full pipeline, regenerate generated_stats.tex and figures
4. M3-S1: Build video collection script (yt-dlp + ffmpeg)
5. M3-S2: Build MediaPipe pose estimation + feature extraction pipeline
6. M3-S3: Run pilot on collected clips, produce figures
7. M3-S4: Draft "Computational Ethology" manuscript section
8. M1-S1: Fix abstract + add Methods subsection (depends on M2)
9. M1-S2: Restructure predictions into tiers + mapping table
10. M1-S3: Expand integration section + reframe self-assessment
11. M1-S4: Update references.bib
12. M1-S5: Integrate video section into main.tex (depends on M3)
13. Assembly: verify PDF compiles, write synthesis report

## Dependency Graph

- M2 (subtasks 1-3) and M3 (subtasks 4-7): INDEPENDENT, run in parallel
- M1-S2, M1-S3, M1-S4 (subtasks 9-11): INDEPENDENT of M2/M3, can start immediately
- M1-S1 (subtask 8): depends on M2-S3 output (updated stats)
- M1-S5 (subtask 12): depends on M3-S4 output (video section draft)
- Assembly (subtask 13): depends on all above

## Stream Allocation

| Manager | Stream Type | Subtasks     | Dependencies      | Async? |
|---------|-------------|--------------|-------------------|--------|
| M2      | parallel    | S1, S2, S3   | None              | Yes    |
| M3      | parallel    | S1, S2, S3, S4 | None           | Yes    |
| M1      | mixed       | S2,S3,S4 now; S1,S5 after M2/M3 | M2, M3 | Partial |

## Complexity Estimate
Large task (13 subtasks). M2 and M3 are compute-heavy. M1 is edit-heavy with cross-stream dependencies. Main risk: video pipeline may fail if clips are unavailable or MediaPipe struggles with broadcast footage quality.
