# Penalty Kick Video Data Sources: Research Report

**Date:** 2026-02-20  
**Purpose:** Identify publicly available penalty kick video data sources (target: 200–500 clips) suitable for pose estimation of both goalkeeper and striker.

---

## 1. Academic Penalty Kick Video Datasets

### 1.1 Multi-Modal Penalty Kick Dataset (755 clips)
| Field | Details |
|-------|---------|
| **Source** | Ranasinghe & Ranasinghe (arXiv:2509.26088) |
| **URL** | https://arxiv.org/abs/2509.26088 |
| **N penalties** | 755 events |
| **Camera angle** | Broadcast (various, from match footage) |
| **Annotations** | Frame-level object detection, shooter keypoints (2D), ball placement |
| **Format** | Video clips + CSV annotations |
| **Access** | Not publicly hosted; contact authors for access |
| **Cost** | Free (academic) |
| **Restrictions** | Likely NDA; derived from broadcast footage |
| **Notes** | Achieved 89% accuracy predicting kick direction. Includes pose-guided attention framework. Best existing dataset for pose-based penalty analysis. |

### 1.2 HAR Penalty Kick Dataset (1,010 clips)
| Field | Details |
|-------|---------|
| **Source** | Universidad de Las Palmas de Gran Canaria (arXiv:2507.12617) |
| **URL** | https://arxiv.org/abs/2507.12617 |
| **N penalties** | 1,010 clips |
| **Camera angle** | Broadcast (official match footage) |
| **Annotations** | Shot direction labels (left/right), manually annotated |
| **Format** | Video clips |
| **Access** | Not publicly hosted; contact authors (SIANI, ULPGC) |
| **Cost** | Free (academic) |
| **Restrictions** | Likely requires agreement; broadcast-derived |
| **Notes** | Largest known annotated penalty clip dataset. Tested 22 backbone models across 7 architectures (MViTv2, SlowFast, I3D, etc.). Up to 63.9% accuracy. |

### 1.3 OpenPose Penalty Analysis Dataset (34 clips)
| Field | Details |
|-------|---------|
| **Source** | Pinheiro et al. (Frontiers in Sports, 2022, DOI: 10.3389/fspor.2022.818556) |
| **URL** | https://www.frontiersin.org/articles/10.3389/fspor.2022.818556 |
| **N penalties** | 34 videos |
| **Camera angle** | Broadcast (European leagues, 2017–2020) |
| **Annotations** | OpenPose body orientation angles (shoulder, hip, nonkicking foot for striker; foot/anticipation for GK) |
| **Format** | Video clips |
| **Access** | Not publicly released; contact authors |
| **Cost** | Free |
| **Restrictions** | Small sample; broadcast-derived |
| **Notes** | **Directly relevant**: analyzed BOTH goalkeeper and striker body pose. OpenPose confidence 0.80 ± 0.14. GK anticipation → 97% strategy prediction accuracy. |

### 1.4 IEEE DataPort – Ball Impact Coordinates
| Field | Details |
|-------|---------|
| **Source** | Patel et al. (IEEE DataPort, DOI: 10.21227/7znr-aj82) |
| **URL** | https://ieee-dataport.org/documents/football-penalty-shootout-ball-impact-coordinates-dataset |
| **N penalties** | Unknown (coordinates only) |
| **Camera angle** | N/A (spatial data, not video) |
| **Format** | CSV (2D ball impact coordinates on goal) |
| **Access** | Public download |
| **Cost** | Free |
| **Restrictions** | Data only, no video |
| **Notes** | Complementary to video datasets—useful for outcome labeling but no video content. |

### 1.5 Goalkeeper Dive Timing Dataset
| Field | Details |
|-------|---------|
| **Source** | Reddy (UT Tyler, Masters Thesis, ScholarWorks) |
| **URL** | https://scholarworks.uttyler.edu/hkdept_grad/32/ |
| **N penalties** | Unknown |
| **Camera angle** | Broadcast |
| **Annotations** | Torso and leg kinematics, dive initiation timing |
| **Format** | Thesis + supplementary data |
| **Access** | Open access thesis; data availability unclear |
| **Cost** | Free |
| **Notes** | GK centroid vertical position → strongest dive commitment predictor (R² = .998). |

---

## 2. YouTube Compilation Channels

### 2.1 Official FIFA/UEFA Channels

| Channel / Playlist | URL | Est. Penalties | Camera | Notes |
|---|---|---|---|---|
| **FIFA "All The Angles" Playlist** | https://www.youtube.com/playlist?list=PLCGIzmTE4d0j5b6jLy7YD-oTHyV8-knYn | 50–100 goals (subset are penalties) | **Multiple angles incl. behind-goal** | Best for behind-goal views; multi-angle replays |
| **FIFA "Anatomy of a Goal"** | https://www.youtube.com/playlist?list=PLCGIzmTE4d0ioH72LxHKIFOY36YwQWpm7 | 30–60 | Multiple analysis angles | Detailed breakdown format |
| **World Cup 2022 Shootouts Playlist** | https://www.youtube.com/playlist?list=PLZOfq5OMX1TR2eB6ZzAzzz_INLQqMwtkJ | ~40–60 shootout penalties | Broadcast + replays | Full shootout sequences |

### 2.2 Major Compilation Videos

| Title | URL | Est. Penalties | Camera | Notes |
|---|---|---|---|---|
| **"EVERY Penalty SCORED In The Premier League 2023/24"** | https://www.youtube.com/watch?v=OGTvmQ7pphY | ~70–90 | Broadcast (wide) | Single-season comprehensive |
| **"World Cup Penalties You Can't Forget"** | https://www.youtube.com/watch?v=V1e5iFdXf0s | 20–40 | Broadcast + replays | Historical highlights |
| **"EPIC PENALTY SHOOTOUTS FIFA World Cup 2022"** | https://www.youtube.com/watch?v=HsV42JyuyPk | ~30–40 | Broadcast | 2022 tournament shootouts |
| **"World Cup penalty kicks, tracked"** | https://www.youtube.com/watch?v=HAuwPue57Vs | Variable | Analysis overlay | Tracking-enhanced footage |

### 2.3 Dedicated Penalty Channels

| Channel | URL | Est. Total Penalties | Notes |
|---|---|---|---|
| **Penalty\*\*shootout** | https://www.youtube.com/@penaltyshootout | 200–500+ across videos | Dedicated penalty content channel |
| **World Cup Goals** | https://www.youtube.com/channel/UC45Vp_sf8RUNhSPNUNMoyLg | 100–300+ | World Cup focused |

### 2.4 YouTube Yield Estimate

By systematically collecting from:
- Premier League "every penalty" compilations (2020–2025): ~350–450 penalties across 5 seasons
- World Cup shootout playlists (2014, 2018, 2022): ~80–120 penalties
- Euro / Copa America shootouts: ~40–60 penalties
- Champions League highlights: ~50–100 penalties

**Total estimated yield from YouTube: 500–700+ individual penalty clips**

### 2.5 Download Method

`yt-dlp` supports time-range extraction:
```bash
yt-dlp <url> --download-sections *START-END -f bestvideo+bestaudio
```
Requires `ffmpeg` for muxing. Can batch-process playlists.

---

## 3. Official Football Archives

### 3.1 FIFA Audiovisual Archives
| Field | Details |
|-------|---------|
| **URL** | https://www.archives.fifa.com |
| **Content** | 50,000+ hours; all FIFA competitions 1930–2023 |
| **Access** | Registration required via FIFA Media Sales team |
| **Format** | Professional broadcast quality video |
| **Cost** | Commercial licensing (likely expensive) |
| **Restrictions** | No perpetuity rights; no public redistribution; requires formal application |
| **Research use** | Must demonstrate purpose; approval not guaranteed |

### 3.2 UEFA Video Archive
| Field | Details |
|-------|---------|
| **URL** | https://www.uefa.com/news-media/news/025e-0f92372c2298-580b6c7931bf-1000--video-archive/ |
| **Content** | All UEFA competitions (Champions League, EURO, youth, women's, futsal) |
| **Access** | Licensing request form → videoarchive@uefa.ch |
| **Cost** | Commercial licensing |
| **Restrictions** | No perpetuity rights; no third-party rights clearance; no social media; UEFA content capped at 30% of program; right of refusal |
| **Research use** | Possible but heavily restricted; formal proposal required |

### 3.3 FIFA Films
| Field | Details |
|-------|---------|
| **URL** | https://www.fifafilms.com |
| **Content** | Documentary and tournament films |
| **Access** | Commercial |
| **Notes** | Less useful for systematic penalty extraction |

---

## 4. Kaggle / GitHub Datasets

### 4.1 Kaggle: Penalty Kick Dataset (2020–2025)
| Field | Details |
|-------|---------|
| **URL** | https://www.kaggle.com/datasets/rodrigoarede2003/penalty-kick-dataset-20202025 |
| **Content** | Penalty kick outcomes from EPL, Serie A, Liga Portugal (2020–2025) |
| **Format** | Tabular data (CSV) — **no video** |
| **Access** | Free download |
| **Notes** | Useful for outcome labels and metadata but does not include video |

### 4.2 Kaggle: Penalty Kick Outcome Prediction Challenge
| Field | Details |
|-------|---------|
| **URL** | https://www.kaggle.com/competitions/penalty-kick-outcome-prediction-challenge |
| **Format** | Competition dataset — likely tabular |
| **Notes** | May contain additional structured features |

### 4.3 GitHub: PenaltyKickPrediction (Masters Thesis)
| Field | Details |
|-------|---------|
| **URL** | https://github.com/gabriellavinco/PenaltyKickPrediction |
| **Content** | YOLO v7 detection + YOLO Pose estimation + LSTM prediction pipeline |
| **Data source** | YouTube-sourced penalty clips |
| **Access** | Code is public; video data likely not redistributed |
| **Notes** | Useful as a **pipeline reference** for building your own extraction workflow |

### 4.4 GitHub: PenaltyShot_Behavior (Nature Communications)
| Field | Details |
|-------|---------|
| **URL** | https://github.com/krm58/PenaltyShot_Behavior |
| **Content** | Analysis code for penalty shot behavior (published in Nature Communications) |
| **Notes** | Code + behavioral data; video availability unclear |

### 4.5 SoccerNet (500+ full matches)
| Field | Details |
|-------|---------|
| **URL** | https://www.soccer-net.org/data |
| **Content** | 500+ broadcast soccer matches at 25 fps, 720p |
| **Annotations** | Action spotting, player tracking, ball tracking, dense captions |
| **Format** | .mkv video files + JSON/CSV annotations |
| **Access** | Free after signing NDA |
| **Cost** | Free |
| **Restrictions** | NDA required; no redistribution |
| **Penalty content** | Penalties are embedded within full matches — must be extracted using action spotting labels |
| **Notes** | **Largest freely available soccer video dataset.** Penalty events need temporal extraction from full match files. SoccerNet's action labels include "penalty" events with timestamps. Estimated 50–150+ penalties across 500+ matches. |

---

## 5. Sports Analytics Companies

### 5.1 StatsBomb
| Field | Details |
|-------|---------|
| **URL** | https://statsbomb.com/what-we-do/hub/free-data/ |
| **Free data** | Event data for Euro 2020, Euro 2022, FA WSL, historical Big 5 leagues |
| **360 data** | Freeze-frame player locations for every event (Euro 2020/2022 free) |
| **Format** | JSON via API (Python: `statsbombpy`, R: `StatsBombR`) |
| **Video** | **No video data** — event/positional data only |
| **Cost** | Free tier available; commercial for full access |
| **Notes** | Excellent for matching penalty metadata (direction, outcome, xG) to video from other sources |

### 5.2 Wyscout (Hudl)
| Field | Details |
|-------|---------|
| **URL** | https://wyscout.com / https://www.hudl.com/en_gb/products/wyscout |
| **Content** | Full match video + tagged events for 600+ competitions |
| **Access** | Commercial subscription (Gold/Diamond tiers) |
| **Cost** | Not publicly listed; contact sales |
| **Academic pricing** | Not explicitly offered; may negotiate |
| **Restrictions** | Strict licensing; no redistribution; content sourced via rights agreements |
| **Notes** | Could systematically filter for all penalty events and clip them. Most comprehensive commercial option. Contact Hudl sales for academic inquiry. |

### 5.3 InStat
| Field | Details |
|-------|---------|
| **Content** | Match video + performance data |
| **Access** | Commercial subscription |
| **Notes** | Used in academic penalty research (OSPAF study). Contact for academic pricing. |

### 5.4 Nacsport
| Field | Details |
|-------|---------|
| **Content** | Video analysis software (not a data source per se) |
| **Notes** | Used in EPL penalty research (952 in-match PKs, 2009/10–2018/19). Software for coding/tagging video, not a video repository. |

---

## 6. Camera Angle Considerations for Pose Estimation

### 6.1 Research Findings

| Camera Angle | Pose Estimation Quality | Pros | Cons |
|---|---|---|---|
| **Behind-goal** | **Best for GK pose** | Clear frontal/rear view of GK; good depth perception; full body visible | Striker may be occluded; limited availability in broadcast |
| **Wide broadcast (elevated side)** | **Moderate for both** | Both players visible; most commonly available; established in SoccerNet | Lateral view limits depth estimation; limb overlap |
| **Side-on (pitch level)** | **Good for run-up kinematics** | Clear sagittal plane; good for gait analysis | Poor for body rotation cues; one player often occluded |
| **Behind-striker** | **Best for striker pose** | Full view of approach angle, hip rotation | GK partially occluded; rare in broadcasts |

### 6.2 Key Research on Angle Effects

- **OpenPose angle study** (Sensors, 2025, 25(3):799): Back-viewing cameras produced **lowest percentage deviations** in joint angle estimation. Accuracy is highly angle- and task-dependent.
- **MediaPipe for sports** (it-jim.com): 33 anatomical landmarks, 30+ FPS on mobile. Struggles with depth estimation and multi-person occlusion.
- **OSPAF recommendation** (Frontiers, 2021): Combine **at least three viewing angles**; aerial + pitch-level views from behind both taker and goalkeeper are optimal.
- **Pinheiro et al. (2022)**: Successfully extracted GK and striker body orientation from **standard broadcast footage** using OpenPose (confidence 0.80 ± 0.14).

### 6.3 Practical Recommendation

For this project (200–500 clips with dual pose estimation):
1. **Primary source**: Standard broadcast footage (wide angle) — most available, sufficient for OpenPose/MediaPipe with confidence >0.75
2. **Supplement with**: Behind-goal replays from FIFA "All The Angles" for validation subset
3. **Minimum resolution**: 720p at 25 fps (SoccerNet standard)
4. **Framework**: MediaPipe Pose (real-time, 33 landmarks) or OpenPose (more robust multi-person) — validated in penalty kick research

---

## 7. Legal / Ethical Considerations

### 7.1 Fair Use Analysis for Academic Penalty Kick Research

| Factor | Assessment | Favors Fair Use? |
|---|---|---|
| **Purpose & character** | Non-commercial academic research; transformative (extracting pose data, not reproducing entertainment value) | **Yes** |
| **Nature of work** | Factual broadcast footage (not highly creative) | **Yes** |
| **Amount used** | Short clips (~5–15 seconds per penalty from multi-hour broadcasts) | **Yes** |
| **Market effect** | No substitute for watching live sports; pose data extraction has no market overlap | **Yes** |

### 7.2 Key Legal Points

- **Transformative use**: Extracting biomechanical pose data from video is highly transformative — the research output (keypoint coordinates, statistical models) bears no resemblance to the original entertainment product. Courts have increasingly favored transformative uses in scholarship (Stanford Fair Use Center; *Google v. Oracle*).
- **De minimis clips**: Using 5–15 second clips from hours-long broadcasts is a minimal portion.
- **Non-commercial**: Pure academic research with no commercial distribution strengthens fair use claims.
- **Precedent**: Multiple published academic papers (Pinheiro 2022; Ranasinghe 2025; ULPGC 2025) have used broadcast penalty footage for pose estimation research without reported legal challenges.

### 7.3 Best Practices to Strengthen Fair Use Position

1. **Use only the minimum footage necessary** (clip to penalty approach + kick only)
2. **Do not redistribute raw video** — share only derived data (keypoints, annotations)
3. **Process locally** — do not host clips on public servers
4. **Cite sources** and acknowledge broadcast rights holders
5. **Obtain IRB/ethics approval** if required by institution (unlikely for public broadcast data, but check)
6. **Use NDA-protected datasets** where available (SoccerNet model)
7. **Document your fair use analysis** in the paper's methods/ethics section

### 7.4 Risk Assessment

| Source | Risk Level | Mitigation |
|---|---|---|
| SoccerNet (NDA) | **Low** | Formal agreement covers use |
| YouTube compilations | **Low-Medium** | Fair use likely applies; don't redistribute video |
| FIFA/UEFA archives | **Low** (if licensed) | Formal licensing eliminates risk |
| Academic dataset sharing | **Low** | Contact authors for terms |

---

## 8. Recommended Data Collection Strategy

### Phase 1: Secure Existing Academic Datasets (Weeks 1–2)

| Action | Expected Yield | Effort |
|---|---|---|
| Contact arXiv:2509.26088 authors for 755-clip dataset | 755 clips | Low (email) |
| Contact arXiv:2507.12617 authors for 1,010-clip dataset | 1,010 clips | Low (email) |
| Sign SoccerNet NDA and extract penalty events | 50–150 clips | Medium (temporal extraction) |
| Contact Pinheiro et al. for 34-clip OpenPose dataset | 34 clips | Low (email) |

**Phase 1 potential: 800–1,900+ clips** (if all authors share)

### Phase 2: YouTube Supplementation (Weeks 2–4)

| Action | Expected Yield | Effort |
|---|---|---|
| Collect EPL "every penalty" compilations (2020–2025) | 350–450 clips | Medium (yt-dlp + manual segmentation) |
| Collect World Cup shootout playlists | 80–120 clips | Low-Medium |
| Collect Euro/Copa America shootouts | 40–60 clips | Low-Medium |
| Collect FIFA "All The Angles" behind-goal clips | 30–50 clips | Low |

**Phase 2 potential: 500–680 clips**

### Phase 3: Annotation & Processing (Weeks 3–6)

1. Standardize all clips: 720p, 25 fps, 5–15 sec duration
2. Run MediaPipe Pose / OpenPose on all clips
3. Extract 33 (MediaPipe) or 25 (OpenPose) keypoints per frame for both GK and striker
4. Label: kick direction, outcome, competition, year, camera angle
5. Quality filter: discard clips where pose confidence < 0.6 for either player

### Priority Ranking

| Priority | Source | Rationale |
|---|---|---|
| **1 (Highest)** | arXiv:2509.26088 (755 clips) | Already annotated with keypoints; most directly relevant |
| **2** | arXiv:2507.12617 (1,010 clips) | Largest existing dataset; direction-labeled |
| **3** | SoccerNet penalty extraction | Free, NDA-covered, standardized quality |
| **4** | YouTube EPL season compilations | High volume, consistent camera, systematic |
| **5** | YouTube shootout compilations | Good for consecutive penalties (sequential analysis) |
| **6** | FIFA "All The Angles" | Behind-goal angles for validation subset |
| **7** | Wyscout (if budget allows) | Most comprehensive; systematic filtering |

### Target: 200–500 Clips

The target of 200–500 clips is **easily achievable** through:
- **Minimum path**: YouTube compilations alone yield 500+ clips
- **Optimal path**: Academic dataset sharing (755 or 1,010 pre-annotated clips) + SoccerNet extraction
- **Gold standard**: Academic datasets + YouTube + SoccerNet, with behind-goal validation subset from FIFA "All The Angles"

---

## Summary Table of All Sources

| Source | Type | N Penalties | Has Video? | Has Pose? | Free? | Access |
|---|---|---|---|---|---|---|
| arXiv:2509.26088 | Academic | 755 | Yes | Yes (2D keypoints) | Yes | Contact authors |
| arXiv:2507.12617 | Academic | 1,010 | Yes | No (direction labels) | Yes | Contact authors |
| Pinheiro 2022 | Academic | 34 | Yes | Yes (OpenPose) | Yes | Contact authors |
| IEEE DataPort PK | Academic | N/A | No | No | Yes | Public download |
| UT Tyler thesis | Academic | Unknown | Partial | Yes (GK kinematics) | Yes | Open access |
| SoccerNet | Academic | 50–150 (est.) | Yes | No | Yes (NDA) | Sign NDA |
| Kaggle PK 2020–25 | Data repo | Large | No | No | Yes | Public download |
| YouTube compilations | Public | 500–700+ | Yes | No | Yes | yt-dlp |
| FIFA Archives | Official | Thousands | Yes | No | $$$ | Licensing |
| UEFA Archives | Official | Thousands | Yes | No | $$$ | Licensing |
| StatsBomb | Analytics | Many | No | No | Free tier | API |
| Wyscout/Hudl | Commercial | Thousands | Yes | No | $$$ | Subscription |

---

*Report compiled via systematic web search across academic databases, video platforms, official archives, and sports analytics providers.*
