### Reviewer memo (Bill Thompson Blair)

**Manuscript**: *The Subspace Penalty Kick: How Shared Neural Representations Create Exploitable Interference in the Goalkeeper–Striker Duel*  
**Author**: Kyle E. Mathewson  
**Reviewer perspective**: mid‑career sports psychologist; decision making + performance under pressure; interested in theory-to-measure alignment and reproducible inference  
**Date**: 2026-02-11  

---

### Overall evaluation (what I like)

This is a creative and genuinely enjoyable manuscript. The core move—mapping an emerging systems‑neuroscience result (shared neural subspaces for compositional tasks) onto an ecologically rich dyadic sport problem (penalty kicks)—is novel and intellectually productive. You do several things well:

- **You generate falsifiable predictions** rather than stopping at metaphor.
- **You try to confront the theory with real data**, and (importantly) you treat nulls as information about measurement limits, not as “theory failure” per se.
- **You write clearly and engagingly**, and the figures are visually compelling.

With revisions, I think this could be a strong “bridge paper” for a sport psych / decision‑making audience: it can help researchers think more mechanistically about (a) commitment dynamics, (b) deception, and (c) sequential dependencies in interactive tasks.

---

### One high-level recommendation (positioning)

Right now the paper oscillates between (i) a conceptual/theoretical framework article and (ii) an empirical paper that “tests predictions.” For a sports psychology journal, you’ll likely land better if you **explicitly position this as a theory/bridge paper with *exploratory* observational checks** and a clear experimental agenda.

- **Practical implication**: tighten the empirical language (especially in the Abstract) and elevate the “Proposed experiments” section into a more formal roadmap.

---

### Major comments and actionable revisions

#### 1) Fix the Abstract’s dataset description so it matches the code + results

There is a **direct mismatch** between the Abstract and the analysis pipeline outputs.

- **Bundesliga dataset size and date range**:
  - The Abstract says \(N=6{,}392\) and “1963–2025”.
  - The analysis and main text later state **\(N=4{,}599\), 1963–2017** (see `data/bundesliga_penalties_1963_2017.csv` and `analysis/prediction2_sequential.py`, `analysis/prediction4_expertise_blindspots.py`).
  - **Action**: update the Abstract to the actual dataset used, or update the dataset and re-run analysis so the Abstract is true. Don’t leave this inconsistent—it’s a desk‑reject trigger.

- **“After a save” vs “after a save/miss”**:
  - In `analysis/prediction2_sequential.py`, the “previous non-goal” category includes **saves and misses** (anything not `Tor`).
  - In the Abstract, you say “after a save,” which implies you excluded misses (you did not).
  - **Action**: revise to “after a previous non-goal (save/miss)” or explicitly split misses vs saves (preferred; see below).

- **“Same-match penalties” claim for the Kaggle keeper repetition**:
  - The code defines “same match” as **consecutive rows in the Kaggle file** where `goalkeeper_name` and `Country` repeat.
  - That may or may not correspond to the same match unless the dataset is strictly match-sorted. Even if it is, readers will (rightly) question this operationalisation.
  - **Action**: either (a) soften the claim (“consecutive penalties faced by the same goalkeeper within this dataset”) or (b) engineer a defensible match identifier (if the raw source supports it), then re-run.

#### 2) Add a proper Methods subsection for the empirical section (and report effect sizes + uncertainty)

Sports psych journals typically expect that even “light” observational analyses include:

- **Data provenance**: where each dataset came from, citation/URL, license/permissions.
- **Inclusion/exclusion rules**: how you handled missing values, duplicates, ambiguous “centre,” etc.
- **Operational definitions**: exactly how you defined “match,” “repeat,” “experience,” “save rate,” etc.
- **Effect sizes + confidence intervals**, not only p-values.

**Actionable suggestion**: In Section 4, insert a short “Data and analysis plan” subsection that lists variables, coding, and tests in a table.

- **For binary outcomes**, add Wilson/Brown binomial CIs or bootstrap CIs (clustered; see #4 below).
- For the chi‑square in Prediction 2, report **risk difference** and/or **odds ratio** with CI. The effect is interesting; quantify it.

#### 3) The key empirical signal (Prediction 2) deserves deeper treatment and stronger alternative-explanations control

Prediction 2 is the one result that will attract attention and scepticism, because it challenges the “serial independence” intuition from equilibrium play.

Right now, the manuscript already notes alternative explanations (team dominance, same taker, keeper preference). That’s good, but you can strengthen it without blowing up scope:

- **Disaggregate “previous non-goal” into misses vs saves**:
  - Psychologically, a save (keeper agency) and a miss (kicker error) plausibly imply different subsequent states (confidence, threat appraisal).
  - Your Bundesliga file has multiple miss categories (`vorbei`, `Pfosten`, `Latte`, etc.). Leverage that.
  - **Action**: extend `analysis/prediction2_sequential.py` to 3 levels of previous outcome (Goal / Save / Miss) and compare.

- **Control for within-match clustering**:
  - Current chi‑square treats each pair as independent.
  - At minimum, compute a **cluster-robust SE / permutation test within match** (shuffle outcomes within match) to see if the signal survives match-level dependence.

- **Control for team side / dominance proxies**:
  - Use existing covariates in the Bundesliga data: `goaldiff`, `minute`, `homegame`, plus `ptexp` and `gkexp` as crude quality proxies.
  - **Action**: logistic regression `goal ~ prev_goal + goaldiff + minute + homegame + ptexp + gkexp` with clustered SEs by `match_id` (or a mixed model if you want to be fancy).

- **Time between penalties**:
  - One of your strongest theoretical hooks is persistence across intervening “blocks.”
  - The Bundesliga data include `minute`. You can compute \(\Delta t\) between penalties and test whether the sequential effect decays with time.
  - **Action**: add a moderator: `prev_goal * delta_minute`.

- **Most important upgrade: add a shootout dataset replication**
  - The theory’s “kick \(n-2\)” ghost is *shootout-structured*. In‑game penalties are a clever proxy, but the paper will be much stronger if you also test in true shootouts.
  - There are public datasets for World Cup / Euros / UCL shootouts (and Kaggle datasets).
  - **Action**: even one supplementary replication analysis would substantially improve publishability.

#### 4) Independence and unit-of-analysis issues (common reviewer objection)

Several analyses treat each penalty as independent when it’s clearly nested (kicks within players, keepers, matches, seasons).

You don’t need a maximal mixed model everywhere, but you should:

- **Acknowledge the nesting explicitly** in the Methods and treat p-values as exploratory.
- Where feasible, use **clustered bootstrap** (by match for Bundesliga; by goalkeeper or player for Kaggle).

Concrete places this matters:

- Prediction 4 (experience): repeated keepers across seasons; naive logit overstates precision.
- Prediction 6 (foot congruence): repeated kickers; also strong team/league heterogeneity.

#### 5) The theory-to-measure alignment is strongest when you say “mechanism needs kinematics”; lean into that

I like your honesty that several predictions are not truly testable with “L/C/R + goal” records. I’d go one step further:

- **Add a short table** (maybe in Section 3 or early Section 4) with columns:
  - Prediction
  - Mechanism (subspace interference / gain modulation / transformation timing)
  - Minimal measurement needed (kinematics, gaze, EEG/EMG, force plates, VR manipulation)
  - Whether your current datasets meet that bar (Yes/No)

This turns a potential weakness (“null results”) into a strength (“clear measurement map for the field”).

#### 6) Consider reframing (or relocating) the “Goalkeeper’s Edge / Striker’s Edge” tip boxes

They’re fun and will appeal to some readers, but in many sport psych journals they’ll read as too coach-facing and potentially too confident given the evidence base.

Options:

- **Option A (my preference)**: keep them but move them into a brief **“Applied implications (speculative)”** section, with explicit caveats.
- **Option B**: keep the boxes but soften language (“may,” “hypothesised,” “suggests”) and avoid prescriptive coaching tone.
- **Option C**: move to Supplementary / Appendix.

#### 7) Figure + image provenance: AI-generated photorealistic images need a publication plan

The photorealistic figures (`manuscript/figures/keeper_motor_pathways.png`, `.../striker_shared_subspace.png`, `.../transformation_window.png`) look great. However, journals increasingly require:

- explicit disclosure that images are AI-generated,
- confirmation that no copyrighted source images were used,
- and sometimes restrictions on photorealistic depictions of “real people.”

**Action**:

- Add a brief statement in Methods/Acknowledgements or a “Generative AI use” note explaining how figures were created (e.g., prompts + model) and that you have rights to publish them.
- Consider a “safe” fallback: convert them into **vector schematics** (or at least remove “stock-photo realism”) so the science survives if a journal objects.

---

### Targeted comments on each empirical prediction (from a sports psych / decision-making lens)

#### Prediction 1 (within- vs cross-axis mismatch)

- **What works**: You identify a classic measurement ceiling: if the keeper dives wrong, goals are near-certain.
- **Key improvement**: your theoretical prediction is about *correction dynamics*, so binary goal outcome is the wrong dependent variable.
- **Actionable next step**: Reframe as “observational check indicates outcome data are insensitive; kinematic measures are required,” and consider dropping the Fisher test emphasis (or keep it as a brief negative control).

#### Prediction 2 (sequential effects / task-history ghosts)

- **This is the empirical anchor**. Strengthen as per Major comment #3.
- **Also consider**: sequential effects could reflect *psychological momentum* and *pressure regulation* as much as neural subspace persistence. You can treat that as a feature: your framework offers a candidate mechanism for why momentum-like patterns arise even with long lags.

#### Prediction 4 (expertise blind spots)

- **Your operationalisation is too coarse** for the mechanistic claim.
  - Experience (seasons) ≠ perceptual attunement or gain modulation.
  - Outcome “save” conflates keeper skill with kicker misses.
- **Actionable improvements** (still feasible with current data):
  - Restrict to on-target penalties (Goal vs Save) for “save rate.”
  - Model `saved ~ gkexp + ptexp + goaldiff + minute` and cluster by keeper.
  - If you keep “unreadable” shots, test an interaction: `gkexp * shot_type`.

#### Prediction 5 (keeper centrality)

- Your analysis correctly flags **severe underpowering** (n=12 central).
- **Action**: rather than treating this as a “failed test,” frame it as a measurement diagnosis: action bias produces too few “stay” cases for inference in this dataset.
- If you want something testable now: analyse **the base rate** of “stay central” (action bias replication) and link it to Bar‑Eli’s norm/regret framing.

#### Prediction 6 (pre-commitment / foot congruence proxy)

- You already say it well: **foot congruence is a weak proxy** for internal commitment state.
- If you keep it, I’d treat it as a “sanity check” rather than a serious test.
- Stronger observational alternatives (if available):
  - Use a dataset with coded keeper-dependent vs keeper-independent strategies (see Noël et al. 2015).
  - Or use video coding / run-up fluency indicators (again, Noël et al.).

---

### Analysis code review (reproducibility + statistical choices)

I reviewed the analysis pipeline in `analysis/` plus the build system.

- **Strengths**:
  - **Clear, modular scripts** per prediction (`analysis/prediction*.py`) and shared utilities (`analysis/utils.py`).
  - **Reproducible build intent** via `Makefile`, auto-written LaTeX macros (`manuscript/generated_stats.tex`), and figure exports to `manuscript/figures/`.
  - **Virtual environment workflow** is present (`make` creates `.venv`).

- **High-impact improvements**:
  - **Pin dependency versions** in `analysis/requirements.txt` (or add a lockfile). Right now it’s unpinned, so exact results may drift.
  - Add a short `README.md` at project root with:
    - `make venv && make analysis && make manuscript`
    - expected outputs
    - data source URLs + license notes
  - `manuscript/compile.sh` currently generates only conceptual figures and then compiles LaTeX; it does **not** run `analysis/run_all.py`. Consider updating it (or clearly documenting that `make` is the canonical build).

- **Statistical choices to flag in the manuscript**:
  - Most analyses are simple and interpretable (good), but **treat nested data as independent**.
  - Several tests would benefit from **CIs** and **cluster-aware inference**.

---

### Figures: what to fix or improve

#### Empirical plots (quick wins)

- **Add uncertainty**: binomial 95% CIs on bar charts (Pred 1/2/5/6) would help readers judge whether differences are meaningful.
- **Reduce title size**: some plot titles are large relative to panel content (journal column widths will make this feel heavy).
- **Legend text glitch**: in `empirical_pred4_expertise.png`, the legend renders as “size □ n” (encoding issue). Change label to “size ∝ n” (plain text) or add a manual legend.

#### Conceptual/AI figures

- They are visually strong, but consider:
  - **Accessibility**: ensure they remain interpretable when printed in grayscale and at column width.
  - **Provenance**: add explicit disclosure and “rights to publish” assurance.
  - **Tone**: for some journals, photorealistic images can read as “popular science.” A schematic alternative might fit better.

---

### References: suggested additions (with stable links) and why they help

Below are references that would strengthen the sports-psych decision-making grounding, provide balance on the game-theory literature, and support the “commitment dynamics” framing. I’ve included DOI links (most stable) and, where helpful, an open PDF link.

- **Mixed strategy equilibrium in penalty kicks (complements Palacios-Huerta; improves credibility with sport + econ reviewers)**  
  Chiappori, P.-A., Levitt, S., & Groseclose, T. (2002). *Testing Mixed-Strategy Equilibria When Players Are Heterogeneous: The Case of Penalty Kicks in Soccer.* **American Economic Review, 92**(4), 1138–1151. DOI: [`https://doi.org/10.1257/00028280260344678`](https://doi.org/10.1257/00028280260344678)  
  Open PDF (author-hosted): [`https://pricetheory.uchicago.edu/levitt/Papers/ChiapporiGrosecloseLevitt2002.pdf`](https://pricetheory.uchicago.edu/levitt/Papers/ChiapporiGrosecloseLevitt2002.pdf)  
  - **How it helps**: shows you know the full penalty‑kick equilibrium literature and provides a second anchor beyond Palacios‑Huerta.

- **Keeper-dependent vs keeper-independent strategies (directly relevant to your P6 “pre-commitment” discussion)**  
  Noël, B., Furley, P., van der Kamp, J., Dicks, M., & Memmert, D. (2015). *The development of a method for identifying penalty kick strategies in association football.* **Journal of Sports Sciences, 33**(1), 1–10. DOI: [`https://doi.org/10.1080/02640414.2014.926383`](https://doi.org/10.1080/02640414.2014.926383)  
  - **How it helps**: gives you an evidence-based operationalisation of “pre-commitment” that is closer to your mechanism than footedness.

- **Late alterations impair accuracy (mechanistic support for “motor purity” / interference claim)**  
  van der Kamp, J. (field experiment; PubMed ID 16608761). *A field simulation study of the effectiveness of penalty kick strategies in soccer: late alterations of kick direction increase errors and reduce accuracy.* **Journal of Sports Sciences.** DOI: [`https://doi.org/10.1080/02640410500190841`](https://doi.org/10.1080/02640410500190841)  
  - **How it helps**: directly supports your claim that mid-run-up changes can degrade execution—exactly the kind of “motor contamination” your framework predicts.

- **Anxiety and attentional control in penalty kicks (sports-psych core; connects to your “task belief/gain” language under pressure)**  
  Wilson, M. R., Wood, G., & Vine, S. J. (2009). *Anxiety, attentional control, and performance impairment in penalty kicks.* **Journal of Sport & Exercise Psychology, 31**(6), 761–775. DOI: [`https://doi.org/10.1123/jsep.31.6.761`](https://doi.org/10.1123/jsep.31.6.761)  
  - **How it helps**: lets you speak to mainstream sport psych mechanisms (threat attention, disrupted control) and then propose how subspace gating might implement them neurally.

- **Choking / self-regulation in shootouts (strong context for sequential effects + timing)**  
  Jordet, G. (2009). *When Superstars Flop: Public Status and Choking Under Pressure in International Soccer Penalty Shootouts.* **Journal of Applied Sport Psychology, 21**(2). DOI: [`https://doi.org/10.1080/10413200902777263`](https://doi.org/10.1080/10413200902777263)  
  - **How it helps**: supports your discussion of pressure-driven behaviour and provides a path to integrate “ghost” effects with self-regulation under threat.

- **Avoidance motivation and negative-valence kicks (a clean shootout dataset paper; dovetails with your sequential framing)**  
  Jordet, G., & Hartman, E. (2008). *Avoidance motivation and choking under pressure in soccer penalty shootouts.* **Journal of Sport & Exercise Psychology, 30**(4), 450–457. DOI: [`https://doi.org/10.1123/jsep.30.4.450`](https://doi.org/10.1123/jsep.30.4.450)  
  - **How it helps**: gives you a validated way to talk about valence/pressure as moderators—useful for explaining heterogeneity in sequential effects.

- **Decision neuroscience of commitment dynamics (supports your “point of no return” and timing claims)**  
  Thura, D., & Cisek, P. (2014). *Context-Dependent Urgency Influences Speed–Accuracy Trade-Offs in Decision-Making and Movement Execution.* **Journal of Neuroscience, 34**(49), 16442–16454. DOI: [`https://doi.org/10.1523/JNEUROSCI.0162-14.2014`](https://doi.org/10.1523/JNEUROSCI.0162-14.2014)  
  - **How it helps**: lets you connect keeper commitment timing to a broader decision‑to‑action literature beyond Tafazoli et al.

- **Neural population dynamics / low-dimensional subspaces in motor control (makes “subspace” language feel canonical, not bespoke)**  
  Shenoy, K. V., Sahani, M., & Churchland, M. M. (2013). *Cortical Control of Arm Movements: A Dynamical Systems Perspective.* **Annual Review of Neuroscience, 36**, 337–359. DOI: [`https://doi.org/10.1146/annurev-neuro-062111-150509`](https://doi.org/10.1146/annurev-neuro-062111-150509)  
  PDF: [`https://web.stanford.edu/~shenoy/GroupPublications/ShenoyEtAlAnnRevNeurosci2013.pdf`](https://web.stanford.edu/~shenoy/GroupPublications/ShenoyEtAlAnnRevNeurosci2013.pdf)  
  - **How it helps**: reinforces the legitimacy of the “subspace/dynamics” framing for readers outside systems neuroscience.

- **Goalkeepers show sequential biases in shootouts (important comparator for your Prediction 2; gives you the canonical “gambler’s fallacy” penalty-kick citation)**  
  Misirlisoy, E., & Haggard, P. (2014). *Asymmetric Predictability and Cognitive Competition in Football Penalty Shootouts.* **Current Biology, 24**(16), 1918–1922. DOI: [`https://doi.org/10.1016/j.cub.2014.07.013`](https://doi.org/10.1016/j.cub.2014.07.013)  
  - **How it helps**: you currently mention gambler’s fallacy without a primary penalty-shootout citation. This paper also provides a substantive “alternative pattern” (alternation) that your repetition result should be compared against.

- **Temporal preparation under pressure in shootouts (directly aligned with your timing/commitment framing and provides a model for video-based measures)**  
  Jordet, G., Hartman, E., & Sigmundstad, E. (2009). *Temporal links to performing under pressure in international soccer penalty shootouts.* **Psychology of Sport and Exercise, 10**(6), 621–627. DOI: [`https://doi.org/10.1016/j.psychsport.2009.03.004`](https://doi.org/10.1016/j.psychsport.2009.03.004)  
  - **How it helps**: shows you can operationalise “commitment timing” with video-derived intervals—exactly the kind of measurement your subspace account predicts will matter.

- **Goalkeeper distraction effects (supports your P5 “centrality/ambiguity” discussion via attentional control theory)**  
  Wood, G., & Wilson, M. R. (2010). *A moving goalkeeper distracts penalty takers and impairs shooting accuracy.* **Journal of Sports Sciences, 28**(9), 937–946. DOI: [`https://doi.org/10.1080/02640414.2010.495995`](https://doi.org/10.1080/02640414.2010.495995)  
  - **How it helps**: strengthens the link from goalkeeper behaviour → striker attention/execution, which is central to your “interference” story.

- **Perceived-control mechanism for “quiet eye” training under pressure (bridges attention + decision confidence to performance)**  
  Wood, G., & Wilson, M. R. (2012). *Quiet-eye training, perceived control and performing under pressure.* **Psychology of Sport and Exercise.** DOI: [`https://doi.org/10.1016/j.psychsport.2012.05.003`](https://doi.org/10.1016/j.psychsport.2012.05.003)  
  - **How it helps**: gives you a validated “pressure → attentional control → accuracy” pathway that your gain-modulation / task-belief language can plausibly implement.

- **Action bias follow-up (useful if you keep Bar‑Eli as a central pillar; shows the broader “socially rational” account)**  
  Bar‑Eli, M., Azar, O. H., & Lurie, Y. (2009). *(Ir)rationality in action: do soccer players and goalkeepers fail to learn how to best perform during a penalty kick?* **Progress in Brain Research, 174.** DOI: [`https://doi.org/10.1016/S0079-6123(09)01309-0`](https://doi.org/10.1016/S0079-6123(09)01309-0)  
  - **How it helps**: gives you a stronger foundation for interpreting the scarcity of “stay central” cases as a stable bias shaped by norms/utility, not simple error.

---

### Potential journal homes (and how to pitch the contribution)

This paper is cross-disciplinary; reviewers will come from different “home” literatures. A few plausible journal targets and how I’d pitch it:

- **Psychology of Sport and Exercise**:
  - **Pitch**: decision making under pressure + interactive opponent dynamics; shared-subspace account as a mechanistic synthesis; empirical checks as exploratory.
  - **Revise for fit**: add stronger pressure/attention citations (e.g., Wilson et al.; Jordet).

- **Journal of Sport & Exercise Psychology**:
  - **Pitch**: theory-driven predictions + applied implications; connect “task belief/gain” to attentional control theory and self-regulation.
  - **Revise for fit**: tone down photorealistic figures; add clear Methods and effect sizes.

- **Journal of Applied Sport Psychology**:
  - **Pitch**: practical implications (your tip boxes) plus a mechanistic model that can generate testable training interventions.
  - **Revise for fit**: keep applied boxes but explicitly label them as speculative and tied to proposed experiments.

- **Human Movement Science / Motor Control**:
  - **Pitch**: commitment dynamics and perception–action coupling in a high-tempo task; subspace framework as a motor decision mechanism.
  - **Revise for fit**: expand kinematics/biomechanics grounding and tighten the observational-statistics section.

Regardless of venue, the safest framing is **“mechanistic synthesis + measurement roadmap”** rather than “we tested and confirmed a neural theory in the wild.”

### Minor / editorial suggestions (quick polish)

- **Tone calibration for journal submission**:
  - Some lines are wonderfully written but might read as “popular science” in a traditional journal (“subspace war,” “neural hygiene,” “No goalkeepers were harmed…”).
  - Consider keeping the voice but trimming the most playful lines, or moving them to a blog/preprint companion.

- **Terminology precision**:
  - Replace “international-level Kaggle penalties” with a more accurate descriptor (league/country composition; you can report a table of `Country` counts).
  - Use “save/miss” rather than “save” when that’s what the coding implies.

- **Citations for gambler’s fallacy / sequential expectations**:
  - You mention gambler’s fallacy without citation. One or two classic references would help (especially for decision-making reviewers).

- **Reference hygiene**:
  - Some BibTeX keys/years look mismatched (e.g., keys that encode 2011/2018 but the year field differs). Not fatal, but it can confuse reviewers if they inspect the `.bib`.

---

### “If you only do three things before submission…”

1) **Make the Abstract perfectly consistent** with the actual datasets, coding, and results.  
2) **Deepen Prediction 2** with at least one robustness check (miss vs save, clustered inference, or a shootout replication).  
3) **Add a brief Methods/Data provenance subsection** + dataset citations and effect sizes/intervals.

Those three changes alone will materially increase the odds of a smooth review process.

---

### Closing

I really like this paper. It’s the kind of cross-disciplinary synthesis that can open up new experimental thinking in sport psych—especially if the empirical component is framed with the right level of humility and methodological clarity. I’d be happy to see it published after tightening the claim–measure link and upgrading the sequential-effects analysis.

— **Bill Thompson Blair**

