# Ten Recent Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exactly ten verified recent papers to the main paper while changing no non-citation text.

**Architecture:** Extend existing citation commands at four claim-bearing locations in `paper/main.tex`, then append ten complete BibTeX records to `paper/references.bib`. Verify provenance from official proceedings and prove prose identity by comparing the citation-stripped SHA-256 hash against the recorded baseline.

**Tech Stack:** LaTeX, BibTeX, Perl read-only validation, PMLR, OpenReview, NeurIPS proceedings.

## Global Constraints

- Modify only citation-command contents in `paper/main.tex` and append records to `paper/references.bib`.
- Do not change prose, equations, tables, figures, captions, results, or claims.
- Add exactly ten new keys; the bibliography must grow from 27 to 37 unique entries.
- Preserve `Stage shaping` and `Self-derived shaping` in Figure 5.
- The citation-stripped `paper/main.tex` SHA-256 baseline is `5159eebe74789780c42b73c445a48f407e1abf2ac2ec56185e7f5d0e88ae23f7`.

---

### Task 1: Add the Ten Verified BibTeX Records

**Files:**
- Modify: `paper/references.bib`

**Interfaces:**
- Consumes: official metadata from PMLR, NeurIPS proceedings, and OpenReview.
- Produces: ten unique BibTeX keys usable by `paper/main.tex`.

- [ ] **Step 1: Append the offline-to-online RL records**

Add:

```bibtex
@inproceedings{nakamoto2023calql,
  title={{Cal-QL}: Calibrated Offline RL Pre-Training for Efficient Online Fine-Tuning},
  author={Nakamoto, Mitsuhiko and Zhai, Simon and Singh, Anikait and Mark, Max Sobol and Ma, Yi and Finn, Chelsea and Kumar, Aviral and Levine, Sergey},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023},
  url={https://proceedings.neurips.cc/paper_files/paper/2023/hash/c44a04289beaf0a7d968a94066a1d696-Abstract-Conference.html}
}

@inproceedings{zhang2023pex,
  title={Policy Expansion for Bridging Offline-to-Online Reinforcement Learning},
  author={Zhang, Haichao and Xu, Wei and Yu, Haonan},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=-Y34L45JR6z}
}

@inproceedings{li2023explore,
  title={Accelerating Exploration with Unlabeled Prior Data},
  author={Li, Qiyang and Zhang, Jason and Ghosh, Dibya and Zhang, Amy and Levine, Sergey},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023},
  url={https://openreview.net/forum?id=Itorzn4Kwf}
}
```

- [ ] **Step 2: Append the policy-refinement records**

Add:

```bibtex
@inproceedings{yuan2025policydecorator,
  title={Policy Decorator: Model-Agnostic Online Refinement for Large Policy Model},
  author={Yuan, Xiu and Mu, Tongzhou and Tao, Stone and Fang, Yunhao and Zhang, Mengke and Su, Hao},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=e5jGTEiJMT}
}

@inproceedings{ren2025dppo,
  title={Diffusion Policy Policy Optimization},
  author={Ren, Allen Z. and Lidard, Justin and Ankile, Lars Lien and Simeonov, Anthony and Agrawal, Pulkit and Majumdar, Anirudha and Burchfiel, Benjamin and Dai, Hongkai and Simchowitz, Max},
  booktitle={International Conference on Learning Representations},
  year={2025},
  url={https://openreview.net/forum?id=mEpqHvbD2h}
}
```

- [ ] **Step 3: Append the temporal-abstraction records**

Add:

```bibtex
@inproceedings{shi2023skimo,
  title={Skill-based Model-based Reinforcement Learning},
  author={Shi, Lucy Xiaoyang and Lim, Joseph J. and Lee, Youngwoon},
  booktitle={Proceedings of the 6th Conference on Robot Learning},
  series={Proceedings of Machine Learning Research},
  volume={205},
  pages={2262--2272},
  year={2023},
  url={https://proceedings.mlr.press/v205/shi23a.html}
}

@inproceedings{zheng2024prise,
  title={{PRISE}: {LLM}-Style Sequence Compression for Learning Temporal Action Abstractions in Control},
  author={Zheng, Ruijie and Cheng, Ching-An and Daum{\'e} III, Hal and Huang, Furong and Kolobov, Andrey},
  booktitle={Proceedings of the 41st International Conference on Machine Learning},
  series={Proceedings of Machine Learning Research},
  volume={235},
  pages={61267--61286},
  year={2024},
  url={https://proceedings.mlr.press/v235/zheng24b.html}
}
```

- [ ] **Step 4: Append the world-model and imagination records**

Add:

```bibtex
@inproceedings{hansen2024tdmpc2,
  title={{TD-MPC2}: Scalable, Robust World Models for Continuous Control},
  author={Hansen, Nicklas and Su, Hao and Wang, Xiaolong},
  booktitle={International Conference on Learning Representations},
  year={2024},
  url={https://openreview.net/forum?id=Oxh5CstDJU}
}

@inproceedings{mazzaglia2023choreographer,
  title={Choreographer: Learning and Adapting Skills in Imagination},
  author={Mazzaglia, Pietro and Verbelen, Tim and Dhoedt, Bart and Lacoste, Alexandre and Rajeswar, Sai},
  booktitle={International Conference on Learning Representations},
  year={2023},
  url={https://openreview.net/forum?id=PhkWyijGi5b}
}

@inproceedings{escontrela2023viper,
  title={Video Prediction Models as Rewards for Reinforcement Learning},
  author={Escontrela, Alejandro and Adeniji, Ademi and Yan, Wilson and Jain, Ajay and Peng, Xue Bin and Goldberg, Ken and Lee, Youngwoon and Hafner, Danijar and Abbeel, Pieter},
  booktitle={Advances in Neural Information Processing Systems},
  volume={36},
  year={2023},
  url={https://openreview.net/forum?id=HWNl9PAYIP}
}
```

- [ ] **Step 5: Verify the bibliography delta**

Run:

```bash
perl -0ne 'while(/^\s*\@\w+\s*\{\s*([^,\s]+)\s*,/mg){$n{$1}++} END{print "entries=",scalar(keys %n)," duplicates=",scalar(grep {$n{$_}>1} keys %n),"\n"}' paper/references.bib
```

Expected: `entries=37 duplicates=0`.

### Task 2: Attach Each Paper to a Supported Existing Claim

**Files:**
- Modify: `paper/main.tex`

**Interfaces:**
- Consumes: the ten keys created in Task 1.
- Produces: a main paper in which every new key is cited.

- [ ] **Step 1: Extend the prior-data sentence**

Change only the citation at the end of “mixing demonstrations with newly
collected transitions” to:

```tex
\cite{ball2023efficient,nakamoto2023calql,zhang2023pex,li2023explore}
```

- [ ] **Step 2: Extend the policy-refinement paragraph**

Add `yuan2025policydecorator` to the citation following “learns an additive
action correction”. Add `ren2025dppo` to the sentence ending “constrained
around the base”. Do not change either sentence.

- [ ] **Step 3: Extend the temporal-abstraction sentence**

Change its citation to:

```tex
\cite{nachum2018hiro,fang2022ptp,shi2023skimo,zheng2024prise}
```

- [ ] **Step 4: Extend the world-model paragraph**

Add `hansen2024tdmpc2` and `mazzaglia2023choreographer` to the citation after
“additional policy-learning experience”. Add `escontrela2023viper` to the
citation after “video model for robot policy improvement”.

- [ ] **Step 5: Prove no non-citation text changed**

Run:

```bash
perl -0pe 's/~?\\cite\{[^}]*\}//g' paper/main.tex | sha256sum
```

Expected:

```text
5159eebe74789780c42b73c445a48f407e1abf2ac2ec56185e7f5d0e88ae23f7  -
```

### Task 3: Validate Citations and Build the Paper

**Files:**
- Verify: `paper/main.tex`
- Verify: `paper/references.bib`
- Generate: `paper/main.pdf`

**Interfaces:**
- Consumes: completed LaTeX and BibTeX edits.
- Produces: a resolved bibliography and successfully compiled PDF.

- [ ] **Step 1: Audit citation-key resolution**

Run a citation/BibTeX key audit and require:

```text
cited=34 bib=37 missing=0 duplicates=0
```

- [ ] **Step 2: Confirm all ten new keys are cited**

Run:

```bash
rg -n 'nakamoto2023calql|zhang2023pex|li2023explore|yuan2025policydecorator|ren2025dppo|shi2023skimo|zheng2024prise|hansen2024tdmpc2|mazzaglia2023choreographer|escontrela2023viper' paper/main.tex paper/references.bib
```

Expected: every key appears once in `references.bib` and at least once in
`main.tex`.

- [ ] **Step 3: Compile the full document**

Run from `paper/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: all four commands exit with status 0.

- [ ] **Step 4: Check final warnings and PDF**

Run:

```bash
rg -n -i 'undefined citations|citation.*undefined|overfull' paper/main.log
pdfinfo paper/main.pdf
```

Expected: no undefined citations or overfull boxes; PDF remains readable.

