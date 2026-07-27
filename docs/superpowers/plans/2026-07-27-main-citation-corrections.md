# Main-Paper Citation Corrections Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the audited citation placements and recent bibliography metadata while preserving the paper's claims and ten-page layout.

**Architecture:** Make narrow prose and citation-key edits in `paper/main.tex`, then complete only the three audited records in `paper/references.bib`. Rebuild the generated bibliography and PDF from those sources and validate semantic assertions, citation integrity, page count, and reference-page placement.

**Tech Stack:** LaTeX, BibTeX, AAAI-27 style, Poppler, Python standard library

## Global Constraints

- Preserve all experimental results, figures, tables, equations, and method claims.
- Preserve unrelated staged, tracked, and untracked user changes.
- Do not commit `paper/main.tex` or `paper/references.bib` because both contained pre-existing user modifications before this task.
- Keep exactly 47 cited keys and 47 bibliography entries with identical sets after removing the unsupported MimicGen co-citation.
- Keep `paper/main.pdf` at exactly ten pages.
- Keep References starting on page 9 and ending on page 10.
- Do not invent RISE pages or a DOI before official metadata is available.

---

### Task 1: Correct Citation Semantics in the Main Text

**Files:**
- Modify: `paper/main.tex:82-101`
- Modify: `paper/main.tex:164-202`
- Modify: `paper/main.tex:443-476`

**Interfaces:**
- Consumes: the approved wording and citation-placement decisions in `docs/superpowers/specs/2026-07-27-main-citation-corrections-design.md`
- Produces: semantically scoped prose with the same scientific claims and citation keys

- [ ] **Step 1: Run the semantic guard before editing**

Run a Python assertion that rejects the audited old phrases:

```bash
python3 -c 'from pathlib import Path; t=Path("paper/main.tex").read_text(); old=[r"more sample efficient~\cite{ball2023efficient,resfit2025residual}", "prior-data methods improve sample efficiency by mixing demonstrations with newly", r"improvement~\cite{yang2026rise,escontrela2023viper}", r"DexMimicGen~\cite{jia2024dexmimicgen,mandlekar2023mimicgen}", r"\emph{(v)~IQL}~\cite{kostrikov2022iql}: a"]; found=[s for s in old if s in t]; print(found); raise SystemExit(bool(found))'
```

Expected: exit 1 and print all five old patterns.

- [ ] **Step 2: Apply the approved narrow prose edits**

Make these exact semantic changes:

```tex
Recent frozen-policy residual methods show
that restricting the trainable component can make online finetuning substantially
more sample efficient~\cite{resfit2025residual}.
```

```tex
Sparse delayed feedback~\cite{arjona2019rudder,harutyunyan2019hindsight,hung2019optimizing,andrychowicz2017hindsight}
makes local credit weak. In our setting, we hypothesize that this weak evidence
allows critic error and residual drift to accumulate over the remaining trajectory.
```

```tex
Online RL can improve this initialization, and prior-data methods improve sample
efficiency by leveraging demonstrations or offline experience during online
learning~\cite{ball2023efficient,nakamoto2023calql,zhang2023pex,li2023explore,rajeswaran2018learning,nair2018overcoming,pertsch2021accelerating}.
```

```tex
RISE uses an action-conditioned video model for robot policy
improvement~\cite{yang2026rise}. Video prediction can also provide an
action-free reward signal for RL~\cite{escontrela2023viper}.
```

Use only `jia2024dexmimicgen` for the named DexMimicGen task provenance.

Describe the final baseline as:

```tex
\emph{(v)~IQL}: we instantiate IQL~\cite{kostrikov2022iql} as a
full-policy off-policy RL baseline that finetunes the entire policy without
retaining a frozen base.
```

- [ ] **Step 3: Run the semantic guard after editing**

Run the Step 1 command again.

Expected: exit 0 and print `[]`.

### Task 2: Complete the Audited Bibliography Records

**Files:**
- Modify: `paper/references.bib:137-160`
- Modify: `paper/references.bib:236-242`

**Interfaces:**
- Consumes: verified arXiv, OpenReview, PMLR, ICRA, and RISE project metadata
- Produces: normalized ResFit, DSRL, and DexMimicGen records without changing their citation keys

- [ ] **Step 1: Run the metadata guard before editing**

```bash
python3 -c 'from pathlib import Path; b=Path("paper/references.bib").read_text(); required=["note={ICLR 2026 Workshop on AI with Recursive Self-Improvement}", "volume={305}", "pages={258--282}", "url={https://proceedings.mlr.press/v305/wagenmaker25a.html}", "pages={16923--16930}", "doi={10.1109/ICRA55743.2025.11127809}"]; missing=[s for s in required if s not in b]; print(missing); raise SystemExit(bool(missing))'
```

Expected: exit 1 and print the missing metadata.

- [ ] **Step 2: Update the three records**

For ResFit, retain the 2025 arXiv article and change only the note to the full
workshop name.

For DSRL, retain the paper-PDF author order and use:

```bibtex
booktitle={Proceedings of the 9th Conference on Robot Learning},
pages={258--282},
year={2025},
volume={305},
series={Proceedings of Machine Learning Research},
publisher={PMLR},
url={https://proceedings.mlr.press/v305/wagenmaker25a.html}
```

For DexMimicGen, add:

```bibtex
pages={16923--16930},
doi={10.1109/ICRA55743.2025.11127809},
url={https://doi.org/10.1109/ICRA55743.2025.11127809}
```

- [ ] **Step 3: Run the metadata guard after editing**

Run the Step 1 command again.

Expected: exit 0 and print `[]`.

### Task 3: Rebuild and Validate the Paper

**Files:**
- Generate: `paper/main.bbl`
- Generate: `paper/main.pdf`
- Inspect: `paper/main.log`

**Interfaces:**
- Consumes: corrected `paper/main.tex` and `paper/references.bib`
- Produces: a verified ten-page PDF with 47 resolved references

- [ ] **Step 1: Build from `paper/`**

```bash
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Expected: all four commands exit 0.

- [ ] **Step 2: Check citation-set integrity**

```bash
python3 -c 'import re; t=open("paper/main.tex").read(); b=open("paper/main.bbl").read(); c=set(k.strip() for g in re.findall(r"\\cite(?:t|p|alp|author|year|yearpar)?\{([^}]+)\}",t) for k in g.split(",")); x=set(re.findall(r"\\bibitem.*?\]\{([^}]+)\}",b,re.S)); print("cited",len(c),"bibliography",len(x),"missing",sorted(c-x),"uncited",sorted(x-c)); raise SystemExit(c!=x or len(c)!=47)'
```

Expected: `cited 47 bibliography 47 missing [] uncited []`, exit 0.

- [ ] **Step 3: Check LaTeX warnings and page contract**

```bash
rg -n 'Citation|undefined references|multiply defined|There were undefined' paper/main.log
pdfinfo paper/main.pdf
pdftotext -f 8 -l 8 -layout paper/main.pdf -
pdftotext -f 9 -l 9 -layout paper/main.pdf -
```

Expected: no citation/reference warning matches; `Pages: 10`; no
`References` heading on page 8; one `References` heading on page 9.

- [ ] **Step 4: Inspect the generated reference pages**

```bash
pdftocairo -f 9 -l 10 -png -r 130 paper/main.pdf /tmp/resfit-citation-audit
```

Inspect both generated images for clipping, overlaps, malformed names, and an
unexpected page 11.

- [ ] **Step 5: Review the scoped diff**

```bash
git diff -- paper/main.tex paper/references.bib
```

Expected: only the approved prose, citation, and bibliography changes from this
plan appear in the task's relevant hunks; unrelated pre-existing changes remain
untouched.
