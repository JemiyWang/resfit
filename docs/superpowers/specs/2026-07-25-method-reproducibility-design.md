# SHORE-RL Method Reproducibility Revision Design

Date: 2026-07-25

## Goal

Revise the SHORE-RL method description so that the main paper is self-contained enough
to answer reproducibility questions about the goal representation, waypoint target,
future-state sampling, value/navigator losses, online/offline mixing, and potential
reward storage. The paper must describe the implementation used by the headline runs
rather than an idealized variant.

## Source of Truth

The implementation and headline run configurations are authoritative:

- `resfit/rl_finetuning/chunk_residual/hiql_gc_value.py`
- `resfit/rl_finetuning/chunk_residual/hiql_high_actor.py`
- `resfit/rl_finetuning/chunk_residual/hiql_subgoal.py`
- `resfit/rl_finetuning/chunk_residual/online_hiql_finetune.py`
- `resfit/rl_finetuning/chunk_residual/online_hiql_store.py`
- `resfit/rl_finetuning/chunk_residual/hiql_potential.py`
- `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
- `resfit/rl_finetuning/off_policy/rl/q_agent.py`
- `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py`
- `resfit/rl_finetuning/chunk_residual/offline_stage_replay.py`
- the task-specific headline run scripts.

Two existing main-paper statements are factually incorrect and must be replaced:

1. The waypoint conditions both the residual actor and the TD3 critic. It is not
   actor-only.
2. The single-state potential value is frozen during residual RL. Its shaped reward is
   computed before a transition is inserted into replay and is not recomputed when the
   transition is sampled.

## Scope

### In scope

- `paper/main.tex`, especially Sections 3, 4.1, 4.2, and 4.3.
- `paper/aaai2027-unified-supp.tex`, especially Sections A.1--A.5, B.2--B.3, and C.5.
- Cross-references, notation, equations, and short hyperparameter statements needed to
  keep those two documents consistent.
- Compilation and PDF-level verification of the main paper and supplement.

### Out of scope

- Figure 1 and every source asset used to build Figure 1. The authors will replace it
  separately.
- Experimental results, plots, tables, TODO values, references, and unrelated prose.
- Changes to the RL implementation.
- Changes to the scientific claims beyond what is required to make them match the
  implementation.

## Main-Paper Revision

### Frozen representation and task goal

Define the representation used by the value modules as a standardized
visual-proprioceptive feature:

\[
e_t = \operatorname{Std}\!\left(
[\operatorname{Pool}(\operatorname{Enc}_{\pi_b}(o_{\le t}));q_t]
\right).
\]

For ACT, the visual component is the mean-pooled frozen encoder-token feature; for
served pi0 policies it is the frozen prefix feature. The representation contains no
simulator object pose in deployed configurations.

Define the fixed task goal \(g_\star\) as a real demonstration terminal feature: among
all demonstration terminal features, choose the one nearest their mean. This medoid
avoids supplying the navigator with an averaged feature that was never observed.

### Goal-conditioned value

Describe the twin-head goal-conditioned value and its normalized bottleneck:

\[
z=\phi(e,g), \qquad \lVert z\rVert_2=\sqrt{d_z}, \qquad
V_i(e,g)=v_i([e;\phi(e,g)]), \quad i\in\{1,2\}.
\]

The bottleneck dimension is \(d_z=10\). The goal-reaching convention is
\(r_g=\mathbf 1[e=g]-1\), with bootstrap mask
\(m_g=1-\mathbf 1[e=g]\). State the EMA target and expectile update compactly:

\[
y_g=r_g+\gamma m_g\min_i \bar V_i(e',g),
\qquad
\mathcal L_V=\sum_i
\mathbb E\!\left[
|\tau-\mathbf 1[y_g-V_i<0]|(y_g-V_i)^2
\right].
\]

If the exact headline checkpoints use the per-head HIQL target rather than the shared
minimum target, the equation must use the per-head form implemented by
`value_loss_mode=hiql`; prose must not claim a shared-min target. The implementation
inspection during execution will resolve the final notation from checkpoint/run
configuration provenance.

Goals are sampled as:

- current feature with probability 0.2;
- a same-trajectory future feature with probability 0.5;
- a uniformly random dataset feature with probability 0.3.

The future branch samples its temporal offset geometrically with parameter
\(1-\gamma\), truncated at the trajectory terminal state.

### Navigator

The high-level navigator is a Gaussian policy
\(\pi_h(z\mid e,g)\). Offline, it is trained by advantage-weighted regression toward the
bottleneck of a realized future waypoint:

\[
\hat z_t=\phi(e_t,e_{w_t}), \qquad
\mathcal L_h=-\mathbb E[
\operatorname{clip}(\exp(\beta A_t),100)
\log\pi_h(\hat z_t\mid e_t,g)].
\]

For the headline dual-arm configuration, the waypoint distance is \(k=15\), and the
clamp-to-goal sampler uses \(w_t=\min(t+k,t_g)\) for a same-trajectory goal. The
high-level random-goal probability is 0.3 and the offline advantage aggregates the two
value heads by their mean.

At deployment, the fixed task goal is \(g_\star\), and the waypoint is the navigator
mean projected to the bottleneck sphere:

\[
z_t=\sqrt{d_z}\,
\frac{\mathbb E[\pi_h(\cdot\mid e_t,g_\star)]}
{\|\mathbb E[\pi_h(\cdot\mid e_t,g_\star)]\|_2+\epsilon}.
\]

### Actor and critic interfaces

Correct the main paper to match the implementation:

- residual actor:
  \(\pi_{\mathrm{res}}(a_t^{\mathrm{res}}\mid
  o_{\le t},a_t^b,z_t)\);
- TD3 critic:
  \(Q(o_{\le t},z_t,a_t)\), where \(a_t\) is the executed base-plus-residual action.

The critic does not receive the base action as a separate input. Both actor and critic
use their trainable multi-view encoder, which is distinct from the frozen base feature
encoder used to construct \(e_t\).

The reward path remains functionally separate in the limited sense that neither the
stage detector nor the single-state potential constructs or scores the waypoint. The
paper must not equate this separation with an actor-only waypoint interface.

### Online joint update and replay mixtures

In waypoint-enabled dual-arm runs:

- the bottleneck encoder \(\phi\) remains frozen;
- the two goal-conditioned value heads and \(\pi_h\) are updated;
- updates start after 2,000 online transitions;
- the value/navigator batch size is 256 with a 50/50 online/demonstration split;
- online goals use geometric future sampling;
- the online learning rates are \(10^{-5}\);
- one joint value/navigator update is performed per environment decision after the
  residual learner's UTD block, not once per individual critic update.

This mixture is distinct from the residual TD3/RLPD batch, which also uses a 50/50
online/demonstration split but has its own replay buffers and objectives.

### Frozen single-state potential and replay semantics

Define the separate single-state value \(V_p\) as an offline expectile value trained
from demonstrations with terminal reward 1 and intermediate reward 0. It is frozen
throughout residual RL.

The potential is:

\[
\Phi(e)=cV_p(e), \qquad
c=\frac{\max(K-1,1)}
{\max(V_{\max}-V_{\min},10^{-6})}.
\]

The potential-shaped reward is:

\[
\tilde r_t=r_t+b[\gamma(1-d_t)\Phi(e_{t+1})-\Phi(e_t)].
\]

The implementation sets the next potential to zero for both termination and
truncation. For online data, the shaped reward is computed during collection before the
transition is inserted into replay. For demonstration data, it is computed once when
the offline replay buffer is built. Because \(V_p\) is frozen, replay sampling does not
recompute or update shaped rewards.

Remove the gradient-update index \(u\), the claim that \(V_p\) is jointly updated, and
the corresponding non-stationary-potential disclaimer from the main paper.

## Supplement Revision

The supplement already contains most implementation details. Revise only factual
inconsistencies and omissions:

- ensure all actor/critic interface statements agree with the implementation;
- state that online navigator updates use fixed-\(k\) waypoint batches with geometric
  goal sampling, while offline pretraining uses clamp-to-goal targets;
- replace “one joint update per residual-critic update” with the implemented schedule:
  one joint update per environment decision after the four-update residual UTD block;
- state explicitly that online and offline potential-shaped rewards are materialized
  before replay insertion and never recomputed at replay sampling;
- ensure the fixed \(V_p\), frozen \(\phi\), and trainable value-head/navigator sets are
  identical in Sections A, B, and C;
- preserve all unresolved experimental TODOs without inventing values.

## Figure 1 Constraint

Do not edit:

- the Figure 1 image;
- its SVG/PDF/PNG/PPTX sources;
- its caption solely to force graphical consistency.

The surrounding prose may become more accurate than the current figure. The authors
will replace Figure 1 separately.

## Verification

1. Compile `paper/main.tex` to `paper/main.pdf`.
2. Compile `paper/aaai2027-unified-supp.tex` to its PDF.
3. Confirm both compilations exit successfully and report no undefined references.
4. Extract text from both PDFs and scan for stale claims:
   - waypoint enters only the actor;
   - the critic does not receive the waypoint;
   - \(V_p\) is updated during residual RL;
   - shaped potential rewards are recomputed from a changing value.
5. Scan for the required new facts:
   - medoid terminal goal;
   - 0.2/0.5/0.3 goal mixture;
   - geometric future sampling;
   - \(k=15\);
   - frozen \(\phi\), trainable value heads and navigator;
   - 50/50 online/offline joint-update batch;
   - frozen \(V_p\);
   - reward materialization before replay insertion.
6. Check page count and visually inspect every modified main-paper page for overflow,
   broken equations, or displaced figures.

## Acceptance Criteria

- The main paper answers each reproducibility concern without requiring source-code
  inspection.
- Every changed scientific statement matches the headline implementation.
- Main paper and supplement agree on actor/critic inputs, trainable/frozen modules,
  sampling, update schedules, and reward storage.
- Figure 1 and its assets are untouched.
- Both PDFs compile cleanly and remain readable.
