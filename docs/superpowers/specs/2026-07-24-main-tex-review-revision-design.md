# SHORE-RL Main-Text Revision Design

## Scope

Revise only `paper/main.tex` in response to the 24 review comments. Preserve
all reported experimental values, figure assets, bibliography entries, and
unrelated working-tree changes. Do not claim that every baseline improves
before collapsing, do not introduce a formal causal theorem, and do not redraw
or rerun experiments in this text-editing pass.

## Central Claim and Evaluation

Use one paper-level claim throughout:

> SHORE-RL achieves high and sustained success rates on long-horizon
> manipulation tasks, while existing RL-finetuning methods collapse below the
> frozen base policy.

Treat late-training collapse as the observed temporal failure pattern. Use
final-window success rate as the quantitative summary of sustained
performance. Learning curves provide the temporal evidence for collapse.
Remove the competing initial-to-final-drop and peak-success framing.

Contribution 1 combines the empirical long-versus-short-horizon observation
with a bounded mechanism analysis: sparse delayed credit gives each residual
correction weak local evidence, while critic error and residual drift can
accumulate over a long rollout. State this as a design explanation rather than
an independently established unique cause. Contribution 2 emphasizes that
SHORE-RL is the first RL-finetuning method in this related-work scope to
condition a frozen-base residual on latent waypoints, and that it is a
base-policy-agnostic plug-in for frozen cloned policies. Contribution 3
summarizes empirical coverage.

## Structural Revisions

- Reduce SHORE-RL-specific detail in the third Related Work paragraph and
  expand the temporal-abstraction gap: prior waypoint methods pair a hierarchy
  with a learned low-level policy, whereas existing frozen-policy RL
  finetuning does not use waypoints to shorten the residual objective horizon.
- Compress the world-model Related Work paragraph because it supports only an
  extension.
- Keep Preliminaries limited to the control problem, frozen-policy residual
  adaptation, and goal-conditioned-value background. Remove duplicated text
  and make all symbols locally defined.
- Cite Figure 1 in the opening Method paragraph.
- Keep online joint finetuning in Section 4.2, where the goal-conditioned value
  and navigator are defined, but integrate it into the subsection rather than
  present it as an isolated method component.
- Define the training-update index `u` before use. Distinguish
  `V_{\mathrm{gc}}` (waypoint selection) from `V_{\mathrm p}` (potential) once,
  then avoid repeated comparisons.
- Compress Section 4.4 into one natural paragraph describing the real-robot
  learned-dynamics extension.

## Notation Audit

- Define the task-goal representation `g`.
- Express the residual actor as
  `a_t^{\mathrm{res}} \sim \pi_{\mathrm{res}}(\cdot \mid
  o_{\le t},a_t^b,z_t)` before composing the executed action.
- Retain `\pi_b` for the frozen base, `\pi_h` for the navigator,
  `V_{\mathrm{gc}}` for the goal-conditioned value, and `V_{\mathrm p}` for the
  single-state potential value.
- Define `u` as the residual-training update index and retain it only where the
  learned potential changes during training.
- Check every symbol for definition-before-use and every section/figure
  reference for validity.

## Experiment Narrative and Names

- Use `ResFit` as the sole prose name for the flat demo-anchored residual
  baseline. Where Figure 3 itself says “Residual RL,” explain in the caption
  that this legend entry denotes ResFit.
- In Section 5.1, define final-window success once. Rename the main-result
  paragraph to “Sustained Long-Horizon Performance,” coordinate it with Figure
  3 and Table 1, reduce anti-collapse rhetoric, and remove references to later
  ablation figures or tables.
- In Section 5.2, make the first paragraph exclusively a protocol-and-coverage
  description. Remove the incorrect “per-stage bonus as the potential”
  phrasing and give a neutral reason for the selected long-horizon ablation
  tasks.
- In the second ablation paragraph, remove the mismatched `.1--.3` summary,
  replace all “Full” labels with “SHORE-RL,” and discuss final-window
  performance only.
- Update the real-robot task order everywhere to `pick_paper_roll`,
  `build_block`, and `pick_cup`, with the corresponding descriptions ordered
  as rolling-object picking, block stacking, and cup grasping/placement.
- Report approximately 1,000 success-only demonstrations per task.
- Remove the requested extension disclaimer and the TODOs in the real-robot
  pipeline, Figure 7 caption, and Conclusion.

## Caption Contract

- Figure 3: describe the plotted long- and short-horizon curves, seeds and
  uncertainty bands, and map “Residual RL” to ResFit. Do not mention markers
  absent from the rendered figure.
- Table 2: describe the matched component arms using “SHORE-RL,” not “Full,”
  and avoid subsection references.
- Figure 5: use “SHORE-RL,” match the five displayed arms, and retain the
  open-circle explanation because one is visible.
- Figure 6: describe seed-mean curves and uncertainty bands rather than
  “per-seed” curves.
- Figure 7: identify panels as `(a) pick_paper_roll`, `(b) build_block`, and
  `(c) pick_cup`; remove the subsection reference and TODO.

## Conclusion

Attribute late-training collapse to the set of existing RL-finetuning
baselines rather than treating all of them as ResFit. State that the main
simulation configuration uses a hand-designed per-stage bonus, without the
incorrect phrase “as the potential.” Retain the tested self-derived alternative
within its demonstrated scope.

## Verification

After editing:

1. Search for `TODO`, `Full`, `initial-to-final`, `peak success`, inconsistent
   task orders, old residual-baseline names, and stale subsection references.
2. Compile with the repository's documented LaTeX sequence.
3. Inspect warnings for undefined references/citations and malformed math.
4. Review the resulting diff against all 24 comments and confirm no unrelated
   files changed during implementation.
