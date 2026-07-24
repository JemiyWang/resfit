# Block Mixed-Replay Fixed BC Weight Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `launch_block_imagination.sh` keep the demo-BC coefficient fixed at `0.1` for the full Block mixed-replay run.

**Architecture:** Use the Trainer's existing fixed-weight behavior: retain `--demo_bc_coef 0.1` and omit `--bc_coef_final`. No Trainer logic, replay construction, cache signature, or other launcher changes are required.

**Tech Stack:** Bash launcher, Python `argparse`, `bash -n`, ripgrep.

## Global Constraints

- Modify only `launch_block_imagination.sh`.
- Keep `--demo_bc_coef 0.1`.
- Remove `--bc_coef_final 0.01`.
- Do not change BC loss computation, mixed replay batch sizes, Trainer defaults, or Offline cache behavior.
- Do not restart any experiment as part of this configuration edit.

---

### Task 1: Fix the Block launcher's BC coefficient

**Files:**
- Modify: `launch_block_imagination.sh:32`
- Test: launcher source assertions, shell syntax, and Trainer argument parsing

**Interfaces:**
- Consumes: `train_chunk_residual.build_parser()`, where omitted `--bc_coef_final` parses as `None`.
- Produces: a Block launcher whose effective BC configuration is `demo_bc_coef=0.1` and `bc_coef_final=None`.

- [ ] **Step 1: Run the source assertion before editing and verify it fails**

Run:

```bash
test "$(rg -o -- '--demo_bc_coef 0\\.1' launch_block_imagination.sh | wc -l)" -eq 1 \
  && ! rg -q -- '--bc_coef_final' launch_block_imagination.sh
```

Expected: non-zero exit status because the launcher currently contains `--bc_coef_final 0.01`.

- [ ] **Step 2: Make the minimal launcher change**

Replace:

```bash
  --demo_bc_coef 0.1 --bc_coef_final 0.01 \
```

with:

```bash
  --demo_bc_coef 0.1 \
```

- [ ] **Step 3: Re-run the source assertion**

Run:

```bash
test "$(rg -o -- '--demo_bc_coef 0\\.1' launch_block_imagination.sh | wc -l)" -eq 1 \
  && ! rg -q -- '--bc_coef_final' launch_block_imagination.sh
```

Expected: exit status `0`.

- [ ] **Step 4: Validate Bash syntax**

Run:

```bash
bash -n launch_block_imagination.sh
```

Expected: exit status `0` with no output.

- [ ] **Step 5: Validate the effective Trainer defaults**

Run:

```bash
PYTHONPATH=/mnt/mnt/data/resfit /mnt/mnt/data/envs/residual/bin/python -c \
'from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser; a = build_parser().parse_args(["--demo_bc_coef", "0.1"]); assert a.demo_bc_coef == 0.1 and a.bc_coef_final is None; print("demo_bc_coef=0.1 bc_coef_final=None")'
```

Expected:

```text
demo_bc_coef=0.1 bc_coef_final=None
```

- [ ] **Step 6: Check the focused diff**

Run:

```bash
git diff --check -- launch_block_imagination.sh
git diff -- launch_block_imagination.sh
```

Expected: no whitespace errors, and the only launcher change is removal of `--bc_coef_final 0.01`.

- [ ] **Step 7: Commit the launcher change**

```bash
git add launch_block_imagination.sh
git commit -m "config: keep block demo BC weight fixed"
```

