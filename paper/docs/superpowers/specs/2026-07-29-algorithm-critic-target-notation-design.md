# Algorithm 1 Critic-Target Notation

## Goal

Shorten line 12 of Algorithm 1 while preserving its central operation: uniformly sample two distinct target critics from the ensemble of ten and use their minimum prediction.

## Design

Replace the expanded n-step target equation with two compact pseudocode statements:

1. Uniformly sample distinct indices `i` and `j` from `{1, ..., 10}`.
2. Define the conservative target estimate as the minimum of target critics `i` and `j`.

The following regression statement will say that the n-step TD target is formed using this minimum estimate before regressing all ten critics. This preserves the training procedure while making the algorithm easier to scan and reducing the width of line 12.

## Validation

Recompile the main paper and check that Algorithm 1 fits without LaTeX errors or undefined references.

## Target-Network Update Notation

Replace the two parameter-update equations on line 16 with the concise statement
“Soft-update critic targets and, on delayed steps, actor targets.” This retains
the delayed actor-target condition while removing implementation-level equations
from the pseudocode.
