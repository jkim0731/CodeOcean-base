# DFF baseline estimation
## Overview
- This project is to develop protocols to estimate fluorescence signal baseline for calculating dFF.

## Issues
- Previously, people use rolling window percentile to calculate dff from neuropil-corrected fluorescence.
- Inhibitory neurons have high baseline firing rate, often correlated with running and stimulus context, making rolling window protocol insufficient
    - Rolling window estimation cannot handle bleaching, drift (non-monotonic, z-drift-driven), and sustained activities altogether.

## Current protocol
- dff.py has 2-step approach.
    1. Estimate global trend with OLS and IRLS.
    2. LOWESS or percentile approach to deal with sustained activities.

## Problems
- There are many parameters.
- The current protocol still does not seem to solve the problem.
- We haven't found a good metric for baseline fitting quality yet.

## Goal
- Find baseline estimation protocol and parameter set that best follows baseline of fluorescence signal.
    - We may need to find another protocol.
    - QC metric is needed.
- QC the results.
    - Make a QC app.


# Behavior
- Think critically and always double check assumptions, codes, and results.
- Look for ambiguity in prompts and data. If it cannot be resolved, always ask for clarification.
- Focus on visual communication with the user. Use jupyter notebook, images, and plots. Consider step by step explanation.
- Use existing tools whenever possible (mostly in /root/capsule/code and /lamf-analysis).
- Divide each session and log at /code/claude-sessions/ with incremental numbering system and title.
