# Development and Analysis Guide

Use this guide for new tools, methods, notebooks, and analyses. Project details live in
`/root/capsule/code/background_information.md`.

## Working principles

- Keep the overall scientific goal and success criteria explicit.
- Check important results independently; do not infer success from command completion alone.
- Consult relevant documentation before guessing. Surface unresolved uncertainty.
- Prefer the simplest robust method and record assumptions, failure modes, and stopping criteria.
- Make focused, readable changes with intermediate checks and useful logging.
- Before publishing, scrub secrets and remove credential files such as `.credentials.json`, `*.pem`,
  `id_rsa`, and `.env`. Rotate any exposed credential at its source.

## Reasoning modes

Use these as roles, not separate mandatory agents:

- **Scientist:** Define strategy, assumptions, baselines, evidence, and failure conditions.
- **Engineer:** Implement correct, maintainable, scalable solutions.
- **Evaluator:** Interpret results, compare methods, and detect misleading improvements.
- **Reviewer:** Check reproducibility, hidden assumptions, robustness, and justified complexity.
- **Coordinator:** Split work, track dependencies, integrate results, and decide when to stop.
- **Summarizer:** Preserve goals, methods, results, failures, and next steps for later sessions.

## Multiple agents and sessions

- Give each concurrent agent a separate branch and worktree.
- Assign clear, non-overlapping ownership; avoid simultaneous edits to the same files.
- Commit coherent checkpoints and communicate the branch, commit, outputs, and remaining work.
- Integrate completed work through the designated main branch. Merge or rebase at integration
  boundaries rather than continuously rebasing every active worktree.
- Delegate only independent work that benefits from separate context. Validate delegated outputs
  before integration.

## Large runs and parallelism

1. Estimate peak memory per worker and leave headroom.
2. Use pools and batches so useful work exceeds process-start and IPC overhead.
3. Parallelize the coarsest independent unit, such as a session or volume.
4. Record worker count, memory estimate, and rationale.

TensorFlow is unsafe with `fork`; use a `spawn` multiprocessing context:

```python
import multiprocessing as mp

ctx = mp.get_context('spawn')
with ctx.Pool(processes=N_SESSIONS) as pool:
    results = pool.map(process_fn, args)

cfg = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=3,
    inter_op_parallelism_threads=1,
)
```

For non-TensorFlow work, `joblib.Parallel` or `ProcessPoolExecutor` may use `fork` when safe. Avoid
thread oversubscription inside process pools.

## Subgoals and handoffs

Define each subgoal by its objective, inputs, expected outputs, success criteria, failure modes, and
stopping condition. A handoff should state:

- branch or commit and generated outputs
- method and validation performed
- key results, limitations, and failures
- recommended next action

Do not rely on implicit session memory.

## Avoid

- skipping validation or hiding failures
- continuing after evidence invalidates the approach
- overfitting to one dataset
- unnecessary complexity
- ignoring coordinate systems, anisotropy, or reproducibility when relevant
