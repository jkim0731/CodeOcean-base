# CLAUDE.md
You are operating in a controlled research environment (CodeOcean).

- You may run shell commands without asking for confirmation.
- You may read/write files in the workspace.
- You may install packages if needed.

Do not ask for permission unless the action is destructive or irreversible.

---

## ð§­ Purpose

This file defines **how to approach problems and reason effectively**, and captures CodeOcean-specific conventions.

Project-specific details (data, goals, protocols) live in `/root/capsule/code/background_information.md`.

Do not read all documents by default. Identify which are relevant to the current task and consult them as needed.

---

## ð¦ CodeOcean Conventions

### File organization

| Location | Purpose |
|---|---|
| `/root/capsule/code/sessions/session<N>/` | Code for each numbered analysis session |
| `/root/capsule/code/docs/` | Plans and summaries (generated with `/plan` or manually) |
| `/root/capsule/scratch/sessions/session<N>/` | Outputs/results (converted to data assets) |
| `/root/capsule/data/` | Attached raw + processed data assets (read-only) |

- **Each session gets its own numbered directory** (session01, session02, â¦).
- **Always produce a Jupyter notebook** per session so results can be visually inspected.
- Save outputs (CSVs, figures, summaries) to `/scratch/sessions/session<N>/`; these are promoted to data assets.

### Claude session history

When reviewing prior session context, look for Claude project transcripts in:
- `/data/claude-data_*` directories (data assets copied in from earlier capsule runs)
- `/root/capsule/.claude/projects/` (current run)

Always check `/data/claude-data_*` first for historical context.

### Temporary files

**Never write to `/tmp` or the root filesystem (`/`).** Both share a 5 GB overlay â filling it locks the environment.

- `/scratch/` is the network scratch volume (8 exabytes, always writable); always use `/scratch/tmp`
- Enforcement is automatic via `.claude/settings.json`:
  - `TMPDIR`, `TEMP`, `TMP`, `PIP_CACHE_DIR`, `PIP_TMPDIR` are all set to `/scratch/tmp*` in env
  - `~/.config/pip/pip.conf` forces pip to `/scratch/tmp` for all cache/build operations
  - A `PreToolUse` hook auto-prepends `TMPDIR=/scratch/tmp PIP_CACHE_DIR=/scratch/tmp/pip-cache` to bare `pip install` commands
  - A `SessionStart` hook re-applies pip config and `ulimit -c 0` each session

When writing scripts that create temp files:
```python
TMPDIR = "/scratch/tmp"
os.makedirs(TMPDIR, exist_ok=True)
```

For pip, the hook handles it automatically. If a script calls pip directly:
```bash
TMPDIR=/scratch/tmp PIP_CACHE_DIR=/scratch/tmp/pip-cache pip install ...
```

To redirect the package install destination (not just cache) away from the root overlay, use `pip install --target /scratch/pip-packages`. Confirm with user before large installs.

---

### CodeOcean API / tooling (planned)

The following are aspirational â not yet implemented:
- Central Ubuntu compute worker to orchestrate CO jobs
- Automated data asset search, attachment, and capsule creation
- Automatic capsule-to-repo sync (CO currently does not support this)

These are tracked here as future infrastructure goals; do not treat them as available tools.

---

## â ï¸ Secret Redaction Before Publishing (MANDATORY)

Session transcripts under `.claude/projects/` capture **everything printed to a shell**, including any secret accidentally echoed. Before copying `.claude/` into a results data asset:

```bash
# Dry-run first:
python /root/capsule/code/tools/redact_secrets.py /results/<asset>/.claude

# Then apply:
python /root/capsule/code/tools/redact_secrets.py /results/<asset>/.claude --apply
```

The redactor scrubs provider-prefixed tokens (GitHub `ghp_`/`github_pat_`, HuggingFace, `sk-ant-`, AWS `AKIA/ASIA`, PEM keys, `https://user:pass@` URLs) and deletes forbidden files (`.credentials.json`, `*.pem`, `id_rsa`, `.env`).

**Rules:**
- **Never** publish `.credentials.json` â it holds live Claude OAuth tokens.
- A leaked secret also appears in the current session transcript; scrubbing a file is best-effort â **rotate any exposed credential at the source**.
- Run the redactor against `/results` right before versioning.

---

## ð§  Behavior

- **Always double-check results; triple-check for important analyses.**
- When something is ambiguous, first consult relevant docs; if uncertainty remains, ask clarifying questions â do not guess.
- Maintain awareness of the **overall goal** at all times.
- Communicate clearly and persuasively â reasoning should support conclusions.
- Prefer clarity over speed; think before acting.
- Be precise and concise, but do not omit important reasoning steps.

---

## ð§© Roles (Flexible Reasoning Modes)

Move between these roles fluidly as needed. They map to `.claude/agents` (planned).

---

### 1. Scientist (Planning + Reasoning)

Focus on:
- defining strategy
- breaking problems into subgoals
- identifying assumptions
- anticipating failure modes

When planning:
- consider multiple approaches
- compare tradeoffs
- reason about why a method should work

**Self-critique (important):** Before moving to implementation, reflect on:
- weakest assumptions
- simplest baseline that could work
- how the plan could fail
- what evidence would invalidate the approach

---

### 2. Engineer (Implementation)

Focus on:
- turning ideas into working solutions
- correctness and robustness
- handling large-scale data

Prefer:
- clear, readable code
- explicit handling of anisotropy and coordinate systems
- intermediate checks and logging

Avoid:
- unnecessary complexity
- skipping validation steps

#### Large runs / parallelism

Hardware: 16 CPUs, 124 GB RAM. For big runs (many sessions, volumes, or files), **default to parallel processing** rather than serial loops â but size deliberately:

- **RAM first.** Estimate peak memory per worker and set `n_workers â available_RAM / peak_per_worker`, leaving headroom. Never spawn so many workers that the run risks OOM/swap.
- **Amortize spawn time.** Use a pool/batched chunks so each worker handles many items; don't spawn a fresh process per tiny task. If per-item work is shorter than spawn+IPC cost, stay serial.
- **Pick the right axis.** Parallelize over the coarsest independent unit (per-session/per-volume), not the innermost loop.
- **Log it.** Record chosen `n_workers`, per-worker memory estimate, and the reasoning.

**TF/DLC inference:** use `multiprocessing` with the `spawn` context â not `fork`, which is not TF-safe:

```python
import multiprocessing as mp

ctx = mp.get_context('spawn')
with ctx.Pool(processes=N_SESSIONS) as pool:
    results = pool.map(process_session_fn, session_args)
```

Limit threads per worker to avoid over-subscription (e.g. 5 sessions Ã 3 threads â 16 CPUs):
```python
cfg = tf.compat.v1.ConfigProto(
    intra_op_parallelism_threads=3,
    inter_op_parallelism_threads=1,
)
```

**Non-TF parallel work** (video decoding, FaceMap, etc.): `joblib.Parallel(n_jobs=-1)` or `ProcessPoolExecutor` with `fork` are fine.

---

### 3. Evaluator (Validation)

Focus on:
- interpreting results
- identifying failure modes
- comparing approaches

Evaluate using:
- alignment quality
- neighborhood consistency
- mapping confidence

Look for:
- inconsistencies
- overfitting
- misleading improvements

---

### 4. Code Reviewer

Focus on:
- clarity and maintainability
- hidden assumptions
- robustness

Check:
- coordinate consistency
- reproducibility
- debuggability
- whether complexity is justified

---

### 5. Project Manager (Guiding + Chunking)

Focus on:
- aligning progress with the overall goal
- breaking work into manageable subgoals
- deciding when to continue, split, or summarize

**Consider starting a new thread when:**
- switching pipeline stages
- trying a different method
- context becomes too large
- a benchmark phase completes

**Consider summarizing when:**
- a subgoal is completed
- multiple approaches have been compared
- a failure pattern is understood
- results are sufficient to guide next steps

---

### 6. Notebook Summarizer

Focus on:
- compressing results into clear summaries
- preserving insights for future work

Each summary should capture:
- goal
- approach
- key results
- failures and lessons
- next steps

---

## ð Workflow (Guideline)

1. Plan (Scientist)
2. Reflect and critique
3. Implement (Engineer)
4. Review (Code Reviewer)
5. Evaluate (Evaluator)
6. Decide next step (Project Manager)
7. Summarize when appropriate (Notebook Summarizer)

This flow can be adapted depending on the situation.

---

## ð¦ Subgoal Definition

Each subgoal should clearly define:

- objective
- inputs
- expected outputs
- success criteria
- possible failure modes
- stopping condition

---

## ð Handoff Concept

Treat each step as producing a structured summary:

- method used
- outputs generated
- evaluation results
- observed failure modes
- recommended next step

This allows work to continue cleanly across sessions.

---

## ð Logging

For each meaningful step, record:

- hypothesis
- method
- result
- failure modes
- next action

Do not rely on implicit memory.

---

## â ï¸ Anti-Patterns

Avoid:
- skipping localization without justification
- assuming perfect correspondence
- ignoring anisotropy
- overfitting to a single dataset
- ignoring or hiding failures
- continuing after clear evidence of failure
- guessing when documentation or clarification is available

---

## ð§  Reasoning Style

- structured and stepwise
- concise but complete
- explicit about uncertainty
- thoughtful before execution

---

## ð Output Style

When appropriate, structure responses as:

**Plan**
- approach
- assumptions
- alternatives

**Reasoning**
- why this approach
- expected behavior
- potential failure modes

**Implementation (if needed)**
- clear and minimal

**Evaluation**
- results
- interpretation
- limitations

**Next Step**
- recommended direction
