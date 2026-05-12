# BenchLoop Spec

## Summary

BenchLoop is a local-first benchmarking system for evaluating open-source and local LLMs on real hardware and through real harnesses.

It has two products:
1. A downloadable CLI that runs benchmarks locally against Ollama, llama.cpp-compatible servers, and selected harnesses.
2. A web app that aggregates uploaded benchmark runs into a community leaderboard.

The core differentiator is that BenchLoop measures both:
- raw model capability
- model performance when attached to real harnesses like OpenClaw, Hermes Agent, Aider, and Continue

That means BenchLoop answers questions existing leaderboards do not answer, such as:
- Which 27B model works best on a 4090?
- Which model performs best in Aider vs OpenClaw?
- What is the best speed-to-quality tradeoff on my machine?
- Does a model that scores well raw also score well inside an agent harness?

## Product Goals

### Primary goals
- Benchmark local models on local hardware with reproducible methodology
- Compare raw model performance and harness-mediated performance
- Combine speed, quality, and reliability into one results format
- Make results easy to compare across model, quantization, machine, and harness
- Create a public leaderboard with community-submitted results

### Non-goals for V1
- Benchmark every proprietary frontier model
- Support every harness at launch
- Build a giant enterprise eval platform
- Replace academic benchmark suites in full

## User stories

### Local power user
As a local model user, I want to run one command and learn which model is best for my hardware and workflow.

### Agent builder
As an agent developer, I want to know whether a model works better through OpenClaw, Hermes Agent, Aider, or direct API.

### Open-source community contributor
As a contributor, I want to run BenchLoop locally and upload my results to a public leaderboard.

### OpenClaw / Hermes operator
As a platform operator, I want first-class benchmark coverage for OpenClaw and Hermes as harnesses.

## Market gap

Existing benchmark properties are fragmented:
- HuggingFace leaderboard: academic quality only
- LMSYS Arena: crowdsourced preference only
- Artificial Analysis: API model speed, price, and quality only
- BFCL: tool-calling only
- Aider leaderboard: Aider-specific coding benchmark only
- geerlingguy / ollama benchmark projects: speed only

No existing product combines:
- local hardware
- local models
- speed + quality
- harness comparison
- tool use / agent loop behavior
- community-submitted machine-specific results

## Core concept

BenchLoop evaluates a tuple:

`model x quantization x machine x harness x test suite`

Examples:
- qwen3.5-27b-q4 on PC1 through raw Ollama
- qwen3.5-27b-q4 on PC1 through OpenClaw
- qwen3.5-27b-q4 on PC1 through Hermes Agent
- qwen3.5-27b-q4 on PC1 through Aider

## Product surfaces

## 1. CLI

### Example commands
```bash
bench-loop run --model qwen3.5:27b-q4 --provider ollama --machine pc1
bench-loop run --model qwen3.5:27b-q4 --provider ollama --machine pc1 --harness raw
bench-loop run --model qwen3.5:27b-q4 --provider ollama --machine pc1 --harness openclaw
bench-loop run --model qwen3.5:27b-q4 --provider ollama --machine pc1 --harness hermes
bench-loop run --model qwen3.5:27b-q4 --provider ollama --machine pc1 --harness aider
bench-loop compare results/*.json
bench-loop dashboard --input results/
bench-loop submit results/latest.json
```

### CLI responsibilities
- discover runtime target
- execute benchmark suites
- collect timing and pass/fail metrics
- save a normalized JSON artifact
- optionally open a local dashboard
- optionally upload validated results to a hosted service

## 2. Web app

### Responsibilities
- accept result uploads
- validate schema and metadata
- deduplicate or version repeated runs
- expose public leaderboard views
- support filtering by hardware, harness, model family, quantization, and date
- visualize speed, quality, and reliability
- highlight best combinations rather than just best raw models

## V1 scope

### Harnesses in V1
Priority order:
1. `raw` - direct Ollama / OpenAI-compatible endpoint baseline
2. `openclaw` - OpenClaw as a harness
3. `hermes` - Hermes Agent as a harness
4. `aider` - Aider as a harness

Stretch for V1.1:
- `continue`
- `opencode`
- `goose`

Do not target VS Code automation first. Cline / Roo / Cursor-like harnesses are valuable but harder to automate reliably.

## Architecture

### High-level architecture

```text
bench-loop/
  bench_loop/
    cli.py
    config.py
    models.py
    runner/
      orchestrator.py
      scorer.py
      result_writer.py
      environment.py
    providers/
      base.py
      ollama.py
      openai_compat.py
    harnesses/
      base.py
      raw.py
      openclaw.py
      hermes.py
      aider.py
    suites/
      base.py
      speed.py
      coding.py
      tool_calling.py
      reasoning.py
      instruction_following.py
      structured_output.py
      agent_loop.py
    tasks/
      coding/
      tool_calling/
      reasoning/
      instruction_following/
      structured_output/
      agent_loop/
    validators/
      json_schema.py
      file_diff.py
      answer_match.py
    sandbox/
      docker_exec.py
      subprocess_exec.py
    report/
      local_dashboard.py
      markdown.py
  results/
  docs/
  tests/
  pyproject.toml
  README.md
```

### Core modules

#### Provider layer
Handles direct model transport.

V1 provider support:
- Ollama API
- generic OpenAI-compatible chat/completions endpoint

Responsibilities:
- send prompt or message list
- capture timing fields when available
- normalize metadata
- expose token counts if provider returns them
- retry transient failures conservatively

#### Harness layer
Each harness adapter implements a standard interface.

```python
class HarnessAdapter:
    name: str

    def setup(self, config: HarnessConfig) -> None:
        ...

    def run_task(self, task: BenchmarkTask) -> HarnessRunResult:
        ...

    def teardown(self) -> None:
        ...
```

`HarnessRunResult` should include:
- final text output
- structured output if applicable
- tool calls attempted
- files changed
- execution transcript or summary
- start/end timestamps
- latency breakdown when available
- failure details

#### Suite layer
A suite defines a family of tasks and a scoring method.

```python
class BenchmarkSuite:
    name: str

    def tasks(self) -> list[BenchmarkTask]:
        ...

    def evaluate(self, run_result: HarnessRunResult, task: BenchmarkTask) -> TaskScore:
        ...
```

#### Runner / orchestrator
Coordinates:
- warmup runs
- repeated trials
- suite execution
- score aggregation
- result persistence

## Benchmark suites

### 1. Speed suite
Purpose: measure raw inference performance and latency.

Metrics:
- TTFT
- prompt eval tokens/sec
- generation tokens/sec
- total latency
- sustained output speed at different target lengths
- cold start load time when detectable

Task shapes:
- short output
- medium output
- long output

Notes:
- run 3 times, discard first run if marked as warmup
- report median and variance

### 2. Coding suite
Purpose: measure practical code generation and code-editing quality.

Task types:
- implement function from spec
- repair broken code
- refactor for correctness
- write unit-tested endpoint or parser

Evaluation:
- run tests in sandbox
- pass/fail correctness
- optional style / lint bonus not required for V1

Inspiration:
- Carlini practical benchmark
- EvalPlus / HumanEval+ patterns
- Aider editing workflows

### 3. Tool-calling suite
Purpose: evaluate schema-correct tool use and argument correctness.

Task types:
- choose correct tool
- produce valid function call JSON
- multi-turn tool use with state
- abstain when no tool should be called

Evaluation:
- schema validation
- exact / semantic arg validation
- call sequencing for multi-turn tasks
- abstention accuracy

Inspiration:
- BFCL

### 4. Reasoning suite
Purpose: evaluate short practical reasoning, not giant academic sweeps.

Task types:
- arithmetic / GSM-like
- small logic chains
- planning problems
- grounded reasoning from provided context

Evaluation:
- exact match or rule-based extraction

### 5. Instruction-following suite
Purpose: test compliance with formatting and behavioral constraints.

Task types:
- produce exact format
- obey word count
- obey forbidden token constraints
- answer in valid list or JSON shape

Evaluation:
- rule checks, not subjective judging

### 6. Structured-output suite
Purpose: test valid JSON / YAML generation from a schema or contract.

Evaluation:
- parse success
- schema validation
- required field accuracy
- nesting correctness

### 7. Agent-loop suite
Purpose: benchmark end-to-end harness behavior.

This is the most important differentiator for harness comparison.

Task types:
- inspect a directory and answer a question
- read file, patch file, explain change
- multi-step tool sequence
- constrained agent task with 2-4 steps

Evaluation:
- task completion
- correctness of final output
- tool efficiency
- unnecessary tool-call penalty
- failure recovery behavior

## Harness details

### Raw harness
- direct provider call
- no extra system prompt beyond suite-defined instructions
- baseline for all models

### OpenClaw harness
- route tasks through OpenClaw chat/completions path
- where possible, use standardized tool definitions for tool and agent-loop tasks
- capture OpenClaw-specific metadata if available

Primary purpose:
- test how models behave when mediated by OpenClaw tool use and context handling

### Hermes harness
- route through Hermes Agent server / gateway path
- exercise tool calling and multi-step execution

Primary purpose:
- compare Hermes orchestration vs OpenClaw orchestration for same underlying model

### Aider harness
- run scripted Aider tasks inside temporary repos
- focus mainly on coding suite and file-edit workflows
- optionally skip suites not meaningful for Aider

Primary purpose:
- compare coding utility, not generic chat quality

## Result schema

Each benchmark run should emit one normalized JSON file.

### Result metadata
- benchmark version
- timestamp
- machine id
- machine hardware summary
- OS
- model id
- model family
- quantization
- provider
- harness
- harness version if known
- suite versions
- total runtime

### Machine fields
- cpu model
- gpu model
- gpu memory
- system memory
- OS
- backend type

### Aggregates
- overall score
- quality score
- speed score
- reliability score
- value score

### Per-suite outputs
For each suite:
- suite score
- task count
- pass count
- failure count
- median latency
- raw task-level records

### Task-level outputs
- task id
- prompt id / fixture id
- input metadata
- output metadata
- score
- pass/fail
- latency
- error if any
- artifact references if any

## Scoring model

V1 scoring should be simple and explainable.

### Aggregate dimensions
- `quality_score`: normalized average of non-speed suites
- `speed_score`: normalized performance score from speed suite
- `reliability_score`: success rate adjusted for crashes, malformed outputs, or harness failures
- `value_score`: combined usefulness metric

Suggested first-pass formula:

```text
overall_score = 0.55 * quality_score + 0.20 * speed_score + 0.25 * reliability_score
```

Optional derived metric:

```text
value_score = quality_score * normalized_generation_speed * reliability_multiplier
```

Important:
- keep formulas transparent
- show raw metrics beside aggregate scores
- never hide the ingredients

## Reproducibility rules

- fixed prompt/task fixtures checked into repo
- deterministic settings by default where possible
- fixed temperature for each suite
- explicit max_tokens
- explicit context size when relevant
- explicit number of trials
- always record software versions
- mark results as comparable only when suite version and core settings match

## Local dashboard

V1 should ship a local dashboard generated from JSON results.

### Local dashboard views
- leaderboard table
- model detail page or panel
- harness comparison chart
- category heatmap
- machine comparison chart
- raw task drilldown

### Recommended implementation
- static HTML generated from results JSON
- lightweight JS charts
- no server required for local mode

## Web app spec

### Core pages
1. Homepage leaderboard
2. Compare page
3. Model detail page
4. Hardware detail page
5. Harness detail page
6. Upload / submit page
7. Trending models page

### Core filters
- GPU / hardware class
- RAM / VRAM bucket
- model family
- parameter size
- quantization
- harness
- benchmark suite
- date window

### Community features for later
- user profiles
- machine profiles
- verified submissions
- comments / notes on models
- saved comparisons
- weekly trending and new model alerts

## Submission flow

### Local -> hosted
1. CLI generates result JSON
2. CLI validates against schema
3. CLI uploads JSON to BenchLoop API
4. API re-validates and stores canonical row set
5. Result appears on leaderboard after passing checks

### Anti-junk protections
- schema validation
- benchmark version checks
- duplicate run detection
- optional signature / hash of fixture set
- mark runs as unverified or verified

## Revision program: BenchLoop v2

BenchLoop v1 proved the CLI and fixture-loading flow. BenchLoop v2 is the cleanup and differentiation pass.

The immediate priority is not more harnesses. The immediate priority is:
1. make every shipped benchmark unmistakably original
2. close the missing-pack gaps with stronger artifact verification
3. lock the scoring and verifier story before visual polish
4. then layer in the FrankenGPU-inspired design system
5. only then shift back to harness-comparative benchmarking

### Phase 1: benchmark cleanup and originality pass

Goal: remove anything that looks derivative, incomplete, or redundant.

Deliverables:
- remove or hide empty legacy surfaces (`coding`, `tool_use`) from user-facing docs and defaults
- rename packs away from public parallel naming where needed
- rewrite prompts, domains, fixture text, and IDs for all current non-speed suites
- standardize task metadata across suites
- explicitly mark every task with verifier type and capability tags

Required metadata fields per task:
- `title`
- `difficulty` (`easy` | `medium` | `hard`)
- `capability_tags`
- `verifier_type` (`exact` | `tolerance` | `json_schema` | `artifact` | `tool_trace`)
- `expected_turns` when applicable
- `notes` for pack-specific evaluator guidance

Phase 1 exit criteria:
- no empty shipped suites remain visible
- all current suite IDs and prompts are original to BenchLoop
- task counts are intentional, not inherited by habit
- README and CLI surfaces reflect the new suite names

### Phase 2: missing pack build-out

Goal: fill the real benchmark gaps before doing any harness matrix work.

New suites to add:
- `loop_code_fix` (artifact-verified bug fixing)
- `loop_schema` (strict structured output)

Suite targets:
- `loop_code_fix`: 10-12 tasks, all with real failing fixtures and executable verifiers
- `loop_schema`: 10-12 tasks, all with parse + schema validation + field-level accuracy checks

Design principles:
- no judge-model scoring for official results
- no toy "hello world" coding tasks unless they represent a real workflow failure mode
- each task must fail for a specific, inspectable reason
- every pass must leave an auditable artifact

Phase 2 exit criteria:
- both suites run end-to-end locally
- each suite has smoke-tested fixtures and deterministic scoring
- BenchLoop can produce at least one clean, publishable run artifact on a local model

### Phase 3: verifier rigor and scoring hardening

Goal: make the benchmark trustworthy before making it pretty.

Deliverables:
- unify suite scoring language around transparent pass / partial / fail semantics
- expose failure reasons in task-level output
- capture turn count, tool count, token usage, and retry count where available
- ensure aggregate scoring formulas match the documented methodology exactly
- add variance reporting for repeated runs where applicable

Verifier requirements:
- artifact verification preferred over prose comparison
- trace invariants allowed, exact tool sequence matching discouraged
- partial credit only when the final outcome actually happened and supporting evidence is incomplete
- headline tables should treat partial as diagnostic, not a full pass

Phase 3 exit criteria:
- per-task failure reasons are inspectable in reports
- aggregate scores are reproducible and documented
- one full benchmark run can be defended methodologically without hand-waving

### Phase 4: design refresh (FrankenGPU-inspired)

Goal: apply the visual system only after the benchmark substance is real.

Visual direction:
- dark background
- green accent palette
- hacker / lab instrumentation aesthetic
- terminal-forward presentation
- stronger topology / telemetry vibe borrowed from FrankenGPU, but cleaner and more productized

Design surfaces in order:
1. CLI output theming
2. local HTML report styling
3. historical comparison dashboard
4. later, public web leaderboard styling

Phase 4 exit criteria:
- the CLI already feels branded in screenshots
- local reports match the FrankenGPU design language
- charts emphasize truth and diagnostics, not generic leaderboard chrome

### Phase 5: harness benchmarking framework

Goal: introduce BenchLoop's actual moat after the benchmark core is solid.

This phase adds a harness-comparative benchmark layer rather than cloning any existing agent pack.

Deliverables:
- capability-tag contract for harness-compatible tasks
- harness adapter contract for raw, OpenClaw, Hermes, Aider, and future harnesses
- BenchLoop-native harness benchmark pack(s)
- side-by-side matrix reporting: model × harness × suite

Rules:
- learn from external methodology, do not clone public pack prompts or IDs
- keep tasks harness-agnostic where possible
- use artifact verification and trace invariants for harness-native tasks
- report harness overhead and reliability separately from raw model quality

Phase 5 exit criteria:
- at least one BenchLoop-native harness suite runs across multiple harnesses
- comparison reports clearly show harness deltas on the same underlying model
- BenchLoop has a differentiated story that is not just "another pack host"

## Immediate execution order

### Now
- write the v2 spec and lock the phase boundaries
- clean up legacy suite naming and empty surfaces
- rewrite current prompt fixtures to be unmistakably BenchLoop

### Next
- build `loop_code_fix`
- build `loop_schema`
- harden scoring and failure reporting

### After that
- apply the FrankenGPU visual system
- then resume harness benchmarking work

## Recommended initial task fixtures

Keep V1 small but representative.

### Speed
- 3 prompt lengths x 3 output targets

### Coding
- 8-12 tasks
- half generate-from-scratch
- half repair/edit

### Tool-calling
- 8-12 tasks
- single-tool, multi-tool, abstain, multi-turn

### Instruction / structured / reasoning
- 5-8 tasks each

### Agent loop
- 4-6 tasks initially

Total V1 target:
- around 35-50 tasks

That is enough to be useful without turning each run into an all-day event.

## Open questions

- Should OpenClaw and Hermes harness tasks be run against the exact same tool schema fixtures, or allow harness-specific variants when needed?
- Do we want Docker mandatory for code execution, or local subprocess fallback for convenience?
- Should the web app support anonymous result uploads at launch, or require an API key / auth?
- Should model naming be normalized centrally to avoid duplicate entries for equivalent quantized aliases?

## Recommendation

For V1:
- use Python for the CLI and orchestration
- keep fixtures in YAML or JSON
- keep scoring transparent and simple
- prioritize raw, OpenClaw, Hermes, and Aider
- ship local dashboard before hosted app
- make hosted app a second step, not a blocker

## Immediate build brief

Build the local-first BenchLoop CLI first.

Must-have deliverables:
1. Python package scaffold
2. Ollama provider
3. raw harness
4. speed suite
5. coding suite with executable checks
6. tool-calling suite with schema validation
7. normalized results JSON schema
8. compare/report command
9. clear README with usage examples

Nice-to-have if time allows:
- stub adapters for OpenClaw, Hermes, and Aider
- local HTML dashboard skeleton

## Key principle

BenchLoop should optimize for useful truth, not benchmark theater. The winning product is the one that tells users which model+harness combination actually works on their hardware in the real world.
