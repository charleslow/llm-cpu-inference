#!/usr/bin/env bash
set -e

ITERATIONS=${1:-20}
TIMEOUT=$(python -c "from constants import TIMEOUT_SECONDS; print(TIMEOUT_SECONDS)")

# Compute next trial number from summary.jsonl line count (matches run_backend.py logic)
next_trial_num() {
    if [ -f results/summary.jsonl ]; then
        grep -c '[^[:space:]]' results/summary.jsonl 2>/dev/null || echo 0
    else
        echo 0
    fi
}

TRIAL_NUM=$(next_trial_num)

echo "=== cpu-llm-bench: $ITERATIONS iterations ==="
echo "Starting from trial $((TRIAL_NUM + 1))"
echo ""

for i in $(seq 1 "$ITERATIONS"); do
    TRIAL_NUM=$((TRIAL_NUM + 1))
    echo "=========================================="
    echo "=== Iteration $i / $ITERATIONS (trial $TRIAL_NUM) ==="
    echo "=========================================="

    # Phase 1: Analysis — research and plan
    echo "--- Phase 1: Analysis ---"
    claude -p "You are the analysis agent for cpu-llm-bench.

Read these files for context:
- program.md (your mission and rules)
- results/summary.jsonl (all past trial results)
- constants.py (benchmark constants)
- backends/ directory (past backend implementations are backends/trial-{N}-{slug}.py)

If trial $TRIAL_NUM > 1 and the last run failed, also read results/experiment.log for debugging.

Your job:
1. Review what has been tried so far and the results
2. Research what to try next — search the web for: small LLMs suitable for CPU inference, GGUF models, quantization options, inference frameworks (llama.cpp, ctransformers, onnxruntime, etc.)
3. Decide on a specific model + framework + quantization to try
4. Write results/analyses/trial-${TRIAL_NUM}.md with:
   - Summary of results so far
   - What this iteration will try and why (hypothesis)
   - Specific model name, framework, quantization, and any install steps

IMPORTANT: Do NOT repeat a model+framework+quantization combo that already appears in summary.jsonl.

Be concrete and specific. Name exact HuggingFace repos and quantization variants." \
        --allowedTools "Edit,Read,Bash,WebSearch" \
        2>&1 | tee results/experiment.log

    # Phase 2: Implementation — code the backend
    echo "--- Phase 2: Implementation ---"
    claude -p "You are the implementation agent for cpu-llm-bench.

Read these files:
- program.md (your mission and rules)
- results/analyses/trial-${TRIAL_NUM}.md (the analysis agent's plan for this trial)
- protocol.py (the InferenceBackend contract)
- backends/hf.py (reference implementation)

Your job:
1. Create a NEW file: backends/trial-${TRIAL_NUM}-{slug}.py (replace {slug} with a descriptive slug matching TRIAL_NAME)
2. Set TRIAL_NAME to a descriptive slug for this trial
3. Install any needed packages (pip install / uv pip install)
4. Run: python run_backend.py --backend backends/trial-${TRIAL_NUM}-{slug}.py --dry-run
5. If the dry run fails, fix the code and retry (up to 3 attempts)
6. Do NOT modify any existing files — only create the new backend file

The Backend class must conform to the InferenceBackend protocol in protocol.py." \
        --allowedTools "Edit,Read,Write,Bash" \
        2>&1 | tee -a results/experiment.log

    # Discover the backend file created by the implementation agent
    BACKEND_FILE=$(ls -1 backends/trial-${TRIAL_NUM}-*.py 2>/dev/null | head -1)
    if [ -z "$BACKEND_FILE" ]; then
        echo "ERROR: No backend file found for trial $TRIAL_NUM — skipping"
        continue
    fi
    echo "Using backend: $BACKEND_FILE"

    # Belt-and-suspenders dry run
    echo "--- Dry run validation ---"
    if ! timeout 120 python run_backend.py --backend "$BACKEND_FILE" --dry-run 2>&1 | tee -a results/experiment.log; then
        echo "Dry run failed for trial $TRIAL_NUM — skipping"
        python score.py --backend "$BACKEND_FILE" --status crash 2>&1 | tee -a results/experiment.log || true
        continue
    fi

    # Full run with hard timeout — pass explicit trial number
    echo "--- Full run ---"
    RUN_OUTPUT=$(timeout "$TIMEOUT" python run_backend.py --backend "$BACKEND_FILE" --trial-num "$TRIAL_NUM" 2>&1 | tee -a results/experiment.log)
    EXIT_CODE=${PIPESTATUS[0]}

    if [ "$EXIT_CODE" -eq 124 ]; then
        echo "Trial $TRIAL_NUM timed out"
        python score.py --backend "$BACKEND_FILE" --status timeout 2>&1 | tee -a results/experiment.log || true
        continue
    elif [ "$EXIT_CODE" -ne 0 ]; then
        echo "Trial $TRIAL_NUM crashed (exit $EXIT_CODE)"
        python score.py --backend "$BACKEND_FILE" --status crash 2>&1 | tee -a results/experiment.log || true
        continue
    fi

    # Extract the completions file path from run output
    COMP_FILE=$(echo "$RUN_OUTPUT" | grep '^COMPLETIONS_FILE=' | tail -1 | cut -d= -f2)

    # Score — pass explicit file if we captured it
    echo "--- Scoring ---"
    if [ -n "$COMP_FILE" ]; then
        python score.py --backend "$BACKEND_FILE" --file "$COMP_FILE" 2>&1 | tee -a results/experiment.log || echo "Scoring failed for trial $TRIAL_NUM"
    else
        echo "WARNING: Could not capture completions file path, falling back to latest"
        python score.py --backend "$BACKEND_FILE" 2>&1 | tee -a results/experiment.log || echo "Scoring failed for trial $TRIAL_NUM"
    fi

    echo ""
done

echo "=========================================="
echo "=== All $ITERATIONS iterations complete ==="
echo "=========================================="
echo "Results in results/summary.jsonl"
