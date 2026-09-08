@echo off
REM ============================================================================
REM  run_explain_v2.bat
REM  Deterministic v2 regeneration wrapper for explain.py (ACSAC 2026, Step C)
REM
REM  This wrapper sets the env vars that MUST exist BEFORE python launches.
REM  Setting PYTHONHASHSEED, OMP_NUM_THREADS, or MKL_NUM_THREADS inside Python
REM  is too late for the current process — they are read at interpreter startup
REM  (PYTHONHASHSEED) or at first MKL/OpenMP initialization (the thread vars).
REM
REM  Sources for the recipe:
REM    - PyTorch reproducibility note (https://docs.pytorch.org/docs/stable/notes/randomness.html)
REM    - Lightning issues #1939 and #2156 (PYTHONHASHSEED programmatic-set fails)
REM    - PyTorch issues #19213, #975, #88718 (in-Python set_num_threads not always honored)
REM    - Intel oneMKL CNR (MKL_CBWR=AUTO,STRICT for same-machine bit-identity)
REM
REM  Usage:
REM    Smoke test (1 query, fast):
REM      run_explain_v2.bat --limit 1
REM
REM    Full v2 regeneration (146 queries, ~10–20 min on CPU):
REM      run_explain_v2.bat
REM
REM    With extra explain.py args (passed through %*):
REM      run_explain_v2.bat --verbose
REM
REM  Run this from the reranker directory:
REM    C:\Users\shane\Downloads\CTI\graph_alignment\reranker\
REM ============================================================================

setlocal EnableExtensions

REM Ensure relative paths resolve from the directory containing this script.
cd /d "%~dp0"

REM ---- Python interpreter startup: PYTHONHASHSEED ----
REM    Locks Python's str/bytes hash randomization. Verified inside explain.py
REM    via sys.flags.hash_randomization and a hash('determinism-canary') print.
SET PYTHONHASHSEED=42

REM ---- CPU thread pinning ----
REM    Both OMP_NUM_THREADS and MKL_NUM_THREADS must be set BEFORE python.
REM    OMP_DYNAMIC=FALSE / MKL_DYNAMIC=FALSE prevent runtime retuning of pools.
SET OMP_NUM_THREADS=1
SET MKL_NUM_THREADS=1
SET OMP_DYNAMIC=FALSE
SET MKL_DYNAMIC=FALSE

REM ---- Intel MKL Conditional Numerical Reproducibility ----
REM    AUTO,STRICT: locks to the best ISA branch of THIS CPU AND adds
REM    thread-count-invariant accumulation order for STRICT-eligible BLAS-3 ops.
REM    For cross-machine reproducibility you would use AVX2,STRICT or COMPATIBLE
REM    instead, at a performance cost. For our same-machine bit-identity goal,
REM    AUTO,STRICT is the right choice (Perplexity report, Section 1.1).
SET MKL_CBWR=AUTO,STRICT

REM ---- Optional: KMP deterministic OpenMP reduction ----
REM    Forces a fixed reduction order in OpenMP-parallel loops. Only meaningful
REM    with static scheduling and fixed thread count, but harmless at thread=1.
SET KMP_DETERMINISTIC_REDUCTION=true

REM ---- cuBLAS workspace config ----
REM    No-op on CPU but defensive — silences spurious warnings if torch was
REM    built against CUDA, and locks the GPU path if env is reused.
SET CUBLAS_WORKSPACE_CONFIG=:4096:8

REM ---- HuggingFace tokenizer parallelism ----
REM    HF tokenizers can spawn worker threads; disable for full control.
SET TOKENIZERS_PARALLELISM=false

REM ---- Optional: pin oneDNN ISA for cross-CPU stability ----
REM    Uncomment to pin to AVX2 (broadest compatibility on modern Intel/AMD).
REM SET ONEDNN_MAX_CPU_ISA=AVX2

REM ---- Echo settings to stdout for the run log ----
echo.
echo ====================================================================
echo  RUN_EXPLAIN_V2.BAT - Step C deterministic v2 regeneration
echo ====================================================================
echo  PYTHONHASHSEED              = %PYTHONHASHSEED%
echo  OMP_NUM_THREADS             = %OMP_NUM_THREADS%
echo  MKL_NUM_THREADS             = %MKL_NUM_THREADS%
echo  OMP_DYNAMIC                 = %OMP_DYNAMIC%
echo  MKL_DYNAMIC                 = %MKL_DYNAMIC%
echo  MKL_CBWR                    = %MKL_CBWR%
echo  KMP_DETERMINISTIC_REDUCTION = %KMP_DETERMINISTIC_REDUCTION%
echo  CUBLAS_WORKSPACE_CONFIG     = %CUBLAS_WORKSPACE_CONFIG%
echo  TOKENIZERS_PARALLELISM      = %TOKENIZERS_PARALLELISM%
echo  CWD                         = %CD%
echo ====================================================================
echo.

REM ---- Sanity check: verify expected paths exist before launching ----
if not exist "checkpoints\best_two_stage_v2\config.json" (
    echo ERROR: checkpoints\best_two_stage_v2\config.json not found.
    echo        Expected v2 model directory at:
    echo        %CD%\checkpoints\best_two_stage_v2\
    echo        Make sure you are running this script from the reranker directory.
    exit /b 10
)
if not exist "data\reranker_pairs_enriched_v2.jsonl" (
    echo ERROR: data\reranker_pairs_enriched_v2.jsonl not found.
    echo        Expected v2 data file at:
    echo        %CD%\data\reranker_pairs_enriched_v2.jsonl
    exit /b 11
)
if not exist "explain.py" (
    echo ERROR: explain.py not found in %CD%.
    exit /b 12
)

echo Pre-flight checks passed. Launching python ...
echo.

REM ---- Launch ----
REM    -u: unbuffered stdout/stderr (clean log)
REM    %*: pass through any additional CLI args (e.g. --limit 1, --verbose)
python -u explain.py ^
    --model checkpoints/best_two_stage_v2 ^
    --data  data/reranker_pairs_enriched_v2.jsonl ^
    --output-dir eval_results_v2 ^
    --seed 42 ^
    %*

set EXITCODE=%ERRORLEVEL%
echo.
echo ====================================================================
echo  explain.py exited with code %EXITCODE%
if "%EXITCODE%"=="0" (
    echo  ^>^>^> Outputs are in:  %CD%\eval_results_v2\
    echo  ^>^>^>   explanations.json
    echo  ^>^>^>   token_importance.csv
    echo  ^>^>^>   explainability_stats.txt
    echo  ^>^>^>   provenance.json
    echo.
    echo  Next step: upload eval_results_v2\token_importance.csv (and ideally
    echo  explanations.json + provenance.json^) for Step C verification.
) else (
    echo  ^>^>^> Run failed. Check the output above for the cause.
    echo  ^>^>^> Common exit codes:
    echo  ^>^>^>    1 = sentence-transformers not installed
    echo  ^>^>^>    2 = test split mismatch (seed propagation broke^)
    echo  ^>^>^>    3 = model not in FP32 (mixed-dtype checkpoint^)
    echo  ^>^>^>   10 = v2 model directory missing
    echo  ^>^>^>   11 = v2 data file missing
    echo  ^>^>^>   12 = explain.py missing
)
echo ====================================================================

endlocal & exit /b %EXITCODE%
