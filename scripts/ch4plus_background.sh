#!/usr/bin/env bash
# CH4+ from ab initio to first-principles effective Hamiltonian and spectra, in the background.
# Survives closing the SSH session / terminal (setsid + nohup), resumes where it stopped (every
# stage is cached), and refuses to start twice.
#
#   scripts/ch4plus_background.sh start --pdf PAPER.pdf [--basis cc-pvtz] [--workers 5] [--threads 4]
#                                       [--llm] [--model opus] [--rounds 4] [--budget 2] [--commit]
#   scripts/ch4plus_background.sh status     # current step, elapsed time, last log lines, alive?
#   scripts/ch4plus_background.sh log        # follow the log (Ctrl-C stops following, not the run)
#   scripts/ch4plus_background.sh stop       # stop the run (cached stages are kept; 'start' resumes)
#
# Steps (each is skipped or resumed when its output already exists):
#   0 check the Python environment (riss_drucker importable with the new modules; else pip install .)
#   1 ab initio stages incl. the reaction plane (scripts/abinitio_ch4plus.py; checkpointed)
#   2 copy the stages to data/abinitio/ch4plus_<label>
#   3 ab initio effective Hamiltonian vs the measured lines (scripts/case_methane_cation_abinitio.py)
#   4 first-principles levels, validation and spectra (scripts/firstprinciples_ch4plus.py)
#   5 (--llm) Claude builds the effective Hamiltonian against the first-principles levels
#     (scripts/llm_build_ch4plus.py --target firstprinciples; resumes its own transcript)
#   6 full stick spectra (scripts/plot_ch4plus_sticks.py)
#   7 verify all recorded case-study numbers (scripts/verify_case_studies.py)
#   8 (--commit) commit the derived files and push
#
# Environment: PYTHON (default ~/rd-venv/bin/python if it exists, else python3).
set -uo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
RUNS="$REPO/runs"
CMD=${1:-status}; shift || true

PDF=""; BASIS="cc-pvtz"; WORKERS=5; THREADS=4; LLM=0; ROUNDS=4; BUDGET=2; COMMIT=0; MODEL=""
while [[ $# -gt 0 ]]; do
  case $1 in
    --pdf) PDF=$(readlink -f "$2"); shift 2;;
    --basis) BASIS=$2; shift 2;;
    --workers) WORKERS=$2; shift 2;;
    --threads) THREADS=$2; shift 2;;
    --llm) LLM=1; shift;;
    --model) MODEL=$2; shift 2;;
    --rounds) ROUNDS=$2; shift 2;;
    --budget) BUDGET=$2; shift 2;;
    --commit) COMMIT=1; shift;;
    *) echo "unknown option $1"; exit 2;;
  esac
done
LABEL=${BASIS//-/}
RUN="$RUNS/ch4plus_$LABEL"
LOG="$RUN/run.log"; STATUS="$RUN/status"; PIDF="$RUN/pid"; CONF="$RUN/config"
PY=${PYTHON:-}
if [[ -z "$PY" ]]; then if [[ -x "$HOME/rd-venv/bin/python" ]]; then PY="$HOME/rd-venv/bin/python"; else PY=python3; fi; fi

alive() { [[ -f "$PIDF" ]] && kill -0 "$(cat "$PIDF")" 2>/dev/null; }

case $CMD in
start)
  [[ -n "$PDF" && -f "$PDF" ]] || { echo "--pdf PAPER.pdf is required (validation against the measured lines)"; exit 2; }
  mkdir -p "$RUN"
  if alive; then echo "already running (pid $(cat "$PIDF")); see: $0 status"; exit 1; fi
  command -v pdftotext >/dev/null || { echo "pdftotext missing: sudo dnf install -y poppler-utils"; exit 1; }
  if [[ $LLM == 1 ]]; then command -v claude >/dev/null || { echo "claude missing (curl -fsSL https://claude.ai/install.sh | bash) or drop --llm"; exit 1; }; fi
  printf 'PDF=%q\nBASIS=%q\nWORKERS=%q\nTHREADS=%q\nLLM=%q\nROUNDS=%q\nBUDGET=%q\nCOMMIT=%q\nPY=%q\nMODEL=%q\n' \
    "$PDF" "$BASIS" "$WORKERS" "$THREADS" "$LLM" "$ROUNDS" "$BUDGET" "$COMMIT" "$PY" "$MODEL" > "$CONF"
  setsid nohup bash "$0" _run --basis "$BASIS" >> "$LOG" 2>&1 < /dev/null &
  echo $! > "$PIDF"
  sleep 1
  echo "started (pid $(cat "$PIDF")); you can close this terminal / SSH session."
  echo "  progress:  $0 status --basis $BASIS"
  echo "  live log:  $0 log --basis $BASIS"
  echo "  stop:      $0 stop --basis $BASIS"
  ;;
status)
  if [[ ! -d "$RUN" ]]; then echo "no run for $BASIS yet ($RUN)"; exit 0; fi
  if alive; then echo "RUNNING (pid $(cat "$PIDF"))"; else echo "NOT RUNNING"; fi
  [[ -f "$STATUS" ]] && cat "$STATUS"
  echo "--- last log lines ($LOG)"
  tail -n 15 "$LOG" 2>/dev/null
  ;;
log)
  tail -n 40 -f "$LOG"
  ;;
stop)
  if alive; then kill -- -"$(cat "$PIDF")" 2>/dev/null || kill "$(cat "$PIDF")"; echo "stopped; cached stages kept ('start' resumes)"
  else echo "not running"; fi
  ;;
_run)
  # shellcheck disable=SC1090
  source "$CONF"
  exec 9>"$RUN/lock"; flock -n 9 || { echo "another run holds the lock"; exit 1; }
  cd "$REPO"
  T0=$(date +%s)
  step() { echo "$(date '+%F %T') step $1: $2 (elapsed $(( ($(date +%s) - T0) / 60 )) min)" | tee -a "$STATUS"; }
  fail() { echo "$(date '+%F %T') FAILED at step $1 (see $LOG); fix and run 'start' again to resume" | tee -a "$STATUS"; exit 1; }
  : > "$STATUS"
  echo "==== $(date '+%F %T') run start: basis $BASIS, workers $WORKERS x $THREADS threads, llm $LLM ${MODEL:+(model $MODEL)}"

  step 0 "python environment"
  if ! "$PY" -c "import riss_drucker.abinitio.firstprinciples, riss_drucker.llm_loop, riss_drucker.molecule; from riss_drucker.data.catalog import levels_from_measured" 2>/dev/null; then
    echo "riss_drucker in $PY is older than this checkout: reinstalling editable (compiles the C++ core)"
    "$PY" -m pip install -e . >> "$LOG" 2>&1 || fail 0
  fi

  step 1 "ab initio stages incl. reaction plane (UMP2/UCCSD(T), $BASIS) - the long step"
  AB="$RUN/abinitio"; DATA="$REPO/data/abinitio/ch4plus_$LABEL"
  mkdir -p "$AB"
  if [[ -d "$DATA" ]]; then for f in "$DATA"/ch4plus_*.json; do [[ -f "$AB/$(basename "$f")" ]] || cp "$f" "$AB/"; done; fi
  OMP_NUM_THREADS=$THREADS "$PY" scripts/abinitio_ch4plus.py --out "$AB" --basis "$BASIS" --workers "$WORKERS" || fail 1

  step 2 "copy stages to $DATA"
  mkdir -p "$DATA"
  for s in min ts irc hess ccsdt neutral plane2d; do cp "$AB/ch4plus_$s.json" "$DATA/" || fail 2; done

  step 3 "ab initio effective Hamiltonian vs measured lines"
  "$PY" scripts/case_methane_cation_abinitio.py --abinitio "$DATA" --pdf "$PDF" \
      --figures "docs/case_studies/figures/methane_cation_$LABEL" \
      --out "docs/case_studies/methane_cation_abinitio_${LABEL}_results.json" > /dev/null || fail 3

  step 4 "first-principles levels, validation, spectra"
  FP="docs/case_studies/ch4plus_firstprinciples_$LABEL.json"
  "$PY" scripts/firstprinciples_ch4plus.py --abinitio "$DATA" --pdf "$PDF" --out "$FP" \
      --figures "docs/case_studies/figures/ch4plus_firstprinciples_$LABEL" || fail 4

  if [[ $LLM == 1 ]]; then
    step 5 "Claude builds the effective Hamiltonian against the first-principles levels"
    TR="docs/case_studies/ch4plus_llm_fp_transcript_$LABEL.json"
    RES=(); [[ -f "$TR" ]] && { cp "$TR" "$RUN/transcript_prev.json"; RES=(--resume "$RUN/transcript_prev.json"); }
    [[ -n "${MODEL:-}" ]] && RES+=(--model "$MODEL")
    "$PY" scripts/llm_build_ch4plus.py --abinitio "$DATA" --pdf "$PDF" --target firstprinciples --fp "$FP" \
        --llm claude-cli --rounds "$ROUNDS" --max-budget-usd "$BUDGET" "${RES[@]}" --transcript "$TR" \
        --out "docs/case_studies/ch4plus_llm_fp_model_$LABEL.json" || fail 5
  else
    step 5 "Claude loop skipped (start with --llm to include it)"
  fi

  step 6 "full stick spectra"
  M="docs/case_studies/ch4plus_llm_fp_model_$LABEL.json"
  if [[ -f "$M" ]]; then
    for opt in "--jet" "--T 50"; do
      # shellcheck disable=SC2086
      "$PY" scripts/plot_ch4plus_sticks.py --model "$M" --pdf "$PDF" $opt \
          --out-dir "docs/case_studies/figures/ch4plus_firstprinciples_$LABEL" || fail 6
    done
  fi

  step 7 "verify recorded case-study numbers"
  "$PY" scripts/verify_case_studies.py --pdf "$PDF" --figures "$RUN/verify_figures" | tail -3 || fail 7

  if [[ $COMMIT == 1 ]]; then
    step 8 "commit and push"
    git add "$DATA" docs/case_studies && git commit -m "CH4+ $LABEL: reaction plane, first-principles levels and spectra" \
      && for i in 1 2 3 4; do git push && break; sleep $((2 ** i)); done || fail 8
  fi
  step done "finished in $(( ($(date +%s) - T0) / 60 )) min"
  ;;
*)
  echo "usage: $0 start|status|log|stop [--basis cc-pvtz] ..."; exit 2;;
esac
