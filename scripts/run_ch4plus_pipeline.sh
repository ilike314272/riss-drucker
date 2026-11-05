#!/usr/bin/env bash
# From finished ab initio stages to the CH4+ effective Hamiltonian and rovibronic spectrum, in one go.
#
#   scripts/run_ch4plus_pipeline.sh RUN_DIR LABEL PAPER.pdf [--llm] [--commit]
#
#   RUN_DIR  output folder of scripts/abinitio_ch4plus.py (e.g. ch4p)
#   LABEL    surface name, e.g. ccpvtz
#   --llm    also let Claude build the model (claude -p on your login; stops cleanly at limits, $2 cap)
#   --commit commit the derived files (stage JSONs, results, figures, line lists) and push
#
# Steps: check all stages are present -> copy them to data/abinitio/ch4plus_LABEL -> ab initio
# effective Hamiltonian, predicted spectrum, Bayesian update, figures and line lists -> (optional)
# Claude-in-the-loop model -> verify every recorded case-study number.
set -euo pipefail
RUN=${1:?run dir}; LABEL=${2:?label, e.g. ccpvtz}; PDF=${3:?paper PDF}; shift 3
LLM=0; COMMIT=0
for a in "$@"; do case $a in --llm) LLM=1;; --commit) COMMIT=1;; *) echo "unknown option $a"; exit 2;; esac; done
PY=${PYTHON:-python}
cd "$(dirname "$0")/.."

for s in min ts irc hess ccsdt neutral; do
  [[ -f "$RUN/ch4plus_$s.json" ]] || { echo "stage '$s' missing in $RUN (is the run finished? grep '^stage' $RUN/run.log)"; exit 1; }
done
DATA=data/abinitio/ch4plus_$LABEL
FIG=docs/case_studies/figures/methane_cation_$LABEL
RES=docs/case_studies/methane_cation_abinitio_${LABEL}_results.json
mkdir -p "$DATA" "$FIG"
cp "$RUN"/ch4plus_{min,ts,irc,hess,ccsdt,neutral}.json "$DATA"/
echo "== 1/3 effective Hamiltonian and spectrum from the $LABEL surface"
$PY scripts/case_methane_cation_abinitio.py --abinitio "$DATA" --pdf "$PDF" --figures "$FIG" --out "$RES" > /dev/null
$PY - "$RES" <<'PYEOF'
import json, sys
r = json.load(open(sys.argv[1]))
print(f"  levels of theory: {r['levels_of_theory']['geometries_hessians']}, {r['levels_of_theory']['energies']}")
print(f"  saddle: {r['stationary_points']['ts_n_imaginary']} imaginary ({r['stationary_points']['ts_imaginary_frequency_cm']:.1f} cm-1)")
for n in ("CH4+", "CD4+"):
    x, e = r[n], r["effective_hamiltonian"][n]
    c = x["rotational_constants_equilibrium"]
    print(f"  {n}: A B C = {c['A']:.3f} {c['B']:.3f} {c['C']:.3f}; barrier {x['barrier_electronic_cm']:.0f} cm-1; "
          f"delta = {x['delta_ab_initio']:.2f} cm-1; spectrum rms (IE only) {e['prediction_no_fit']['rms_cm']:.2f}; "
          f"after update {e['bayesian_update']['rms_cm']:.2f} (delta {e['bayesian_update']['delta']:.2f})")
print(f"  adiabatic IE {r['ionization_energy']['adiabatic_cm']:.0f} cm-1; delta ratio {r['isotope_ratio_delta']:.1f}")
PYEOF
echo "  figures and line lists: $FIG"
if [[ $LLM == 1 ]]; then
  echo "== 2/3 Claude builds the model (stops cleanly at usage limits; resume with --resume)"
  $PY scripts/llm_build_ch4plus.py --abinitio "$DATA" --pdf "$PDF" --llm claude-cli --rounds 4 --max-budget-usd 2 \
      --out "docs/case_studies/ch4plus_llm_model_${LABEL}.json" --transcript "docs/case_studies/ch4plus_llm_transcript_${LABEL}.json" \
      --figures "$FIG"
else
  echo "== 2/3 Claude loop skipped (add --llm)"
fi
echo "== 3/3 verify recorded case-study numbers"
$PY scripts/verify_case_studies.py --pdf "$PDF" --figures "$(mktemp -d)" | tail -3
if [[ $COMMIT == 1 ]]; then
  git add "$DATA" "$FIG" "$RES" docs/case_studies/ch4plus_llm_*"${LABEL}"*.json 2>/dev/null || true
  git commit -m "CH4+ $LABEL surface: stages, effective Hamiltonian, spectra" && git push
fi
