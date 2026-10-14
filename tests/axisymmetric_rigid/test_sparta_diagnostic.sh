#!/usr/bin/env bash
# Execute the real SPARTA axisymmetric example with the optional audit.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
build="${1:?usage: test_sparta_diagnostic.sh BUILD_DIRECTORY}"
binary="$(find "$build" -type f \( -name spa_kokkos -o -name spa_kokkos_* -o -name spa_* \) -executable | head -n 1)"
if [[ -z "$binary" ]]; then
  echo "SPARTA Kokkos executable not found under $build" >&2
  exit 1
fi
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
cp "$root/examples/axi/"{in.axi,air.species,air.vss,data.circle} "$tmp/"
# A short smoke test; 101 steps checks that the device grain state survives the collision timestep.
sed -i 's/^run[[:space:]]*1000/run 101/' "$tmp/in.axi"
(cd "$tmp" && "$binary" -k on -sf kk -in in.axi > run.log 2>&1) || {
  cat "$tmp/run.log" >&2
  exit 1
}
cat "$tmp/run.log" | grep 'GRAIN_DIAGNOSTIC' || {
  echo "Expected instrumented Kokkos diagnostic was not printed" >&2
  exit 1
}
python3 - "$tmp/run.log" <<'PY'
import math
import re
import sys
log=open(sys.argv[1],encoding='utf-8',errors='replace').read()
prescribed=re.findall(r'GRAIN_PRESCRIBED_TEST step=(\d+) reaction_Jx=([^\s]+) grain_vx=([^\s]+) budget_residual=([^\s]+)',log)
if not prescribed or int(prescribed[-1][0])!=100:
    raise SystemExit('Missing prescribed real-particle axial write test')
reaction=float(prescribed[-1][1])
if not math.isfinite(reaction) or reaction>=0.:
    raise SystemExit(f'Invalid sign or magnitude of prescribed grain reaction: {reaction}')
velocity=float(prescribed[-1][2])
residual=float(prescribed[-1][3])
if not math.isfinite(velocity) or not math.isfinite(residual):
    raise SystemExit('Nonfinite prescribed grain velocity or momentum budget')
if abs(1.e-6*velocity-reaction)>1.e-10*abs(reaction):
    raise SystemExit('Prescribed gas-grain momentum exchange failed')
if abs(residual)>1.e-10*abs(reaction):
    raise SystemExit('Prescribed gas-grain reported budget nonzero')
persistence=re.findall(r'GRAIN_PERSISTENCE_TEST step=(\d+) grain_vx=([^\s]+)',log)
if not persistence or int(persistence[-1][0])!=101:
    raise SystemExit('Missing next-step device grain persistence check')
if not math.isclose(float(persistence[-1][1]),velocity,rel_tol=1.e-12,abs_tol=0.):
    raise SystemExit('Kokkos grain state did not persist to the next timestep')



matches=re.findall(r'GRAIN_DIAGNOSTIC step=(\d+) molecules=([^ ]+) gas_px=([^ ]+) translational_E=([^\s]+)',log)
if not matches:
    raise SystemExit('No correctly formatted diagnostic line')
step,n,px,energy=matches[-1]
vals=[float(n),float(px),float(energy)]
if int(step)!=100 or not all(map(math.isfinite, vals)) or n=='nan':
    raise SystemExit(f'Unexpected step or nonfinite diagnostic: {matches[-1]}')
if vals[0]<=0 or vals[2]<=0:
    raise SystemExit(f'Nonphysical molecule count or translational energy: {matches[-1]}')
print('PASS full SPARTA Kokkos axisymmetric diagnostic smoke test:',matches[-1])
PY
