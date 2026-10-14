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
trap 'status=$?; if (( status != 0 )); then echo "=== SPARTA diagnostic output (last 120 lines) ===" >&2; tail -n 120 "$tmp/run.log" >&2 || true; fi; rm -rf "$tmp"' EXIT
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
persistence=re.findall(r'GRAIN_PERSISTENCE_TEST step=(\d+) grain_vx=([^\s]+) grain_x=([^\s]+)',log)
if not persistence or int(persistence[-1][0])!=101:
    raise SystemExit('Missing next-step device grain persistence check')
if not math.isclose(float(persistence[-1][1]),velocity,rel_tol=1.e-12,abs_tol=0.):
    raise SystemExit('Kokkos grain state did not persist to the next timestep')
position=float(persistence[-1][2])
if not math.isfinite(position) or not math.isclose(position,velocity*1.e-6,rel_tol=1.e-10,abs_tol=1.e-25):
    raise SystemExit(f'Grain free-flight displacement incorrect: {position}')
second=re.findall(r'GRAIN_SECOND_GRAIN_TEST step=(\d+) grain_vx=([^\s]+) grain_x=([^\s]+)',log)
if not second or int(second[-1][0])!=101:
    raise SystemExit('Missing independent second-grain diagnostic')
if float(second[-1][1])!=0. or float(second[-1][2])!=0.:
    raise SystemExit('Unforced second grain unexpectedly moved')
overlap=re.findall(r'GRAIN_OVERLAP_TEST step=(\d+) force0=([^\s]+) force1=([^\s]+) v0=([^\s]+) v1=([^\s]+)',log)
if not overlap or int(overlap[-1][0])!=101:
    raise SystemExit('Missing finite-overlap grain contact diagnostic')
force0,force1,v0,v1=map(float,overlap[-1][1:])
if not all(math.isfinite(x) for x in (force0,force1,v0,v1)):
    raise SystemExit('Nonfinite grain contact force or velocity')
if not force0<0.0 or not force1>0.0:
    raise SystemExit('Expected strictly nonzero, repulsive contact forces')
# For R=75 um, center spacing=100 um and kn=1000 N/m,
# the exact initial pair force is 1000*(150-100) um = 0.05 N.
if not math.isclose(force0,-0.05,rel_tol=1.e-10,abs_tol=1.e-12):
    raise SystemExit(f'Incorrect contact stiffness or overlap: force0={force0}')
if not math.isclose(force1,0.05,rel_tol=1.e-10,abs_tol=1.e-12):
    raise SystemExit(f'Incorrect contact stiffness or overlap: force1={force1}')
if not math.isclose(force0+force1,0.0,abs_tol=1.e-12):
    raise SystemExit('Pairwise contact forces fail Newton third law')
if not v0<0.0 or not v1>0.0:
    raise SystemExit('Repulsive contact did not separate grains')
if not math.isclose(v0+v1,0.0,abs_tol=1.e-12):
    raise SystemExit('Pair contact introduced net axial momentum')






wall=re.findall(r'GRAIN_WALL_TEST step=(\d+) normal_force=([^\s]+) wall_reaction=([^\s]+) vx=([^\s]+)',log)
if not wall or int(wall[-1][0])!=101:
    raise SystemExit('Missing real-SPARTA grain-wall contact test')
fn,jwall,vwall=map(float,wall[-1][1:])
if not all(map(math.isfinite,(fn,jwall,vwall))):
    raise SystemExit('Nonfinite wall contact diagnostics')
if not math.isclose(fn,0.025,rel_tol=1.e-10,abs_tol=1.e-12):
    raise SystemExit(f'Incorrect initial wall spring force: {fn}')
if not math.isclose(jwall,-fn,rel_tol=1.e-12,abs_tol=1.e-12):
    raise SystemExit('Wall reaction does not balance grain force')
if vwall<=0.:
    raise SystemExit('Overlapping grain failed to accelerate away from wall')

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
