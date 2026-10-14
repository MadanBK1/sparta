#!/usr/bin/env bash
# Standalone unit tests; these do NOT imply a working axisymmetric SPARTA solver.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
compiler="${CXX:-g++}"
for name in test_axisymmetric_body_geometry test_axisymmetric_axial_coupling test_axisymmetric_sphere_convergence test_axisymmetric_moving_axial_plane test_grain_beam_reference test_grain_maxwellian_reference test_grain_poisson_reference test_grain_axial_elastic_event test_grain_exposure_reference test_grain_axisymmetric_cell_reference test_grain_annular_encounter_diagnostic; do
  "$compiler" -std=c++11 -Wall -Wextra -Werror -I"$root/src" \
    "$root/tests/axisymmetric_rigid/$name.cpp" -o "$out/$name"
  "$out/$name"
done
