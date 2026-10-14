# PR #613: 2D axisymmetric rigid-body adaptation (translation-only)

**Base**: exact PR #613 head commit `4f87d1f43731afb2aa4fe3d51cc821835f50637d`.
This branch is independent of `rigid-axisymmetric-kokkos` and does not
include its diagnostic/prototype work.

## Intended model

- DSMC gas remains SPARTA's 2D axisymmetric (x,r) solver.
- Individual finite-radius circles move in the (x,r) meridional plane.
- Reuse PR #613's moving-surface collision/reflection and rigid force
  accumulation rather than implementing a second collision kernel.
- Start with translation (vx,vr). Rotation and torque require a separate
  model/validation.
- A circle with off-axis center is NOT an axisymmetric solid of revolution.
  Its meridional projection becomes a torus when revolved; this is a
  reduced-order 2D momentum-coupling approximation and not resolved
  gas collisions around isolated 3D spheres.

## Source gates found in the PR

1. `FixRigid::FixRigid()` explicitly rejects axisymmetric domains
   (`src/fix_rigid.cpp`).
2. `Update::rigid_init()` rejects `global rigid` in axisymmetry
   (`src/update.cpp`).
3. The axisymmetric CPU mover is dispatched as `move<1,1,0,0>`,
   while 2D rigid cases use `move<2,1,0,1>`. Merely removing
   (1)-(2) would still bypass the RIGID mover specialization.
4. The Kokkos mover selects its axisymmetric dispatch independently.
   Its rigid/kk and moving-surface geometry paths must be audited
   before enabling an accelerator.
5. `FixRigid::init()` expects the existing `compute surf` force/torque
   channels and requires `global rigid`. Those force channels must
   have consistent physical units for the reduced 2D representation.

## Physics/implementation gates before any guard is removed

- Identify which moving-line intersection and collision routines
  handle the `DIM=1, RIGID=1` specialization correctly. Axisymmetric
  particle remapping and curved radial-cell crossings must remain intact.
- Specify how a 2D circle represents a physical sphere or a ring.
  Select an explicit effective azimuthal measure for gas collision
  sampling, wall force scaling, and rigid mass; do not confuse
  2D projected impulse with 3D physical force.
- Define **translation only** and prevent unsupported rotational DOFs.
  Restrict centers to r>=0 and handle crossing the symmetry axis.
- Verify weighted particle momentum: SPARTA `fnum` and cell weights.
  Do not double-multiply annular collision weights or rigid force totals.
- Check solid geometry remapping, dynamic cell/surface assignments,
  and collision persistence on MPI ownership boundaries.
- Benchmark: (a) static circle gas collision vs normal axisym mover,
  (b) moving circle gas reaction equal/opposite to rigid acceleration,
  (c) single/two MPI ranks, (d) Kokkos Serial before CUDA/HIP/SYCL.

## Scope of this first commit

This branch documents the existing implementation and preserves all
runtime safety guards. No unverified physical solver is enabled.


## Concrete CPU mover incompatibility (source inspection)

The existing `Update::move<DIM,SURF,OPT,RIGID>()` rigid
moving-surface intersection only applies to `DIM == 2` and `DIM == 3`.
For `DIM == 2`, the code calls
`Geometry::line_line_moving_intersect(...,fr->xcm,fr->vcm,fr->omega,...)`.
For `DIM == 1` (axisymmetric), it instead calls
`Geometry::axi_line_intersect(...)` on a **static** line, without
passing any `FixRigid` position, velocity, angular velocity, or
surface-frame transform. Merely dispatching `move<1,1,0,1>` would
continue testing static geometry during a rigid-body move. In addition,
the `DIM == 1` path obtains a transformed collision-point velocity
(`vc`) from the axisymmetric intersection function. That result must
not be confused with the wall velocity returned by the moving rigid
collision routine.

**Specific next implementation task:** devise an axisymmetric
moving-line intersection that consistently combines the axisymmetric
trajectory, time-dependent rigid surface geometry, collision normal,
and wall velocity. Reuse the existing gas-surface scattering law and
force reaction after a valid moving-surface hit is found. Test with
constant rigid velocity and zero rotation *before* enabling feedback
from collision forces. No axisymmetric rigid-body safety guard may be
removed until this collision path is implemented and tested.

For an off-axis circle, this remains a reduced 2D closure: the
geometrical intersection of 2D particle tracks and moving circles is
not by itself equivalent to the physical collision statistics of a
3D axisymmetric gas interacting with isolated off-axis spheres.
