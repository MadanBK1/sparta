# Axisymmetric moving rigid bodies (development branch)

This branch is based on upstream SPARTA PR #613 (`surf-rigid-body`) at
`4f87d1f43731afb2aa4fe3d51cc821835f50637d`.

**Status:** design and validation specification only. Axisymmetric `fix rigid`
is **not implemented or enabled yet**. Do not use this branch to claim
axisymmetric gas–solid conservation or a validated regolith model.

## Initial target

Implement two-way, collision-resolved momentum exchange between SPARTA's
2D axisymmetric DSMC gas and a *surface of revolution* represented by a
moving 2D meridional profile. In the first milestone, constrain the body to
axial translation, leaving existing non-axisymmetric `fix rigid` behavior
untouched. No LAMMPS dependency.

A circular meridional profile centered on the axis represents a sphere.
A circular profile wholly off axis represents a torus, **not** an
independent spherical regolith grain. Individual off-axis grains require a
different reduced-order statistical coupling and are not in milestone 1.

## Audit before changing physics

1. In `src/fix_rigid.cpp`, the constructor currently rejects
   `domain->axisymmetric`. Do not delete this guard until the following
   invariants have implementations and tests.
2. Identify the axisymmetric mover's position/velocity transformation,
   weight handling, and interactions with the rigid moving-surface path
   in `src/update.cpp` and `src/KOKKOS/update_kokkos.cpp`.
3. Trace impulse units and DSMC physical-particle weighting through
   `compute surf`, the moving-surface collision model, and `fix rigid`.
   Never multiply collision tallies by `2*pi*r` a second time if the
   mover's statistical weights already represent annular volumes.
4. Replace the planar-area mass/inertia evaluation with geometry of
   revolution: `dV = 2*pi*r*dA`, ensuring correct mass for a centered
   sphere and correct toroidal mass for an off-axis section.
5. Constrain the first implementation to `v_r = 0`, axial translation,
   and no arbitrary-body rotation. A radial shift of an axisymmetric
   object generally breaks the represented geometry and must be rejected.
6. Verify moving-surface swept collision lists, cut-cell volumes, and
   grid remapping in the axisymmetric mesh, including the `r = 0`
   boundary and bodies approaching the axis.
7. Extend the validated CPU path to `rigid/kk`; measure host/device
   synchronization cost. Do not assume GPU support from successful
   non-axisymmetric Kokkos compilation alone.

## Required tests

- **Geometry:** centered spherical profile gives
  `V = 4*pi*R^3/3` and `M = rho*V`. Off-axis circular section
  gives the volume of the corresponding torus.
- **Collision impulse:** gas momentum change and rigid-body momentum
  change match with opposite sign when summed with SPARTA's *actual*
  physical-particle weights. Account for external/boundary fluxes.
- **Drag:** stationary and translating axis-centered sphere in a
  free-molecular reference flow; compare with an appropriate analytical
  solution using the same accommodation model.
- **Numerics:** no leaked or deleted gas particles at the moving
  surface; reproducible timestep-refinement trend; correct cut-cell
  remapping.
- **Execution:** single rank, multi-rank CPU, and Kokkos consistency
  before Intel PVC scaling.

## Do not confuse these models

A 2D axisymmetric **resolved surface** models a solid of revolution.
It cannot directly resolve arbitrary isolated, off-axis 3D grains.
The first milestone therefore supports physically axisymmetric objects,
not a realistic arbitrary 3D granular bed. Extending to granular contact
mechanics and off-axis grains requires a separate model and validation.

## Repository isolation

Development takes place on `MadanBK1/sparta:rigid-axisymmetric-kokkos`.
Do not push to `sparta/sparta` or to the fork's `master`.
