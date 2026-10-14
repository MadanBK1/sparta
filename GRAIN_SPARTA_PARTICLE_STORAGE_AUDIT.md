# SPARTA particle-storage audit for individual-grain coupling

Status: architecture audit; **no production DSMC grain collision operator**.

## Verified from this branch's source

- `src/particle.h`: `Particle::OnePart` contains `id`, `ispecies`,
  `icell`, `flag`, `x[3]`, `v[3]`, `erot`, `evib`,
  `dtremain`, and `weight`.
- `src/particle.cpp`: `Particle::post_weight()` compares
  `particles[i].weight` to `grid->cinfo[icell].weight` and **stochastically
  deletes or clones whole particles** according to their ratio. It does not
  interpret `OnePart::weight` as an independently assignable arbitrary
  number of physical molecules in a variable-weight DSMC swarm.
- `src/KOKKOS/particle_kokkos.h`: device storage is
  `ParticleKokkos::k_particles`, a dual view of `OnePart`.
  `add_particle_kokkos()` initializes particle fields but does not assign
  `weight` or `dtremain`; a newly created grain-split particle must
  explicitly initialize **all** state fields and applicable custom data.
- `src/KOKKOS/particle_kokkos.cpp`: `post_weight_device()` performs
  a count/scan/scatter on the GPU, manages particle capacity and copied
  custom arrays, and issues fresh IDs when cloning. This implementation
  provides a useful pattern for a future device-side split/scatter, but
  is **not** a callable weighted-grain-split API.

## Blocking semantic issue

The standalone helper `grain_weighted_split_reference.h` produces two
**mathematical populations with fractional weights**. Do **not** write
these two weights into `OnePart::weight` and assume correct SPARTA
semantics: that field is used in cell-weight resampling. Arbitrarily
overloading it could cause erroneous particle cloning/deletion and wrong
physical momentum transfer.

A compatible grain operator must first decide whether to:
1. introduce explicit per-representative *physical statistical weight*
   independent of SPARTA's cell weight (and propagate it through migration,
   collisions, emission, adaptation, restart, diagnostics, and custom data);
2. use stochastic whole-representative collision updates at a compatible
   resolution, analyzing noise and bias and preserving the weight
   assumptions of SPARTA's existing axisymmetric algorithms; or
3. represent sub-particle participating populations in separate moment
   accumulators with a validated reconstruction/closure.

None is selected or implemented yet.

## Minimum integration acceptance tests

1. Read a particle on host/device and show exactly how its effective
   physical molecule count depends on `fnum`, `weight`, grid cell, and
   axisymmetric mover conventions; verify with a controlled diagnostic.
2. Test copy, splitting, migration, restart, custom arrays and post-weight
   resampling for any newly introduced statistical-weight field.
3. Compare number, momentum and energy budgets before/after grain
   interaction using **axial** momentum first. Azimuthally averaged radial
   momentum has different global symmetry and requires separate treatment.
4. Require no host/device round trip per particle or per collision;
   benchmark kernel speed and memory on Serial, CUDA/HIP/SYCL.
5. Define a closure for off-axis individual spheres in an axisymmetric gas.
   The resolved three-dimensional wake cannot exist on this grid.

## Hard rule

Do not enable `fix rigid` in axisymmetric geometry by merely deleting
the existing guards: a meridional circle off the axis represents a **torus**,
not an individual spherical regolith grain.
