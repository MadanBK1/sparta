# Kokkos axisymmetric grain operator: integration gate

**Status: disabled; this document is not a claim of a working solver.**

## Source-verified timestep order

In `src/KOKKOS/update_kokkos.cpp`, `UpdateKokkos::run()` performs:

1. `modify->start_of_step()`;
2. `particle->pre_weight()` when cell weighting is enabled;
3. `(this->*moveptr)()`, possibly with iterative migration;
4. `particle->post_weight()`;
5. `particle_kk->sort_kokkos()` when collision or reorder is active;
6. `collide->collisions()`, then `collide_react_update()` when enabled;
7. `modify->end_of_step()`;
8. output and synchronization as needed.

A *candidate* future gas/grain event stage is the point **after**
`collide_react_update()` and **before** diagnostic `end_of_step()`, now
annotated in the Kokkos update source. This is not necessarily the
appropriate final operator splitting and has not been validated.

## Reasons the stage is not enabled

- The annular-cell mixing approximation does not resolve individual
  off-axis spherical grain wakes or angular grain/gas positions.
- SPARTA's physical representative mass in existing rigid recoil is
  `fnum * species.mass * (cellweightflag ? particle.weight : 1)`.
  The `weight` field also participates in grid-based cloning/deletion,
  so it cannot simply be reinterpreted as an arbitrary fractional mass.
- The mean-field partial-collision kernel changes the mean velocity but
  removes velocity variance; physical DSMC scattering needs a validated
  closure or collision-population reconstruction.
- `ParticleKokkos::k_particles.view_device()` requires correct
  `sync(Device,PARTICLE_MASK)` and `modify(Device,PARTICLE_MASK)` around
  a writer. Particle-splitting will additionally require capacity
  management, custom-attribute handling, sorting and MPI migration.
- Independent grain ownership and MPI impulse reductions must be specified
  before any multi-rank coupling.
- Gas collisions sampled after transport differ from resolved collisions
  along trajectories; a time-step convergence and wall-contact check is
  required before choosing the operator split.
- All grain impulses must be applied *once*, synchronized with the grain
  kick/drift/finish kick stage, not twice.

## Proposed verification ladder

1. Validate identical particle mass and momentum on real
   `ParticleKokkos::k_particles` and a CPU reference.
2. Test zero-collision no-op with real particle arrays and correct Kokkos
   modification flags.
3. Test a prescribed collision on one rank with axial momentum auditing,
   then repeat with independent grain ownership across ranks.
4. Establish a 3D molecular/sphere statistical reference to validate
   the axisymmetric annular closure before enabling random impacts.
5. Only then add GPU event selection, particle representation closure,
   integration and time-step convergence tests on representative hardware.

No new user-facing `fix`, input-script command, or runtime feature was
added by the source comment marking this integration point.
