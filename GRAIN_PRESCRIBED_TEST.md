# Prescribed axial gas-particle write test (development only)

Macro `SPARTA_GRAIN_PRESCRIBED_AXIAL_TEST` is **disabled by default**.
When enabled, `UpdateKokkos::run()` adds exactly `+0.001 m/s` to the axial
velocity of the first locally owned DSMC representative on each rank at
step 100 (only on ranks owning at least one particle). It computes the
**opposite** reaction impulse using the current SPARTA representative
particle mass `fnum * species.mass * (cellweightflag ? particle.weight : 1)`.
The updated particle view is marked device-modified.

This is explicitly **not physical gas–grain coupling**:
- A temporary one-element Kokkos grain-velocity view now receives the MPI-reduced reaction impulse using a prescribed mass of 1e-6 kg. It is not retained as a simulation grain across timesteps.
- The perturbed particle is selected by rank-local array position rather
  than proximity to a grain. MPI rank count changes the chosen particles.
- The update is an imposed velocity perturbation. Its opposite impulse is applied to a temporary test grain velocity, with an explicit axial momentum residual check; this does not validate physical collision statistics.
- This is not an actual stochastic collision, particle splitting, or
  statistically validated representation of off-axis spheres.
- The test deliberately changes the simulated gas and must **never** be
  enabled in physical production runs.

The `axisymmetric-grain-tests.yml` full-SPARTA job builds this macro
alongside the read-only diagnostic and tests the expected reaction
sign/finite magnitude at step 100. It is a data-path smoke test only;
the numerical coupling and physical collision probability remain open.

The diagnostic prints `grain_vx` and `budget_residual`. The automated smoke test checks that `1e-6 * grain_vx = reaction_Jx` within tolerance. The MPI-reduced impulse is replicated on every rank for test purposes; it is **not** one independently simulated grain per rank. Full SPARTA compile/run of this change has not yet been verified.
