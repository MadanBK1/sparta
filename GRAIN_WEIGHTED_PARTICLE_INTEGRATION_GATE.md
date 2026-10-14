# Weighted collision population storage: implementation gate

This audit applies to the current SPARTA Kokkos source in this branch.

## Source evidence

`ParticleKokkos::post_weight_device()` in
`src/KOKKOS/particle_kokkos.cpp` reads
`d_particles_l[i].weight / d_cinfo[icell].weight` and probabilistically
discards or clones representatives. Its new copies are assigned different
particle IDs. Thus `OnePart::weight` is **not** a general-purpose
independently selected physical-molecule population weight; it participates
in spatial cell-weight normalization.

The same method uses a device count array, a 64-bit guarded prefix scan,
scratch `d_newparticles`, and a device scatter before copying back to
`k_particles`. For every custom integer/double vector or array, it also
permutes the corresponding storage through the same offsets. It sets
`sorted_kk=0`, modifies `PARTICLE_MASK`, and updates `nlocal`.

`ParticleKokkos::grow()` resizes particle storage AND every registered
custom attribute, updates `maxlocal`, and refreshes views/pointers.
Consequently, growing `k_particles` alone is not a safe splitting
operation.

## Required design decision

**Do not** implement fractional gas/grain encounters by writing
`OnePart::weight = W_colliding / fnum` into existing SPARTA particles.
That would conflate the independent collision fraction with SPARTA's
cell-weight bookkeeping and its resampling algorithm.

Choose and validate one of two explicit architectures:

1. **Discrete conservative population extension**. Add a genuine
   independent *physical representative multiplicity* (distinct from
   SPARTA cell weights) with complete support in creation, cloning,
   merging, sorting, MPI migration, restarts and diagnostics. Insert
   separate colliding and uncollided states using an ownership-safe
   prefix-scan/scatter process. Verify that cell weighting and the new
   multiplicity do not double count molecules.
2. **Subgrid momentum/energy source model**. Do not modify a whole
   DSMC representative's velocity; use carefully validated stochastic
   conservative source terms for gas and grain and quantify the
   distribution-function error. This is a different physical closure
   and must not masquerade as exact particle splitting.

Neither architecture is implemented here. The standalone
`kokkos_grain_weighted_split.h` performs only arithmetic bookkeeping.

## Non-negotiable validation gates

- Single and two MPI ranks, cell weights enabled and disabled.
- Conservation of physical gas multiplicity, gas+grain momentum, and
  kinetic/internal energy for an elastic reference collision.
- Consistency of particle custom attributes, sorted-cell index, and
  species after resizing, reordering, and communication.
- No negative or zero physical weights written to a live representative.
- Test a split where colliding fraction is much smaller than 1/W, and
  a regime with expected encounter count much larger than one.
- Confirm no unintended particle resampling in subsequent SPARTA steps.
- CUDA, HIP and SYCL device execution, separately from Serial CI.
- For off-axis physical spheres, establish an axisymmetric statistical
  closure and validate it against a full 3D reference.

Until those tests pass, keep actual gas-to-grain collision modification
and scientific production mode disabled.
