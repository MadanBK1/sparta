# Optional read-only SPARTA Kokkos diagnostic

An opt-in C++ compile definition `SPARTA_GRAIN_READONLY_DIAGNOSTIC` enables a diagnostic within `UpdateKokkos::run()` after gas collisions and before end-of-step fixes. It is **disabled by default** and does **not** run any gas–grain collision or modify particles.

Every 100 timesteps, the diagnostic uses `ParticleKokkos::k_particles.view_device()` and `k_species.view_device()` to compute local gas axial momentum with SPARTA's representative mass convention, `fnum * species.mass * (cellweightflag ? particle.weight : 1)`. An MPI all-reduction produces the global momentum. Rank zero prints `GRAIN_DIAGNOSTIC step=... gas_px=...` to the SPARTA screen stream. This is **not** a closed-system conservation measurement unless boundary/source momentum fluxes are accounted for.

The diagnostic requires a custom SPARTA Kokkos compilation with the macro defined, such as passing `-DSPARTA_GRAIN_READONLY_DIAGNOSTIC` through the appropriate compiler flags of your build. It has not been compiled or exercised on the branch's CI; the existing standalone regression workflow does **not** build `update_kokkos.cpp`.

The global MPI reduction is a deliberate synchronization and communication point and is unsuitable for performance benchmarking when run frequently. Do not interpret this instrumented path as device-only computation with zero host coordination.

**Acceptance before enabling physics:** compile the full SPARTA executable using a real Kokkos backend, run an axisymmetric no-grain case with the macro both off and on, verify identical DSMC state evolution, and compare the diagnostic with an independent host computation. Next add conserved grain impulses only after verifying an actual collision law and treatment of DSMC superparticle weighting.
