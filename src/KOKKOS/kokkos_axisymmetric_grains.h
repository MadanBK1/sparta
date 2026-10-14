/* Portable Kokkos grain state and integration prototype.
   This is an independent kernel module, not yet wired to SPARTA's mover.
   x is axial, y is radial; grains carry physical spherical mass.
   Axisymmetric gas-grain collision rules remain to be derived/validated. */
#ifndef SPARTA_KOKKOS_AXISYMMETRIC_GRAINS_H
#define SPARTA_KOKKOS_AXISYMMETRIC_GRAINS_H
#include <Kokkos_Core.hpp>
#include <cmath>
namespace SPARTA_NS {
namespace AxisymmetricGrains {
template<class ExecSpace>
struct State {
  using memory_space = typename ExecSpace::memory_space;
  using view_type = Kokkos::View<double*[2], memory_space>;
  using scalar_type = Kokkos::View<double*, memory_space>;
  view_type position,velocity,force;
  scalar_type mass,radius;
  int n;
  explicit State(int count) :
    position("grain_position",count), velocity("grain_velocity",count),
    force("grain_force",count),mass("grain_mass",count),
    radius("grain_radius",count),n(count) {}
};
template<class ExecSpace>
void zero_forces(State<ExecSpace>& state) {
  auto force=state.force;
  Kokkos::parallel_for("grain_zero_force",
    Kokkos::RangePolicy<ExecSpace>(0,state.n),KOKKOS_LAMBDA(int i) {
      force(i,0)=0.;force(i,1)=0.;
    });
}
template<class ExecSpace>
void kick_drift(State<ExecSpace>& state, double dt) {
  auto p=state.position, v=state.velocity, f=state.force, m=state.mass;
  Kokkos::parallel_for("grain_kick_drift",
    Kokkos::RangePolicy<ExecSpace>(0,state.n), KOKKOS_LAMBDA(int i) {
      const double invm=1.0/m(i);
      for(int d=0;d<2;d++) {
        v(i,d)+=0.5*dt*f(i,d)*invm;
        p(i,d)+=dt*v(i,d);
      }
    });
}
template<class ExecSpace>
void finish_kick(State<ExecSpace>& state, double dt) {
  auto v=state.velocity, f=state.force, m=state.mass;
  Kokkos::parallel_for("grain_finish_kick",
    Kokkos::RangePolicy<ExecSpace>(0,state.n),KOKKOS_LAMBDA(int i) {
      for(int d=0;d<2;d++) v(i,d)+=0.5*dt*f(i,d)/m(i);
    });
}
// Pairwise frictionless normal contact for a reduced 2D meridional
// grain model. Linear spring-dashpot; overlap > 0 only.
// This is a prototype O(Npairs) kernel; caller supplies unique unordered
// (i,j) pairs, with 0<=i<j<n. No periodic azimuthal contacts are modeled.
// Forces use atomic additions because grains can share neighbors.
// The pair list should be built using a GPU neighbor search before this
// is used in a large simulation.
template<class ExecSpace, class PairView>
void linear_normal_contacts(State<ExecSpace>& state, const PairView& pairs,
                            double stiffness, double damping) {
  auto x=state.position, v=state.velocity, f=state.force, r=state.radius;
  const int npairs=static_cast<int>(pairs.extent(0));
  Kokkos::parallel_for("grain_linear_normal_contacts",
    Kokkos::RangePolicy<ExecSpace>(0,npairs), KOKKOS_LAMBDA(int k) {
      const int i=pairs(k,0), j=pairs(k,1);
      const double dx=x(j,0)-x(i,0), dy=x(j,1)-x(i,1);
      const double d2=dx*dx+dy*dy;
      if (d2<=0.0) return; // coincident centers require a separate policy
      const double dist=Kokkos::sqrt(d2);
      const double overlap=r(i)+r(j)-dist;
      if (overlap<=0.0) return;
      const double nx=dx/dist, ny=dy/dist;
      const double rel=(v(j,0)-v(i,0))*nx+(v(j,1)-v(i,1))*ny;
      double fn=stiffness*overlap-damping*rel;
      if (fn<0.0) fn=0.0; // non-adhesive contact
      Kokkos::atomic_add(&f(i,0),-fn*nx);
      Kokkos::atomic_add(&f(i,1),-fn*ny);
      Kokkos::atomic_add(&f(j,0), fn*nx);
      Kokkos::atomic_add(&f(j,1), fn*ny);
    });
}
// A force kernel must refresh forces between kick_drift and finish_kick.
// All mass entries must be positive; setup is responsible for validating.
} // AxisymmetricGrains
} // SPARTA_NS
#endif
