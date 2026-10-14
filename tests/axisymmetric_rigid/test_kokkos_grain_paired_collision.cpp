#include "kokkos_grain_paired_collision.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
int main(int argc,char**argv) {
 Kokkos::initialize(argc,argv);
 int fail=0;
 {
  using Ex=Kokkos::DefaultExecutionSpace;
  using Mem=typename Ex::memory_space;
  SPARTA_NS::AxisymmetricGrains::PairedCollisionEvents<Ex> e(2);
  Kokkos::View<double*[2],Mem> gas("gas",2), grain("grain",1);
  auto g=Kokkos::create_mirror_view(gas);
  auto a=Kokkos::create_mirror_view(e.new_gas_velocity);
  auto ix=Kokkos::create_mirror_view(e.gas_index);
  auto iy=Kokkos::create_mirror_view(e.grain_index);
  auto m=Kokkos::create_mirror_view(e.gas_molecular_mass);
  auto w=Kokkos::create_mirror_view(e.physical_weight);
  g(0,0)=10.;g(0,1)=1.;g(1,0)=5.;g(1,1)=-1.;
  a(0,0)=8.;a(0,1)=2.;a(1,0)=3.;a(1,1)=0.;
  ix(0)=0;ix(1)=1;iy(0)=iy(1)=0;
  m(0)=m(1)=0.5;w(0)=w(1)=4.;
  Kokkos::deep_copy(gas,g); Kokkos::deep_copy(e.new_gas_velocity,a);
  Kokkos::deep_copy(e.gas_index,ix);
  Kokkos::deep_copy(e.grain_index,iy);
  Kokkos::deep_copy(e.gas_molecular_mass,m);
  Kokkos::deep_copy(e.physical_weight,w);
  Kokkos::deep_copy(grain,0.);
  SPARTA_NS::AxisymmetricGrains::validate_paired_collisions(e,gas,grain);
  // Duplicate gas IDs are unsafe for the parallel update; reject them.
  ix(1)=0;
  Kokkos::deep_copy(e.gas_index,ix);
  bool rejected=false;
  try { SPARTA_NS::AxisymmetricGrains::validate_paired_collisions(e,gas,grain); }
  catch(const std::invalid_argument&) { rejected=true; }
  if(!rejected) fail=1;
  ix(1)=1;Kokkos::deep_copy(e.gas_index,ix);
  SPARTA_NS::AxisymmetricGrains::validate_paired_collisions(e,gas,grain);
  SPARTA_NS::AxisymmetricGrains::apply_paired_collisions(e,gas,grain);
  auto gi=Kokkos::create_mirror_view(grain);
  Kokkos::deep_copy(gi,grain);Kokkos::deep_copy(g,gas);
  // Gas weighted axial delta = -4-4=-8, radial delta = +2+2=4.
  if(std::abs(gi(0,0)-8.)>1e-12 ||
     std::abs(gi(0,1)+4.)>1e-12 ||
     std::abs(g(0,0)-8.)>1e-12 || std::abs(g(1,0)-3.)>1e-12) fail=1;
 }
 Kokkos::finalize();
 if(fail) {std::cerr<<"FAIL paired collisions\n";return 1;}
 std::cout<<"PASS paired gas/grain collision bookkeeping\n";
}
