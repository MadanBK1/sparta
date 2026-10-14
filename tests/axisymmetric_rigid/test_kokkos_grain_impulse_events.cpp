#include "kokkos_grain_impulse_events.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <iostream>
int main(int argc,char**argv) {
  Kokkos::initialize(argc,argv);
  int status=0;
  {
    using Exec=Kokkos::DefaultExecutionSpace;
    using Mem=typename Exec::memory_space;
    SPARTA_NS::AxisymmetricGrains::ImpulseEvents<Exec> e(3);
    Kokkos::View<double*[2],Mem> impulse("impulse",2);
    Kokkos::deep_copy(impulse,0.0);
    auto ids=Kokkos::create_mirror_view(e.grain);
    auto dp=Kokkos::create_mirror_view(e.delta_gas_p);
    auto w=Kokkos::create_mirror_view(e.physical_weight);
    ids(0)=0;ids(1)=1;ids(2)=0;
    dp(0,0)=2.;dp(0,1)=-1.;w(0)=3.;
    dp(1,0)=-4.;dp(1,1)=2.;w(1)=2.;
    dp(2,0)=1.;dp(2,1)=1.;w(2)=2.;
    Kokkos::deep_copy(e.grain,ids);
    Kokkos::deep_copy(e.delta_gas_p,dp);
    Kokkos::deep_copy(e.physical_weight,w);
    SPARTA_NS::AxisymmetricGrains::accumulate_grain_impulses(e,impulse);
    Kokkos::deep_copy(dp,impulse);
    if(std::abs(dp(0,0)+8.)>1e-12 ||
       std::abs(dp(0,1)-1.)>1e-12 ||
       std::abs(dp(1,0)-8.)>1e-12 ||
       std::abs(dp(1,1)+4.)>1e-12) status=1;
  }
  Kokkos::finalize();
  if(status) {std::cerr<<"FAIL impulse accumulation\n";return 1;}
  std::cout<<"PASS weighted multi-event impulse accumulation\n";
}
