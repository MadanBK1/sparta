#include "kokkos_grain_split_plan.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
int main(int argc,char**argv) {
 Kokkos::initialize(argc,argv);
 int bad=0;
 {
 using Exec=Kokkos::DefaultExecutionSpace;
 using Mem=typename Exec::memory_space;
 Kokkos::View<double*,Mem> w("physical_molecules",5),remaining("remaining",5),colliding("colliding",5);
 Kokkos::View<int*,Mem> events("physical_impacts",5),slots("slots",5);
 auto hw=Kokkos::create_mirror_view(w);
 auto he=Kokkos::create_mirror_view(events);
 for(int i=0;i<5;++i) hw(i)=1000000.;
 he(0)=0;he(1)=1;he(2)=100;he(3)=1000000;he(4)=-1;
 Kokkos::deep_copy(w,hw);Kokkos::deep_copy(events,he);
 SPARTA_NS::GrainSplitPlan::plan<Exec>(w,events,slots,remaining,colliding);
 auto hs=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),slots);
 auto hr=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),remaining);
 auto hc=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),colliding);
 for(int i=0;i<4;++i) {
   if(hs(i)!=(i==1||i==2?2:1) ||
      std::fabs(hr(i)+hc(i)-1.e6)>1.e-9 ||
      std::fabs(hc(i)-he(i))>1.e-9) bad=1;
 }
 if(hs(4)!=-1 || hr(4)!=-1. || hc(4)!=-1.) bad=1;
 }
 Kokkos::finalize();
 if(bad) std::fprintf(stderr,"Kokkos weighted split plan regression failed\n");
 return bad;
}
