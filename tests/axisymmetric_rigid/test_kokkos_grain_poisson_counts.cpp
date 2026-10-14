#include "kokkos_grain_poisson_counts.h"
#include <Kokkos_Core.hpp>
#include <cmath>
#include <cstdio>
int main(int argc,char** argv) {
  Kokkos::initialize(argc,argv);
  int failed=0;
  {
    using Exec=Kokkos::DefaultExecutionSpace;
    using Mem=typename Exec::memory_space;
    const int n=100000;
    Kokkos::View<double*,Mem> means("means",n);
    Kokkos::View<int*,Mem> counts("counts",n);
    Kokkos::deep_copy(means,2.0);
    SPARTA_NS::GrainPoissonKokkos::sample_counts<Exec>(means,counts,12345,100);
    auto a=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),counts);
    SPARTA_NS::GrainPoissonKokkos::sample_counts<Exec>(means,counts,12345,100);
    auto b=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),counts);
    double sum=0.0, sum2=0.0; int zero=0;
    for(int i=0;i<n;++i) {
      if(a(i)!=b(i) || a(i)<0) failed=1;
      sum+=a(i);sum2+=double(a(i))*a(i);
      if(a(i)==0) ++zero;
    }
    const double avg=sum/n, variance=sum2/n-avg*avg;
    if(std::fabs(avg-2.0)>0.025 || std::fabs(variance-2.0)>0.06 ||
       std::fabs(double(zero)/n-std::exp(-2.0))>0.008) failed=1;
    Kokkos::deep_copy(means,0.0);
    SPARTA_NS::GrainPoissonKokkos::sample_counts<Exec>(means,counts,12345,100);
    auto z=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),counts);
    for(int i=0;i<n;++i) if(z(i)!=0) {failed=1;break;}
    Kokkos::deep_copy(means,17.0);
    SPARTA_NS::GrainPoissonKokkos::sample_counts<Exec>(means,counts,12345,100);
    auto invalid=Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(),counts);
    for(int i=0;i<n;++i) if(invalid(i)!=-1) {failed=1;break;}
  }
  Kokkos::finalize();
  if(failed) std::fprintf(stderr,"Grain Poisson sampler regression FAILED\n");
  return failed;
}
