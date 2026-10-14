/* Nonmutating split-size planner for weighted gas-grain populations.
   Keeps collision multiplicity separate from SPARTA's cell weight.
   This file DOES NOT create OnePart objects or adjust particle.weight.
   Input colliding count is physical molecules, 0 <= n <= W.
   A partial encounter needs TWO distinct populations; an exact 0 or W
   encounter needs just ONE. Invalid entries yield -1 to block scatter.
*/
#ifndef SPARTA_KOKKOS_GRAIN_SPLIT_PLAN_H
#define SPARTA_KOKKOS_GRAIN_SPLIT_PLAN_H
#include <Kokkos_Core.hpp>
namespace SPARTA_NS { namespace GrainSplitPlan {
template<class ExecSpace,class WeightView,class CountView,class SlotsView,
         class RemainView,class CollideView>
void plan(const WeightView& physical_weight,const CountView& impact_count,
          const SlotsView& slots,const RemainView& remaining,
          const CollideView& colliding) {
  const int n=static_cast<int>(physical_weight.extent(0));
  Kokkos::parallel_for("grain_collision_split_plan",
    Kokkos::RangePolicy<ExecSpace>(0,n),KOKKOS_LAMBDA(const int i) {
      const double w=physical_weight(i);
      const double k=static_cast<double>(impact_count(i));
      if (!(w>0.0 && k>=0.0 && k<=w) ||
          !(w<Kokkos::Experimental::infinity_v<double>)) {
        slots(i)=-1;remaining(i)=-1.0;colliding(i)=-1.0;
        return;
      }
      // Represented masses add to w, even for large W and small k.
      colliding(i)=k;
      remaining(i)=w-k;
      slots(i)=(k>0.0 && k<w) ? 2 : 1;
    });
}
}} // namespace
#endif
