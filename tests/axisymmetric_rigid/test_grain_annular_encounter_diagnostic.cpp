#include "grain_annular_encounter_diagnostic.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
using namespace SPARTA_NS::GrainAnnularEncounterDiagnostic;
int main() {
 const double pi=std::acos(-1.);
 const double volume=2.*pi;
 const auto a=evaluate(0.01,100.,1.e-3,volume,1.e5);
 const auto b=evaluate(0.01,100.,1.e-3,volume,1.e8);
 if(std::abs(a.optical_depth-(pi*1.e-4*100.*1.e-3/volume))>1.e-15) return 1;
 if(std::abs(a.single_molecule_probability-b.single_molecule_probability)>1.e-15) return 2;
 if(std::abs(b.expected_physical_impacts/a.expected_physical_impacts-1000.)>1.e-10) return 3;
 if(evaluate(0.01,0.,1.e-3,volume,1.e5).expected_physical_impacts!=0.) return 4;
 try {evaluate(0.01,100.,1.e-3,0.,1.e5);return 5;}
 catch(const std::invalid_argument&) {}
 std::cout<<"PASS annular gas representative exposure diagnostic\n";
}
