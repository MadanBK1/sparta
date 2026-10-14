cmake_minimum_required(VERSION 3.20)
project(sparta_axisymmetric_grain_prototypes LANGUAGES CXX)
find_package(Kokkos CONFIG REQUIRED)
enable_testing()
foreach(t IN ITEMS
  test_kokkos_grain_integration
  test_kokkos_grain_contact
  test_kokkos_grain_wall_contact
  test_kokkos_grain_impulse_events
  test_kokkos_grain_impulse_force
  test_kokkos_grain_paired_collision
  test_kokkos_grain_gas_indexed_exchange
  test_kokkos_grain_mean_field_partial_exchange
  test_kokkos_grain_particle_momentum_audit
  test_kokkos_grain_momentum_budget_audit)
  add_executable(${t} ${t}.cpp)
  target_compile_features(${t} PRIVATE cxx_std_17)
  target_include_directories(${t} PRIVATE ${PROJECT_SOURCE_DIR}/../../src/KOKKOS)
  target_link_libraries(${t} PRIVATE Kokkos::kokkos)
  add_test(NAME ${t} COMMAND ${t})
endforeach()
