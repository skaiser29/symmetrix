#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <cstring>
#include <type_traits>

// TODO: this currently references libsymmetrix/tools_kokkos
#include "tools_kokkos.hpp"
#include "Kokkos_Core.hpp"

namespace py = pybind11;

template<typename T, int Flags>
Kokkos::View<const std::remove_const_t<T>*> create_kokkos_view(
    std::string label,
    py::array_t<T, Flags> array)
{
    using Scalar = std::remove_const_t<T>;
    py::array_t<Scalar, py::array::c_style | py::array::forcecast> contiguous(array);
    Kokkos::View<Scalar*> d_array(label, contiguous.size());
    auto h_array = Kokkos::create_mirror_view(d_array);
    std::memcpy(h_array.data(), contiguous.data(), sizeof(Scalar) * contiguous.size());
    Kokkos::deep_copy(d_array, h_array);
    return d_array;
}

template<typename T, int Flags>
void set_kokkos_view(
    Kokkos::View<T*>& view,
    py::array_t<T, Flags> array)
{
    py::array_t<T, py::array::c_style | py::array::forcecast> contiguous(array);
    if (view.size() != contiguous.size())
        Kokkos::realloc(view, contiguous.size());
    auto h_array = Kokkos::create_mirror_view(view);
    std::memcpy(h_array.data(), contiguous.data(), sizeof(T) * contiguous.size());
    Kokkos::deep_copy(view, h_array);
}

template<typename T, int Flags>
void set_kokkos_view(
    Kokkos::View<T**,Kokkos::LayoutRight>& view,
    py::array_t<T, Flags> array,
    const int N0,
    const int N1)
{
    py::array_t<T, py::array::c_style | py::array::forcecast> contiguous(array);
    if (view.extent(0) != N0 or view.extent(1) != N1)
        Kokkos::realloc(view, N0, N1);
    auto h_array = Kokkos::create_mirror_view(view);
    std::memcpy(h_array.data(), contiguous.data(), sizeof(T) * N0 * N1);
    Kokkos::deep_copy(view, h_array);
}

template<typename T, int Flags>
void set_kokkos_view(
    Kokkos::View<T***,Kokkos::LayoutRight>& view,
    py::array_t<T, Flags> array,
    const int N0,
    const int N1,
    const int N2)
{
    py::array_t<T, py::array::c_style | py::array::forcecast> contiguous(array);
    if (view.extent(0) != N0 or view.extent(1) != N1 or view.extent(2) != N2)
        Kokkos::realloc(view, N0, N1, N2);
    auto h_array = Kokkos::create_mirror_view(view);
    std::memcpy(h_array.data(), contiguous.data(), sizeof(T) * N0 * N1 * N2);
    Kokkos::deep_copy(view, h_array);
}
