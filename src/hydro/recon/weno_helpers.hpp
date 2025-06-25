#ifndef WENO_HELPERS_HPP
#define WENO_HELPERS_HPP

#include <iostream> 
#include <memory>
#include <string>
#include <vector>
#include <cmath>

// Parthenon headers
#include <parthenon/package.hpp>

// WENO5-HJ Function
KOKKOS_INLINE_FUNCTION
Real WENO5_HJ(const Real &a, const Real &b, const Real &c, const Real &d) {
    
    const Real epsilon = 1E-6;

    Real IS0 = 13 * std::pow(a - b, 2) + 3 * std::pow(a - 3 * b, 2);
    Real IS1 = 13 * std::pow(b - c, 2) + 3 * std::pow(b + c, 2);
    Real IS2 = 13 * std::pow(c - d, 2) + 3 * std::pow(3 * c - d, 2);

    Real alpha0 = 1.0 / std::pow(epsilon + IS0, 2);
    Real alpha1 = 6.0 / std::pow(epsilon + IS1, 2);
    Real alpha2 = 3.0 / std::pow(epsilon + IS2, 2);

    Real omega0 = alpha0 / (alpha0 + alpha1 + alpha2);
    Real omega2 = alpha2 / (alpha0 + alpha1 + alpha2);

    Real phi = (1.0 / 3.0) * omega0 * (a - 2 * b + c) + (1.0 / 6.0) * (omega2 - 0.5) * (b - 2 * c + d);

    return phi;
}

// HJ-FLUX Function
KOKKOS_INLINE_FUNCTION
void HJ_FLUX(Real &u_x_plus, Real &u_x_minus, 
  const Real &q_im3, const Real &q_im2, const Real &q_im1, const Real &q_i,
  const Real &q_ip1, const Real &q_ip2, const Real &q_ip3, const Real &dx) {

    // Compute the First Derivatives
    Real der1 = (q_im1 - q_im2) / dx; // i-1 - i-2
    Real der2 = (q_i   - q_im1) / dx; // i   - i-1
    Real der3 = (q_ip1 - q_i)   / dx; // i+1 - i
    Real der4 = (q_ip2 - q_ip1) / dx; // i+2 - i+1

    // Compute the Common Term in Equation (2.6)
    Real common = (-der1 + 7 * der2 + 7 * der3 - der4) / 12.0;

    // Compute the Second Derivatives
    Real secder1 = (q_ip3 - 2 * q_ip2 + q_ip1) / dx; // i+3 - 2i+2 + i+1
    Real secder2 = (q_ip2 - 2 * q_ip1 + q_i)   / dx; // i+2 - 2i+1 + i
    Real secder3 = (q_ip1 - 2 * q_i   + q_im1) / dx; // i+1 - 2i   + i-1
    Real secder4 = (q_i   - 2 * q_im1 + q_im2) / dx; // i   - 2i-1 + i-2
    Real secder5 = (q_im1 - 2 * q_im2 + q_im3) / dx; // i-1 - 2i-2 + i-3

    // Compute the WENO Reconstruction
    Real weno_plus_flux = WENO5_HJ(secder1, secder2, secder3, secder4);
    u_x_plus = common + weno_plus_flux;   // Equation (2.9)
    Real weno_minus_flux = WENO5_HJ(secder5, secder4, secder3, secder2);
    u_x_minus = common - weno_minus_flux; // Equation (2.6)
    
}

#endif // WENO_HELPERS_HPP
