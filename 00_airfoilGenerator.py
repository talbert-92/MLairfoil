import os
import numpy as np
import random
import math
from scipy.special import comb

# --- CONFIGURATION ---
# General settings
OUTPUT_DIRECTORY = "generated_airfoils"
TARGET_NUM_AIRFOILS = 1000 # Desired final count of valid airfoils
NUM_POINTS_PER_SURFACE = 50
# This will generate 2 * NUM_POINTS_PER_SURFACE - 1 points (e.g., 99 points for NUM_POINTS_PER_SURFACE = 50).
# These points are derived from NUM_POINTS_PER_SURFACE cosine-spaced x-coordinates from x=0 to x=1.
# The list order is: Upper surface (TE to LE), then Lower surface (LE+1 to TE).

AIRFOIL_TYPES_TO_GENERATE = ['NACA4', 'NACA5', 'CST'] # Options: 'NACA4', 'NACA5', 'CST'

# NACA 4-Digit Parameters (Ranges for random generation)
# m: max camber in % chord (1st digit of NACA MPTT name)
# p: position of max camber in tenths of chord (2nd digit of NACA MPTT name)
# t: thickness in % chord (last two digits TT of NACA MPTT name)
NACA4_RANGES = {
    'm': (0, 9),  # Max camber (0% to 9% of chord), e.g., m=2 for NACA 2xxx
    'p': (0, 9),  # Position of max camber (0.0c to 0.9c), e.g., p=4 for NACA x4xx
                  # If m=0 (symmetric), p is usually 0 (NACA 00xx).
                  # If m>0 and p=0, formulas might be ill-defined or imply LE max camber.
                  # Code handles (m>0, p=0) by producing a symmetric airfoil.
    't': (6, 24)  # Thickness (6% to 24% of chord), e.g., t=12 for NACA xx12
}

# NACA 5-Digit Parameters (Ranges for random generation - NACA LPQXX)
# L: Design lift coefficient factor (L is the first digit). C_L_design approx L * 0.15.
# P: Position of max camber designator (P is the second digit). Relates to x_f/c = P * 0.05.
# Q: Camber line type (0 for standard, 1 for reflex). Fixed to 0 for this script.
# XX: Thickness in % chord (t/c * 100)
NACA5_RANGES = {
    'L': (0, 5),  # Corresponds to C_L_design up to approx 0.75. L=0 is symmetric.
    'P': (1, 5),  # Corresponds to x_f/c from 0.05 to 0.25. These are standard P values for NACA 5-digit.
                  # (P=1 => xf/c=0.05, P=2 => xf/c=0.10, ..., P=5 => xf/c=0.25)
    'Q': [0],     # Q=0 for standard camber line. Reflex camber (Q=1) is not implemented.
    'XX': (6, 24) # Thickness (6% to 24% of chord)
}

# CST Airfoil Parameters
CST_NUM_CONTROL_POINTS_UPPER = 6 # Number of Bernstein polynomial control points for upper surface (N_order = Num_points - 1)
CST_NUM_CONTROL_POINTS_LOWER = 6 # Number of Bernstein polynomial control points for lower surface

# Ranges for Bernstein polynomial weights A_upper[0]...A_upper[N], A_lower[0]...A_lower[N]
# A0 typically controls LE radius/shape (small positive for upper, small negative for lower for round LE).
# AN typically controls TE angle/shape contribution from Bernstein polynomials.
# Class function C(x) = x^N1 * (1-x)^N2 with N1=0.5 (round LE), N2=1.0 (sharp TE) is used.
CST_UPPER_COEFFS_RANGES = [(0.05, 0.25)] + [(-0.15, 0.15)] * (CST_NUM_CONTROL_POINTS_UPPER - 2) + [(-0.05, 0.1)]
CST_LOWER_COEFFS_RANGES = [(-0.25, -0.05)] + [(-0.15, 0.15)] * (CST_NUM_CONTROL_POINTS_LOWER - 2) + [(-0.1, 0.05)]
CST_TE_THICKNESS_RANGE = (0.0, 0.005) # Trailing edge thickness as fraction of chord (t_te/c).
# CST_LEADING_EDGE_RADIUS_RANGE is indirectly controlled by the ranges for the first coefficients (A_upper[0], A_lower[0]).
# For N1=0.5, R_le/c is approximately 0.5 * A_0^2 for symmetric sections.

# Geometric validation parameters
MIN_ANGLE_DEGREES_SMOOTHNESS = 160 # Minimum internal angle (degrees) between three consecutive points to be considered smooth.
# Lower values allow sharper turns. 180 is perfectly flat.

# --- THEORETICAL DOCUMENTATION AND GENERATION FUNCTIONS ---

def generate_cosine_spacing(n_points):
    """Generates n_points x-coordinates (from 0.0 to 1.0) using cosine spacing."""
    return 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_points)))

# --- NACA 4-Digit Airfoil ---
# Theory:
# NACA MPTT airfoils, where:
# M  = maximum camber in percent chord (M = m_param).
# P  = position of maximum camber in tenths of chord (P = p_param * 0.1c).
# TT = maximum thickness in percent chord (TT = t_param).
#
# Camber Line (y_c/c):
# If M=0 or P=0 (for m_param > 0, p_param=0 means max camber at LE, though usually handled as symmetric):
#   The code treats m_param=0 or p_param=0 as symmetric (y_c=0).
# For 0 <= x/c <= p:
#   y_c/c = (m/p^2) * (2*p*(x/c) - (x/c)^2)
# For p < x/c <= 1:
#   y_c/c = (m/(1-p)^2) * ((1-2*p) + 2*p*(x/c) - (x/c)^2)
# where m = m_param/100 and p = p_param/10.
#
# Thickness Distribution (y_t/c): Symmetric about the camber line.
#   y_t/c = (t/0.2) * (0.2969*sqrt(x/c) - 0.1260*(x/c) - 0.3516*(x/c)^2 + 0.2843*(x/c)^3 - 0.1015*(x/c)^4)
# where t = t_param/100. This formula results in a small finite trailing edge thickness.
# (Using -0.1036 for the last coefficient would close the trailing edge).
#
# Final Coordinates (x_u, y_u), (x_l, y_l):
#   theta = atan(dy_c/dx)
#   x_u/c = x/c - (y_t/c)*sin(theta);  y_u/c = y_c/c + (y_t/c)*cos(theta)
#   x_l/c = x/c + (y_t/c)*sin(theta);  y_l/c = y_c/c - (y_t/c)*cos(theta)
# The derivative dy_c/dx:
# For 0 <= x/c <= p: dy_c/dx = (2m/p^2) * (p - x/c)
# For p < x/c <= 1: dy_c/dx = (2m/(1-p)^2) * (p - x/c)

def _naca_4digit_camber_line_and_slope(x_over_c_array, m_param, p_param):
    yc_over_c = np.zeros_like(x_over_c_array)
    dyc_dx = np.zeros_like(x_over_c_array)

    m = m_param / 100.0
    p_chord_pos = p_param / 10.0

    # If m=0 (no camber) or p=0 (max camber at LE, treat as symmetric for simplicity here), yc=0 and dyc/dx=0.
    if m_param == 0 or p_param == 0:
        return yc_over_c, dyc_dx

    for i, x_c in enumerate(x_over_c_array):
        if x_c <= p_chord_pos:
            yc_over_c[i] = (m / (p_chord_pos**2)) * (2 * p_chord_pos * x_c - x_c**2)
            dyc_dx[i] = (2 * m / (p_chord_pos**2)) * (p_chord_pos - x_c)
        else:
            yc_over_c[i] = (m / ((1 - p_chord_pos)**2)) * ((1 - 2 * p_chord_pos) + 2 * p_chord_pos * x_c - x_c**2)
            dyc_dx[i] = (2 * m / ((1 - p_chord_pos)**2)) * (p_chord_pos - x_c)
    return yc_over_c, dyc_dx

def _naca_thickness_distribution(x_over_c_array, t_param):
    t = t_param / 100.0
    # Ensure x_over_c_array is non-negative for sqrt
    x_c_safe = np.maximum(x_over_c_array, 0)
    
    yt_over_c = (t / 0.2) * (
        0.2969 * np.sqrt(x_c_safe)
        - 0.1260 * x_c_safe
        - 0.3516 * x_c_safe**2
        + 0.2843 * x_c_safe**3
        - 0.1015 * x_c_safe**4  # Standard coefficient for finite TE
    )
    return np.maximum(yt_over_c, 0) # Ensure thickness is non-negative

def generate_naca_4digit(m_param, p_param, t_param, n_points_surface):
    name = f"NACA{m_param}{p_param}{str(t_param).zfill(2)}"
    if t_param < 1: # Avoid issues with zero or negative thickness
        return None, None, None, None, name 

    x_c_profile = generate_cosine_spacing(n_points_surface) # x/c from 0 to 1

    yc_over_c, dyc_dx = _naca_4digit_camber_line_and_slope(x_c_profile, m_param, p_param)
    yt_over_c = _naca_thickness_distribution(x_c_profile, t_param)

    theta = np.arctan(dyc_dx)
    
    xu_c = x_c_profile - yt_over_c * np.sin(theta)
    yu_c = yc_over_c + yt_over_c * np.cos(theta)
    xl_c = x_c_profile + yt_over_c * np.sin(theta)
    yl_c = yc_over_c - yt_over_c * np.cos(theta)

    # Force LE at (0,0) as NACA definitions ensure y_c(0)=0 and y_t(0)=0
    yu_c[0] = 0.0
    yl_c[0] = 0.0
    xu_c[0] = 0.0
    xl_c[0] = 0.0
    
    # Ensure TE x-coordinates are 1.0
    xu_c[-1] = 1.0
    xl_c[-1] = 1.0

    final_x_coords = np.concatenate((xu_c[::-1], xl_c[1:]))
    final_y_coords = np.concatenate((yu_c[::-1], yl_c[1:]))
    
    return final_x_coords, final_y_coords, x_c_profile, yu_c, yl_c, name

# --- NACA 5-Digit Airfoil ---
# Theory:
# NACA LPQXX airfoils:
# L  = Design lift coefficient factor. C_L_design approx. L_param * 0.15.
# P  = Position of max camber designator. x_f/c (position of max camber) = P_param * 0.05.
# Q  = Camber line type (0 for standard, 1 for reflex). This script implements Q=0 only.
# XX = Max thickness in % chord (XX_param).
#
# Camber Line (y_c/c) for Q=0:
# Parameters m_slope (a constant related to LE slope) and k1_const are determined by P_param.
# These constants are typically defined for a reference C_L_design (e.g., 0.15, when L=1).
# The final y_c ordinates are scaled by L_param.
#
# For 0 <= x/c <= p_val (where p_val = P_param * 0.05, the location of max camber):
#   y_c/c = (k1_const/6) * [ (x/c)^3 - 3*m_slope*(x/c)^2 + m_slope^2*(3-m_slope)*(x/c) ]
# For p_val < x/c <= 1:
#   y_c/c = (k1_const * m_slope^3 / 6) * (1 - x/c)
# These y_c/c values are for L=1. They are then multiplied by L_param for the actual camber.
#
# dy_c/dx (slope of camber line):
# For 0 <= x/c <= p_val: dy_c/dx = (k1_const/6) * [ 3*(x/c)^2 - 6*m_slope*(x/c) + m_slope^2*(3-m_slope) ]
# For p_val < x/c <= 1: dy_c/dx = - (k1_const * m_slope^3 / 6)
# These dy_c/dx values are also for L=1 and scaled by L_param.
#
# Thickness distribution is the same as for NACA 4-digit airfoils.
# Final coordinates are calculated similarly using theta = atan(dy_c/dx).

NACA5_CONSTANTS = { # P_digit: (p_val=xf/c, m_slope_param, k1_const for L=1 / Cl_design=0.15)
    1: (0.05,   0.0580, 361.40),  # For P=1 (e.g., NACA 210XX), xf/c = 0.05
    2: (0.10,   0.1260, 51.640), # For P=2 (e.g., NACA 220XX), xf/c = 0.10
    3: (0.15,   0.2025, 15.957), # For P=3 (e.g., NACA 230XX), xf/c = 0.15
    4: (0.20,   0.2900, 6.643),   # For P=4 (e.g., NACA 240XX), xf/c = 0.20
    5: (0.25,   0.3910, 3.230)    # For P=5 (e.g., NACA 250XX), xf/c = 0.25
} # k1 values based on Sheldahl & Klimas, SAND80-1969.

def _naca_5digit_camber_line_and_slope(x_over_c_array, L_param, P_param):
    if L_param == 0: # Symmetric airfoil
        return np.zeros_like(x_over_c_array), np.zeros_like(x_over_c_array)

    if P_param not in NACA5_CONSTANTS:
        raise ValueError(f"NACA 5-digit P_param {P_param} is not supported. Use keys from NACA5_CONSTANTS (1-5).")

    p_val_actual_xf, m_slope_const, k1_const = NACA5_CONSTANTS[P_param]

    yc_over_c_ref = np.zeros_like(x_over_c_array)
    dyc_dx_ref = np.zeros_like(x_over_c_array)

    for i, x_c in enumerate(x_over_c_array):
        if x_c <= p_val_actual_xf:
            term_xc = x_c * x_c
            yc_over_c_ref[i] = (k1_const / 6.0) * (term_xc * x_c - 3 * m_slope_const * term_xc + m_slope_const**2 * (3 - m_slope_const) * x_c)
            dyc_dx_ref[i] = (k1_const / 6.0) * (3 * term_xc - 6 * m_slope_const * x_c + m_slope_const**2 * (3 - m_slope_const))
        else:
            yc_over_c_ref[i] = (k1_const * m_slope_const**3 / 6.0) * (1 - x_c)
            dyc_dx_ref[i] = - (k1_const * m_slope_const**3 / 6.0)
    
    # Scale by L_param (since k1_const values are for L=1, effectively C_L_design=0.15)
    yc_over_c = L_param * yc_over_c_ref
    dyc_dx = L_param * dyc_dx_ref
    
    return yc_over_c, dyc_dx

def generate_naca_5digit(L_param, P_param, Q_param, XX_param, n_points_surface):
    if Q_param != 0:
        # This script only implements Q=0 (standard camber)
        Q_param = 0 
    name = f"NACA{L_param}{P_param}{Q_param}{str(XX_param).zfill(2)}"

    if XX_param < 1:
        return None, None, None, None, name
    if P_param not in NACA5_CONSTANTS: # Should be caught by random generator choice, but good check
        return None, None, None, None, name

    x_c_profile = generate_cosine_spacing(n_points_surface)

    yc_over_c, dyc_dx = _naca_5digit_camber_line_and_slope(x_c_profile, L_param, P_param)
    yt_over_c = _naca_thickness_distribution(x_c_profile, XX_param)
    
    theta = np.arctan(dyc_dx)

    xu_c = x_c_profile - yt_over_c * np.sin(theta)
    yu_c = yc_over_c + yt_over_c * np.cos(theta)
    xl_c = x_c_profile + yt_over_c * np.sin(theta)
    yl_c = yc_over_c - yt_over_c * np.cos(theta)

    yu_c[0], yl_c[0] = 0.0, 0.0 # Force LE at (0,0)
    xu_c[0], xl_c[0] = 0.0, 0.0
    xu_c[-1], xl_c[-1] = 1.0, 1.0 # Force TE x-coord to 1.0

    final_x_coords = np.concatenate((xu_c[::-1], xl_c[1:]))
    final_y_coords = np.concatenate((yu_c[::-1], yl_c[1:]))

    return final_x_coords, final_y_coords, x_c_profile, yu_c, yl_c, name

# --- CST (Class Shape Transformation) Airfoil ---
# Theory:
# y(x/c) = C(x/c) * S(x/c) + (x/c) * dz_TE
# where:
#   x/c = non-dimensional chord position (0 to 1).
#   C(x/c) = Class function. Typically C(x/c) = (x/c)^N1 * (1 - x/c)^N2.
#     N1 = 0.5 for a rounded leading edge.
#     N2 = 1.0 for a sharp trailing edge (if dz_TE also accounts for closure).
#   S(x/c) = Shape function. S(x/c) = sum_{i=0}^{N_order} [ A_i * B_i(x/c) ]
#     A_i     = Bernstein polynomial coefficients (weights).
#     N_order = Order of the Bernstein polynomial (Number of control points = N_order + 1).
#     B_i(x/c) = Bernstein basis polynomial: B_i(x/c) = K_i * (x/c)^i * (1 - x/c)^(N_order - i).
#     K_i     = Binomial coefficient: K_i = comb(N_order, i) = N_order! / (i! * (N_order - i)!).
#   dz_TE = y-coordinate of the trailing edge point (as fraction of chord).
#           For upper surface: dz_TE = te_thickness_abs / 2.0
#           For lower surface: dz_TE = -te_thickness_abs / 2.0
#
# Separate shape functions S_upper(x/c) and S_lower(x/c) are defined using distinct sets of coefficients A_ui and A_li.
# y_upper(x/c) = C(x/c) * S_upper(x/c) + (x/c) * (te_thickness_abs / 2.0)
# y_lower(x/c) = C(x/c) * S_lower(x/c) + (x/c) * (-te_thickness_abs / 2.0)

def _class_function(x_over_c, N1=0.5, N2=1.0):
    """Class function C(x) = x^N1 * (1-x)^N2."""
    # np.power handles x=0 and x=1 correctly if N1, N2 are positive.
    term1 = np.power(x_over_c, N1)
    term2 = np.power(1.0 - x_over_c, N2)
    return term1 * term2

def _shape_function_sum(x_over_c, coeffs):
    """Shape function S(x) = sum(A_i * B_i(x))."""
    N_order = len(coeffs) - 1
    if N_order < 0: return np.zeros_like(x_over_c) # No coefficients

    S_x = np.zeros_like(x_over_c)
    for i, A_i in enumerate(coeffs):
        K_i = comb(N_order, i)
        term_x_i = np.power(x_over_c, i)
        term_1_minus_x_N_minus_i = np.power(1.0 - x_over_c, N_order - i)
        B_i_x = K_i * term_x_i * term_1_minus_x_N_minus_i
        S_x += A_i * B_i_x
    return S_x

def generate_cst_airfoil(coeffs_upper, coeffs_lower, te_thickness_abs, n_points_surface):
    N_upper_pts = len(coeffs_upper)
    N_lower_pts = len(coeffs_lower)
    
    name_parts = ["CST", f"Nu{N_upper_pts}", f"Nl{N_lower_pts}"]
    name_parts.append(f"TE{te_thickness_abs:.4f}".replace('.', 'p'))
    # Example of adding first coefficients to name for more detail (can make filenames long)
    # name_parts.append(f"Au0_{coeffs_upper[0]:.2f}_Al0_{coeffs_lower[0]:.2f}".replace('.', 'p').replace('-', 'm'))
    name = "_".join(name_parts)
    name = name.replace('-', 'neg') # Ensure filename characters are valid

    x_c_profile = generate_cosine_spacing(n_points_surface)

    C_x = _class_function(x_c_profile, N1=0.5, N2=1.0)

    S_upper_x = _shape_function_sum(x_c_profile, coeffs_upper)
    S_lower_x = _shape_function_sum(x_c_profile, coeffs_lower)

    y_te_upper_offset = te_thickness_abs / 2.0
    y_te_lower_offset = -te_thickness_abs / 2.0

    y_upper = C_x * S_upper_x + x_c_profile * y_te_upper_offset
    y_lower = C_x * S_lower_x + x_c_profile * y_te_lower_offset
    
    # Enforce LE closure at (0,0) and TE y-coordinates
    # C(0)=0 and x*dz_TE=0 at x=0, so y_upper[0] and y_lower[0] should be 0.
    y_upper[x_c_profile < 1e-9] = 0.0
    y_lower[x_c_profile < 1e-9] = 0.0
    # C(1)=0, so y_upper[x=1] = 1.0 * y_te_upper_offset, similar for lower.
    y_upper[x_c_profile > 1.0 - 1e-9] = y_te_upper_offset
    y_lower[x_c_profile > 1.0 - 1e-9] = y_te_lower_offset

    # Ensure x-coordinates are exactly 0 and 1 at ends
    x_final = x_c_profile.copy()
    x_final[0] = 0.0
    x_final[-1] = 1.0

    final_x_coords = np.concatenate((x_final[::-1], x_final[1:]))
    final_y_coords = np.concatenate((y_upper[::-1], y_lower[1:]))
    
    return final_x_coords, final_y_coords, x_c_profile, y_upper, y_lower, name

# --- GEOMETRIC CHECKS ---
def check_self_intersection(x_profile_coords, y_upper_surf, y_lower_surf):
    """
    Checks for self-intersection: y_upper must be >= y_lower for all x in (0,1).
    Also checks if total thickness is excessively small or negative.
    """
    if len(x_profile_coords) != len(y_upper_surf) or len(x_profile_coords) != len(y_lower_surf):
         # This case should ideally not be reached if inputs are correct
         return False 

    # Check y_upper >= y_lower for x in (0, 1), ignoring LE and TE points for strict inequality
    # Using a small tolerance for floating point comparisons
    min_thickness_allowed = -1e-5 # Allow very slight negative thickness due to numerics
    if np.any(y_upper_surf[1:-1] < y_lower_surf[1:-1] + min_thickness_allowed):
        return False # Intersection detected
    
    # Check for overall non-positive thickness (collapsed airfoil)
    thickness = y_upper_surf - y_lower_surf
    if np.all(thickness[1:-1] < 1e-6): # If nearly zero thickness everywhere except ends
        return False
        
    return True

def check_smoothness(x_combined_coords, y_combined_coords, min_angle_deg):
    """
    Checks for profile smoothness by examining angles between successive segments.
    An angle close to 180 degrees (pi radians) is smooth. Small angles indicate kinks.
    """
    if len(x_combined_coords) < 3:
        return True # Not enough points to form an angle

    for i in range(1, len(x_combined_coords) - 1):
        p_prev = np.array([x_combined_coords[i-1], y_combined_coords[i-1]])
        p_curr = np.array([x_combined_coords[i],   y_combined_coords[i]])
        p_next = np.array([x_combined_coords[i+1], y_combined_coords[i+1]])

        vec1 = p_prev - p_curr
        vec2 = p_next - p_curr

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 < 1e-9 or norm2 < 1e-9: # Points are coincident, indicates a problem
            return False 

        dot_product = np.dot(vec1, vec2)
        cos_theta = dot_product / (norm1 * norm2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0) # Avoid domain errors with acos
        
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)

        if angle_deg < min_angle_deg:
            return False # Kink detected
    return True

def check_geometry_validity(x_combined, y_combined, x_profile_orig, y_upper_surf, y_lower_surf):
    """Combines all geometric checks for an airfoil."""
    if not check_self_intersection(x_profile_orig, y_upper_surf, y_lower_surf):
        return False
    if not check_smoothness(x_combined, y_combined, min_angle_deg=MIN_ANGLE_DEGREES_SMOOTHNESS):
        return False
    return True

# --- UTILITY FUNCTIONS ---
def save_airfoil_dat(airfoil_name, x_coords, y_coords, directory):
    """Saves airfoil coordinates to a .dat file in the specified format."""
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    filepath = os.path.join(directory, f"{airfoil_name}.dat")
    
    with open(filepath, 'w') as f:
        f.write(f"{airfoil_name}\n")
        for i in range(len(x_coords)):
            # Format: X.XXXX       Y.YYYYY (example: 1.0000       0.00063)
            # x_coord: 6 characters wide, 4 decimal places
            # y_coord: 7 characters wide, 5 decimal places
            # Spacing: "       " (7 spaces)
            f.write(f"{x_coords[i]:6.4f}       {y_coords[i]:7.5f}\n")

def randomly_generate_naca4_params():
    m = random.randint(NACA4_RANGES['m'][0], NACA4_RANGES['m'][1])
    p = random.randint(NACA4_RANGES['p'][0], NACA4_RANGES['p'][1])
    t = random.randint(NACA4_RANGES['t'][0], NACA4_RANGES['t'][1])
    return {'m': m, 'p': p, 't': t}

def randomly_generate_naca5_params():
    L = random.randint(NACA5_RANGES['L'][0], NACA5_RANGES['L'][1])
    P = random.choice(list(NACA5_CONSTANTS.keys())) # P must be one of the defined keys for standard NACA5 series.
    Q = random.choice(NACA5_RANGES['Q']) # Currently fixed to [0]
    XX = random.randint(NACA5_RANGES['XX'][0], NACA5_RANGES['XX'][1])
    return {'L': L, 'P': P, 'Q': Q, 'XX': XX}

def randomly_generate_cst_params():
    coeffs_upper = [random.uniform(r[0], r[1]) for r in CST_UPPER_COEFFS_RANGES]
    coeffs_lower = [random.uniform(r[0], r[1]) for r in CST_LOWER_COEFFS_RANGES]
    te_thickness = random.uniform(CST_TE_THICKNESS_RANGE[0], CST_TE_THICKNESS_RANGE[1])
    return {'coeffs_upper': coeffs_upper, 'coeffs_lower': coeffs_lower, 'te_thickness': te_thickness}

# --- MAIN SCRIPT LOGIC ---
def main():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created output directory: {OUTPUT_DIRECTORY}")

    valid_airfoils_count = 0
    generated_attempts = 0
    generated_names = set() # To avoid saving duplicate airfoils if parameters repeat

    print(f"🚀 Starting airfoil generation...")
    print(f"🎯 Target: {TARGET_NUM_AIRFOILS} valid airfoils.")
    print(f"💾 Output directory: {os.path.abspath(OUTPUT_DIRECTORY)}")
    print(f"⚙️  Points per surface definition: {NUM_POINTS_PER_SURFACE} (total {2*NUM_POINTS_PER_SURFACE-1} points in .dat file)")
    print(f"✈️  Airfoil types to generate: {', '.join(AIRFOIL_TYPES_TO_GENERATE)}")
    print("-" * 40)

    max_attempts = TARGET_NUM_AIRFOILS * 100 # Stop if rejection rate is too high
    
    while valid_airfoils_count < TARGET_NUM_AIRFOILS and generated_attempts < max_attempts:
        generated_attempts += 1
        
        airfoil_type = random.choice(AIRFOIL_TYPES_TO_GENERATE)
        
        combined_x, combined_y = None, None
        profile_x_orig, profile_y_upper, profile_y_lower = None, None, None
        airfoil_name = "Unknown"

        if airfoil_type == 'NACA4':
            params = randomly_generate_naca4_params()
            combined_x, combined_y, profile_x_orig, profile_y_upper, profile_y_lower, airfoil_name = generate_naca_4digit(
                params['m'], params['p'], params['t'], NUM_POINTS_PER_SURFACE
            )
        elif airfoil_type == 'NACA5':
            params = randomly_generate_naca5_params()
            combined_x, combined_y, profile_x_orig, profile_y_upper, profile_y_lower, airfoil_name = generate_naca_5digit(
                params['L'], params['P'], params['Q'], params['XX'], NUM_POINTS_PER_SURFACE
            )
        elif airfoil_type == 'CST':
            params = randomly_generate_cst_params()
            combined_x, combined_y, profile_x_orig, profile_y_upper, profile_y_lower, airfoil_name = generate_cst_airfoil(
                params['coeffs_upper'], params['coeffs_lower'], params['te_thickness'], NUM_POINTS_PER_SURFACE
            )

        if combined_x is None: # Generation function itself failed (e.g., invalid params)
            # print(f"Attempt {generated_attempts}: Generation failed for {airfoil_name}.") # Optional debug
            continue
        
        if airfoil_name in generated_names:
            # print(f"Attempt {generated_attempts}: Airfoil {airfoil_name} already generated. Skipping.") # Optional debug
            continue

        is_valid = False
        if profile_x_orig is not None and profile_y_upper is not None and profile_y_lower is not None:
             is_valid = check_geometry_validity(combined_x, combined_y, profile_x_orig, profile_y_upper, profile_y_lower)
        
        if is_valid:
            save_airfoil_dat(airfoil_name, combined_x, combined_y, OUTPUT_DIRECTORY)
            generated_names.add(airfoil_name)
            valid_airfoils_count += 1
            print(f"✅ Generated ({valid_airfoils_count}/{TARGET_NUM_AIRFOILS}): {airfoil_name} (Attempt {generated_attempts})")
        else:
            # print(f"Attempt {generated_attempts}: Invalid geometry for {airfoil_name}. Skipping.") # Optional debug
            pass
        
        if generated_attempts % (TARGET_NUM_AIRFOILS // 10 or 1) == 0 and generated_attempts > 0: # Progress update
             if valid_airfoils_count > 0:
                rejection_rate = (generated_attempts - valid_airfoils_count) / generated_attempts * 100
                print(f"   Progress: {valid_airfoils_count}/{TARGET_NUM_AIRFOILS} valid. Attempts: {generated_attempts}. Rejection: {rejection_rate:.1f}%")


    print("-" * 40)
    if valid_airfoils_count >= TARGET_NUM_AIRFOILS:
        print(f"🎉 Success! Generated {valid_airfoils_count} valid airfoils in {generated_attempts} attempts.")
    else:
        print(f"⚠️ Warning: Generation stopped. Only {valid_airfoils_count}/{TARGET_NUM_AIRFOILS} valid airfoils generated after {generated_attempts} attempts.")
    print(f"💾 Files saved in: {os.path.abspath(OUTPUT_DIRECTORY)}")

if __name__ == '__main__':
    main()