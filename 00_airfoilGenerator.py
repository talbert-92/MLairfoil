import os
import numpy as np
import random
import math
from itertools import product

# ==============================================================================
# --- USER CONFIGURATION: The "Control Panel" for Airfoil Generation ---
# ==============================================================================
# This is the main section for any user to modify.
# - To change the number of airfoils, edit 'num_to_generate'.
# - To change how airfoils are shaped, edit the 'range' for each parameter.
# - The 'help' text explains what each parameter does.
# ==============================================================================

GENERATION_PARAMETERS = {
    'NACA4': {
        'num_to_generate': 500,
        'parameters': {
            'm': {
                'range': (0, 9),
                'type': 'int',
                'help': 'Max camber in percent of chord (e.g., 2 for NACA 2xxx). This is the M in MPTT.'
            },
            'p': {
                'range': (1, 9),
                'type': 'int',
                'help': 'Position of max camber in tenths of chord (e.g., 4 for NACA x4xx). This is the P in MPTT.'
            },
            't': {
                'range': (6, 24),
                'type': 'int',
                'help': 'Max thickness in percent of chord (e.g., 12 for NACA xx12). This is the TT in MPTT.'
            }
        }
    },
    'NACA5': {
        'num_to_generate': 500,
        'parameters': {
            'L': {
                'range': (1, 6),
                'type': 'int',
                'help': 'Design lift coefficient factor (CL ~ L * 0.15). This is the L in LPQXX.'
            },
            'P': {
                'choices': [1, 2, 3, 4, 5],
                'type': 'choice',
                'help': 'Position of max camber code (e.g., P=2 for xf/c=0.10). This is the P in LPQXX.'
            },
            'Q': {
                'choices': [0],
                'type': 'choice',
                'help': 'Camber line type (0 for standard, 1 for reflex). This is the Q in LPQXX. Fixed to 0.'
            },
            'XX': {
                'range': (6, 25),
                'type': 'int',
                'help': 'Max thickness in percent of chord (e.g., 12 for NACA xx12). This is the XX in LPQXX.'
            }
        }
    },
    'NACA6': {
        'num_to_generate': 500,
        'parameters': {
            'S': {
                'range': range(1, 10),
                'type': 'range',
                'help': 'Series & min pressure location code (e.g., S=4 for NACA 64-xxx series at 40% chord).'
            },
            'L': {
                'range': range(1, 7),
                'type': 'range',
                'help': 'Design lift coefficient in tenths (e.g., L=2 for a design CL of 0.2).'
            },
            'tt': {
                'range': range(6, 19),
                'type': 'range',
                'help': 'Max thickness in percent of chord (e.g., tt=12 for 12% thickness).'
            }
        }
    }
}

# General script settings (usually no need to change)
OUTPUT_DIRECTORY = "generated_airfoils"
NUM_POINTS_PER_SURFACE = 100
MIN_ANGLE_DEGREES_SMOOTHNESS = 90

# ==============================================================================
# --- END OF USER CONFIGURATION ---
# The rest of the script contains the generation logic.
# ==============================================================================


# --- THEORETICAL DOCUMENTATION AND GENERATION FUNCTIONS ---

def generate_cosine_spacing(n_points):
    """Generates n_points x-coordinates (from 0.0 to 1.0) using cosine spacing."""
    return 0.5 * (1.0 - np.cos(np.linspace(0, np.pi, n_points)))

# --- NACA 4-Digit ---
def _naca_4digit_camber_line_and_slope(x_over_c_array, m_param, p_param):
    yc_over_c = np.zeros_like(x_over_c_array)
    dyc_dx = np.zeros_like(x_over_c_array)
    m = m_param / 100.0
    p_chord_pos = p_param / 10.0
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
    x_c_safe = np.maximum(x_over_c_array, 0)
    yt_over_c = (t / 0.2) * (
        0.2969 * np.sqrt(x_c_safe)
        - 0.1260 * x_c_safe
        - 0.3516 * x_c_safe**2
        + 0.2843 * x_c_safe**3
        - 0.1015 * x_c_safe**4
    )
    return np.maximum(yt_over_c, 0)

def generate_naca_4digit(m_param, p_param, t_param, n_points_surface):
    name = f"NACA{m_param}{p_param}{str(t_param).zfill(2)}"
    if t_param < 1: return None, None, None, None, None, name
    x_c_profile = generate_cosine_spacing(n_points_surface)
    yc_over_c, dyc_dx = _naca_4digit_camber_line_and_slope(x_c_profile, m_param, p_param)
    yt_over_c = _naca_thickness_distribution(x_c_profile, t_param)
    theta = np.arctan(dyc_dx)
    xu_c, yu_c = x_c_profile - yt_over_c * np.sin(theta), yc_over_c + yt_over_c * np.cos(theta)
    xl_c, yl_c = x_c_profile + yt_over_c * np.sin(theta), yc_over_c - yt_over_c * np.cos(theta)
    yu_c[0], yl_c[0], xu_c[0], xl_c[0] = 0.0, 0.0, 0.0, 0.0
    xu_c[-1], xl_c[-1] = 1.0, 1.0
    final_x_coords = np.concatenate((xu_c[::-1], xl_c[1:]))
    final_y_coords = np.concatenate((yu_c[::-1], yl_c[1:]))
    return final_x_coords, final_y_coords, x_c_profile, yu_c, yl_c, name

# --- NACA 5-Digit ---
NACA5_CONSTANTS = {
    1: (0.05, 0.0580, 361.40), 2: (0.10, 0.1260, 51.640), 3: (0.15, 0.2025, 15.957),
    4: (0.20, 0.2900, 6.643), 5: (0.25, 0.3910, 3.230)
}

def _naca_5digit_camber_line_and_slope(x_over_c_array, L_param, P_param):
    if L_param == 0: return np.zeros_like(x_over_c_array), np.zeros_like(x_over_c_array)
    if P_param not in NACA5_CONSTANTS: raise ValueError(f"NACA 5-digit P_param {P_param} is not supported.")
    p_val_actual_xf, m_slope_const, k1_const = NACA5_CONSTANTS[P_param]
    yc_over_c_ref, dyc_dx_ref = np.zeros_like(x_over_c_array), np.zeros_like(x_over_c_array)
    for i, x_c in enumerate(x_over_c_array):
        if x_c <= p_val_actual_xf:
            term_xc = x_c * x_c
            yc_over_c_ref[i] = (k1_const / 6.0) * (term_xc * x_c - 3 * m_slope_const * term_xc + m_slope_const**2 * (3 - m_slope_const) * x_c)
            dyc_dx_ref[i] = (k1_const / 6.0) * (3 * term_xc - 6 * m_slope_const * x_c + m_slope_const**2 * (3 - m_slope_const))
        else:
            yc_over_c_ref[i] = (k1_const * m_slope_const**3 / 6.0) * (1 - x_c)
            dyc_dx_ref[i] = - (k1_const * m_slope_const**3 / 6.0)
    return L_param * yc_over_c_ref, L_param * dyc_dx_ref

def generate_naca_5digit(L_param, P_param, Q_param, XX_param, n_points_surface):
    name = f"NACA{L_param}{P_param}{Q_param}{str(XX_param).zfill(2)}"
    if XX_param < 1 or P_param not in NACA5_CONSTANTS: return None, None, None, None, None, name
    x_c_profile = generate_cosine_spacing(n_points_surface)
    yc_over_c, dyc_dx = _naca_5digit_camber_line_and_slope(x_c_profile, L_param, P_param)
    yt_over_c = _naca_thickness_distribution(x_c_profile, XX_param)
    theta = np.arctan(dyc_dx)
    xu_c, yu_c = x_c_profile - yt_over_c * np.sin(theta), yc_over_c + yt_over_c * np.cos(theta)
    xl_c, yl_c = x_c_profile + yt_over_c * np.sin(theta), yc_over_c - yt_over_c * np.cos(theta)
    yu_c[0], yl_c[0], xu_c[0], xl_c[0] = 0.0, 0.0, 0.0, 0.0
    xu_c[-1], xl_c[-1] = 1.0, 1.0
    final_x_coords, final_y_coords = np.concatenate((xu_c[::-1], xl_c[1:])), np.concatenate((yu_c[::-1], yl_c[1:]))
    return final_x_coords, final_y_coords, x_c_profile, yu_c, yl_c, name

# --- NACA 6-Digit ---
NACA6_BASE_COORDS_X = np.array([
    1.0000, 0.9500, 0.9000, 0.8000, 0.7000, 0.6000, 0.5000, 0.4000, 0.3000,
    0.2500, 0.2000, 0.1500, 0.1000, 0.0750, 0.0500, 0.0250, 0.0125, 0.0000
])
NACA6_BASE_COORDS_Y = np.array([
    0.0010, 0.0053, 0.0099, 0.0189, 0.0279, 0.0365, 0.0446, 0.0514, 0.0565,
    0.0577, 0.0575, 0.0556, 0.0508, 0.0468, 0.0413, 0.0315, 0.0232, 0.0000
])

def _naca_6digit_camber_line_and_slope(x_over_c, design_cl, x_p):
    yc_over_c, dyc_dx = np.zeros_like(x_over_c), np.zeros_like(x_over_c)
    if abs(1.0 - x_p) < 1e-9: return yc_over_c, dyc_dx
    mask = x_over_c <= x_p
    x_front = x_over_c[mask]
    yc_over_c[mask] = (design_cl / (6.0 * (1 - x_p))) * (-x_front**3 + 3 * x_p * x_front**2 - 3 * x_p**2 * x_front)
    dyc_dx[mask] = (design_cl / (2.0 * (1 - x_p))) * (-x_front**2 + 2 * x_p * x_front - x_p**2)
    yc_over_c[~mask], dyc_dx[~mask] = (design_cl * x_p**3) / (6.0 * (1 - x_p)), 0.0
    return yc_over_c, dyc_dx

def generate_naca_6digit(S_param, L_param, tt_param, n_points_surface):
    name = f"NACA6{S_param}{L_param}{tt_param:02d}"
    if tt_param < 1: return None, None, None, None, None, name
    x_c_profile = generate_cosine_spacing(n_points_surface)
    design_cl, x_p = L_param / 10.0, S_param / 10.0
    yc_over_c, dyc_dx = _naca_6digit_camber_line_and_slope(x_c_profile, design_cl, x_p)
    base_yt = np.interp(x_c_profile, NACA6_BASE_COORDS_X[::-1], NACA6_BASE_COORDS_Y[::-1])
    yt_over_c = base_yt * (tt_param / 12.0)
    theta = np.arctan(dyc_dx)
    xu_c, yu_c = x_c_profile - yt_over_c * np.sin(theta), yc_over_c + yt_over_c * np.cos(theta)
    xl_c, yl_c = x_c_profile + yt_over_c * np.sin(theta), yc_over_c - yt_over_c * np.cos(theta)
    yu_c[0], yl_c[0], xu_c[0], xl_c[0] = 0.0, 0.0, 0.0, 0.0
    xu_c[-1], xl_c[-1] = 1.0, 1.0
    final_x_coords, final_y_coords = np.concatenate((xu_c[::-1], xl_c[1:])), np.concatenate((yu_c[::-1], yl_c[1:]))
    return final_x_coords, final_y_coords, x_c_profile, yu_c, yl_c, name

# --- GEOMETRIC CHECKS AND UTILITIES ---
def check_geometry_validity(x_combined, y_combined, x_profile_orig, y_upper_surf, y_lower_surf):
    if x_combined is None: return False
    # Check for self-intersection
    if np.any(y_upper_surf[1:-1] < y_lower_surf[1:-1] - 1e-5): return False
    # Check for collapsed profile
    if np.all((y_upper_surf - y_lower_surf)[1:-1] < 1e-6): return False
    # Check for smoothness (kinks)
    if len(x_combined) < 3: return True
    for i in range(1, len(x_combined) - 1):
        p_prev, p_curr, p_next = np.array([x_combined[i-1], y_combined[i-1]]), np.array([x_combined[i], y_combined[i]]), np.array([x_combined[i+1], y_combined[i+1]])
        vec1, vec2 = p_prev - p_curr, p_next - p_curr
        norm1, norm2 = np.linalg.norm(vec1), np.linalg.norm(vec2)
        if norm1 < 1e-9 or norm2 < 1e-9: return False
        cos_theta = np.clip(np.dot(vec1, vec2) / (norm1 * norm2), -1.0, 1.0)
        if np.degrees(np.arccos(cos_theta)) < MIN_ANGLE_DEGREES_SMOOTHNESS: return False
    return True

def save_airfoil_dat(airfoil_name, x_coords, y_coords, directory):
    if not os.path.exists(directory): os.makedirs(directory)
    filepath = os.path.join(directory, f"{airfoil_name}.dat")
    with open(filepath, 'w') as f:
        f.write(f"{airfoil_name}\n")
        for i in range(len(x_coords)):
            f.write(f"{x_coords[i]:6.4f}       {y_coords[i]:7.5f}\n")

# --- PARAMETER GENERATION LOGIC ---
def get_random_params(family_key):
    """Generates a dictionary of random parameters for a given family key."""
    params = {}
    param_config = GENERATION_PARAMETERS[family_key]['parameters']
    for key, config in param_config.items():
        if config['type'] == 'int':
            params[key] = random.randint(config['range'][0], config['range'][1])
        elif config['type'] == 'choice':
            params[key] = random.choice(config['choices'])
    return params

def get_naca6_parameter_list():
    """Generates a shuffled list of ALL unique NACA 6-digit parameter combinations."""
    config = GENERATION_PARAMETERS['NACA6']['parameters']
    s_range, l_range, tt_range = config['S']['range'], config['L']['range'], config['tt']['range']
    all_combinations = list(product(s_range, l_range, tt_range))
    param_list = [{'S': S, 'L': L, 'tt': tt} for S, L, tt in all_combinations]
    random.shuffle(param_list)
    return param_list

# --- MAIN SCRIPT LOGIC ---
def main():
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.makedirs(OUTPUT_DIRECTORY)
        print(f"Created output directory: {OUTPUT_DIRECTORY}")

    families_to_generate = GENERATION_PARAMETERS.keys()
    overall_target = sum(config['num_to_generate'] for config in GENERATION_PARAMETERS.values())
    total_generated_count = 0

    print(f"🚀 Starting controlled airfoil generation...")
    print(f"🎯 Overall Target: {overall_target} airfoils across {len(families_to_generate)} families.")
    print(f"💾 Output directory: {os.path.abspath(OUTPUT_DIRECTORY)}")

    for family_type in families_to_generate:
        config = GENERATION_PARAMETERS[family_type]
        target_count = config['num_to_generate']
        
        print("-" * 40)
        print(f"🔥 Generating {target_count} airfoils for family: {family_type}")
        
        valid_count_for_family = 0
        generated_names_for_family = set()
        
        if family_type in ['NACA4', 'NACA5']:
            # Strategy: Random sampling. Good for large parameter spaces.
            attempt_limit = target_count * 80  # Safety break after many attempts
            attempts = 0
            while valid_count_for_family < target_count and attempts < attempt_limit:
                attempts += 1
                params = get_random_params(family_type)
                
                # Generate name and skip if already processed
                if family_type == 'NACA4':
                    name = f"NACA{params['m']}{params['p']}{str(params['t']).zfill(2)}"
                else: # NACA5
                    name = f"NACA{params['L']}{params['P']}{params['Q']}{str(params['XX']).zfill(2)}"
                
                if name in generated_names_for_family:
                    continue
                
                # Generate airfoil coordinates
                if family_type == 'NACA4':
                    gen_output = generate_naca_4digit(params['m'], params['p'], params['t'], NUM_POINTS_PER_SURFACE)
                else: # NACA5
                    gen_output = generate_naca_5digit(params['L'], params['P'], params['Q'], params['XX'], NUM_POINTS_PER_SURFACE)
                
                combined_x, combined_y, profile_x, yu, yl, name_from_func = gen_output
                generated_names_for_family.add(name) # Mark as processed

                if check_geometry_validity(combined_x, combined_y, profile_x, yu, yl):
                    save_airfoil_dat(name, combined_x, combined_y, OUTPUT_DIRECTORY)
                    valid_count_for_family += 1
                    print(f"✅ Generated {family_type} ({valid_count_for_family}/{target_count}): {name}")
            
            if valid_count_for_family < target_count:
                 print(f"⚠️ Warning: Reached attempt limit ({attempt_limit}) for {family_type}. Generated {valid_count_for_family}/{target_count}.")

        elif family_type == 'NACA6':
            # Strategy: Exhaustive search of all unique combinations.
            all_param_combos = get_naca6_parameter_list()
            num_possible = len(all_param_combos)
            print(f"🔥 Found {num_possible} unique parameter combinations for NACA6. Testing them...")
            
            for params in all_param_combos:
                if valid_count_for_family >= target_count:
                    break # Target has been met

                gen_output = generate_naca_6digit(params['S'], params['L'], params['tt'], NUM_POINTS_PER_SURFACE)
                combined_x, combined_y, profile_x, yu, yl, name = gen_output

                if name in generated_names_for_family:
                    continue
                generated_names_for_family.add(name)
                
                if check_geometry_validity(combined_x, combined_y, profile_x, yu, yl):
                    save_airfoil_dat(name, combined_x, combined_y, OUTPUT_DIRECTORY)
                    valid_count_for_family += 1
                    print(f"✅ Generated {family_type} ({valid_count_for_family}/{target_count}): {name}")

            if valid_count_for_family < target_count:
                print(f"⚠️ Warning: Exhausted all {num_possible} unique parameter sets for {family_type}. Generated {valid_count_for_family}/{target_count}.")

        total_generated_count += valid_count_for_family

    print("-" * 40)
    if total_generated_count >= overall_target:
        print(f"🎉 Success! Generated all {total_generated_count} requested airfoils.")
    else:
        print(f"⚠️ Generation finished. Generated {total_generated_count}/{overall_target} valid airfoils.")
    print(f"💾 Files saved in: {os.path.abspath(OUTPUT_DIRECTORY)}")

if __name__ == '__main__':
    main()