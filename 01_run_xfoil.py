"""Runs an XFOIL analysis for a given airfoil and flow conditions
   with automatic convergence checking and bad result filtering"""
import os
import subprocess
import numpy as np
import shutil
import re
import time
import sys  # Added for sys.executable
import subprocess
import time
from subprocess import PIPE, STARTUPINFO, STARTF_USESHOWWINDOW, SW_HIDE


# Function to kill any stuck XFOIL processes
def kill_process_by_name(process_name="xfoil.exe"):
    """
    Force kill any running instances of a process by name
    Helps clear stuck XFOIL processes
    """
    killed = False
    try:
        for proc in psutil.process_iter(['pid', 'name']):
            # Check if process name contains xfoil
            if process_name.lower() in proc.info['name'].lower():
                print(f"Killing stuck process: {proc.info['name']} (PID {proc.info['pid']})")
                try:
                    process = psutil.Process(proc.info['pid'])
                    process.terminate()
                    killed = True
                except Exception as e:
                    print(f"Failed to terminate process: {e}")
    except Exception as e:
        print(f"Error scanning for processes: {e}")
    
    return killed

# Try to install psutil if not present
try:
    import psutil
except ImportError:
    print("Installing required package: psutil")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
        import psutil
        print("Successfully installed psutil")
    except Exception as e:
        print(f"Warning: Could not install psutil: {e}")
        print("The script will continue but with limited process management capabilities")
        
        # Define a dummy function if psutil isn't available
        def kill_process_by_name(process_name="xfoil.exe"):
            print("Process management unavailable - psutil not installed")
            # Use alternative Windows-only taskkill as fallback
            if os.name == 'nt':
                try:
                    subprocess.run(f"taskkill /F /IM {process_name}", 
                                 shell=True, stdout=subprocess.DEVNULL, 
                                 stderr=subprocess.DEVNULL)
                    print(f"Attempted to kill {process_name} using taskkill")
                except:
                    pass
            return False

alpha_i = -2
alpha_f = 8
alpha_step = 0.25
reynolds_list = [1e5, 2e5, 3e5]

# Maximum allowed absolute value for aerodynamic coefficients
# Results with coefficients exceeding this value will be discarded
MAX_COEF_VALUE = 2.0  

# 2) DISCOVER ALL VALID .dat FILES IN THE CURRENT FOLDER
pattern = re.compile(r'^.+\.dat$', re.IGNORECASE)
airfoil_list = [
    fname[:-4].lower()
    for fname in os.listdir('.')
    if pattern.match(fname)
]
print("Airfoils to run:", airfoil_list)
        
n_iter = 200

# Create subfolder for output polars
polar_output_folder = "polar_outputs"
os.makedirs(polar_output_folder, exist_ok=True)

# Create a subfolder for skipped/problematic airfoils
skipped_output_folder = "skipped_airfoils"
os.makedirs(skipped_output_folder, exist_ok=True)

def validate_polar_data(polar_filename):
    """
    Validates polar data to check if coefficients are within reasonable bounds.
    Returns True if data is valid, False otherwise.
    """
    try:
        # Load the polar data
        polar_data = np.loadtxt(polar_filename, skiprows=12)
        
        # Check columns typically found in XFOIL output
        # Column indices: 0=alpha, 1=CL, 2=CD, 3=CDp, 4=CM, etc.
        if polar_data.shape[1] >= 5:  # Ensure we have at least alpha, CL, CD, CDp, CM
            # Check if any coefficient exceeds the maximum allowed value
            cl_values = polar_data[:, 1]  # CL column
            cd_values = polar_data[:, 2]  # CD column
            cm_values = polar_data[:, 4]  # CM column
            
            # Check for NaN values or values exceeding threshold
            if (np.isnan(cl_values).any() or 
                np.isnan(cd_values).any() or 
                np.isnan(cm_values).any() or
                np.max(np.abs(cl_values)) > MAX_COEF_VALUE or
                np.max(np.abs(cd_values)) > MAX_COEF_VALUE or
                np.max(np.abs(cm_values)) > MAX_COEF_VALUE):
                
                print(f"[Warning] Invalid coefficients detected in {polar_filename}")
                return False
                
            # Data seems valid
            return True
        else:
            print(f"[Error] Unexpected data format in {polar_filename}")
            return False
            
    except Exception as e:
        print(f"[Error] Failed to validate polar data: {e}")
        return False


def run_xfoil_hidden(input_filename, timeout=30):

    startupinfo = STARTUPINFO()
    startupinfo.dwFlags |= STARTF_USESHOWWINDOW
    startupinfo.wShowWindow = SW_HIDE

    try:
        process = subprocess.Popen(
            f"xfoil.exe < {input_filename}",
            shell=True,
            stdout=PIPE,
            stderr=PIPE,
            text=True,
            startupinfo=startupinfo,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        stdout, stderr = process.communicate(timeout=timeout)
        time.sleep(0.5)
        return True, stdout, stderr
    except subprocess.TimeoutExpired:
        subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1)
        return False, "", "Timeout"
    except Exception as e:
        return False, "", str(e)
            
    except Exception as e:
        print(f"[Error] XFOIL execution failed: {e}")
        
        # Ensure process is terminated if it exists
        if process:
            try:
                process.kill()
            except:
                pass
                
        return False, "", str(e)

def run_xfoil_with_timeout(input_filename, timeout=30):
    """
    Run XFOIL with a timeout to prevent hanging on problematic airfoils
    Ensures proper process termination to avoid file access issues
    """
    process = None
    try:
        # Start the process
        process = subprocess.Popen(f"xfoil.exe < {input_filename}", 
                                  shell=True, 
                                  stdout=subprocess.PIPE,
                                  stderr=subprocess.PIPE,
                                  text=True)
        
        # Wait for the process to complete or timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            # Wait a bit to ensure file handles are released
            time.sleep(0.5)
            return True, stdout, stderr
        except subprocess.TimeoutExpired:
            print(f"[Warning] XFOIL execution timed out after {timeout} seconds")
            
            # Force terminate process and any child processes
            try:
                # On Windows, we need to be more aggressive with termination
                if os.name == 'nt':
                    # Use taskkill to forcefully terminate the process tree
                    subprocess.run(f"taskkill /F /T /PID {process.pid}", shell=True, 
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                else:
                    process.kill()
                
                # Wait for process to fully terminate
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    print("[Warning] Process termination timed out")
                
                # Wait additional time to ensure file handles are released
                time.sleep(1)
            except Exception as e:
                print(f"[Error] Failed to terminate process: {e}")
                
            return False, "", "Timeout"
            
    except Exception as e:
        print(f"[Error] XFOIL execution failed: {e}")
        
        # Ensure process is terminated if it exists
        if process:
            try:
                process.kill()
            except:
                pass
                
        return False, "", str(e)

# Process each airfoil
print("\n===== Starting XFOIL Analysis =====\n")

# First ensure no existing XFOIL processes are running
kill_process_by_name("xfoil.exe")

for airfoil_name in airfoil_list:
    airfoil_valid = True  # Track if this airfoil has any valid Reynolds results
    
    for Re in reynolds_list:
        polar_filename = f"polar_{airfoil_name}_Re{int(Re):d}.txt"
        input_filename = f"input_file_Re{int(Re):d}.in"

        # Remove previous polar file if exists
        if os.path.exists(polar_filename):
            os.remove(polar_filename)

        # Write XFOIL input file
        with open(input_filename, 'w') as input_file:
            input_file.write(f"LOAD {airfoil_name}.dat\n")
            input_file.write("PANE\n")
            input_file.write("OPER\n")
            input_file.write(f"Visc {int(Re)}\n")  # set Reynolds number
            input_file.write("Mach 0\n")          # ensure Mach number is zero
            input_file.write(f"Iter {n_iter}\n")  # set iteration limit

            # Start polar dump
            input_file.write("PACC\n")
            input_file.write(f"{polar_filename}\n\n")

            # Sweep angle of attack
            input_file.write(f"ASeq {alpha_i} {alpha_f} {alpha_step}\n")

            # Close polar dump
            input_file.write("PACC\n\n")

            # Exit XFoil
            input_file.write("QUIT\n")

        # Run XFOIL with timeout protection
        print(f"Running XFOIL for {airfoil_name} at Re={int(Re)}")
        #success, stdout, stderr = run_xfoil_with_timeout(input_filename)
        success, stdout, stderr = run_xfoil_hidden(input_filename)
        
        # If execution failed or timed out, kill any stuck processes
        if not success:
            # Try to clean up stuck processes
            kill_process_by_name("xfoil.exe")
            # Wait a moment for system to release resources
            time.sleep(1)
            print(f"[Error] XFOIL execution failed for {airfoil_name} at Re={int(Re)}")
            airfoil_valid = False
            continue

        # Check if polar file was created
        if os.path.exists(polar_filename):
            # Validate the polar data
            if validate_polar_data(polar_filename):
                # Move valid polar file to output folder
                shutil.move(polar_filename, os.path.join(polar_output_folder, polar_filename))
                print(f"[OK] Valid polar data for {airfoil_name} at Re={int(Re)}")
            else:
                # Move invalid polar file to skipped folder for reference
                shutil.move(polar_filename, os.path.join(skipped_output_folder, polar_filename))
                print(f"[Skipped] Invalid polar data for {airfoil_name} at Re={int(Re)}")
                airfoil_valid = False
        else:
            print(f"[Warning] Polar file not found for {airfoil_name} at Re={int(Re)}")
            airfoil_valid = False
    
    # Cleanup input files after processing this airfoil
    for Re in reynolds_list:
        input_filename = f"input_file_Re{int(Re):d}.in"
        if os.path.exists(input_filename):
            try:
                os.remove(input_filename)
                print(f"Removed input file: {input_filename}")
            except PermissionError:
                print(f"[Warning] Could not remove {input_filename} - file is still in use")
                # Wait and try again
                try:
                    time.sleep(2)
                    os.remove(input_filename)
                    print(f"Successfully removed input file on second attempt: {input_filename}")
                except Exception as e:
                    print(f"[Error] Failed to remove input file: {e}")
                    # Just continue - this isn't critical
            except Exception as e:
                print(f"[Error] Failed to remove input file: {e}")
                # Just continue - this isn't critical

    # Summary for this airfoil
    if airfoil_valid:
        print(f"✓ {airfoil_name}: Successfully processed")
    else:
        print(f"✗ {airfoil_name}: Processing failed or had invalid results")

print("XFOIL processing complete!")
print(f"Valid results saved to: {polar_output_folder}")
print(f"Invalid results saved to: {skipped_output_folder}")