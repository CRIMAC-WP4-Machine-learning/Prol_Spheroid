import subprocess

print('IAM HERE=======================================')

#1. Compile the Fortran code
#gfortran your_program.f90 -o your_program
#2. Run the Fortran executable from Python:
## Replace 'your_program' with the actual name of your compiled Fortran executable
#executable_path = './profcn_bkh5'  # You may need to provide the full path if it's not in the same directory
## Run the Fortran executable
#process = subprocess.run([executable_path], stdout=subprocess.PIPE, text=True)

# Compile the Fortran program !!! Linux
fortran_file_path = 'profcn_II.f90'
# Compile the Fortran code
compile_command = f'gfortran {fortran_file_path} -o profcn_II'
subprocess.run(compile_command, shell=True)

# Check if the compilation was successful
#if compile_result.returncode == 0:
#    print("Compilation successful")
#else:
#    print(f"Compilation failed: {compile_result.stderr.decode()}")

# Run the compiled Fortran executable !!! Linux
run_command = './profcn_II'
subprocess.run(run_command, shell=True)

#%% Windows
#compile_command = ["gfortran", "profcn_II.f90", "-o", "profcn_II.exe"]
#compile_result = subprocess.run(compile_command, check=True, capture_output=True)

## Run the compiled Fortran executable !!! Windows 
#run_command = './profcn_II.exe'
#subprocess.run(run_command, check=True)