import subprocess

print('=======================================')

## Define the Fortran source file and the output executable name
source_file = 'param.f90'

# Command to compile the Fortran source file into an object file
compile_command = ['gfortran', source_file, '-c']

## Define the Fortran source file and the output module file
##fortran_source = 'param.f90'
##output_module = 'param.mod'
#
## Compile the Fortran program !!! Linux
#fortran_source = 'param.f90'
## Compile the Fortran code
#compile_command = f'gfortran {fortran_source} -o param'
#subprocess.run(compile_command, shell=True)
## Compile the Fortran module
##compile_command = ['gfortran', '-c', fortran_source]
#
## Run the compilation command
##result = subprocess.run(compile_command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
##print("Compilation successful.")
#
## Run the compiled Fortran executable !!! Linux
##run_command = './param'
##subprocess.run(run_command, shell=True)

try:
    result = subprocess.run(compile_command, check=True, capture_output=True, text=True)
    print("Compilation to object file successful.")
    print("Output:", result.stdout)
except subprocess.CalledProcessError as e:
    print("Compilation failed.")
    print("Error:", e.stderr)