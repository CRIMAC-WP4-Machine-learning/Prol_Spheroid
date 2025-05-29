# High-Precision model and open-source software for acoustic backscattering by liquid- and gas-filled prolate spheroid across a wide frequency range and tilt angles: implications for fisheries acoustics

<b>Code for backscattering calculations of penetrable (gas or fluid) prolate spheroids 
with high precision for a wide range of tilt angles and large frequency ranges used 
in fisheries acoustics.</b>

The code accompany the paper "High-Precision model for acoustic backscattering by liquid and gas filled prolate spheroid over a wide frequency range and tilt angles – Implications in fisheries acoustics" published in the Journal of Sound and Vibration. See "Citing the code" for reference.

The current release "ProlateSpheroidv0.9" is the release corresponding to the submission of the manuscript.

The code is written in Python and uses "A.L. Van-Buren, Mathieu and spheroidal wave 
functions fortran programs for their accurate calculation., (n.d.). https://github.com/mathieuandspheroidalwavefunctions." for the expansion coefficients with the Meixner and Schafke normalization scheme, 
and the prolate spheroidal radial functions of the first and second kind and their derivatives, <br><br>

A.L. Van Buren, J.E. Boisvert, Accurate calculation of prolate spheroidal radial functions of the first kind and their first derivatives, Quarterly of Applied Mathematics LX (2002) 589–599.<br>
A.L. Van Buren, J.E. Boisvert, Improved calculation of Prolate Spheroidal Radial functions of the second kind and their first derivatives, Quarterly of Applied Mathematics LXII (2004) 493–507.<br>

Iterative refinement (Ziaeemehr, 2021) and the biconjugate gradient stabilized (Bi-CGSTAB) (Van der Vorst, 1992) methods with incomplete LU-factorization (ILU) as preconditioning is also used. <br><br>
A. Ziaeemehr, Solvers - terative Refinement method, (2021). https://github.com/Ziaeemehr/solvers/blob/master/py/iterativeRefinement.ipynb <br>
H.A. Van der Vorst, Bi-CGSTAB: A fast and smoothly converging variant of Bi-CG for the solution of nonsymmetric linear systems., SIAM Journal on Scientific and Statistical Computing, 13 (1992) 631-644<br>

For more details, see the paper (<it>in review</it>).

## How to run the script to model the backscattering from a liquid- or gas-filled prolate spheroid

1 - Set the parameters of prolate spheroid and water in the src/settings.py

2 - Run one of the scripts src/script_liquid_filled.py or src/script_air_filled.py which are set up to use the liquid- and  gas-filled settings respectively. The result is the TS as a function of frequency which saves the csv file in the temp folder.

```
python3 script_liquid_filled.py
```

or

```
python3 script_air_filled.py
```

Optionally you can define your own settings in src/script_user_defined.py, and run 

```
python3 script_user_defined.py
```

The scripts assume that they are run from the src directory. Running the scripts outside of this directory causes errors. 

A Fortran compiler (e.g., gfortran) must be installed for the scripts to run. This is not included in requirements.txt 

## Optimized version 
The improved performance version is accessible in the vectorized branch.

## Verification results

A number of test cases has been calculated using COMSOL Multiphysics. These 
resutlts are found in /COMSOL_results.

The acoustic properties and dimentions of the prolate spheroids are

| Material | Density (kg.m-3) | Sound speed (m/s) | Semi-major axis a (mm) | Semi-minor axis b (mm) |
|----------|------------------|-------------------|------------------------|------------------------|
| Water	   | 1027	          | 1500	          | NA                     | NA                     |
| Liquid   | 1027×1.05	      | 1500×1.05	      | 80	                   | 20                     |
| Gas	   | 10	              | 343          	  | 30	                   | 10                     | 

Material of the prolate spheroid, aspect ratio, angle of incidence, and acoustic frequencies are also found in the file titles.

E.g. Air_Prol_B1cm_AspR2_ro10_c343_Theta15_f38_300kHz.csv (air, b, aspect ratio, density, soundspeed, angle, frequency from:to).

## Dependencies

Requirements to run the code are found in requirements.txt. 

E.g run:

```
pip install -r requirements.txt
```

## Citing the code

If you use the code for you research, please cite the paper published in the ournal of Sound and Vibration.

```
@article{KHODABANDELOO2025119227,
title = {High-precision model and open-source software for acoustic backscattering by liquid- and gas-filled prolate spheroids across a wide frequency range and incident angles: Implications for fisheries acoustics},
journal = {Journal of Sound and Vibration},
pages = {119227},
year = {2025},
issn = {0022-460X},
doi = {https://doi.org/10.1016/j.jsv.2025.119227},
url = {https://www.sciencedirect.com/science/article/pii/S0022460X25003013},
author = {Babak Khodabandeloo and Yngve Heggelund and Bjørnar Ystad and Sander Andre Berg Marx and Geir Pedersen},
keywords = {Target Strength, Prolate Spheroid, Spheroidal wave functions, Fish body, Swimbladder, Modal series solution},
abstract = {Among the few geometries with analytical scattering solutions, the prolate spheroid is perhaps one of the best representative models for various marine organisms or their dominant scattering organs. The mathematical formulation for the exact solution of scattering from prolate spheroids is well-established and known. However, solving these equations presents challenges, including difficulties in efficiently and stably estimating prolate spheroidal wave functions (PSWFs), handling numerical overflow or underflow, and addressing ill-conditioned systems of equations. These issues are addressed in this work, and the model provides stable and precise solutions for both gas- and liquid-filled prolate spheroids across all incident angles and over a wide frequency range. Additionally, the required number of terms to truncate the infinite series of scattering modes is investigated, and empirical formulas are provided. The calculated backscattering results are validated for aspect ratios (i.e., ratio of the longest to the shortest dimension of the prolate spheroid) of up to 10 by comparison with those estimated using finite element methods (FEM). An open-source software package developed with Python and Fortran, is provided alongside this paper.}
}
```
