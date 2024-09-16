# High-Precision model for acoustic backscattering by liquid and gas filled prolate spheroid over a wide frequency range and tilt angles

Accompanging code to the manuscript "High-Precision model for acoustic backscattering by liquid and gas filled prolate spheroid over a wide frequency range and tilt angles – Implications in fisheries acoustics" (Khodabandeloo et al.) 
submitted to Journal of Sound and Vibration for review.

The current release "ProlateSpheroidv0.9" is the release corresponding to the submission of the manuscript.

Code for backscattering calculations of penetrable (gas or fluid) prolate spheroids 
with high precision for a wide range of tilt angles and large frequency ranges used 
in the fisheries acoustics. 


The code is written in Python and uses "A.L. Van-Buren, Mathieu and spheroidal wave 
functions fortran programs for their accurate calculation., (n.d.). https://github.com/mathieuandspheroidalwavefunctions." for the expansion coefficients with the Meixner and Schafke normalization scheme, 
and the prolate spheroidal radial functions of the first and second kind and their derivatives, 

References:<br>
A.L. Van Buren, J.E. Boisvert, Accurate calculation of prolate spheroidal radial functions of the first kind and their first derivatives, Quarterly of Applied Mathematics LX (2002) 589–599.<br>
A.L. Van Buren, J.E. Boisvert, Improved calculation of Prolate Spheroidal Radial functions of the second kind and their first derivatives, Quarterly of Applied Mathematics LXII (2004) 493–507.<br>


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

Optional - Define your own settings in src/script_user_defined.py, and run 

```
python3 script_user_defined.py
```

The scripts assumes that they are run from the src directory.

## Dependencies

Requirements to run the code are found in requirements.txt. 

E.g run:

```
pip install -r requirements.txt
```

## Citing the code?

If you use the code for you research, please cite the article once published (currently under review)

```
@article{khodabandeloo2024prolate,
  title={High-Precision model for acoustic backscattering by liquid and gas filled prolate spheroid over a wide frequency range and tilt angles – Implications in fisheries acoustics},
  author={},
  journal={xxxx},
  pages={xxxx},
  year={xxxx},
  publisher={xxxx}
}
```