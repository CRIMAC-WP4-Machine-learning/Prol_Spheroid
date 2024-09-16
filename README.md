# High-Precision model for acoustic backscattering by liquid and gas filled prolate spheroid over a wide frequency range and tilt angles

Accompanging code to the manuscript "High-Precision model for acoustic backscattering by liquid and gas filled prolate spheroid over a wide frequency range and tilt angles – Implications in fisheries acoustics" (Khodabandeloo et al.) 
submitted to Journal of Sound and Vibration for review.

The current release "ProlateSpheroidv0.9" is the release corresponding to the submission of the manuscript.

## How to run the script to model the backscattering from a liquid- or gas-filled prolate spheroid

1 - Set the parameters of prolate spheroid and water in the src/settings.py

2 - Run one of the scripts src/script_liquid_filled.py or src/script_air_filled.py which are set up to use the liquid- and  gas-filled settings respectively. The result is the TS as a function of frequency which saves the csv file in the temp folder.

Optional - Define your own settings in src/script_user_defined.py, and run 

## Dependencies

Requirements to run the code are found in requirements.txt. E.g run 

```
pip install -r requirements.txt
```

## Citing the code?

If you use the code for you research, please cite the article once published (currently under review)

```
@article{khodabandeloo2024prolate,
  title={High-Precision model for acoustic backscattering by liquid and gas filled prolate spheroid over a wide frequency range and tilt angles – Implications in fisheries acoustics},
  author={},
  journal={Journal of Sound and Vibration},
  pages={xxxx},
  year={xxxx},
  publisher={Elsevier}
}
```