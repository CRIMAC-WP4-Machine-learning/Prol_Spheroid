How to run the script to model the backscattering from a liquid- or gas-filled prolate spheroid. 

1 - Set the parameters of prolate spheroid and water in the src/settings.py

2 - Run one of the scripts src/script_liquid_filled.py or src/script_air_filled.py which are set up to use the liquid- and 
gas-filled settings respectively. The result is the TS as a function of frequency which saves the csv file in the temp folder.

Optional - Define your own settings in src/script_user_defined.py, and run 
