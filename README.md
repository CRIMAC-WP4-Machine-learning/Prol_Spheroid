How to run the script to model the backscattering from a liquid- or gas-filled prolate spheroid. 

1 - Set the parameters of prolate spheroid and water in the src/setings.py
2 - In the script prol_spheroid.py, under Inputs determine which one you want to model, liquid or airfilled spheroid by 

   	settings = LiquidFilledSettings()
	or
	settings = AirFilledSettings()

3 - run the prol_spheroid.py script to generate the TS as a function of frequency which saves the csv file in the temp folder. 
