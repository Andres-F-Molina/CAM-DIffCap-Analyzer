# CAM DiffCap Analyser
## Project description  
The DiffCap analyser is a program that extract the H2-H3 phase transistion peak from differential capacity curves and fits it using a Gaussian function. After the fitting is done, the
program reports the peak potential, amplitude, area and full width at half maximum (FWHM). In this version, the program only extract the peak observed in the charge cycle.  

The program takes directly the raw *.mpr files from GPCL (Galvanostatic cycling with Potential Limitation) tests in Biologic cyclers.  

## Installation
1. Clone the repo.
2. Specify all the directories in the file 'directory_config_template.yaml'.
3. Change the name to 'directory_config.yaml'.
4. Install all the packages from the requirements.txt file
5. Run the main.
