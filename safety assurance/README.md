This folder contains two formulations for safety assurance

Original formulation:

The general pipline is shown in safety_assurance.ipynb

Issues:
1. problem A is easy to stuck in local optimum which means even if the optimum v(s_fov) is greater than v0, there may exist solutions by which the car can be stopped inside fov.
2. optimizer is very sensitive so it is easy to fail during online optimizing.

New formulation as suggested by Prof. Callies:

P.py contains an function for solving problem P_i as indicated as formulation 1 in the file formulation_sheet.pdf

formulation_rainer_callies.ipynb implements the pipline that solves formulation 1
