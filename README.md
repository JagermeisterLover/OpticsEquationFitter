# OpticsEquationFitter
Equation fitter for various aspheric surface used in optical design. Uses lmfit library to fit equations.
Script expects surface sag data in tempsurfacedata.txt (heights r and corresponding z sag values) and ConvertSettings.txt.

Radius R is always fixed for fitting, normalization factor H is user-defined, eccentricity squared e2 and conic constant are either variable (1) or user-defined fixed values (0). Algorithm for fitting can be automatically chosen based on best chi-square result, or user-defined (leastq, least_squares, powell or nelder are supported). Number of aspheric terms used for fitting is user defined, 0 if only e2 or conic constant should be used as variables during fitting.
