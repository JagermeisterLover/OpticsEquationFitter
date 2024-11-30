import numpy as np
import lmfit
import matplotlib.pyplot as plt

def check_for_nan_or_inf(data, label):
    if np.any(np.isnan(data)) or np.any(np.isinf(data)):
        raise ValueError(f"{label} contains NaN or infinite values.")

def even_asphere_sag(r, R, k, *coeffs):
    discriminant = 1 - (1 + k) * r**2 / R**2
    discriminant = np.maximum(discriminant, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (r**2) / (R * (1 + np.sqrt(discriminant)))
        term1 = np.where(np.isfinite(term1), term1, 0)
    term2 = sum(A * r**(4 + 2*i) for i, A in enumerate(coeffs))
    return term1 + term2

def extended_asphere_sag(r, R, k, *coeffs):
    discriminant = 1 - (1 + k) * r**2 / R**2
    discriminant = np.maximum(discriminant, 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (r**2) / (R * (1 + np.sqrt(discriminant)))
        term1 = np.where(np.isfinite(term1), term1, 0)
    term2 = sum(A * r**(3 + i) for i, A in enumerate(coeffs))
    return term1 + term2

def opal_universal_z(r, R, H, e2, *coeffs):
    z = r**2 / (2 * R)
    for _ in range(10):
        w = z / H
        Q = sum(A * w**(3 + i) for i, A in enumerate(coeffs))
        z_new = ((r**2) + (1 - e2) * (z**2)) / (2 * R) + Q
        z_new = np.where(np.isfinite(z_new), z_new, 0)
        z = z_new
    return z

def opal_universal_u(r, R, H, e2, *coeffs):
    z = r**2 / (2 * R)
    for _ in range(10):
        w = r**2 / H**2
        Q = sum(A * w**(2 + i) for i, A in enumerate(coeffs))
        z_new = ((r**2) + (1 - e2) * (z**2)) / (2 * R) + Q
        z_new = np.where(np.isfinite(z_new), z_new, 0)
        z = z_new
    return z

def opal_polynomial_z(r, R, e2, *coeffs):
    A1 = 2 * R
    A2 = e2 - 1
    z = r**2 / A1
    for _ in range(10):
        Q = sum(A * (z**(3 + i)) for i, A in enumerate(coeffs))
        z_new = (r**2 + A2 * (z**2)) / A1 + Q
        z_new = np.where(np.isfinite(z_new), z_new, 0)
        z = z_new
    return z

def main():
    
    A1 = None
    A2 = None
    
    data = np.loadtxt("tempsurfacedata.txt")
    print(f"Data shape: {data.shape}") 
    r_data, z_data = data[:, 0], data[:, 1]
    check_for_nan_or_inf(r_data, "r_data")
    check_for_nan_or_inf(z_data, "z_data")
    settings = {}
    with open("ConvertSettings.txt", "r") as file:
        for line in file:
            key, value = line.strip().split('=')
            settings[key] = value

    equation_choice = settings['SurfaceType']
    R = float(settings['Radius'])
    H = float(settings['H'])
    e2_isVariable = int(settings['e2_isVariable'])
    e2_value = float(settings['e2'])
    conic_isVariable = int(settings['conic_isVariable'])
    conic_value = float(settings['conic'])
    num_terms = int(settings['TermNumber'])
    optimization_algorithm = settings.get('OptimizationAlgorithm', 'leastsq')
    auto_algorithm = int(settings.get('AutoAlgorithm', '0'))

    params = lmfit.Parameters()

    if equation_choice == '1':
        if conic_isVariable == 0:
            params.add('k', value=conic_value, vary=False)
        else:
            params.add('k', value=-1.0, vary=True)

        if num_terms > 0:
            for i in range(num_terms):
                params.add(f'A{4 + 2*i}', value=0.0)
        else:
            # If TermNumber is 0, do not add higher order coefficients
            params.add('k', value=conic_value, vary=True)

        def objective(params, r, z):
            k = params['k']
            coeffs = [params[f'A{4 + 2*i}'] for i in range(num_terms)] if num_terms > 0 else []
            model = even_asphere_sag(r, R, k, *coeffs)
            check_for_nan_or_inf(model, "even_asphere_sag output")
            return model - z

    elif equation_choice == '2':
        if conic_isVariable == 0:
            params.add('k', value=conic_value, vary=False)
        else:
            params.add('k', value=-1.0, vary=True)

        if num_terms > 0:
            for i in range(num_terms):
                params.add(f'A{3 + i}', value=0.0)
        else:
            params.add('k', value=conic_value, vary=True)

        def objective(params, r, z):
            k = params['k']
            coeffs = [params[f'A{3 + i}'] for i in range(num_terms)] if num_terms > 0 else []
            model = extended_asphere_sag(r, R, k, *coeffs)
            check_for_nan_or_inf(model, "extended_asphere_sag output")
            return model - z

    elif equation_choice == '3':
        if e2_isVariable == 0:
            params.add('e2', value=e2_value, vary=False)
        else:
            params.add('e2', value=1.0, vary=True)

        if num_terms > 0:
            for i in range(num_terms):
                params.add(f'A{3 + i}', value=0.0)
        else:
            params.add('e2', value=e2_value, vary=True)

        def objective(params, r, z):
            e2 = params['e2'].value
            coeffs = [params[f'A{3 + i}'].value for i in range(num_terms)] if num_terms > 0 else []
            model = opal_universal_z(r, R, H, e2, *coeffs)
            check_for_nan_or_inf(model, "opal_universal_z output")
            return model - z

    elif equation_choice == '4':
        if e2_isVariable == 0:
            params.add('e2', value=e2_value, vary=False)
        else:
            params.add('e2', value=1.0, vary=True)

        if num_terms > 0:
            for i in range(num_terms):
                params.add(f'A{2 + i}', value=0.0)
        else:
            params.add('e2', value=e2_value, vary=True)

        def objective(params, r, z):
            e2 = params['e2'].value
            coeffs = [params[f'A{2 + i}'].value for i in range(num_terms)] if num_terms > 0 else []
            model = opal_universal_u(r, R, H, e2, *coeffs)
            check_for_nan_or_inf(model, "opal_universal_u output")
            return model - z

    elif equation_choice == '5':
        if e2_isVariable == 0:
            params.add('e2', value=e2_value, vary=False)
        else:
            params.add('e2', value=1.0, vary=True)

        if num_terms > 0:
            for i in range(num_terms):
                params.add(f'A{3 + i}', value=0.0)
        else:
            params.add('e2', value=e2_value, vary=True)

        def objective(params, r, z):
            e2 = params['e2'].value
            coeffs = [params[f'A{3 + i}'].value for i in range(num_terms)] if num_terms > 0 else []
            model = opal_polynomial_z(r, R, e2, *coeffs)
            check_for_nan_or_inf(model, "opal_polynomial_z output")
            return model - z

    else:
        print("Invalid choice. Exiting.")
        return

    methods = ['leastsq', 'least_squares', 'nelder', 'powell']

    if auto_algorithm == 1:
        best_result = None
        best_method = None
        best_chisqr = np.inf

        for method in methods:
            try:
                result = lmfit.minimize(objective, params, args=(r_data, z_data), method=method, max_nfev=10000)
                print(f"\nResults for method: {method}")
                lmfit.report_fit(result)

                if result.chisqr < best_chisqr:
                    best_chisqr = result.chisqr
                    best_result = result
                    best_method = method
            except Exception as e:
                print(f"Method {method} failed with error: {e}")

        if best_result is not None:
            print(f"\nBest method: {best_method} with chi-square: {best_chisqr}")
            result = best_result
        else:
            print("All methods failed.")
            return
    else:
        if optimization_algorithm in ['leastsq', 'least_squares']:
            result = lmfit.minimize(objective, params, args=(r_data, z_data), method=optimization_algorithm, max_nfev=10000, xtol=1e-12, ftol=1e-12)
        else:
            result = lmfit.minimize(objective, params, args=(r_data, z_data), method=optimization_algorithm, max_nfev=10000)

        lmfit.report_fit(result)

    if equation_choice == '1':
        fitted_z = even_asphere_sag(r_data, R, result.params['k'], *[result.params[f'A{4 + 2*i}'] for i in range(num_terms)] if num_terms > 0 else [])
    elif equation_choice == '2':
        fitted_z = extended_asphere_sag(r_data, R, result.params['k'], *[result.params[f'A{3 + i}'] for i in range(num_terms)] if num_terms > 0 else [])
    elif equation_choice == '3':
        e2 = result.params['e2'].value
        fitted_z = opal_universal_z(r_data, R, H, e2, *[result.params[f'A{3 + i}'] for i in range(num_terms)] if num_terms > 0 else [])
    elif equation_choice == '4':
        e2 = result.params['e2'].value
        fitted_z = opal_universal_u(r_data, R, H, e2, *[result.params[f'A{2 + i}'] for i in range(num_terms)] if num_terms > 0 else [])
    elif equation_choice == '5':
        e2 = result.params['e2'].value
        fitted_z = opal_polynomial_z(r_data, R, e2, *[result.params[f'A{3 + i}'] for i in range(num_terms)] if num_terms > 0 else [])

        # Calculate A1 and A2
        A1 = 2 * R
        A2 = e2 - 1

    deviations = fitted_z - z_data

    print("\nFitted values and deviations:")
    print(f"{'r':<10}{'Original z':<20}{'Fitted z':<20}{'Deviation':<20}")
    for r, oz, fz, dev in zip(r_data, z_data, fitted_z, deviations):
        print(f"{r:<10.6f}{oz:<20.12f}{fz:<20.12f}{dev:<20.12f}".replace('.', ','))

    print("\nFormatted Variables:")
    print(f"R: {R:.8f} (fixed)".replace('.', ','))
    if A1 is not None and A2 is not None:
        print(f"A1: {A1:.8f} (calculated)".replace('.', ','))
        print(f"A2: {A2:.8e} (calculated)".replace('.', ','))

    for name, param in result.params.items():
        if name in ['k', 'e2']:
            # Print k and e2 in decimal form
            print(f"{name}: {param.value:.8f} (init = {param.init_value:.8f})".replace('.', ','))
        else:
            # Print other parameters in scientific notation
            print(f"{name}: {param.value:.8e} (init = {param.init_value:.8e})".replace('.', ','))




    plt.figure()
    plt.scatter(r_data, z_data, label='Data')
    plt.plot(r_data, fitted_z, label='Fit', color='red')
    plt.xlabel('Height (r)')
    plt.ylabel('Sag (z)')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    try:
        main()
    except FileNotFoundError as e:
        print(f"\nFile not found error:\n{str(e)}")
    except ValueError as e:
        print(f"\nValue error (possibly invalid data format):\n{str(e)}")
    except Exception as e:
        print(f"\nAn unexpected error occurred:\n{str(e)}")
    finally:
        input("\nPress Enter to exit...")