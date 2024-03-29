#Add 10% (low level) / 20% (high level) noise to the spectra

import numpy as np

def modify_array(arr,noise_level):
    # Get non-zero elements
    non_zero_elements = arr[arr != 0].astype(float)

    # Calculate the number of elements in each third
    total_elements = len(non_zero_elements)
    one_third = total_elements // 3

    # Randomly choose indices for each third
    indices = np.random.permutation(total_elements)

    # Modify the values according to the noise level
    if noise_level == 0.1:
        non_zero_elements[indices[:one_third]] *= 1.1
        non_zero_elements[indices[one_third:2*one_third]] *= 0.9
    if noise_level == 0.2:
        non_zero_elements[indices[:one_third]] *= 1.2
        non_zero_elements[indices[one_third:2*one_third]] *= 0.8

    # Create a new array with modified values
    modified_arr = np.zeros_like(arr).astype(float)
    modified_arr[arr != 0] = non_zero_elements

    # Renormalize to the range 0 to 100
    modified_arr = np.clip(modified_arr, 0, 100)

    # Round to the closest integer
    modified_arr = np.round(modified_arr).astype(int)

    return modified_arr

