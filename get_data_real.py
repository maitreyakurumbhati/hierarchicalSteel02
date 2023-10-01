import numpy as np


def get_data_real(folder_location):

    mill_string = ["A", "B", "C"]
    coupon_string = list(map(str, np.arange(4, 10, 1)))
    grade_string = list(map(str, np.arange(6, 8, 1)))

    strain_all_tests = []
    stress_all_tests = []

    counter = 1

    for mill in mill_string:
        for grade in grade_string:
            for coupon in coupon_string:
                coupon_name = mill + grade + coupon

                data = np.genfromtxt(folder_location + str(counter) + "_" + coupon_name + ".csv", delimiter=' ')
                strain_history = data[:, 0]
                stress_data = data[:, 1]

                strain_all_tests.append(strain_history)
                stress_all_tests.append(stress_data)
                counter += 1

    return strain_all_tests, stress_all_tests
