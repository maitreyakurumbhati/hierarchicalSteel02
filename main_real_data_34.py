import numpy as np
from get_data_real import get_data_real
from hierarchical_inference_sampling_v2 import HierarchicalInferenceSampling
import pickle


location = 'G:/My Drive/Research/Hierarchical Bayesian and Model Class Selection/Code/Data/cyclic_filtered/'

coupon_strain_data, coupon_stress_data = get_data_real(location)

'''
remove the last test data
'''

coupon_strain_data.pop()
coupon_strain_data.pop(0)
coupon_stress_data.pop()
coupon_stress_data.pop(0)

norm_pars = np.array([480, 205000, 0.019, 0.847, 0.183, 0.029, 0.017])

mean_theta = norm_pars/norm_pars * 0.1

D_theta = np.diag(0.2 * mean_theta)
R_theta = np.diag(np.ones(7))
cov_theta = D_theta @ R_theta @ D_theta

start_point = [mean_theta for coupon in range(len(coupon_stress_data))]

start_point.append(cov_theta)
start_point.append(mean_theta)

noise = [80.0 for coupon in range(len(coupon_stress_data))]

start_point.append(noise)

n_samples = 50000
tuning_interval = 1000
burn_in = 20000

inf_object = HierarchicalInferenceSampling(coupon_strain_data, coupon_stress_data, start_point, n_samples,
                                           burn_in, tuning_interval)

if __name__ == '__main__':

    trace, time = inf_object.perform_mcmc_within_gibbs()

    data_to_dump = [trace, inf_object, time]
    pickle.dump(data_to_dump, open("realdata_filtered_01_sep_noise.pickle", "wb"))
