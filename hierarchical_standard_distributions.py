import scipy.special
import scipy.stats


def get_scaled_inverse_chi_squared_sample(dof, scale_par):

    samp = scipy.stats.invgamma.rvs(a=dof/2, scale=dof*scale_par/2, size=1)

    return samp[0]


def get_inverse_wishart_sample(dof, scale_matrix):

    return scipy.stats.invwishart.rvs(df=dof, scale=scale_matrix, size=1)


def get_multivariate_normal_sample(mean_vector, cov_matrix):

    sample = scipy.stats.multivariate_normal.rvs(mean=mean_vector, cov=cov_matrix, size=1).flatten()

    return sample.tolist()
