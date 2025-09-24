"""
Sampler functions for Bayesian NonParametric Step (BNP-Step)

Alex Rojewski, 2023

"""

import numpy as np
import scipy as sp
from scipy import special
from distributions import MultivariateGaussian


def sample_b(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec, t_m_vec, f_vec, eta_vec, gamma, rng,
             temp):
    """
    Samples all loads b_m as part of BNP-Step's Gibbs sampler.

    Arguments:
    weak_limit -- Maximum number of possible steps
    num_data -- Number of observations
    data_points -- Observations
    data_times -- Time points corresponding to each observation
    b_m_vec -- Previous b_m samples
    h_m_vec -- Previous h_m samples
    t_m_vec -- Previous t_m samples
    f_vec -- Previous F_bg samples
    eta_vec -- Previous eta samples
    gamma -- Hyperparameter for priors on b_m
    rng -- Random number generator from BNPStep object
    temp -- Temperature (for simulated annealing)

    Returns:
    load_matrix -- a numpy array containing each new b_m sample for 1:M (where M is weak_limit)
                   Note: as vectorization results in a matrix of M identical copies of the samples, we 
                         only need to return the first row of this matrix.
    """
    # Pre-calculate all matrices which will never change element values
    # during load sampling
    times_matrix = np.broadcast_to(t_m_vec[-1], (num_data, weak_limit))
    obs_time_matrix = np.broadcast_to(data_times, (weak_limit, num_data)).T
    height_matrix = np.broadcast_to(h_m_vec[-1], (num_data, weak_limit))
    # Pre-calculate Heaviside terms times associated heights
    height_heaviside_mat = np.multiply(height_matrix,
                                       np.heaviside((-1 * (obs_time_matrix - times_matrix)),
                                                    np.ones((num_data, weak_limit))))

    # Initial assignment of load matrix - use the values from the previous sample.
    broadcast_loads = np.broadcast_to(b_m_vec[-1], (num_data, weak_limit))
    load_matrix = broadcast_loads.copy()

    # Pre-generate an array of Gumbel variables for sampling
    gumbel_variables = -np.log(-np.log(rng.random((weak_limit, 2))))

    # Shuffle the order of sampling each load
    sampling_order = np.arange(weak_limit)
    rng.shuffle(sampling_order)

    # Sample each load
    for i in range(weak_limit):
        # Calculate exponent value for b_m = 1
        load_matrix[:, sampling_order[i]] = 1
        bht_matrix = np.multiply(load_matrix, height_heaviside_mat)
        bht_sum = f_vec[-1] + np.sum(bht_matrix, axis=1)
        exponent_on = -(eta_vec[-1] / 2) * np.sum(np.square(data_points - bht_sum))

        # Calculate exponent value for b_m = 0
        load_matrix[:, sampling_order[i]] = 0
        bht_matrix = np.multiply(load_matrix, height_heaviside_mat)
        bht_sum = f_vec[-1] + np.sum(bht_matrix, axis=1)
        exponent_off = -(eta_vec[-1] / 2) * np.sum(np.square(data_points - bht_sum))

        # Calculate un-normalized log probabilities
        prob_on = ((1 / temp) * np.log(gamma / weak_limit)) + ((1 / temp) * exponent_on)
        prob_off = ((1 / temp) * np.log(1 - (gamma / weak_limit))) + ((1 / temp) * exponent_off)

        # Sample the b_m
        new_bm = np.argmax(np.asarray([(prob_off + gumbel_variables[i, 0]), (prob_on + gumbel_variables[i, 1])]))
        load_matrix[:, sampling_order[i]] = new_bm

    return load_matrix[0]


def sample_fh(weak_limit, num_data, data_points, data_times, b_m_vec, t_m_vec, eta_vec, psi, chi, f_ref, h_ref, rng,
              temp):
    """
    Samples F_bg and all h_m as part of BNP-Step's Gibbs sampler.

    Arguments:
    weak_limit -- Maximum number of possible steps
    num_data -- Number of observations
    data_points -- Observations
    data_times -- Time points corresponding to each observation
    b_m_vec -- Previous b_m samples
    h_m_vec -- Previous h_m samples
    t_m_vec -- Previous t_m samples
    f_vec -- Previous F_bg samples
    eta_vec -- Previous eta samples
    psi -- Variance hyperparameter for prior on F_bg
    chi -- Variance hyperparameter for priors on h_m
    f_ref -- Mean hyperparameter for prior on F_bg
    h_ref -- Mean hyperparameter for priors on h_m
    rng -- Random number generator from BNPStep object
    temp -- Temperature (for simulated annealing)

    Returns:
    new_fh -- a numpy array containing the new F_bg sample as the first element, followed by the new
              h_m samples.
    """
    # Record indices of on and off loads for later reconstruction of the vectors
    on_loads = []
    off_loads = []
    for i in range(weak_limit):
        if b_m_vec[-1][i] == 1:
            on_loads.append(i)
        else:
            off_loads.append(i)
    on_loads = np.asarray(on_loads)
    off_loads = np.asarray(off_loads)

    # Generate vectors with only entries corresponding to "on" loads - we
    # use this to draw posterior samples for the "on" loads.
    on_load_times = []
    on_loads_vector = np.ones(on_loads.size)
    num_on_loads = int(on_loads.size)
    for i in range(on_loads.size):
        on_load_times.append(t_m_vec[-1][int(on_loads[i])])
    on_load_times = np.asarray(on_load_times)

    # Pre-calculate all matrices which will never change element values
    # during height/background sampling
    times_matrix = np.broadcast_to(on_load_times, (num_data, num_on_loads))
    obs_time_matrix = np.broadcast_to(data_times, (num_on_loads, num_data)).T
    load_matrix = np.broadcast_to(on_loads_vector, (num_data, num_on_loads))
    # Pre-calculate Heaviside terms multiplied by associated loads
    load_heaviside_mat = np.multiply(load_matrix,
                                     np.heaviside((-1 * (obs_time_matrix - times_matrix)),
                                                  np.ones((num_data, num_on_loads))))

    # Calculate precision matrix
    precision_matrix = eta_vec[-1] * np.matmul(load_heaviside_mat.T, load_heaviside_mat) + (
            chi * np.eye(num_on_loads, num_on_loads))
    precision_first_row = eta_vec[-1] * np.sum(load_heaviside_mat, axis=0)
    precision_first_column = np.reshape(precision_first_row, (num_on_loads, 1))
    precision_first_row = np.insert(precision_first_row, 0, 0)
    precision_matrix = np.hstack((precision_first_column, precision_matrix))
    precision_matrix = np.vstack((precision_first_row, precision_matrix))
    precision_matrix[0, 0] = (eta_vec[-1] * num_data) + psi

    # Calculate q matrix - see SI for description of what this is
    q_matrix = (eta_vec[-1] * np.matmul(data_points, load_heaviside_mat)) + (chi * h_ref)
    q_matrix = np.insert(q_matrix, 0, (eta_vec[-1] * np.sum(data_points) + (psi * f_ref)))
    q_matrix = np.reshape(q_matrix, ((num_on_loads + 1), 1))

    # Find covariance matrix and mean vector
    covariance = temp * np.linalg.inv(precision_matrix)
    mean_vector = np.matmul(covariance, q_matrix)

    # Jointly sample F and the h_m corresponding to on loads from a multivariate Normal
    h_tmp = np.ravel(MultivariateGaussian.sample(mean_vector, sigma=covariance, epsilon=0.00000001))

    # Rebuild the full sample vector - includes both F and h_m
    fh_size = weak_limit + 1
    new_fh = np.zeros(fh_size)
    ind = 0
    for i in range(fh_size):
        is_zero = False
        for j in range(off_loads.size):
            if (off_loads[j] + 1) == i:
                is_zero = True
        if is_zero:
            # For all off loads, sample the new height from the prior
            new_fh[i] = rng.normal(h_ref, np.sqrt(1 / chi))
        else:
            new_fh[i] = h_tmp[ind]
            ind += 1

    return new_fh


def sample_fh_fast(weak_limit, num_data, data_points, data_times,
                   b_m_vec, t_m_vec, eta_vec, psi, chi, f_ref, h_ref, rng, temp,
                   Wacc=None):
    """
    Accelerated version of sample_fh using closed-form precision matrix
    from S19.pdf.

    Arguments:
    weak_limit -- maximum number of candidate steps (M)
    num_data -- number of observations (N)
    data_points -- observed data (w_1..N)
    data_times -- observation times (t_1..N), must be monotonic increasing
    b_m_vec -- previous b_m samples (take last row)
    t_m_vec -- previous t_m samples (take last row, real times)
    eta_vec -- previous eta samples (take last element)
    psi, chi -- precision hyperparameters for Fbg, h_m priors
    f_ref, h_ref -- prior means for Fbg, h_m
    rng -- numpy Generator
    temp -- annealing temperature
    Wacc -- optional precomputed prefix sum of data_points
    """
    b = np.asarray(b_m_vec[-1], dtype=np.int8)
    tau_real = np.asarray(t_m_vec[-1], dtype=np.float64)
    eta = float(eta_vec[-1])

    if Wacc is None:
        Wacc = np.cumsum(np.asarray(data_points, dtype=np.float64))
    else:
        Wacc = np.asarray(Wacc, dtype=np.float64)

    # indices of active steps (b=1), sorted by tau
    on_idx = np.where(b == 1)[0]
    if on_idx.size == 0:
        # no active steps: F ~ N(f_ref, psi^-1), h_m ~ N(h_ref, chi^-1)
        new_fh = np.empty(weak_limit + 1)
        new_fh[0] = rng.normal(f_ref, np.sqrt(1.0/psi))
        for m in range(weak_limit):
            new_fh[m+1] = rng.normal(h_ref, np.sqrt(1.0/chi))
        return new_fh

    tau_idx = np.searchsorted(data_times, tau_real[on_idx], side="left")
    order = np.argsort(tau_idx, kind="stable")
    on_idx = on_idx[order]
    tau_idx = tau_idx[order]
    ml = on_idx.size

    # build precision matrix Λ and q vector
    dim = ml + 1
    Lambda = np.zeros((dim, dim))
    q = np.zeros(dim)

    # (7) Λ11
    Lambda[0, 0] = eta * num_data + psi
    # (11) q1
    q[0] = eta * Wacc[-1] + psi * f_ref

    # fill rows/cols for active steps
    for j in range(ml):
        m = on_idx[j]
        tau = tau_idx[j]
        # (8) Λi1
        Lambda[j+1, 0] = eta * tau
        Lambda[0, j+1] = eta * tau
        # (9) Λii
        Lambda[j+1, j+1] = eta * tau + chi
        # (12) qi
        q[j+1] = eta * Wacc[tau] + chi * h_ref
        # (10) Λij for i != j
        for k in range(j):
            tau_k = tau_idx[k]
            val = eta * min(tau, tau_k)
            Lambda[j+1, k+1] = val
            Lambda[k+1, j+1] = val

    # covariance and mean
    cov = temp * np.linalg.inv(Lambda)
    mean = cov @ q

    # joint Gaussian sample
    fh_sample = rng.multivariate_normal(mean, cov)

    # build full vector of length weak_limit+1
    new_fh = np.empty(weak_limit + 1)
    new_fh[0] = fh_sample[0]  # Fbg
    # assign active steps in order
    for j, m in enumerate(on_idx):
        new_fh[m+1] = fh_sample[j+1]
    # inactive steps: sample from prior
    off_idx = np.where(b == 0)[0]
    for m in off_idx:
        new_fh[m+1] = rng.normal(h_ref, np.sqrt(1.0/chi))

    return new_fh


def sample_t(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
             t_m_vec, f_vec, eta_vec, rng, temp):
    """
    Samples all t_m as part of BNP-Step's Gibbs sampler.

    Arguments:
    weak_limit -- Maximum number of possible steps
    num_data -- Number of observations
    data_points -- Observations
    data_times -- Time points corresponding to each observation
    b_m_vec -- Previous b_m samples
    h_m_vec -- Previous h_m samples
    t_m_vec -- Previous t_m samples
    f_vec -- Previous F_bg samples
    eta_vec -- Previous eta samples
    rng -- Random number generator from BNPStep object
    temp -- Temperature (for simulated annealing)

    Returns:
    times_matrix -- a numpy array containing each new t_m sample for 1:M (where M is weak_limit)
                    Note: as vectorization results in a matrix of M identical copies of the samples, we 
                          only need to return the first row of this matrix.
    """
    # Pre-calculate all matrices which will never change element values
    # during jump time sampling
    load_matrix = np.broadcast_to(b_m_vec[-1], (num_data, weak_limit))
    obs_time_matrix = np.broadcast_to(data_times, (weak_limit, num_data)).T
    height_matrix = np.broadcast_to(h_m_vec[-1], (num_data, weak_limit))
    # Pre-calculate product of b_m and h_m term-wise
    bh_matrix = np.multiply(load_matrix, height_matrix)

    # Initial assignment of times matrix - use the values from the previous sample.
    # These will act as our t_olds for Metropolis updates.
    broadcast_times = np.broadcast_to(t_m_vec[-1], (num_data, weak_limit))
    times_matrix = broadcast_times.copy()

    # Shuffle the order of sampling each jump time
    sampling_order = np.arange(weak_limit)
    rng.shuffle(sampling_order)

    # Pre-generate a bunch of random values to use with Metropolis step
    u_value = rng.random(weak_limit)

    # Iterate through each t_m and generate new samples
    for i in range(weak_limit):
        # If the corresponding load is off, just sample from the prior
        if b_m_vec[-1][sampling_order[i]] == 0:
            times_matrix[:, sampling_order[i]] = rng.choice(data_times)
        # Otherwise, perform a Metropolis update
        else:
            # Generate a proposal - draw from the prior
            t_prop = rng.choice(data_times)
            t_old = times_matrix[0, sampling_order[i]]
            # Calculate exponent value for t_old
            # Calculate Heaviside terms for likelihood
            bht_matrix = np.multiply(bh_matrix, np.heaviside((-1 * (obs_time_matrix - times_matrix)),
                                                             np.ones((num_data, weak_limit))))
            bht_sum = f_vec[-1] + np.sum(bht_matrix, axis=1)
            exponent_old = -(eta_vec[-1] / 2) * np.sum(np.square(data_points - bht_sum))

            # Calculate exponent value for t_prop
            times_matrix[:, sampling_order[i]] = t_prop
            # Calculate Heaviside terms for likelihood
            bht_matrix = np.multiply(bh_matrix, np.heaviside((-1 * (obs_time_matrix - times_matrix)),
                                                             np.ones((num_data, weak_limit))))
            bht_sum = f_vec[-1] + np.sum(bht_matrix, axis=1)
            exponent_prop = (eta_vec[-1] / 2) * np.sum(np.square(data_points - bht_sum))

            # Calculate acceptance ratio
            acceptance_ratio = (1 / (np.exp(exponent_prop + exponent_old))) ** (1 / temp)

            # See if we accept the proposal - if so, we've already updated the relevant element.
            # If not, then we need to replace t_old back into the times matrix.
            if u_value[i] > acceptance_ratio:
                times_matrix[:, sampling_order[i]] = t_old

    return times_matrix[0]

def _softmax_sample_from_logits(logits, rng):
    """
    Numerically stable softmax sampling:
    - logits: 1D array of real-valued scores
    - returns: one index sampled according to softmax(logits)
    """
    z = logits - np.max(logits)                  # log-sum-exp stabilization
    p = np.exp(z, dtype=np.float64)
    s = p.sum()
    if not np.isfinite(s) or s <= 0.0:           # extreme underflow/NaN guard
        # fall back to argmax, which matches the limit of an extremely peaked softmax
        return int(np.argmax(logits))
    p /= s
    # sample via inverse-CDF to be robust when p is very sparse
    u = rng.random()
    cdf = np.cumsum(p)
    return int(np.searchsorted(cdf, u, side="left"))

def sample_t_softmax_strict(weak_limit, num_data, data_points, data_times,
                            b_m_vec, h_m_vec, t_m_vec, f_vec, eta_vec, rng, temp,
                            Wacc=None, eps=1e-12):
    """
    Strict softmax Gibbs update for all tau_m (sampling efficiency prioritized).
      - Only compute softmax for active steps (b_m=1); inactive steps sample from prior
      - After each tau_m update, recompute H1/H2 using the latest tau_{-m} (strict Gibbs)
      - Numerically stable: log-sum-exp, clipping, tie-handling, temperature scaling

    Args:
      weak_limit: M (max number of steps)
      num_data: N (number of observations)
      data_points: array of length N (w_1..N)
      data_times: array of length N (assumed non-decreasing)
      b_m_vec, h_m_vec, t_m_vec, f_vec, eta_vec: sampler histories (use last entries)
      rng: numpy.random.Generator
      temp: temperature for simulated annealing (>=1 typically)
      Wacc: optional prefix sums of data_points; if None, computed via cumsum
      eps: small epsilon for numerical guards

    Returns:
      t_new: array (length M) with updated real-valued tau times
    """
    # -- Current state (use last samples) --
    b   = np.asarray(b_m_vec[-1], dtype=np.int8)        # (M,)
    h   = np.asarray(h_m_vec[-1], dtype=np.float64)     # (M,)
    t   = np.asarray(t_m_vec[-1], dtype=np.float64)     # (M,) real times
    F   = float(f_vec[-1])
    eta = float(eta_vec[-1])
    M   = int(weak_limit)
    N   = int(num_data)

    # -- Precompute Wacc and index vector 1..N --
    if Wacc is None:
        Wacc = np.cumsum(np.asarray(data_points, dtype=np.float64))
    else:
        Wacc = np.asarray(Wacc, dtype=np.float64)
    n1 = np.arange(1, N + 1, dtype=np.float64)

    # Bounds for clipping (searchsorted safety)
    tmin, tmax = float(data_times[0]), float(data_times[-1])

    # -- Partition active and inactive steps --
    on_mask = (b == 1)
    on_idx  = np.where(on_mask)[0]
    off_idx = np.where(~on_mask)[0]

    # Initialize result with old values
    t_new = t.copy()

    # Inactive steps: sample from prior (uniform over data_times) to match original behavior
    for m in off_idx:
        t_new[m] = rng.choice(data_times)

    # If no active steps, we are done
    if on_idx.size == 0:
        return t_new

    # -- Temperature scaling in energy --
    scale = -eta / (2.0 * max(float(temp), eps))

    # -- Random-scan Gibbs (better mixing) --
    order_scan = np.array(on_idx, copy=True)
    rng.shuffle(order_scan)

    for m in order_scan:
        hm = float(h[m])

        # (1) Map latest tau to indices and build a stable order among active steps
        tau_idx_all = np.searchsorted(
            data_times,
            np.clip(t_new[on_idx], tmin, tmax),
            side="left"
        )
        tau_idx_all = np.clip(tau_idx_all, 0, N - 1)

        # Stable sort by (tau_idx, original index)
        order = np.lexsort((on_idx, tau_idx_all))
        idx_ord = on_idx[order]
        tau_ord = tau_idx_all[order].astype(np.float64)
        h_ord   = h[idx_ord].astype(np.float64)

        # Enforce non-decreasing and add tiny jitter if exact ties remain
        tau_ord = np.maximum.accumulate(tau_ord)
        if np.any(np.diff(tau_ord) <= 0):
            tau_ord += eps * np.arange(tau_ord.size, dtype=np.float64)

        # (2) Locate current step m and compute H1/H2 in real-time (O(B))
        pos = int(np.where(idx_ord == m)[0][0])
        H1 = float(np.dot(h_ord[:pos], tau_ord[:pos])) if pos > 0 else 0.0
        H2 = float(np.sum(h_ord[pos+1:])) if pos + 1 < h_ord.size else 0.0

        # (3) Build logits(n) = scale * [ a*Wacc[n] + b*n1 + c ]
        a = -2.0 * hm
        bcoef = 2.0 * F * hm + hm * hm + 2.0 * hm * H2
        c = 2.0 * hm * H1
        logits = scale * (a * Wacc + bcoef * n1 + c)  # length N

        # (4) Sample index n* via softmax and write back the real time
        n_star = _softmax_sample_from_logits(logits, rng)
        t_new[m] = data_times[n_star]

    return t_new

def sample_eta(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec, t_m_vec, f_vec,
               phi, eta_ref, rng, temp):
    """
    Samples eta as part of BNP-Step's Gibbs sampler.

    Arguments:
    weak_limit -- Maximum number of possible steps
    num_data -- Number of observations
    data_points -- Observations
    data_times -- Time points corresponding to each observation
    b_m_vec -- Previous b_m samples
    h_m_vec -- Previous h_m samples
    t_m_vec -- Previous t_m samples
    f_vec -- Previous F_bg samples
    phi -- Shape hyperparameter for Gamma prior on eta
    eta_ref -- Scale hyperparameter (as eta_ref/phi) for Gamma prior on eta
    rng -- Random number generator from BNPStep object
    temp -- Temperature (for simulated annealing)

    Returns:
    new_eta -- the new sample for eta
    """
    # Pre-calculate all matrices which will never change element values
    # during eta sampling, which is all of them.
    times_matrix = np.broadcast_to(t_m_vec[-1], (num_data, weak_limit))
    obs_time_matrix = np.broadcast_to(data_times, (weak_limit, num_data)).T
    height_matrix = np.broadcast_to(h_m_vec[-1], (num_data, weak_limit))
    load_matrix = np.broadcast_to(b_m_vec[-1], (num_data, weak_limit))
    # Pre-calculate product of b_m and h_m term-wise
    bh_matrix = np.multiply(load_matrix, height_matrix)
    # Pre-calculate Heaviside terms times loads and heights
    bht_matrix = np.multiply(bh_matrix,
                             np.heaviside((-1 * (obs_time_matrix - times_matrix)),
                                          np.ones((num_data, weak_limit))))
    # Calculate large sum term for beta
    bht_sum = f_vec[-1] + np.sum(bht_matrix, axis=1)
    beta_sum_term = (1 / 2) * np.sum(np.square(data_points - bht_sum))

    # Calculate beta hyperparameter
    beta = temp / ((phi / eta_ref) + beta_sum_term)
    # Calculate alpha hyperparameter
    alpha = (1 / temp) * ((num_data / 2) + phi + temp - 1)

    # Sample new eta
    new_eta = rng.gamma(alpha, beta)

    return new_eta


def calculate_loglikelihood(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                            t_m_vec, f_vec, eta_vec):
    """
    Calculates the log likelihood given a dataset and a set of associated samples from BNP-Step

    Arguments:
    weak_limit -- Maximum number of possible steps
    num_data -- Number of observations
    data_points -- Observations
    data_times -- Time points corresponding to each observation
    b_m_vec -- Previous b_m samples
    h_m_vec -- Previous h_m samples
    t_m_vec -- Previous t_m samples
    f_vec -- Previous F_bg samples
    eta_vec -- Previous eta samples

    Returns:
    new_log_likelihood -- the log likelihood for the provided samples and observations
    """
    # Calculate matrices required for vectorized sum calculations
    times_matrix = np.broadcast_to(t_m_vec[-1], (num_data, weak_limit))
    obs_time_matrix = np.broadcast_to(data_times, (weak_limit, num_data)).T
    height_matrix = np.broadcast_to(h_m_vec[-1], (num_data, weak_limit))
    load_matrix = np.broadcast_to(b_m_vec[-1], (num_data, weak_limit))

    # Calculate product of b_m and h_m term-wise
    bh_matrix = np.multiply(load_matrix, height_matrix)
    # Calculate Heaviside terms times loads and heights
    bht_matrix = np.multiply(bh_matrix,
                             np.heaviside((-1 * (obs_time_matrix - times_matrix)),
                                          np.ones((num_data, weak_limit))))
    # Calculate sum term for the exponent
    bht_sum = f_vec[-1] + np.sum(bht_matrix, axis=1)
    exponent_term = (eta_vec[-1] / 2) * np.sum(np.square(data_points - bht_sum))
    # Calculate log likelihood
    new_log_likelihood = ((num_data / 2) * np.log(eta_vec[-1] / (2 * np.pi))) - exponent_term

    return new_log_likelihood


def calculate_logposterior(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                           t_m_vec, f_vec, eta_vec, chi, h_ref, gamma, phi, eta_ref, psi, f_ref):
    """
    Calculates the log posterior given a dataset and a set of associated samples from BNP-Step

    Arguments:
    weak_limit -- Maximum number of possible steps
    num_data -- Number of observations
    data_points -- Observations
    data_times -- Time points corresponding to each observation
    b_m_vec -- Previous b_m samples
    h_m_vec -- Previous h_m samples
    t_m_vec -- Previous t_m samples
    f_vec -- Previous F_bg samples
    eta_vec -- Previous eta samples
    chi -- Variance hyperparameter for priors on h_m
    h_ref -- Mean hyperparameter for priors on h_m
    gamma -- Hyperparameter for priors on b_m
    phi -- Shape hyperparameter for Gamma prior on eta
    eta_ref -- Scale hyperparameter (as eta_ref/phi) for Gamma prior on eta
    psi -- Variance hyperparameter for prior on F_bg
    f_ref -- Mean hyperparameter for prior on F_bg

    Returns:
    log_posterior -- the log posterior for the provided samples and observations
    """
    # Calculate the log likelihood
    log_likelihood = calculate_loglikelihood(weak_limit, num_data, data_points, data_times, b_m_vec, h_m_vec,
                                             t_m_vec, f_vec, eta_vec)

    # Calculate priors on b_m
    b_m = np.asarray(b_m_vec[-1])
    on_prior = (gamma / weak_limit) * b_m
    off_prior = (np.ones(weak_limit) - b_m) - ((gamma / weak_limit) * (np.ones(weak_limit) - b_m))
    prior_b_m = np.sum(np.log(on_prior + off_prior))

    # Calculate priors on h_m
    h_m = np.asarray(h_m_vec[-1])
    prior_h_m = ((weak_limit / 2) * np.log(chi / (2 * np.pi))) - ((chi / 2) * np.sum(np.square(h_m - h_ref)))

    # Calculate priors on t_m
    prior_t_m = -weak_limit * np.log(num_data)

    # Calculate prior on eta
    prior_eta = ((phi - 1) * np.log(eta_vec[-1])) - ((phi * eta_vec[-1]) / eta_ref) - np.log(sp.special.gamma(phi)) - \
                (phi * np.log(eta_ref / phi))

    # Calculate prior on F
    prior_f = ((1 / 2) * np.log(psi / (2 * np.pi))) - ((psi / 2) * ((f_vec[-1] - f_ref) ** 2))

    log_posterior = log_likelihood + prior_b_m + prior_h_m + prior_t_m + prior_eta + prior_f

    return log_posterior
