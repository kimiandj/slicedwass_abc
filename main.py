# Author: Kimia Nadjahi

import pyabc
import tempfile
import os
import numpy as np
import utils
from scipy.stats import gaussian_kde, invgamma
import pickle
import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd

# Matplotlib settings
plt.rcParams.update(plt.rcParamsDefault)
plt.style.use('bmh')
fontsize = 10
params = {'axes.labelsize': fontsize + 2,
          'font.size': fontsize + 2,
          'legend.fontsize': fontsize,
          'xtick.labelsize': fontsize,
          'ytick.labelsize': fontsize,
          'text.usetex': True,
          'errorbar.capsize': 4}
plt.rcParams.update(params)
plt.rc('font', family='serif')
plt.rcParams['axes.facecolor'] = 'white'


def distance_fn(type, n_proj):
    if type == 'summary_euclidean_scale':
        return lambda x, y: np.linalg.norm((x["data"].var(axis=0) - y["data"].var(axis=0)) * x["data"].shape[0])
    elif type == 'hilbert':
        return lambda x, y: utils.hilbert_distance(x["data"], y["data"], p=2)
    elif type == 'swapping':
        return lambda x, y: utils.swap_distance(x["data"], y["data"], n_sweeps=100, tol=1e-8, p=2)
    elif type == 'sliced_w':
        return lambda x, y: utils.sw_distance(x["data"], y["data"], n_montecarlo=1, L=n_proj, p=2)
    elif type == 'kl':
        return lambda x, y: utils.kl_empirical(x["data"], y["data"])
    else:
        raise ValueError(
            "Distance type should be 'summary_euclidean_scale', 'hilbert', "
            "'swapping', 'sliced_w', or 'kl'.")


def save_results(history, times, dirname):
    # Create directory that will contain the results
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    for it in range(history.max_t+1):
        # Save the posterior distribution at each ABC iteration
        filename = 'posterior_it=' + str(it) + '.csv'
        df, w = history.get_distribution(m=0, t=it)
        df['weight'] = w
        df.to_csv(os.path.join(dirname, filename))

        # Save extended information at each iteration, including weighted distances that the parameter samples achieve
        filename = 'info_it=' + str(it) + '.csv'
        df = history.get_population_extended(m=0, t=it)
        df.to_csv(os.path.join(dirname, filename))

    # Save information on the evolution of epsilon, the number of sample attempts per iteration and the iteration times
    filename = 'all_populations.csv'
    df = history.get_all_populations()
    # df['times'] = np.insert(times, 0, 0)
    df.to_csv(os.path.join(dirname, filename))


def plot_posterior(param, dim, n_obs, n_it, n_particles, n_proj, types, labels):
    # Matplotlib settings
    plt.rcParams['lines.linewidth'] = 1

    directory = os.path.join("results", param + "_dim=" + str(dim) + "_n_obs=" + str(n_obs)
                             + "_n_particles=" + str(n_particles) + "_n_proj=" + str(n_proj)
                             + "_n_it=" + str(n_it))

    # Plot true posterior pdf
    fig = plt.figure(0, figsize=(4, 2))
    with open(os.path.join(directory, "true_posterior"), "rb") as f:
        post_samples = pickle.load(f)
    pyabc.visualization.plot_kde_1d(
        pd.DataFrame({"post_samples": post_samples}), np.ones(post_samples.shape[0]) / post_samples.shape[0],
        xmin=0, xmax=10, ax=plt.gca(), x="post_samples", color='darkgray', linestyle='--', numx=1000, label="True posterior")
    t = np.linspace(0, 10, 1000)
    plt.fill_between(t, plt.gca().lines[0].get_ydata(), facecolor='gray', alpha=0.4)

    # Plot ABC posteriors
    for i in range(len(types)):
        df = pd.read_csv(os.path.join(directory, types[i], 'all_populations.csv'))
        max_it = df['t'].iloc[-1]
        df = pd.read_csv(os.path.join(directory, types[i], 'posterior_it=' + str(max_it) + '.csv'))
        w = df['weight'].values
        df = df[df.columns.difference(['weight'])]
        pyabc.visualization.plot_kde_1d(
            df, w,
            xmin=0, xmax=10, ax=plt.gca(),
            x="scale", numx=1000, label=labels[i])
        plt.fill_between(t, plt.gca().lines[-1].get_ydata(),
                         facecolor=plt.gca().lines[-1].get_color(), alpha=0.2)
        plt.xlabel(r'$\sigma^2$', fontsize=12)
        plt.ylabel("density", fontsize=14)
    plt.legend(fontsize=8)
    plt.savefig(os.path.join(directory, "abc_posteriors.pdf"), bbox_inches='tight')
    plt.close(fig)


def plot_wass(param, dim, n_obs, n_it, n_particles, n_proj, types, labels):
    # Open true posterior data
    directory = os.path.join("results", param + "_dim=" + str(dim) + "_n_obs=" + str(n_obs)
                             + "_n_particles=" + str(n_particles) + "_n_proj=" + str(n_proj)
                             + "_n_it=" + str(n_it))
    with open(os.path.join(directory, "true_posterior"), "rb") as f:
        post_samples = pickle.load(f)

    # Plot Wasserstein distance to true posterior
    wass1 = np.zeros((len(types), n_it))
    for i in range(len(types)):
        df = pd.read_csv(os.path.join(directory, types[i], 'all_populations.csv'))
        max_it = df['t'].iloc[-1]
        with open(os.path.join(directory, "times_" + str(types[i])), "rb") as f:
            times = pickle.load(f)
        for t in range(max_it+1):
            df = pd.read_csv(os.path.join(directory, types[i], 'posterior_it=' + str(t) + '.csv'))
            df = df[df.columns.difference(['id', 'weight'])]
            wass1[i, t] = utils.wass_distance(df.values, post_samples[:, None], order=1, type="exact")
            x = [times[i] - times[0] for i in range(max_it+1)]
        plt.figure(1, figsize=(4, 2))
        plt.plot(range(1, max_it+2), wass1[i, :max_it+1], label=labels[i], marker='.')
        plt.figure(2, figsize=(4, 2))
        plt.plot(x, wass1[i, :max_it+1], label=labels[i], marker='.')
    plt.figure(1)
    plt.xlabel("number of iteration", fontsize=12)
    plt.ylabel(r'$\mathbf{W}_1$', fontsize=14)
    plt.xticks(range(1, max_it+2, 2))
    plt.legend()
    plt.savefig(os.path.join(directory, "wass1_comparison_it.pdf"), bbox_inches='tight')
    plt.figure(2)
    plt.xlabel("time (s)", fontsize=12)
    plt.xscale('symlog')
    plt.ylabel(r'$\mathbf{W}_1$', fontsize=14)
    plt.legend()
    plt.savefig(os.path.join(directory, "wass1_comparison_time.pdf"), bbox_inches='tight')
    with open(os.path.join(directory, "wass1"), "wb") as f:
        pickle.dump(wass1, f, pickle.HIGHEST_PROTOCOL)


def main(param, dim, n_obs, n_procs, n_it, n_particles, n_proj, max_time, types, labels):
    # Create directory that will contain the results
    directory = os.path.join("results", param + "_dim=" + str(dim) + "_n_obs=" + str(n_obs)
                             + "_n_particles=" + str(n_particles) + "_n_proj=" + str(n_proj)
                             + "_n_it=" + str(n_it))
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Define data-generating parameters
    true_mean = np.random.normal(size=dim)
    true_scale = 4
    Sigma_likelihood = true_scale * np.eye(dim)
    # Define priors on the scale parameter
    alph = 1
    prior_args = {"scale": pyabc.RV("invgamma", alph)}
    prior = pyabc.Distribution(prior_args)

    # Generate observations
    observations = np.random.multivariate_normal(true_mean, Sigma_likelihood, size=n_obs)
    # Save the dataset of observations
    with open(os.path.join(directory, "dataset"), "wb") as f:
        pickle.dump(observations, f, pickle.HIGHEST_PROTOCOL)

    # Define parameters of the true posterior
    alph_post = alph + 0.5 * (n_obs * dim)
    beta_post = 1 + 0.5 * ((observations - true_mean)*(observations - true_mean)).sum()
    # Generate parameter samples from the true posterior
    post_samples = invgamma.rvs(a=alph_post, scale=beta_post, size=n_particles)
    # Save the result
    with open(os.path.join(directory, "true_posterior"), "wb") as f:
        pickle.dump(post_samples, f, pickle.HIGHEST_PROTOCOL)

    # Define generative model used in ABC to generate synthetic data
    def model(parameter):
        Sigma = (parameter["scale"]) * np.eye(dim)
        return {"data": np.random.multivariate_normal(true_mean, Sigma, size=n_obs)}

    for i in range(len(types)):
        print("Running ABC-SMC with " + str(labels[i]) + " distance...")
        distance = distance_fn(types[i], n_proj)
        abc = pyabc.ABCSMC(models=model,
                           parameter_priors=prior,
                           distance_function=distance,
                           population_size=n_particles,  # nb of particles
                           sampler=pyabc.sampler.MulticoreEvalParallelSampler(n_procs=n_procs),
                           eps=pyabc.epsilon.QuantileEpsilon(alpha=0.5))

        # Setting the observed data for the ABC-SMC object
        db_path = ("sqlite:///" +
                   os.path.join(tempfile.gettempdir(), "test.db"))
        abc_id = abc.new(db_path, {"data": observations})

        # Run ABC-SMC
        history, times = abc.run(minimum_epsilon=0.01, max_nr_populations=n_it, max_time=max_time*60.0)

        # Save results
        print("Done! Saving results for ABC-SMC with " + str(labels[i]) + " distance...")
        save_results(history, times, os.path.join(directory, types[i]))

        # with open(os.path.join(directory, "results_" + str(types[i])), "wb") as f:
        #     pickle.dump(history, f, pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(directory, "times_" + str(types[i])), "wb") as f:
            pickle.dump(times, f, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", type=int, default=2, help="data dimension")    
    parser.add_argument("--n_obs", type=int, default=100, help="number of observations")
    parser.add_argument("--n_procs", type=int, default=pyabc.sge.nr_cores_available(),
                        help='number of processors to use for parallelization')
    parser.add_argument("--n_it", type=int, default=10, help="number of ABC iterations")
    parser.add_argument("--n_particles", type=int, default=100, help="number of particles")
    parser.add_argument("--n_proj", type=int, default=10, help="number of projections for the SW")
    parser.add_argument("--max_time", type=float, default=10.0, help="maximum running time (in min)")
    args = parser.parse_args()

    # Try different distances on ABC-SMC
    test_types = ["summary_euclidean_scale", "hilbert", "swapping", "sliced_w", "kl"]
    test_labels = ["Euclidean-ABC", "WABC-Hilbert", "WABC-Swapping", "SW-ABC", "KL-ABC"]

    main(param="scale", dim=args.dim, n_obs=args.n_obs, n_procs=args.n_procs, n_it=args.n_it,
         n_particles=args.n_particles, n_proj=args.n_proj, max_time=args.max_time,
         types=test_types, labels=test_labels)

    print("Plotting the final posterior distribution...")
    plot_posterior(param="scale", dim=args.dim, n_obs=args.n_obs,
                   n_it=args.n_it, n_particles=args.n_particles, n_proj=args.n_proj,
                   types=test_types, labels=test_labels)

    print("Computing the Wasserstein distance...")
    plot_wass(param="scale", dim=args.dim, n_obs=args.n_obs,
              n_it=args.n_it, n_particles=args.n_particles, n_proj=args.n_proj,
              types=test_types, labels=test_labels)
