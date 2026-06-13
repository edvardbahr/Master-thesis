import matplotlib
matplotlib.use("TkAgg")  # Interactive backend: opens a separate plot window.
from matplotlib import pyplot as plt
import numpy as np
import math
import pandas as pd

from scipy.special import kve as _bessel_kve
from scipy.stats import beta as _beta_dist
from scipy.stats import geninvgauss as _gig_dist
from scipy.stats import invgamma as _invgamma_dist
from scipy.stats import t as _student_t_dist


_LOG_TWO = math.log(2.0)
_LOG_PI = math.log(math.pi)


def GH_skew_student_t_pdf(x, nu, mu, delta, beta, log=False):
    if nu <= 0.0 or delta <= 0.0:
        raise ValueError("nu and delta must be positive")

    if beta == 0.0:
        scale = delta / math.sqrt(nu)
        if log:
            return _student_t_dist.logpdf(x, df=nu, loc=mu, scale=scale)
        return _student_t_dist.pdf(x, df=nu, loc=mu, scale=scale)

    x = np.asarray(x, dtype=float)
    centered = x - mu
    radius = np.hypot(delta, centered)
    abs_beta = abs(beta)
    order = 0.5 * (nu + 1.0)
    z = abs_beta * radius

    log_pdf = (
        0.5 * (1.0 - nu) * _LOG_TWO
        + nu * math.log(delta)
        + order * math.log(abs_beta)
        + np.log(_bessel_kve(order, z)) - z
        + beta * centered
        - math.lgamma(0.5 * nu)
        - 0.5 * _LOG_PI
        - order * np.log(radius)
    )

    if log:
        return log_pdf

    return np.exp(log_pdf)


def GH_skew_student_t_rvs(nu, mu, delta, beta, size=None, random_state=None):
    if nu <= 0.0 or delta <= 0.0:
        raise ValueError("nu and delta must be positive")

    if beta == 0.0:
        scale = delta / math.sqrt(nu)
        return _student_t_dist.rvs(df=nu, loc=mu, scale=scale, size=size, random_state=random_state)

    rng = np.random.default_rng(random_state)
    w = _invgamma_dist.rvs(0.5 * nu, scale=0.5 * delta * delta, size=size, random_state=rng)
    z = rng.standard_normal(size=size)
    return mu + beta * w + np.sqrt(w) * z



if __name__ == "__main__":
    # Degrees of freedom. To be kept positive. Expectation exists when nu > 2,
    # variance exists when nu > 4, skew when nu > 6 and kurtosis when nu > 8.
    #nu = 11.934015443794806
    # Acts as a location parameter when beta = 0.0 or delta = 0.0. (note that Gh skew t becomes a regular t distribution when beta = 0.0)
    #mu = -0.11096648143597644
    # Scale parameter. To be kept positive.
    #delta = 2.0573811824740798
    # Skewness parameter. When beta > 0, the distribution is skewed to the right, and when beta < 0, it is skewed to the left.
    #beta = 0.26042113901859876

    # MLE of s,r,nu : 0.655707, 0.008672, 11.782924
    # s is the standard deviation of the distribution
    s_values = np.array([0.65, 0.65, 0.65])
    # r is the fraction of variance explained by skewness vs tail heaviness. To be kept in [0, 1). When r = 0, the distribution is a regular t distribution, and when r approaches 1, the distribution becomes more skewed and less heavy-tailed.
    r_values = np.array([0.01, 0.5, 0.9999])
    # nu controls tail heaviness. To be kept positive. Expectation exists when nu > 2, variance exists when nu > 4, skew when nu > 6 and kurtosis when nu > 8.
    nu_values = np.array([8.0, 8.0, 8.0])

    assert len(s_values) == len(r_values) == len(nu_values), "s, r and nu must have the same length"


    # nu min should be 6 or greater and should only exceed 30 with probability 1%. Its mean should be 12
    # 

    delta = s_values * np.sqrt((nu_values - 2) * (1 - r_values))
    beta = np.sqrt(r_values * (nu_values - 4) / 2) / (s_values * (1 - r_values))
    mu = -s_values * np.sqrt(r_values * (nu_values - 4) / 2)

    sequence_length = 200
    samples = np.empty((len(s_values), sequence_length))
    for i in range(len(s_values)):
        samples[i] = GH_skew_student_t_rvs(
            nu_values[i],
            mu[i],
            delta[i],
            beta[i],
            size=sequence_length,
            random_state=2,
        )
    x = np.linspace(*np.quantile(samples[0], [0.001, 0.999]), 1000)

    if False:
        #plt.hist(samples, bins=120, density=True, alpha=0.35, label="GH skew-t samples")
        plt.plot(
            x,
            GH_skew_student_t_pdf(x, nu_values[0], mu[0], delta[0], beta[0]),
            linewidth=2.0,
            label="GH skew-t PDF",
        )
        plt.title("GH Skew Student's t Samples vs PDF")
        plt.xlabel("x")
        plt.ylabel("Density")
        plt.grid()
        plt.legend()
        plt.show()
    
    if True:

        np.random.seed(1)

        SV = {
            "mu": -9.896525,
            "phi": 0.95,
            #"phi": 0.8238304,
        }

        h = np.empty((len(s_values), sequence_length))
        y = np.empty((len(s_values), sequence_length))
        epsilon = np.random.normal(loc=0.0, scale=1.0, size=sequence_length)
        GHST_mean = mu + beta * delta ** 2 / (nu_values - 2.0)
        h[:, 0] = SV["mu"] + GHST_mean / (1 - SV["phi"])
        y[:, 0] = np.exp(0.5 * h[:, 0]) * epsilon[0]

        for i in range(1, sequence_length):
            h[:, i] = SV["mu"] + SV["phi"] *( h[:, i-1] - SV["mu"]) + samples[:, i]
            y[:, i] = np.exp(0.5 * h[:, i]) * epsilon[i]
        

        fig, axes = plt.subplots(3, 2, figsize=(12, 8))

        for idx, (s, r, nu) in enumerate(zip(s_values, r_values, nu_values)):
            
            axes[idx, 0].plot(h[idx], label="SV process")
            axes[idx, 0].set_title(f"SV process with s={s}, r={r}, nu={nu}")
            axes[idx, 0].set_xlabel("Time")
            axes[idx, 0].set_ylabel("h")
            axes[idx, 0].grid()
            axes[idx, 0].legend()
            axes[idx, 1].plot(y[idx], label="Observations")
            axes[idx, 1].set_title(f"Observations from SV process with s={s}, r={r}, nu={nu}")
            axes[idx, 1].set_xlabel("Time")
            axes[idx, 1].set_ylabel("y")
            axes[idx, 1].grid()
            axes[idx, 1].legend()
            plt.tight_layout()

        plt.show()

    
    if False:



        s = 3.0

        # create a plot of sub plots with different values of r
        fig, axes = plt.subplots(3, 3, figsize=(15, 10))
        r_values = [0.9, 0.99, 0.9999]
        nu_values = [5, 10, 20]
        for i in range(3):
            for j in range(3):
                r = r_values[j]
                nu = nu_values[i]
                delta = s * math.sqrt((nu - 2) * (1 - r))
                beta = math.sqrt(r * (nu - 4) / 2) / (s * (1 - r))
                mu = -s * math.sqrt(r * (nu - 4) / 2)

                x = np.linspace(-5, 5, 1000)
                y = GH_skew_student_t_pdf(x, nu, mu, delta, beta)

                axes[i, j].plot(
                    x,
                    y,
                    linewidth=2.0,
                    label="GH skew-t PDF",
                )
                axes[i, j].set_title(f"r = {r}, nu = {nu}")
                axes[i, j].set_xlabel("x")
                axes[i, j].set_ylabel("Density")
                axes[i, j].grid()
                axes[i, j].legend()

                samples = GH_skew_student_t_rvs(nu, mu, delta, beta, size=100_000, random_state=123)

                quantiles = np.quantile(samples, [0.001, 0.999])

                print(f"r = {r}, nu = {nu}, 0.1% quantile: {quantiles[0]}, 99.9% quantile: {quantiles[1]}")
        plt.tight_layout()
        plt.show()


    
    if True:


        df = pd.read_csv("ghst_em_bootstrap_estimates.csv")
        

        df  = df[df["converged"] == True][["s", "r", "nu"]]



        # create a df of mean and var of s, r and nu
        summary_df = df.agg(["mean", "var"])

        print(summary_df)

        # r is assumed to be beta distributed. Use method of moments to estimate the parameters of the beta distribution
        r_mean = summary_df.loc["mean", "r"]
        r_var = summary_df.loc["var", "r"]

        concentration = r_mean * (1.0 - r_mean) / r_var - 1.0
        r_alpha = r_mean * concentration
        r_beta = (1.0 - r_mean) * concentration

        print(f"Estimated beta distribution parameters for r: alpha = {r_alpha}, beta = {r_beta}")

        
        x = np.linspace(np.finfo(float).eps, 1.0 - np.finfo(float).eps, 1000)
        plt.plot(x, _beta_dist.pdf(x, r_alpha, r_beta), label="Estimated beta PDF for r")
        plt.title("Estimated Beta Distribution for r")
        plt.show()


        nu_mean = summary_df.loc["mean", "nu"]
        nu_std = np.sqrt(summary_df.loc["var", "nu"])

        # Method of moments estimation for shifted exponential distribution parameters
        lambda_ = 1/nu_std
        nu_min = nu_mean - nu_std


        print(f"Estimated Exp distribution parameters for nu: lambda = {lambda_}, nu_min = {nu_min}")


        x = np.linspace(nu_min, nu_mean + 5*nu_std, 1000)
        plt.plot(x, lambda_ * np.exp(-lambda_ * (x - nu_min)), label="Estimated shifted Exp PDF for nu")
        plt.title("Estimated Shifted Exponential Distribution for nu")
        plt.show()


        s_mean = summary_df.loc["mean", "s"]
        s_var = summary_df.loc["var", "s"]
