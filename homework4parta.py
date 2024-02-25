# Chase Whitfield
# MAE 3403 Spring 2024
# Homework 4 Problem A
# 02/26/2024

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def plot_normal_distribution(mu, sigma, x_values, pdf=None, cdf=None, shaded_area=None, title='', prob_text='',
                             is_pdf=True):
    """
    Plot the Probability Density Function (PDF) and Cumulative Distribution Function (CDF)
    for a normal distribution with given mean (mu) and standard deviation (sigma).

    Parameters:
        mu (float): Mean of the normal distribution.
        sigma (float): Standard deviation of the normal distribution.
        x_values (array_like): Values of x to plot against.
        pdf (array_like, optional): Probability Density Function values corresponding to x_values.
        cdf (array_like, optional): Cumulative Distribution Function values corresponding to x_values.
        shaded_area (tuple, optional): Tuple specifying the range to be shaded (x_start, x_end).
        title (str, optional): Title for the plot.
        prob_text (str, optional): Text to display the probability equation on the plot.
        is_pdf (bool, optional): Indicates whether the plot is for PDF (True) or CDF (False).
    """
    plt.figure(figsize=(10, 6))

    if is_pdf:
        plt.plot(x_values, pdf, label='PDF')
        if shaded_area:
            shaded_mask = (x_values >= shaded_area[0]) & (x_values <= shaded_area[1])
            plt.fill_between(x_values, 0, pdf, where=shaded_mask, alpha=0.2)
        plt.title('Probability Density Function (PDF)')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.text(0.05, 0.9, prob_text, transform=plt.gca().transAxes)
        plt.legend()
    else:
        if x_values is not None and cdf is not None:
            plt.plot(x_values, cdf, label='CDF')
            if shaded_area:
                plt.scatter(shaded_area[1], stats.norm.cdf(shaded_area[1], mu, sigma), color='red',
                            label=f'({shaded_area[1]:.2f}, {stats.norm.cdf(shaded_area[1], mu, sigma):.2f})', zorder=5)
                plt.text(shaded_area[1], stats.norm.cdf(shaded_area[1], mu, sigma), f'P= {stats.norm.cdf(shaded_area[1], mu, sigma):.2f}', fontsize=10, ha='right')
            plt.title('Cumulative Distribution Function (CDF)')
            plt.xlabel('x')
            plt.ylabel('Cumulative Probability')
            plt.text(0.05, 0.9, prob_text, transform=plt.gca().transAxes)
            plt.legend()
        else:
            print("Error: x_values or cdf is None")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

def main():
    """
    Main function to execute the code.
    """
    # Define the parameters for the normal distributions
    mu1, sigma1 = 0, 1
    mu2, sigma2 = 175, 3

    # Generate x values for plotting
    x_values = np.linspace(-10, 200, 1000)

    # Calculate the probability density function (PDF) for the first normal distribution
    pdf1 = stats.norm.pdf(x_values, mu1, sigma1)

    # Calculate the cumulative distribution function (CDF) for the first normal distribution
    cdf1 = stats.norm.cdf(x_values, mu1, sigma1)

    # Calculate the probability x < -0.5 given the first normal distribution
    prob_x_less_than_neg_05 = stats.norm.cdf(-0.5, mu1, sigma1)
    prob_text1 = f'P(x<-0.50|N(0.00, 1.00)) = {prob_x_less_than_neg_05:.2f}'

    # Calculate the probability x > 181 given the second normal distribution
    prob_x_gt_181 = 1 - stats.norm.cdf(181, mu2, sigma2)
    prob_text2 = f'P(x>181.00|N(175.00, 3.00)) = {prob_x_gt_181:.2f}'

    # Plotting the first normal distribution PDF
    plot_normal_distribution(mu1, sigma1, x_values, pdf1, title='Normal Distribution N(0, 1)', shaded_area=(-10, -0.5), prob_text=prob_text1)

    # Calculate the probability density function (PDF) for the second normal distribution
    pdf2 = stats.norm.pdf(x_values, mu2, sigma2)

    # Calculate the cumulative distribution function (CDF) for the second normal distribution
    cdf2 = stats.norm.cdf(x_values, mu2, sigma2)

    # Calculate the probability x < 1 given the first normal distribution
    prob_x_less_than_1 = stats.norm.cdf(1, mu1, sigma1)
    prob_text3 = f'P(x<1.00|N(0.00, 1.00)) = {prob_x_less_than_1:.2f}'

    # Calculate the probability x > μ + 2σ given the second normal distribution
    prob_x_gt_mu_plus_2sigma = 1 - stats.norm.cdf(mu2 + 2 * sigma2, mu2, sigma2)
    prob_text4 = f'P(x>{mu2 + 2 * sigma2:.2f}|N({mu2:.2f}, {sigma2:.2f})) = {prob_x_gt_mu_plus_2sigma:.2f}'

    # Plotting the second normal distribution PDF
    plot_normal_distribution(mu2, sigma2, x_values, pdf2, title='Normal Distribution N(175, 3)', shaded_area=(181, 200), prob_text=prob_text2)

    # Create new figures for plotting CDF
    plot_normal_distribution(mu1, sigma1, x_values, None, cdf1, title='Normal Distribution N(0, 1)', prob_text=prob_text3, is_pdf=False, shaded_area=(None, 1))
    plot_normal_distribution(mu2, sigma2, x_values, None, cdf2, title='Normal Distribution N(175, 3)', prob_text=prob_text4, is_pdf=False, shaded_area=(None, mu2 + 2 * sigma2))

if __name__ == "__main__":
    main()

# Used ChatGPT to help create the graphs
# Used ChatGPT to help create the main function 
