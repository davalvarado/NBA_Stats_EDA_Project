import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm, chisquare

# Load Data
data = pd.read_csv('seasons_stats.csv', encoding='ISO-8859-1')

# Define the variables for analysis
variables = ['PPG', 'PER', 'AST%', 'TS%', 'USG%']

# Create new feature PPG (Points Per Game)
data['PPG'] = data['PTS'] / data['G']

# Filter out NaN values from selected variables
data_filtered = data[variables].dropna()

# Plot histograms and calculate descriptive statistics
for var in variables:
    plt.figure(figsize=(8, 5))
    sns.histplot(data_filtered[var], kde=True, bins=30, color='blue', alpha=0.7)
    plt.title(f'Histogram of {var}')
    plt.xlabel(var)
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

    # Descriptive Statistics for each variable:
    mean_val = data_filtered[var].mean()
    median_val = data_filtered[var].median()
    mode_val = data_filtered[var].mode().iloc[0]
    std_dev = data_filtered[var].std()
    min_val = data_filtered[var].min()
    max_val = data_filtered[var].max()

    print(f"Variable: {var}")
    print(f"Mean: {mean_val:.2f}")
    print(f"Median: {median_val:.2f}")
    print(f"Mode: {mode_val:.2f}")
    print(f"Standard Deviation (Spread): {std_dev:.2f}")
    print(f"Min: {min_val}, Max: {max_val}")
    print(f"Tails (5th & 95th percentiles): {np.percentile(data_filtered[var], [5, 95])}")
    print("-" * 50)

# PMF for PPG grouped by Age:
pmf_data = data[['PPG', 'Age']].dropna()  # Check that PPG and Age are non-null

young_players = pmf_data[pmf_data['Age'] <= 25]['PPG']  # Players 25 and younger
older_players = pmf_data[pmf_data['Age'] > 25]['PPG']  # Players older than 25

# Calc PMF:
pmf_young = young_players.value_counts(normalize=True).sort_index()
pmf_older = older_players.value_counts(normalize=True).sort_index()

# Display PMF:
plt.figure(figsize=(10, 6))
plt.bar(pmf_young.index, pmf_young.values, alpha=0.7, label='Age ≤ 25', color='blue')
plt.bar(pmf_older.index, pmf_older.values, alpha=0.7, label='Age > 25', color='red')
plt.title('PMF of PPG: Young Players vs Older Players')
plt.xlabel('Points Per Game (PPG)')
plt.ylabel('Probability')
plt.legend()
plt.grid(True)
plt.show()


# Calculate CDF:
sorted_ppg = np.sort(data_filtered['PPG'].dropna())
cdf = np.arange(1, len(sorted_ppg) + 1) / len(sorted_ppg)

# Display CDF
plt.figure(figsize=(10, 6))
plt.plot(sorted_ppg, cdf, marker='.', linestyle='none', color='green')
plt.title('CDF of Points Per Game (PPG)')
plt.xlabel('Points Per Game (PPG)')
plt.ylabel('Cumulative Probability')
plt.grid(True)
plt.show()

# Interpretation of CDF:
median_ppg = np.median(sorted_ppg)
percentile_90 = np.percentile(sorted_ppg, 90)
print(f"The median PPG is {median_ppg:.2f}, indicating that half of the players score less than this value.")
print(f"The 90th percentile of PPG is {percentile_90:.2f}, meaning that only 10% of players score above this value.")


# Fit and Plot Analytical Distribution - Normal Distribution:
mu, sigma = norm.fit(data_filtered['PPG'])  # Fit the PPG data to a normal distribution

# Generate a range of values for the fitted distribution
x = np.linspace(min(data_filtered['PPG']), max(data_filtered['PPG']), 100)
pdf_fitted = norm.pdf(x, mu, sigma)

# Plot histogram data with fitted normal distribution:
plt.figure(figsize=(10, 6))
sns.histplot(data_filtered['PPG'], bins=30, kde=False, color='blue', label='PPG Data', stat='density', alpha=0.7)
plt.plot(x, pdf_fitted, 'r-', label=f'Normal Fit (μ={mu:.2f}, σ={sigma:.2f})', linewidth=2)

# Display Plot
plt.title('PPG Distribution with Fitted Normal Distribution')
plt.xlabel('Points Per Game (PPG)')
plt.ylabel('Density')
plt.legend()
plt.grid(True)
plt.show()

# Create Scatter Plots - PPG vs. PER:
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_filtered['PER'], y=data_filtered['PPG'], color='blue', alpha=0.7)
plt.title('Scatter Plot: PPG vs. PER')
plt.xlabel('Player Efficiency Rating (PER)')
plt.ylabel('Points Per Game (PPG)')
plt.grid(True)
plt.show()

# Covariance and Pearson's Correlation - PPG vs. PER:
cov_per_pgg = np.cov(data_filtered['PER'], data_filtered['PPG'])[0, 1]
pearson_per_ppg = np.corrcoef(data_filtered['PER'], data_filtered['PPG'])[0, 1]
print("-" * 50)
print(f"Covariance (PPG, PER): {cov_per_pgg:.2f}")
print(f"Pearson's Correlation (PPG, PER): {pearson_per_ppg:.2f}")

# Scatter Plot - PPG vs. USG%:
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_filtered['USG%'], y=data_filtered['PPG'], color='green', alpha=0.7)
plt.title('Scatter Plot of PPG vs. USG%')
plt.xlabel('Usage Percentage (USG%)')
plt.ylabel('Points Per Game (PPG)')
plt.grid(True)
plt.show()

# Covariance and Pearson Correlation - PPG vs. USG%:
cov_usg_ppg = np.cov(data_filtered['USG%'], data_filtered['PPG'])[0, 1]
pearson_usg_ppg = np.corrcoef(data_filtered['USG%'], data_filtered['PPG'])[0, 1]
print("-" * 50)
print(f"Covariance (PPG, USG%): {cov_usg_ppg:.2f}")
print(f"Pearson Correlation (PPG, USG%): {pearson_usg_ppg:.2f}")

# Calculate Chi-Squared Test:
# Bin the data for chi-square test
observed_freq, bin_edges = np.histogram(data_filtered['PPG'], bins=10, density=False)

# Expected frequencies from the fitted normal distribution
expected_freq = len(data_filtered['PPG']) * np.diff(norm.cdf(bin_edges, loc=mu, scale=sigma))

# Scale expected frequencies to match the total of observed frequencies
expected_freq = expected_freq * (observed_freq.sum() / expected_freq.sum())

# Perform chi-square test
chi2_stat, p_value = chisquare(f_obs=observed_freq, f_exp=expected_freq)

# Print Results/Interpretation:
print("-" * 50)
print(f"Chi-Square Test Statistic: {chi2_stat:.2f}")
print(f"P-Value: {p_value}")

if p_value < 0.05:
    print("Result: Reject the null hypothesis. The data does not follow a normal distribution.")
else:
    print("Result: Fail to reject the null hypothesis. The data follows a normal distribution.")

# Regression Analysis:
""" 
Define dependent and explanatory variables:
dependent_var = 'PPG' --> outcome trying to predict
explanatory_var = 'USG%' --> measures the % of a team's plays that a player is involved in
"""

regression_data = data_filtered[['PPG', 'USG%']].dropna()

# Fit linear regression model
model = smf.ols(formula='PPG ~ Q("USG%")', data=regression_data)
results = model.fit()

print(results.summary())

# Plot Regression
plt.figure(figsize=(10, 6))
sns.scatterplot(x=regression_data['USG%'], y=regression_data['PPG'],
                color='blue', alpha=0.7, label='Data Points')
sns.lineplot(x=regression_data['USG%'], y=results.fittedvalues,
             color='red', label='Regression Line')

plt.title('Regression Analysis: PPG vs USG%')
plt.xlabel('USG%')
plt.ylabel('PPG')
plt.grid(True)
plt.legend()
plt.show()



