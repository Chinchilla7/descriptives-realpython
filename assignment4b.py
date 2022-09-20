import researchpy as rp
import pandas as pd

#load in data
df = pd.read_csv('data/data.csv')
df.columns

#Prints out descriptive information for DataFrame.
rp.codebook(df)

#getting descriptive of single/continuous variables
rp.summary_cont(df[['Age', 'HR', 'sBP']])

#getting descriptive of categorical variables
rp.summary_cat(df[['Group', 'Smoke']])


#code in realpython-descriptive stats page:
import math
import statistics
import numpy as np
import scipy.stats
import pandas as pd
#create data
x = [8.0, 1, 2.5, 4, 28.0]
x_with_nan = [8.0, 1, 2.5, math.nan, 4, 28.0]
x

x_with_nan

math.isnan(np.nan), np.isnan(math.nan)

math.isnan(y_with_nan[3]), np.isnan(y_with_nan[3])

#create np.ndarray and pd.Series objects that correspond to x and x_with_nan:
y, y_with_nan = np.array(x), np.array(x_with_nan)
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
y
y_with_nan

z
z_with_nan

#calculate mean
mean_ = sum(x) / len(x)
mean_

mean_ = statistics.mean(x)
mean_

mean_ = statistics.fmean(x)
mean_

# if there are nan values among your data, 
# then statistics.mean() and statistics.fmean() will return nan as the output:
mean_ = statistics.mean(x_with_nan)
mean_

mean_ = statistics.fmean(x_with_nan)
mean_

#using numpy to get mean
mean_ = np.mean(y)
mean_

mean_ = y.mean()
mean_

np.mean(y_with_nan)

y_with_nan.mean()

#ignore nan values
np.nanmean(y_with_nan)

#pd.series with .mean()
mean_ = z.mean()
mean_

z_with_nan.mean()

#weigthed mean
0.2 * 2 + 0.5 * 4 + 0.3 * 8
#implement weighted mean
x = [8.0, 1, 2.5, 4, 28.0]
w = [0.1, 0.2, 0.3, 0.25, 0.15]
wmean = sum(w[i] * x[i] for i in range(len(x))) / sum(w)
wmean

wmean = sum(x_ * w_ for (x_, w_) in zip(x, w)) / sum(w)
wmean

#np.average
y, z, w = np.array(x), pd.Series(x), np.array(w)
wmean = np.average(y, weights=w)
wmean

wmean = np.average(z, weights=w)
wmean

(w * y).sum() / w.sum()

#nan values in dataset
w = np.array([0.1, 0.2, 0.3, 0.0, 0.2, 0.1])
(w * y_with_nan).sum() / w.sum()

np.average(y_with_nan, weights=w)

np.average(z_with_nan, weights=w)

#harmonic mean
hmean = len(x) / sum(1 / item for item in x)
hmean

hmean = statistics.harmonic_mean(x)
hmean

#nan, 0, negative number in dataset
statistics.harmonic_mean(x_with_nan)

statistics.harmonic_mean([1, 0, 2])

statistics.harmonic_mean([1, 2, -2])

#3rd way to calculate harmonic mean
scipy.stats.hmean(y)

scipy.stats.hmean(z)

#geometric mean
#pure python
gmean = 1
for item in x:
    gmean *= item

gmean **= 1 / len(x)
gmean
#converts all values to floating-point numbers and returns their geometric mean:
gmean = statistics.geometric_mean(x)
gmean
#data with nan values
gmean = statistics.geometric_mean(x_with_nan)
gmean
# another way to get geometric mean
scipy.stats.gmean(y)
scipy.stats.gmean(z)

#Median
#pure python
n = len(x)
if n % 2:
    median_ = sorted(x)[round(0.5*(n-1))]
else:
    x_ord, index = sorted(x), round(0.5 * n)
    median_ = 0.5 * (x_ord[index-1] + x_ord[index])

median_
#median with statistics.median
median_ = statistics.median(x)
median_

median_ = statistics.median(x[:-1])
median_

statistics.median_low(x[:-1])

statistics.median_high(x[:-1])

#work with nan values
statistics.median(x_with_nan)

statistics.median_low(x_with_nan)

statistics.median_high(x_with_nan)

#get the median with np.median()
median_ = np.median(y)
median_

median_ = np.median(y[:-1])
median_

#use nanmedian() to ignore all nan values
np.nanmedian(y_with_nan)

np.nanmedian(y_with_nan[:-1])

#pandas ignore nan values by default
z.median()

z_with_nan.median()

#Mode
#pure python
u = [2, 3, 2, 8, 12]
mode_ = max((u.count(item), item) for item in set(u))[1]
mode_
#get mode with statistics.mode() and statistics.multimode()
mode_ = statistics.mode(u)
mode_
mode_ = statistics.multimode(u)
mode_
#If there’s more than one modal value, 
# then mode() raises StatisticsError, 
# while multimode() returns the list with all modes
v = [12, 15, 12, 15, 21, 15, 12]
statistics.mode(v)  # Raises StatisticsError
statistics.multimode(v)

#nan values as regular values
statistics.mode([2, math.nan, 2])

statistics.multimode([2, math.nan, 2])

statistics.mode([2, math.nan, 0, math.nan, 5])

statistics.multimode([2, math.nan, 0, math.nan, 5])

#mode with scipy.stats.mode()
u, v = np.array(u), np.array(v)
mode_ = scipy.stats.mode(u)
mode_

mode_ = scipy.stats.mode(v)
mode_

#get mode and number of occurrence as NumPy arrays
mode_.mode

mode_.count

#pandas with multimodal values, ignores nan by default
u, v, w = pd.Series(u), pd.Series(v), pd.Series([2, 2, math.nan])
u.mode()
v.mode()
w.mode()

#variance
#pure python
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
var_
#statistics.variance()
var_ = statistics.variance(x)
var_
#nan values in data
statistics.variance(x_with_nan)
# numpy np.var() or var()
var_ = np.var(y, ddof=1)
var_

var_ = y.var(ddof=1)
var_

#1 degree of freedom and nan values
np.var(y_with_nan, ddof=1)

y_with_nan.var(ddof=1)

#skip nan values
np.nanvar(y_with_nan, ddof=1)

# pandas skip nan values by default
z.var(ddof=1)
z_with_nan.var(ddof=1)

# Standard deviation
#pure python
std_ = var_ ** 0.5
std_
#statistics.stdev()
std_ = statistics.stdev(x)
std_
#numpy with standard deviation
np.std(y, ddof=1)

y.std(ddof=1)

np.std(y_with_nan, ddof=1)

y_with_nan.std(ddof=1)

np.nanstd(y_with_nan, ddof=1)

#pandas with standard deviation
z.std(ddof=1)

z_with_nan.std(ddof=1)

# Skewness
#pure python
x = [8.0, 1, 2.5, 4, 28.0]
n = len(x)
mean_ = sum(x) / n
var_ = sum((item - mean_)**2 for item in x) / (n - 1)
std_ = var_ ** 0.5
skew_ = (sum((item - mean_)**3 for item in x)
         * n / ((n - 1) * (n - 2) * std_**3))
skew_

#skewness with scipy.stats.skew():
y, y_with_nan = np.array(x), np.array(x_with_nan)
scipy.stats.skew(y, bias=False)

scipy.stats.skew(y_with_nan, bias=False)

#pandas with skewness
z, z_with_nan = pd.Series(x), pd.Series(x_with_nan)
z.skew()

z_with_nan.skew()

# Percentiles
x = [-5.0, -1.1, 0.1, 2.0, 8.0, 12.8, 21.0, 25.8, 41.0]
statistics.quantiles(x, n=2)

statistics.quantiles(x, n=4, method='inclusive')

# numpy and percentiles
y = np.array(x)
np.percentile(y, 5)

np.percentile(y, 95)

np.percentile(y, [25, 50, 75])

np.median(y)

#numpy ignore nan values
y_with_nan = np.insert(y, 2, np.nan)
y_with_nan

np.nanpercentile(y_with_nan, [25, 50, 75])

#quantile() and nanquantile()
np.quantile(y, 0.05)

np.quantile(y, 0.95)

np.quantile(y, [0.25, 0.5, 0.75])

np.nanquantile(y_with_nan, [0.25, 0.5, 0.75])

#pandas and percentiles with quantile()
z, z_with_nan = pd.Series(y), pd.Series(y_with_nan)
z.quantile(0.05)

z.quantile(0.95)

z.quantile([0.25, 0.5, 0.75])

z_with_nan.quantile([0.25, 0.5, 0.75])

# Range
#np.ptp():
np.ptp(y)

np.ptp(z)

np.ptp(y_with_nan)

np.ptp(z_with_nan)

np.amax(y) - np.amin(y)

#calculate maxima and minima of sequences
np.nanmax(y_with_nan) - np.nanmin(y_with_nan)

y.max() - y.min()

z.max() - z.min()

z_with_nan.max() - z_with_nan.min()

#interquartile range
quartiles = np.quantile(y, [0.25, 0.75])
quartiles[1] - quartiles[0]

quartiles = z.quantile([0.25, 0.75])
quartiles[0.75] - quartiles[0.25]

# Summary of descriptive stats
result = scipy.stats.describe(y, ddof=1, bias=False)
result
#nobs: the number of observations or elements in your dataset
#minmax: the tuple with the minimum and maximum values of your dataset
#mean: the mean of your dataset
#variance: the variance of your dataset
#skewness: the skewness of your dataset
#kurtosis: the kurtosis of your dataset

result.nobs

result.minmax[0]  # Min

result.minmax[1]  # Max

result.mean

result.variance

result.skewness

result.kurtosis

# pandas description stats summary 
result = z.describe()
result
#
result['mean']

result['std']

result['min']

result['max']

result['25%']

result['50%']

result['75%']

# Measure of correlation bewtween pairs of data
#create lists
x = list(range(-10, 11))
y = [0, 2, 2, 2, 2, 3, 3, 6, 7, 4, 7, 6, 6, 9, 4, 5, 5, 10, 11, 12, 14]
x_, y_ = np.array(x), np.array(y)
x__, y__ = pd.Series(x_), pd.Series(y_)

#covariance
#pure python
n = len(x)
mean_x, mean_y = sum(x) / n, sum(y) / n
cov_xy = (sum((x[k] - mean_x) * (y[k] - mean_y) for k in range(n))
          / (n - 1))
cov_xy

#numpy covariance
cov_matrix = np.cov(x_, y_)
cov_matrix

x_.var(ddof=1)

y_.var(ddof=1)

cov_xy = cov_matrix[0, 1]
cov_xy

cov_xy = cov_matrix[1, 0]
cov_xy

#pandas covariance
cov_xy = x__.cov(y__)
cov_xy

cov_xy = y__.cov(x__)
cov_xy

# Correlation Coefficient
#pure python
var_x = sum((item - mean_x)**2 for item in x) / (n - 1)
var_y = sum((item - mean_y)**2 for item in y) / (n - 1)
std_x, std_y = var_x ** 0.5, var_y ** 0.5
r = cov_xy / (std_x * std_y)
r
#scipy.stats
r, p = scipy.stats.pearsonr(x_, y_)
r
p #p-value

#np.corrcoef()
corr_matrix = np.corrcoef(x_, y_)
corr_matrix

#
r = corr_matrix[0, 1]
r

r = corr_matrix[1, 0]
r

#scipy.stats.linregress()
scipy.stats.linregress(x_, y_)
#access particular values from the result of linregress(), including correlation coefficient
result = scipy.stats.linregress(x_, y_)
r = result.rvalue
r

#pandas with correlation coefficient
r = x__.corr(y__)
r

r = y__.corr(x__)
r

# Working With 2D Data
# creating a 2D NumPy array:
a = np.array([[1, 1, 1],
              [2, 3, 1],
              [4, 9, 2],
              [8, 27, 4],
              [16, 1, 1]])
a
#apply Python statistics functions
np.mean(a)
a.mean()
np.median(a)
a.var(ddof=1)
#axis=0 with np.mean()
np.mean(a, axis=0)
a.mean(axis=0)
#axis=1 to mean()
np.mean(a, axis=1)
a.mean(axis=1)
#parameter axis with other numpy functions
np.median(a, axis=0)
np.median(a, axis=1)
a.var(axis=0, ddof=1)
a.var(axis=1, ddof=1)

#SciPy statistics functions
scipy.stats.gmean(a)  # Default: axis=0
scipy.stats.gmean(a, axis=0)
#axis=1, get calculations across all columns
scipy.stats.gmean(a, axis=1)
#statistics for entire dataset, axis=None:
scipy.stats.gmean(a, axis=None)

#scipy.stats.describe()
scipy.stats.describe(a, axis=None, ddof=1, bias=False)
scipy.stats.describe(a, ddof=1, bias=False)  # Default: axis=0
scipy.stats.describe(a, axis=1, ddof=1, bias=False)
#axis=1, describe() returns summary for all rows
result = scipy.stats.describe(a, axis=1, ddof=1, bias=False)
result.mean

# Dataframes
# create dataframe
row_names = ['first', 'second', 'third', 'fourth', 'fifth']
col_names = ['A', 'B', 'C']
df = pd.DataFrame(a, index=row_names, columns=col_names)
df
#without arguments, DataFrame will return results for each column
df.mean()
df.var()
#results for each row, specify the parameter axis=1:
df.mean(axis=1)
df.var(axis=1)
#isolate each column of a DataFrame
df['A']
# apply methods
df['A'].mean()
df['A'].var()
#get all data from a DataFrame with .values or .to_numpy():
df.values
df.to_numpy()
#statistics summary for all columns
df.describe()
#specify value of optional parameter percentiles, access each item of the summary:
df.describe().at['mean', 'A']
df.describe().at['50%', 'B']

# Visualizing Data

import matplotlib.pyplot as plt
plt.style.use('ggplot')

#box plot
#create data
np.random.seed(seed=0)
x = np.random.randn(1000)
y = np.random.randn(100)
z = np.random.randn(10)
#box.plot()
fig, ax = plt.subplots()
ax.boxplot((x, y, z), vert=False, showmeans=True, meanline=True,
           labels=('x', 'y', 'z'), patch_artist=True,
           medianprops={'linewidth': 2, 'color': 'purple'},
           meanprops={'linewidth': 2, 'color': 'red'})
plt.show()

#histograms
#np.histogram()
hist, bin_edges = np.histogram(x, bins=10)
hist
bin_edges
#histogram() calculates, .hist() shows graphically:
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=False)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()
#get the histogram with the cumulative numbers of items 
# if you provide the argument cumulative=True to .hist():
fig, ax = plt.subplots()
ax.hist(x, bin_edges, cumulative=True)
ax.set_xlabel('x')
ax.set_ylabel('Frequency')
plt.show()

#pie charts
#data
x, y, z = 128, 256, 1024
#pie()
fig, ax = plt.subplots()
ax.pie((x, y, z), labels=('x', 'y', 'z'), autopct='%1.1f%%')
plt.show()

#bar charts
#data sets
x = np.arange(21)
y = np.random.randint(21, size=21)
err = np.random.randn(21)
#
fig, ax = plt.subplots()
ax.bar(x, y, yerr=err)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()


#X-Y plots
#2 data sets
x = np.arange(21)
y = 5 + 2 * x + 2 * np.random.randn(21)
slope, intercept, r, *__ = scipy.stats.linregress(x, y)
line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
#plot() to get x-y plot:
fig, ax = plt.subplots()
ax.plot(x, y, linewidth=0, marker='s', label='Data points')
ax.plot(x, intercept + slope * x, label=line)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.legend(facecolor='white')
plt.show()

#Heatmap for a covariance matrix with .imshow():

matrix = np.cov(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()

# Heatmap for the correlation coefficient matrix
matrix = np.corrcoef(x, y).round(decimals=2)
fig, ax = plt.subplots()
ax.imshow(matrix)
ax.grid(False)
ax.xaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.yaxis.set(ticks=(0, 1), ticklabels=('x', 'y'))
ax.set_ylim(1.5, -0.5)
for i in range(2):
    for j in range(2):
        ax.text(j, i, matrix[i, j], ha='center', va='center', color='w')
plt.show()