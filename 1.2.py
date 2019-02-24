import os
import pandas as pd
import requests
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import statsmodels.api as sm


plt.style.use('ggplot')
# % matplotlib inline

PATH = r'iris/iris.data'
r = requests.get('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data')

with open(PATH, 'w') as f:
    f.write(r.text)

# os.chdir(PATH)
df = pd.read_csv(PATH, names=['sepal length', 'sepal width', 'petal length', 'petal width', 'class'])
df.head()
#    sepal length  sepal width     ...       petal width        class
# 0           5.1          3.5     ...               0.2  Iris-setosa
# 1           4.9          3.0     ...               0.2  Iris-setosa
# 2           4.7          3.2     ...               0.2  Iris-setosa
# 3           4.6          3.1     ...               0.2  Iris-setosa
# 4           5.0          3.6     ...               0.2  Iris-setosa
#
# [5 rows x 5 columns]

# 通过列名选择某一列数据
a = df['sepal length']
# 0      5.1
# 1      4.9
# 2      4.7
# 3      4.6
# 4      5.0

# 前4行 前2列
b = df.ix[:3, :2]
#    sepal length  sepal width
# 0           5.1          3.5
# 1           4.9          3.0
# 2           4.7          3.2
# 3           4.6          3.1

# 前4行 包含width的列
c = df.ix[:3, [x for x in df.columns if 'width' in x]]
#    sepal width  petal width
# 0          3.5          0.2
# 1          3.0          0.2
# 2          3.2          0.2
# 3          3.1          0.2

# 所有可用的唯一类
d = df['class'].unique()
# ['Iris-setosa' 'Iris-versicolor' 'Iris-virginica']

# class = Iris-virginica 的数据
e = df[df['class'] == 'Iris-virginica']
#      sepal length  sepal width       ...        petal width           class
# 100           6.3          3.3       ...                2.5  Iris-virginica
# 101           5.8          2.7       ...                1.9  Iris-virginica
# 102           7.1          3.0       ...                2.1  Iris-virginica
# 103           6.3          2.9       ...                1.8  Iris-virginica
# 104           6.5          3.0       ...                2.2  Iris-virginica
# 105           7.6          3.0       ...                2.1  Iris-virginica
# 106           4.9          2.5       ...                1.7  Iris-virginica
# 107           7.3          2.9       ...                1.8  Iris-virginica
# 108           6.7          2.5       ...                1.8  Iris-virginica
# 109           7.2          3.6       ...                2.5  Iris-virginica
# 110           6.5          3.2       ...                2.0  Iris-virginica
# ...
# [50 rows x 5 columns]

# 所有数据计数
f = df.count()
# sepal length    150
# sepal width     150
# petal length    150
# petal width     150
# class           150
# dtype: int64

# class = Iris-virginica 的数据计数
g = df[df['class'] == 'Iris-virginica'].count()
# sepal length    50
# sepal width     50
# petal length    50
# petal width     50
# class           50
# dtype: int64

# 重置索引
virginica = df[df['class'] == 'Iris-virginica'].reset_index(drop=True)
#     sepal length  sepal width       ...        petal width           class
# 0            6.3          3.3       ...                2.5  Iris-virginica
# 1            5.8          2.7       ...                1.9  Iris-virginica
# 2            7.1          3.0       ...                2.1  Iris-virginica
# 3            6.3          2.9       ...                1.8  Iris-virginica
# 4            6.5          3.0       ...                2.2  Iris-virginica
# 5            7.6          3.0       ...                2.1  Iris-virginica
# 6            4.9          2.5       ...                1.7  Iris-virginica
# 7            7.3          2.9       ...                1.8  Iris-virginica
# 8            6.7          2.5       ...                1.8  Iris-virginica
# 9            7.2          3.6       ...                2.5  Iris-virginica
# 10           6.5          3.2       ...                2.0  Iris-virginica
# ...
# [50 rows x 5 columns]

# class=Iris-virginica并且petal width>2.2的数据
aa = df[(df['class'] == 'Iris-virginica') & (df['petal width'] > 2.2)]
#      sepal length  sepal width       ...        petal width           class
# 100           6.3          3.3       ...                2.5  Iris-virginica
# 109           7.2          3.6       ...                2.5  Iris-virginica
# 114           5.8          2.8       ...                2.4  Iris-virginica
# 115           6.4          3.2       ...                2.3  Iris-virginica
# 118           7.7          2.6       ...                2.3  Iris-virginica
# 120           6.9          3.2       ...                2.3  Iris-virginica
# 135           7.7          3.0       ...                2.3  Iris-virginica
# 136           6.3          3.4       ...                2.4  Iris-virginica
# 140           6.7          3.1       ...                2.4  Iris-virginica
# 141           6.9          3.1       ...                2.3  Iris-virginica
# 143           6.8          3.2       ...                2.3  Iris-virginica
# 144           6.7          3.3       ...                2.5  Iris-virginica
# 145           6.7          3.0       ...                2.3  Iris-virginica
# 148           6.2          3.4       ...                2.3  Iris-virginica
#
# [14 rows x 5 columns]

# 快速的描述性统计：数量、平均值、最大值、最小值、标准差等
bb = df.describe()
#        sepal length  sepal width  petal length  petal width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.054000      3.758667     1.198667
# std        0.828066     0.433594      1.764420     0.763161
# min        4.300000     2.000000      1.000000     0.100000
# 25%        5.100000     2.800000      1.600000     0.300000
# 50%        5.800000     3.000000      4.350000     1.300000
# 75%        6.400000     3.300000      5.100000     1.800000
# max        7.900000     4.400000      6.900000     2.500000

# 自定义百分比
cc = df.describe(percentiles=[.20, .40, .80, .90, .95])
#        sepal length  sepal width  petal length  petal width
# count    150.000000   150.000000    150.000000   150.000000
# mean       5.843333     3.054000      3.758667     1.198667
# std        0.828066     0.433594      1.764420     0.763161
# min        4.300000     2.000000      1.000000     0.100000
# 20%        5.000000     2.700000      1.500000     0.200000
# 40%        5.600000     3.000000      3.900000     1.160000
# 50%        5.800000     3.000000      4.350000     1.300000
# 80%        6.520000     3.400000      5.320000     1.900000
# 90%        6.900000     3.610000      5.800000     2.200000
# 95%        7.255000     3.800000      6.100000     2.300000
# max        7.900000     4.400000      6.900000     2.500000

# 相关性
dd = df.corr()
#               sepal length  sepal width  petal length  petal width
# sepal length      1.000000    -0.109369      0.871754     0.817954
# sepal width      -0.109369     1.000000     -0.420516    -0.356544
# petal length      0.871754    -0.420516      1.000000     0.962757
# patal width       0.817954    -0.356544      0.962757     1.000000
ee = df.corr(method='spearman')

# # 创建 宽6 高4的插图
# [fig, ax] = plt.subplots(figsize=(6, 4))
# # 依照petal width数据绘制直方图 颜色黑色
# ax.hist(df['petal width'], color='black')
# # 设置 x y 的标签 字体大小12
# ax.set_ylabel('count', fontsize=12)
# ax.set_xlabel('width', fontsize=12)
# # 设置全图标题 y的参数调整为y轴方向相对于图片顶部位置 字号14
# plt.title('iris petal width', fontsize=14, y=1.01)
#
# # 创建 宽6 高4的插图
# fig1, ax1 = plt.subplots(2, 2, figsize=(6, 4))
# # 依照petal width数据绘制直方图 颜色黑色
# ax1[0][0].hist(df['petal width'], color='black')
# # 设置 x y 的标签 字体大小12
# ax1[0][0].set_ylabel('count', fontsize=12)
# ax1[0][0].set_xlabel('width', fontsize=12)
# # 设置全图标题 y的参数调整为y轴方向相对于图片顶部位置 字号14
# ax1[0][0].set_title('iris petal width', fontsize=14, y=1.01)
#
# # 依照petal length数据绘制直方图 颜色黑色
# ax1[0][1].hist(df['petal length'], color='black')
# # 设置 x y 的标签 字体大小12
# ax1[0][1].set_ylabel('count', fontsize=12)
# ax1[0][1].set_xlabel('width', fontsize=12)
# # 设置全图标题 y的参数调整为y轴方向相对于图片顶部位置 字号14
# ax1[0][1].set_title('iris petal length', fontsize=14, y=1.01)
#
# # 依照petal width数据绘制直方图 颜色黑色
# ax1[1][0].hist(df['sepal width'], color='black')
# # 设置 x y 的标签 字体大小12
# ax1[1][0].set_ylabel('count', fontsize=12)
# ax1[1][0].set_xlabel('width', fontsize=12)
# # 设置全图标题 y的参数调整为y轴方向相对于图片顶部位置 字号14
# ax1[1][0].set_title('iris sepal width', fontsize=14, y=1.01)
#
# # 依照petal length数据绘制直方图 颜色黑色
# ax1[1][1].hist(df['sepal length'], color='black')
# # 设置 x y 的标签 字体大小12
# ax1[1][1].set_ylabel('count', fontsize=12)
# ax1[1][1].set_xlabel('width', fontsize=12)
# # 设置全图标题 y的参数调整为y轴方向相对于图片顶部位置 字号14
# ax1[1][1].set_title('iris sepal length', fontsize=14, y=1.01)
#
# plt.tight_layout()

# 散点图
# fig2, ax2 = plt.subplots(figsize=(6, 6))
# ax2.scatter(df['petal width'], df['petal length'], color='green')
# ax2.set_xlabel('petal width')
# ax2.set_ylabel('petal length')
# ax2.set_title('petal scatterplot')

# 线图
# fig3, ax3 = plt.subplots(figsize=(6, 6))
# ax3.plot(df['petal length'], color='blue')
# ax2.set_xlabel('spicemen number')
# ax2.set_ylabel('petal length')
# ax2.set_title('petal length plot')

# 条形图
# fig4, ax4 = plt.subplots(figsize=(6, 6))
# bar_width = .8
# labels = [x for x in df.columns if 'length' in x or 'width' in x]
# ver_y = [df[df['class'] == 'Iris-versicolor'][x].mean() for x in labels]
# vir_y = [df[df['class'] == 'Iris-virginica'][x].mean() for x in labels]
# set_y = [df[df['class'] == 'Iris-setosa'][x].mean() for x in labels]
# x = np.arange(len(labels))
# ax4.bar(x, vir_y, bar_width, bottom=set_y, color='darkgrey')
# ax4.bar(x, set_y, bar_width, bottom=ver_y, color='white')
# ax4.bar(x, ver_y, bar_width, color='black')
# ax4.set_xticks(x + (bar_width/2))
# ax4.set_xticklabels(labels, rotation=-70, fontsize=12)
# ax4.set_title('mean feature measurment by class', y=1.01)
# ax4.legend(['virginica', 'setosa', 'versicolor'])

# seaborn库
# sns.pairplot(df, hue='class')

# 小提琴图
# fig, ax = plt.subplots(2, 2, figsize=(7, 7))
# sns.set(style='white', palette='muted')
# sns.violinplot(x=df['class'], y=df['sepal length'], ax=ax[0, 0])
# sns.violinplot(x=df['class'], y=df['sepal width'], ax=ax[0, 1])
# sns.violinplot(x=df['class'], y=df['petal length'], ax=ax[1, 0])
# sns.violinplot(x=df['class'], y=df['petal width'], ax=ax[1, 1])
# fig.suptitle('violin plots', fontsize=16, y=1.03)
# for i in ax.flat:
#     plt.setp(i.get_xticklabels(), rotation=-90)
# fig.tight_layout()
# plt.show()

# 处理数据 map apply applymap groupby
df['short class'] = df['class'].map({'Iris-setosa': 'SET', 'Iris-virginica': 'VIR', 'Iris-versicolor': 'VER'})
#      sepal length  sepal width  petal length  petal width class
# 0             5.1          3.5           1.4          0.2   SET
# 1             4.9          3.0           1.4          0.2   SET
# 2             4.7          3.2           1.3          0.2   SET
# 3             4.6          3.1           1.5          0.2   SET
# 4             5.0          3.6           1.4          0.2   SET
# 5             5.4          3.9           1.7          0.4   SET
# 6             4.6          3.4           1.4          0.3   SET
# ...
# [150 rows x 5 columns]

df['wide petal'] = df['petal width'].apply(lambda v: 1 if v >= 1.3 else 0)
#      sepal length  sepal width     ...      class  wide petal
# 0             5.1          3.5     ...        SET           0
# 1             4.9          3.0     ...        SET           0
# 2             4.7          3.2     ...        SET           0
# 3             4.6          3.1     ...        SET           0
# ...
# [150 rows x 6 columns]

df['petal area'] = df.apply(lambda r: r['petal length'] * r['petal width'], axis=1)
#      sepal length  sepal width     ...      wide petal  petal area
# 0             5.1          3.5     ...               0        0.28
# 1             4.9          3.0     ...               0        0.28
# 2             4.7          3.2     ...               0        0.26
# 3             4.6          3.1     ...               0        0.30
# 4             5.0          3.6     ...               0        0.28
# 5             5.4          3.9     ...               0        0.68
# ...
# [150 rows x 7 columns]

df.applymap(lambda m: np.log(m) if isinstance(m, float) else m)

# 按照class 进行划分，并且提供每个特征的均值
df.groupby('short class').mean()
#        sepal length  sepal width     ...      wide petal  petal area
# class                                ...
# SET           5.006        3.418     ...             0.0      0.3628
# VER           5.936        2.770     ...             0.7      5.7204
# VIR           6.588        2.974     ...             1.0     11.2962
#
# [3 rows x 6 columns]

df.groupby('short class').describe()
#       petal area                                  ...  wide petal
#            count     mean       std   min     25% ...         min  25%  50%  75%  max
# class                                             ...
# SET         50.0   0.3628  0.183248  0.11  0.2650 ...         0.0  0.0  0.0  0.0  0.0
# VER         50.0   5.7204  1.368403  3.30  4.8600 ...         0.0  0.0  1.0  1.0  1.0
# VIR         50.0  11.2962  2.157412  7.50  9.7175 ...         1.0  1.0  1.0  1.0  1.0
#
# [3 rows x 48 columns]

df.groupby('petal width')['short class'].unique().to_frame()
#                   class
# petal width
# 0.1               [SET]
# 0.2               [SET]
# 0.3               [SET]
# 0.4               [SET]
# 0.5               [SET]
# 0.6               [SET]
# 1.0               [VER]
# 1.1               [VER]
# 1.2               [VER]
# 1.3               [VER]
# 1.4          [VER, VIR]
# 1.5          [VER, VIR]
# 1.6          [VER, VIR]
# 1.7          [VER, VIR]
# 1.8          [VER, VIR]
# 1.9               [VIR]
# 2.0               [VIR]
# 2.1               [VIR]
# 2.2               [VIR]
# 2.3               [VIR]
# 2.4               [VIR]
# 2.5               [VIR]


df.groupby('class')['petal width'].agg({'delta': lambda x: x.max() - x.min(), 'max': np.max, 'min': np.min})
#                  delta  max  min
# class
# Iris-setosa        0.5  0.6  0.1
# Iris-versicolor    0.8  1.8  1.0
# Iris-virginica     1.1  2.5  1.4

# 建模和评估 statsmodels scikit-learn
y = df['sepal length'][:50]
x = df['sepal width'][:50]
X = sm.add_constant(x)
results = sm.OLS(y, X).fit()
print(results.summary())
#                            OLS Regression Results
# ==============================================================================
# Dep. Variable:           sepal length   R-squared:                       0.558
# Model:                            OLS   Adj. R-squared:                  0.548
# Method:                 Least Squares   F-statistic:                     60.52
# Date:                Thu, 15 Nov 2018   Prob (F-statistic):           4.75e-10
# Time:                        14:35:06   Log-Likelihood:                 2.0879
# No. Observations:                  50   AIC:                           -0.1759
# Df Residuals:                      48   BIC:                             3.648
# Df Model:                           1
# Covariance Type:            nonrobust
# ===============================================================================
#                   coef    std err          t      P>|t|      [0.025      0.975]
# -------------------------------------------------------------------------------
# const           2.6447      0.305      8.660      0.000       2.031       3.259
# sepal width     0.6909      0.089      7.779      0.000       0.512       0.869
# ==============================================================================
# Omnibus:                        0.252   Durbin-Watson:                   2.517
# Prob(Omnibus):                  0.882   Jarque-Bera (JB):                0.436
# Skew:                          -0.110   Prob(JB):                        0.804
# Kurtosis:                       2.599   Cond. No.                         34.0
# ==============================================================================
#
# Warnings:
# [1] Standard Errors assume that the covariance matrix of the errors is correctly specified.

fig, ax = plt.subplots(figsize=(7, 7))
ax.plot(x, results.fittedvalues, label='regression line')
ax.scatter(x, y, label='data point', color='r')
ax.set_ylabel('sepal length')
ax.set_xlabel('sepal width')
ax.set_title('setosa sepal width vs sepal length', fontsize=14, y=1.02)
ax.legend(loc=2)
plt.show()