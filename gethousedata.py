import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt


pd.set_option("display.max_columns", 30)
pd.set_option("display.max_colwidth", 100)
pd.set_option("display.precision", 3)

csv_path = "magic.csv"
df = pd.read_csv(csv_path)
# 输出标题行
# print(df.columns)

# .T转置数据框并垂直的显示
# print(df.T)

# multiple units
mu = df[df['listingtype_value'].str.contains('Apartments For')]
# single units
su = df[df['listingtype_value'].str.contains('Apartment For')]

# print(len(mu), len(su))

# print(su['propertyinfo_value'])

m = len(su[~(su['propertyinfo_value'].str.contains('Studio')|su['propertyinfo_value'].str.contains('bd'))])
# print(m)
n = len(su[~(su['propertyinfo_value'].str.contains('ba'))])
# print(n)

# 选择拥有浴室的房源
no_baths = su[~(su['propertyinfo_value'].str.contains('ba'))]

# 再排除那些确实了浴室信息的房源
sucln = su[~su.index.isin(no_baths.index)]


def parse_info(row):
    if not 'sqft' in row:
        bd, ba = row.split('•')[:2]
        sqft = np.nan
    else:
        bd, ba, sqft = row.split('•')[:3]
    return pd.Series({'Beds': bd, 'Baths': ba, 'Sqft': sqft})

attr = sucln['propertyinfo_value'].apply(parse_info)
# print(attr)

# 在取值中将字符串删除
attr_cln = attr.applymap(lambda x: x.strip().split(' ')[0] if isinstance(x, str) else np.nan)
# print(attr_cln)

sujnd = sucln.join(attr_cln)
# print(sujnd.T)


# parse out zip, floor
def parse_addy(r):
    so_zip = re.search(', NY(\d+)', r)
    so_flr = re.search('(?:APT|#)\s+(\d+)[A-Z]+,', r)
    if so_zip:
        zipc = so_zip.group(1)
    else:
        zipc = np.nan
    if so_flr:
        flr = so_flr.group(1)
    else:
        flr = np.nan
    return pd.Series({'Zip': zipc, 'Floor': flr})

flrzip = sujnd['routable_link/_text'].apply(parse_addy)
suf = sujnd.join(flrzip)
# print(suf.T)

# 将数据减少为所需要的那些列
sudf = suf[['propertyinfo_value', 'Beds', 'Baths', 'Sqft', 'Floor', 'Zip']]
# 清理奇怪的列名，并重置索引
sudf.rename(columns={'pricelarge_value_prices': 'Rent'}, inplace=True)
sudf.reset_index(drop=True, inplace=True)
print(sudf)