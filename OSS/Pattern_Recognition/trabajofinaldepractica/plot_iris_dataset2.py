import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
df = pd.DataFrame(np.random.randn(1000, 4 ), columns=['a', 'b', 'c', 'd'])
df["a_cat"] = pd.cut(df.a, bins=np.linspace(-3.5, 3.5, 8))
g = sns.pairplot(df, hue="a_cat", hue_order=df.a_cat.cat.categories, palette="YlGnBu")
g.savefig("pairplot.png")
