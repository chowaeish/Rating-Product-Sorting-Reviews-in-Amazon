import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
df=pd.read_csv("amazon_review.csv")
df.head(15)
# TASK-1
df["overall"].mean()
#overall mean=4.5875

df["reviewTime"].dtype
df["reviewTime"] = pd.to_datetime(df["reviewTime"])
df["reviewTime"].max()
current_date = pd.to_datetime("2014-12-08 0:0:0")
df["days_diff"] = (current_date - df["reviewTime"]).dt.days


df["days_qcut"] = pd.qcut(df["days_diff"], [0,0.25,0.50,0.75,1], ["A","B","C","D"])
df["days_qcut"].value_counts()
df.head()

df.loc[df["days_qcut"]=="A", "overall"].mean() * 28 / 100 + \
df.loc[df["days_qcut"] == "B", "overall"].mean() * 26 / 100 + \
df.loc[df["days_qcut"] == "C", "overall"].mean() * 24 / 100 + \
df.loc[df["days_qcut"] == "D", "overall"].mean() * 22 / 100
# new_mean = 4.5955

#Task-2
df["helpful_no"]=df["total_vote"]-df["helpful_yes"]
df.head()

df["score_up_down_diff"]=df["helpful_yes"]-df["helpful_no"]
df.sort_values(by="score_up_down_diff",ascending=False)



def wilson_lower_bound(up, down, confidence=0.95):
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.sort_values(by="wilson_lower_bound",ascending=False).head(20)


