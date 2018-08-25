import pandas as pd
import data_frames

df = data_frames.t_shirts()
pd.get_dummies(df, drop_first = True)