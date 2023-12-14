import numpy as np
import pandas as pd
import math
import torch


x = np.linspace(0,1,256)
df_x=pd.DataFrame(x)
df_x.to_csv('../data/x.csv',index=False)

t = np.linspace(0,1,100)
df_t=pd.DataFrame(t)
df_t.to_csv('../data/t.csv',index=False)

u_real=[]
for one_x in (list(x)):
    for one_t in (list(t)):
        u_real.append(math.exp(-one_t) * math.sin(one_x * math.pi))

print(u_real)
u_real=np.array(u_real).reshape(256,100)

df_ureal=pd.DataFrame(u_real)
df_ureal.to_csv('../data/usol.csv',index=False)


