import pandas as pd
import numpy as np
data=pd.read_csv("student_score.csv")
avg_sc=np.mean(data,axis=0)
print(avg_sc)
high_sub=np.argmax(avg_sc)
subject=['math','science','english','history']
print(subject[high_sub],'=',max(avg_sc))

