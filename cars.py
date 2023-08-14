import numpy as np
fuel_eff=np.array([150,250,100,30,90,110])
avg=np.mean(fuel_eff)
improvement=((fuel_eff[1]-fuel_eff[0])/fuel_eff[1])*100
print("percentage improvement between two car models is",improvement)
print("average fuel efficiency: ",avg)
