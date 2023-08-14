from collections import Counter
data=open(r"textfile.txt")
c=data.read()
d=c.split()
counts=Counter(d)
print(counts)
