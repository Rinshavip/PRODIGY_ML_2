import pandas as pnd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler  
from sklearn.cluster import KMeans

try:
    tdst = pnd.read_csv("Mall_Customers.csv")
except FileNotFoundError:
    print("No such file exist please put the Mall_Customers.csv File in the folder with main.py")

allspend = tdst[["Age","Annual Income (k$)","Spending Score (1-100)"]]

sr = StandardScaler()
st = sr.fit_transform(allspend)

clus = KMeans(n_clusters=5)
tl = clus.fit_predict(st)

tdst['clusters'] = tl
fig = plt.figure()
axs = plt.axes(projection='3d')

for i in range(5):
    cluster_data = tdst[tdst['clusters'] == i]
    axs.scatter(cluster_data['Age'], cluster_data['Annual Income (k$)'], cluster_data['Spending Score (1-100)'], label=f'Cluster {i}')

axs.set_xlabel('Age')
axs.set_ylabel('Annual Income (k$)')
axs.set_zlabel('Spending Score (1-100)')
axs.set_title('K-Means Clustering (Age, Income, Spending Score)')
axs.legend()
plt.show()
