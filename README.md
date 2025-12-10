# k-means
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Dataset
data = {
    "Institution": [
        "Baku State University",
        "Azerbaijan State University of Economics",
        "Azerbaijan State Oil and Industry University",
        "Azerbaijan Technical University",
        "Baku Engineering University",
        "Khazar University",
        "Azerbaijan University of Architecture and Construction",
        "Sumgait State University",
        "ADA University"
    ],
    "Academic Reputation": [22, 21.2, 13.1, 19.1, 24.9, 7.8, 25.7, 14.1, 8.4],
    "Employer Reputation": [20.1, 37.2, 19, 25.5, 37.5, 15.1, 20.7, 12.6, 22.4],
    "Faculty Student": [94, 36.7, 73.3, 82.2, 37.1, 60.6, 42.1, 97.9, 62.3],
    "Citations per Faculty": [1.9, 1.5, 1.5, 1.4, 1.5, 3.3, 1, 1.1, 1.3],
    "International Faculty": [13.4, 32, 46.8, 13.9, 21.3, 82.3, 9.9, 2.6, 20.2],
    "International Students": [7, 10.7, 9.7, 13.4, 8.7, 37.1, 14.5, 1.4, 7.7],
    "International Students Diversity": [11.2, 13.9, 13.9, 16.5, 12.3, 35.8, 17.1, 5.8, 11.9],
    "International Research Network": [22.1, 11.7, 8.5, 4.2, 3.8, 7.6, 1.3, 1.8, 5.2],
    "Employment Outcomes": [64.1, 51.5, 45, 11.6, 11.7, 11.9, 11.8, 12, 12],
    "Sustainability": [43.3, 39.2, 31, 27.9, 29.4, 27.5, 25.8, 23, 20]
}

df = pd.DataFrame(data)
X = df.iloc[:, 1:]

# Standartlaşdırma
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# K-means (K=3)
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(X_scaled)
df['Cluster'] = clusters

# PCA ilə 2D vizualizasiya
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(8,6))
for cluster in range(3):
    plt.scatter(
        X_pca[df['Cluster']==cluster, 0], 
        X_pca[df['Cluster']==cluster, 1], 
        label=f'Cluster {cluster}', 
        s=100
    )

# Universitet adlarını göstərmək
for i, txt in enumerate(df['Institution']):
    plt.annotate(txt, (X_pca[i,0]+0.05, X_pca[i,1]+0.05), fontsize=8)

plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('University Clusters (K-means)')
plt.legend()
plt.show()
