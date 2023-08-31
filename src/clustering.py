from sklearn.cluster import KMeans

def perform_clustering(data):
    kmeans = KMeans(n_clusters=6, n_init=100, random_state=0).fit(data)
    data['Cluster'] = kmeans.labels_
    return data