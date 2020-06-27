import matplotlib.pyplot as plt
import random
from sklearn.cluster import KMeans
import torch

class extrautils:
    def __init__():
        super().__init__()
        
    
        
    def kmeans_wcss(X, seed_range, init, max_iter, n_init, random_state):
        wcss = []
        
        for i in range(1, seed_range + 1):
            kmeans = KMeans(n_clusters=i, init=init, max_iter=max_iter, n_init=n_init, random_state=random_state)
            kmeans.fit(X)
            wcss.append(kmeans.inertia_)
		
        return wcss
        
        
		
    def read_class (root_path, filename):
        classes = []
        url = root_path + filename
        
        with open (url) as file:
           classes = file.read().strip().split('\n')
		   
        return classes
		
    def read_class_desc (root_path, filename):
        classes_desc = {}
        url = root_path + filename

        file = open(url, "r")

        for line in file:
            line_desc = line.split("\t")
            classes_desc[line_desc[0]] = line_desc[1]

        file.close()

        return classes_desc

    def get_train_data (root_path, classes):
        train_data = []
        train_labels = []

        url = root_path + 'train'

        for class_name in classes:
            for i in range (500):
                train_data.append(plt.imread( (url + "/{}/images/{}_{}.JPEG".format(class_name, class_name, str(i))), 'RGB'))
                train_labels.append(class_name)
                
        return train_data, train_labels
        
        
    def get_val_data (root_path):
        val_data = []
        val_labels = []

        url = root_path + 'val'

        file = open(url + "/val_annotations.txt", "r")

        for line in file:
            line_desc = line.strip().split("\t")
            val_data.append (plt.imread(url+"/images/{}".format(line_desc[0]), 'RGB'))
            val_labels.append (line_desc[1])

        file.close()
                
        return val_data, val_labels
        
    def concatenate (data1, data2):
        return data1 + data2
        
    def data_shuffle_and_split(data, data_labels, split_perc):
        i = [j for j in range (len(data))]
        random.shuffle(i)
        
        split_size = slice(0,int(split_perc*len(data)))
        split_end = slice(int(split_perc*len(data)), len(data))

        train_data = [data[j] for j in i[split_size]]
        train_labels = [data_labels[j] for j in i[split_size]]

        val_data = [data[j] for j in i[split_end]]
        val_labels = [data_labels[j] for j in i[split_end]]
        
        return train_data, train_labels, val_data, val_labels
        
    def dataset_shuffle_and_split(dataset, split_perc):
        return torch.utils.data.random_split(dataset, [int(split_perc*(len(dataset))), int((1-split_perc)*(len(dataset)))])
