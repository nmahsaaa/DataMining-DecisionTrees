#!/usr/bin/env python
# coding: utf-8

# In[32]:


import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.model_selection import KFold


# In[33]:


dataset = pd.read_csv("C:/Users/Mahsa/Desktop/tst/expanded.txt",
                      names=['class','cap-shape','cap-surface','cap-color','bruises?','odor',
                                                   'gill-attachment','gill-spacing','gill-size','gill-color','stalk-shape',
                                                  'stalk-root','stalk-surface-above-ring','stalk-surface-below-ring','stalk-color-above-ring',
                                                     'stalk-color-below-ring','veil-type','veil-color','ring-number','ring-type',
                                                     'spore-print-color','population','habitat',])


# In[34]:


def entropy(target_col):
    elements,counts = np.unique(target_col,return_counts = True)
    entropy = np.sum([(-counts[i]/np.sum(counts))*np.log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
    return entropy 


# In[35]:


def InfoGain(data,split_attribute_name,target_name="class"):
    
    total_entropy = entropy(data[target_name])
    vals,counts= np.unique(data[split_attribute_name],return_counts=True)
    Weighted_Entropy = np.sum([(counts[i]/np.sum(counts))*entropy(data.where(data[split_attribute_name]==vals[i]).dropna()[target_name]) for i in range(len(vals))])
    Information_Gain = total_entropy - Weighted_Entropy
    return Information_Gain


# In[36]:


def ID3(data,originaldata,features,target_attribute_name="class",parent_node_class = None):

    if len(np.unique(data[target_attribute_name])) <= 1:
        return np.unique(data[target_attribute_name])[0]
    
    #If the dataset is empty, return the mode target feature value in the original dataset
    elif len(data) ==0:
        return np.unique(originaldata[target_attribute_name])[np.argmax(np.unique(originaldata[target_attribute_name],return_counts=True)[1])]
    
    #If the feature space is empty, return the mode target feature value of the direct parent node
    elif len(features) ==0:
        return parent_node_class
    
    #If none of the above holds true, grow the tree!
    else:
        #Set the default value for this node
        parent_node_class = np.unique(data[target_attribute_name])[np.argmax(np.unique(data[target_attribute_name],return_counts=True)[1])]
        
        #Select the feature which best splits the dataset
        item_values = [InfoGain(data,feature,target_attribute_name) for feature in features]
        best_feature_index = np.argmax(item_values)
        best_feature = features[best_feature_index]
        
        #Create the tree structure. The root gets the name of the feature (best_feature) with the maximum information
        tree = {best_feature:{}}
        
        #Remove the feature with the best inforamtion gain from the feature space
        features = [i for i in features if i != best_feature]
        
        #Grow a branch under the root node for each possible value of the root node feature
        for value in np.unique(data[best_feature]):
            value = value
            #Split the dataset along the value of the feature with the largest information gain and therwith create sub_datasets
            sub_data = data.where(data[best_feature] == value).dropna()
            
            #Call the ID3 algorithm for each of those sub_datasets with the new parameters
            subtree = ID3(sub_data,dataset,features,target_attribute_name,parent_node_class)
            
            #Add the sub tree, grown from the sub_dataset to the tree under the root node
            tree[best_feature][value] = subtree
            
        return(tree)


# In[37]:


def predict(query,tree,default = 'p'):
   
    for key in list(query.keys()):
        if key in list(tree.keys()):
            try:
                result = tree[key][query[key]] 
            except:
                return default
            
            result = tree[key][query[key]]
            
            if isinstance(result,dict):
                return predict(query,result)
            else:
                return result


# In[38]:


def train_test_split(dataset):
    X = dataset.values[:, 1:23]
    kf = KFold(n_splits=6)
    kf.get_n_splits(X)
    for train_index, test_index in kf.split(X):
        print("TRAIN:", train_index, "TEST:", test_index)
        training_data, testing_data = X[train_index], X[test_index]
    return training_data, testing_data


# In[39]:


def test(data,tree):
    #Create new query instances and convert it to a dictionary
    queries = data.iloc[:,:-1].to_dict(orient = "records")
    
    #Create a empty DataFrame in which columns the prediction of the tree are stored
    predicted = pd.DataFrame(columns=["predicted"]) 
    
    #Calculate the prediction accuracy
    for i in range(len(data)):
        predicted.loc[i,"predicted"] = predict(queries[i],tree,'p') 
    print('The prediction accuracy is: ',(np.sum(predicted["predicted"] == data["class"])/len(data))*100,'%')


# In[40]:


train_test_split(dataset)
tree = ID3(training_data,training_data,training_data.columns[-22:])
pprint(tree)
test(testing_data,tree)


# In[ ]:




