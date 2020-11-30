import torch
from torch.utils.data import TensorDataset

#returns the value after every kth period.
#For example, if period_id = 0, and list_of_ids = [1,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0,1,1,1,0], and k=2, it would return [9,17]
def indices_of_every_kth_period(list_of_ids, period_id, k):
    num_periods = 0
    every_kth_period = []
    for i in range(len(list_of_ids)):
        if list_of_ids[i] == period_id:
            num_periods += 1

            if num_periods == k:
                num_periods = 0
                every_kth_period.append(i+1)

    return every_kth_period



def split_list_on_indices(list_to_split, indices):
    #indices.append(len(arr))
    indices.insert(0, 0)

    split_list = [list_to_split[indices[i]:indices[i+1]] for i in range(len(indices)-1)]
    return split_list




def get_padded_size_n(list_of_subarrs, n):
    features = []
    labels = []
    for subarr in list_of_subarrs:
        for i in range(1, len(subarr)+1):
            padding = [0] * (n-i)
            start = max(i-n,0) #max(i-k, 0)
            end = i

            padded_subarr = padding + subarr[start:end]
            features.append(padded_subarr[0:-1])
            labels.append(padded_subarr[-1])


    return features, labels





def get_tensor_dataset(list_of_ids, period_id, n, k):
    period_indices = indices_of_every_kth_period(list_of_ids, period_id, k)
    split_list = split_list_on_indices(list_of_ids, period_indices)
    features, labels = get_padded_size_n(split_list, n)

    #return features, labels



    return TensorDataset(torch.tensor(features), torch.tensor(labels))
