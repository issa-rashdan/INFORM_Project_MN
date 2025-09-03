import numpy as np

def db(data, epsilon=1e-10):
    return 10 * np.log10(data + epsilon)

def clip(data, clip_min=-100, clip_max=0):
    return np.clip(data, clip_min, clip_max)

def standardize_min_max(data, min_val=-100, max_val=0):
    return (data - min_val) / (max_val - min_val)


def standardize_log_normal(batch):
    for f in range(batch.shape[1]):
        batch[:, f, :, :] -= np.nanmin(batch[:, f, :, :])
        batch[:, f, :, :] = np.log(batch[:, f, :, :] + 1e-10)
        batch[:, f, :, :] -= np.mean(batch[:, f, :, :])
        batch[:, f, :, :] *= 1 / (np.std(batch[:, f, :, :]) + 1e-10)
    return batch

def sigmoid_log_visualize_echogram(data, frequencies):
    if len(data.shape) != 3:
        print('sigmoid_log_visualize_echogram function requires that the number of input dimensions is 3. ', len(data.shape), ' were given.')
        return None
    elif data.shape[2] != 4:
        print('sigmoid_log_visualize_echogram function requires 4 input frequency channels. ', data.shape[2], ' were given.')
        if frequencies != [18, 38, 120, 200]:
            print('visualize function must use input parameter frequencies == [18, 38, 120, 200]. ', frequencies, ' were given.')
        return None
    else:
        eps = 1e-25
        k = np.array(preprocess_params()['k']).reshape(1, 1, 4)
        a = np.array(preprocess_params()['a']).reshape(1, 1, 4)
        data += eps
        data = 1 / (1 + k * np.power(data, a))
        return data
    

# Binary segmentation (background, sandeel)
'''
def preprocess_params():
    k = [2.019e-04, 2.506e-04, 1.781e-04, 1.292e-04]
    a = [-0.779, -0.812, -0.681, -0.578]
    params = {'k': k, 'a': a}
    return params
'''

# Segmentation with three classes (background, sandeel, other fish)
def preprocess_params():
    k = [4.153e-04, 3.464e-04, 2.692e-04, 2.613e-04]
    a = [-0.669, -1.115, -0.693, -0.593]

    params = {'k': k, 'a': a}
    return params