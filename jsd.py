import numpy as np
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon

data = np.load('/home/lasii/Research/dataset/alz/processed/data.npy')
label = np.load('/home/lasii/Research/dataset/alz/processed/labels.npy')

data_aug = np.load('/home/lasii/Research/dataset/alz/processed/aug_data.npy')
label_aug = np.load('/home/lasii/Research/dataset/alz/processed/aug_labels.npy')

def softmax(logits):
    exp_logits = np.exp(logits)
    probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probs

A = np.where(label==0)
class1_logits = data[A]

A_aug = np.where(label_aug==0)
class2_logits = data_aug[A_aug]

probs_class1 = softmax(class1_logits)
probs_class2 = softmax(class2_logits)

class1_probs = np.mean(probs_class1, axis=0)
class2_probs = np.mean(probs_class2, axis=0)

A_kl_12 = entropy(class1_probs, class2_probs)

A_kl_21 = entropy(class2_probs, class1_probs)

jsd_A = jensenshannon(class1_probs, class2_probs)



B = np.where(label==1)
class1_logits = data[B]

B_aug = np.where(label_aug==1)
class2_logits = data_aug[B_aug]

probs_class1 = softmax(class1_logits)
probs_class2 = softmax(class2_logits)

class1_probs = np.mean(probs_class1, axis=0)
class2_probs = np.mean(probs_class2, axis=0)

B_kl_12 = entropy(class1_probs, class2_probs)

B_kl_21 = entropy(class2_probs, class1_probs)

jsd_B = jensenshannon(class1_probs, class2_probs)


C = np.where(label==2)
class1_logits = data[C]

C_aug = np.where(label_aug==2)
class2_logits = data_aug[C_aug]

probs_class1 = softmax(class1_logits)
probs_class2 = softmax(class2_logits)

class1_probs = np.mean(probs_class1, axis=0)
class2_probs = np.mean(probs_class2, axis=0)

C_kl_12 = entropy(class1_probs, class2_probs)

C_kl_21 = entropy(class2_probs, class1_probs)

jsd_C = jensenshannon(class1_probs, class2_probs)
