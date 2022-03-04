import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class GridHashing(object):

    def __init__(self, scale, obs_processed_flat_dim, bucket_sizes=None):
        # Hashing function: SimHash
        if bucket_sizes is None:
            # Large prime numbers
            bucket_sizes = [999931, 999953, 999959, 999961, 999979, 999983]

        prime = 100001
        mods_list = []
        for bucket_size in bucket_sizes:
            mod = 1
            mods = []
            #for _ in range(dim_key):
            for _ in range(obs_processed_flat_dim):
                mods.append(mod)
                mod = (mod * prime) % bucket_size
            mods_list.append(mods)
        self.bucket_sizes = np.asarray(bucket_sizes)
        self.mods_list = np.asarray(mods_list).T
        # print("largest range", scale * prime)
        self.tables = np.zeros((len(bucket_sizes), np.max(bucket_sizes)))
        #self.projection_matrix = np.random.normal(size=(obs_processed_flat_dim, dim_key))
        self.scale = scale

    def compute_keys(self, obss):
        binaries = np.asarray(obss) // self.scale
        keys = np.cast['int'](binaries.dot(self.mods_list)) % self.bucket_sizes
        return keys

    def inc_hash(self, obss):
        keys = self.compute_keys(obss)
        for idx in range(len(self.bucket_sizes)):
            np.add.at(self.tables[idx], keys[:, idx], 1)

    def inc_multi(self, obss, inc_num):
        keys = self.compute_keys(obss)
        for idx in range(len(self.bucket_sizes)):
            self.tables[idx, keys[:, idx]] += inc_num


    def query_hash(self, obss):
        keys = self.compute_keys(obss)
        all_counts = []
        for idx in range(len(self.bucket_sizes)):
            all_counts.append(self.tables[idx, keys[:, idx]])
        return np.asarray(all_counts).min(axis=0)

    def fit_before_process_samples(self, obs):
        if len(obs.shape) == 1:
            obss = [obs]
        else:
            obss = obs
        before_counts = self.query_hash(obss)
        self.inc_hash(obss)

    def predict(self, obs):
        counts = self.query_hash(obs)
        return counts
        # return 1. / np.maximum(1., np.sqrt(counts))
        # return counts[0]

    def predict_rewards(self, obs):
        counts = self.query_hash(obs)
        # return counts
        return 1. / np.maximum(1., np.sqrt(counts))


if __name__=="__main__":
    hash = GridHashing(0.1, 2)

    x_list = []
    y_list=[]
    pred_list=[]
    list=[]
    for i in range(1000):
        pos = np.random.randint(0,36,[10, 2])
        hash.inc_hash(pos)
        x_list.append(pos[:,0])
        y_list.append(pos[:,1])
        list.append(pos)


    for pos in list:
        eps = np.random.random([1,2])
        pred = hash.predict(pos + eps)
        # print(pos + eps)
        # print(pred)
        pred_list.append(pred)
    x_array = np.array(x_list)
    y_array=np.array(y_list)
    z_array = np.array((pred_list))
    ax=plt.subplot(111, projection='3d')
    ax.scatter(x_array,y_array,z_array,c='g')

    # plt.show()

    bx=plt.subplot(111)
    bx.scatter(x_array,y_array,c='r')
    # plt.show()
