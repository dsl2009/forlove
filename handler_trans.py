import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from config import disct


def gen_data():
    ix = list(range(98))
    df = pd.read_csv('/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/transition_train.csv')
    df = df.replace(to_replace=disct, value=ix)
    print(df.tail(10))
    data = np.zeros(shape=(274, 98, 98), dtype=np.float32)
    df_gp = df.groupby(by='date_dt')
    for ix, x in enumerate(df_gp.count().index):
        d = df[df['date_dt'] == x]

        dt = d[['o_district_code', 'd_district_code', 'cnt']].values

        for k in range(dt.shape[0]):
            x, y, value = dt[k, 0], dt[k, 1], dt[k, 2]
            print(x, y, value)
            data[ix, int(x), int(y)] = value
    np.save('trans', data)

def show():
    img = np.load('trans.npy')
    tp = np.zeros(shape=98, dtype=np.float32)
    plt.plot(img[:,0,1])
    plt.show()

    for k in range(274):
        print((4+k)%7)
        #plt.plot(img[k, :,1])
        plt.plot(img[k, 1,:])

        plt.show()


show()

