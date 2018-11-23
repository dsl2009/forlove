import pandas as pd
from matplotlib import pyplot as plt
from datetime import datetime
import numpy as np
max_num = 6000
img = np.zeros(shape=[400, 274],dtype=np.float32)

flow_file = '/media/dsl/20d6b919-92e1-4489-b2be-a092290668e4/dsl/flow_train.csv'
flow_data = pd.read_csv(flow_file)
flow_group = flow_data.groupby(by='district_code')





for ix, x in enumerate(flow_group.count().index):

    d = flow_data[flow_data['district_code']==x]
    dwell = d['dwell'].values
    flow_in = d['flow_in'].values
    flow_out = d['flow_out'].values
    pad = np.zeros(shape=[274],dtype=np.float32)
    dataes = np.vstack((dwell,flow_in,flow_out, pad))
    img[ix*4:(ix+1)*4,:] = dataes
img = img/max_num
img = np.reshape(img, [20,20, 274])
for x in range(274):
    plt.imshow(img[:,:,x])
    plt.show()


