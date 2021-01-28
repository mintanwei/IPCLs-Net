import seaborn
import pandas as pd
import matplotlib.pyplot as plt

result_file = "./experiments/total.txt"
file = open(result_file).readlines()

seaborn.set_style("whitegrid")

# data = {
#     'k':["1"]*9,
#     'p':[0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.5, 0.75, 1],
#     'F2':[0.846]*9,
#     'Sensitivity':[0.868]*9,
#     'Precision':[0.802]*9,
#     'Recall':[0.857]*9
# }
#
# for stat in file:
#     stat = stat.strip().split(",")
#     for n in stat:
#         n = n.split("=")
#         if n[0].strip() == "k":
#             data[n[0].strip()].append(n[1])
#         else:
#             data[n[0].strip()].append(float(n[1]))
#
# frame = pd.DataFrame(data)
# print(frame[:40])
#
# # sns.palplot(sns.hls_palette(8, l=.7, s=.9))
#
# f = plt.figure()
# f.add_subplot(1, 2, 1)
# pl = seaborn.lineplot(x="k", y="F2", hue="p", data=frame, palette=seaborn.hls_palette(9, s=.5))
# f.add_subplot(1, 2, 2)
# pl2 = seaborn.lineplot(x="p", y="F2", hue="k", data=frame, palette=seaborn.hls_palette(7, s=.5))
# plt.show()

import numpy as np
#
data = {
    'Image type': ['Ground truth', 'Prediction', 'Ground truth', 'Prediction', 'Ground truth', 'Prediction', 'Ground truth', 'Prediction'],
    'Lesion type':['A','A','B1','B1','B2','B2','B3','B3'],
    'Proportion (%)':[9.7, 7.3, 78.4, 77.3, 10.5, 11.1, 1.4, 4.3]
}
# data = {
#     'Image type': ['Ground truth', 'Prediction', 'Ground truth', 'Prediction', 'Ground truth', 'Prediction', 'Ground truth', 'Prediction'],
#     'Lesion type':['A','A','B1','B1','B2','B2','B3','B3'],
#     'Proportion (%)':[18.7, 15.1, 76.4, 82.4, 4.9, 2.5, 0.000, 0.000]
# }

df = pd.DataFrame(data, columns=['Image type', 'Lesion type', 'Proportion (%)'])

b = seaborn.barplot(x='Lesion type', y='Proportion (%)', data=df, hue='Image type')

for index,row in df.iterrows():
    if row.name % 2 == 0:
        b.text(row.name//2 - 0.2,row.values[2]+1,round(row.values[2],1),color="black",ha="center")
    else:
        b.text(row.name//2 + 0.2, row.values[2] + 1, round(row.values[2], 1), color="black", ha="center")
plt.show()