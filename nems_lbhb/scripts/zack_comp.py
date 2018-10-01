
#Compare models fit by NEMS to my regression on pupil data.
#ZPS 2018-09-28

import sys
import pandas as pd
import matplotlib.pyplot as plt

#sys.path.append('/auto/users/schwarza/code/nems_db/nems_lbhb/scripts/')
from nems_lbhb.scripts.pup_results import pup_pred_sum

nems_models = pup_pred_sum(batch=294, fs=4, jkn=20)

#sys.path.append('/auto/users/schwarza/code/pupil/multiple_regression/')
my_models = pd.read_csv('/auto/users/schwarza/code/pupil/multiple_regression/nems_comparison.csv')

models = pd.merge(my_models, nems_models, on="cellid")
models['r2'] = models['r'] ** 2

fig, axes = plt.subplots(1, 3)
axes[0].scatter(models["r2"], models["r_squared"]) #accuracy
axes[0].set_aspect('equal','box')
axes[0].set_xlabel('zs r2')
axes[0].set_ylabel('nems r2')

axes[1].scatter(models["b_3"], models["d"]) #dc terms
axes[1].set_xlabel('zs dc')
axes[1].set_ylabel('nems dc')

axes[2].scatter(models["b_4"], models["g"]) #gain terms
axes[2].set_xlabel('zs gain')
axes[2].set_ylabel('nems gain')

plt.show()

models["accuracy_difference"] = models["r2"] - models["r_squared"]
models = models.sort_values(by = "accuracy_difference")
models.to_csv("nems_accuracy_difference.csv")

