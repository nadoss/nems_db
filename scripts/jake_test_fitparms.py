
from nems_db.utilities.params import fitted_params_per_cell

cellids = ['TAR010c-18-1', 'TAR010c-15-3', 'TAR010c-58-2',
           'TAR017b-09-1', 'TAR017b-43-1']
batch = 271
modelname = 'ozgf100ch18_wc18x1_fir1x15_lvl1_dexp1_fit01'
df = fitted_params_per_cell(cellids, batch, modelname, include_stats=True)
print(df)

# example output (truncated)
"""                                                                                    mean  \
weight_channels.basic---coefficients         [[-0.15906195968842768, 0.3843768871847452, 0....
fir.basic---coefficients                     [[0.1656700154496499, 0.8316530120326318, 0.55...
levelshift.levelshift---level                                                         0.203298
nonlinearity.double_exponential---amplitude                                           0.218866
nonlinearity.double_exponential---base                                               0.0515607
nonlinearity.double_exponential---kappa                                               0.117831
nonlinearity.double_exponential---shift                                              -0.118553

                                                                                           std  \
weight_channels.basic---coefficients         [[0.40403571595725013, 0.7331578015069575, 0.9...
fir.basic---coefficients                     [[0.29994481838872183, 0.17224095449207594, 0....
levelshift.levelshift---level                                                         0.968149
nonlinearity.double_exponential---amplitude                                           0.291951
nonlinearity.double_exponential---base                                               0.0491721
nonlinearity.double_exponential---kappa                                               0.240207
nonlinearity.double_exponential---shift                                               0.943362

                                                                                  TAR010c-18-1  \
weight_channels.basic---coefficients         [[0.0887608158389488, 0.14670809100674873, 0.1...
fir.basic---coefficients                     [[0.12310242089016128, 0.7140680798391459, 0.3...
levelshift.levelshift---level                                                        -0.678747
nonlinearity.double_exponential---amplitude                                           0.795443
nonlinearity.double_exponential---base                                                0.140065
nonlinearity.double_exponential---kappa                                              0.0326327
nonlinearity.double_exponential---shift                                               0.825605

                                                                                  TAR010c-15-3  \
weight_channels.basic---coefficients         [[-0.07748810621554607, -0.10302684755519784, ...
fir.basic---coefficients                     [[0.27883152277346396, 0.8797231551859458, 0.4...
levelshift.levelshift---level                                                          1.99263
nonlinearity.double_exponential---amplitude                                          0.0796707
nonlinearity.double_exponential---base                                               0.0639726
nonlinearity.double_exponential---kappa                                               0.530775
nonlinearity.double_exponential---shift                                               -1.85929
"""