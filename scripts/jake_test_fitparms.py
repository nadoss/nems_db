
from nems_db.utilities.params import fitted_params_per_cell

cellids = ['TAR010c-18-1', 'TAR010c-15-3', 'TAR010c-58-2',
           'TAR017b-09-1', 'TAR017b-43-1']
batch = 271
modelname = 'ozgf100ch18_wc18x1_fir1x15_lvl1_dexp1_fit01'
df = fitted_params_per_cell(cellids, batch, modelname, include_stats=True)
print(df)

# example output (truncated)
"""
                                                                                          mean  \
weight_channels.basic---coefficients         [[0.0887608158389488, 0.14670809100674873, 0.1...
fir.basic---coefficients                     [[0.12310242089016128, 0.7140680798391459, 0.3...
levelshift.levelshift---level                                                        -0.678747
nonlinearity.double_exponential---amplitude                                           0.795443
nonlinearity.double_exponential---base                                                0.140065
nonlinearity.double_exponential---kappa                                              0.0326327
nonlinearity.double_exponential---shift                                               0.825605

                                                                                           std  \
weight_channels.basic---coefficients         [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,...
fir.basic---coefficients                     [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,...
levelshift.levelshift---level                                                                0
nonlinearity.double_exponential---amplitude                                                  0
nonlinearity.double_exponential---base                                                       0
nonlinearity.double_exponential---kappa                                                      0
nonlinearity.double_exponential---shift                                                      0

                                                                                  TAR010c-18-1  \
weight_channels.basic---coefficients         [[[0.0887608158389488, 0.14670809100674873, 0....
fir.basic---coefficients                     [[[0.12310242089016128, 0.7140680798391459, 0....
levelshift.levelshift---level                                                        -0.678747
nonlinearity.double_exponential---amplitude                                           0.795443
nonlinearity.double_exponential---base                                                0.140065
nonlinearity.double_exponential---kappa                                              0.0326327
nonlinearity.double_exponential---shift                                               0.825605

                                                                                  TAR010c-15-3  \
weight_channels.basic---coefficients         [[[0.0887608158389488, 0.14670809100674873, 0....
fir.basic---coefficients                     [[[0.12310242089016128, 0.7140680798391459, 0....
levelshift.levelshift---level                                                        -0.678747
nonlinearity.double_exponential---amplitude                                           0.795443
nonlinearity.double_exponential---base                                                0.140065
nonlinearity.double_exponential---kappa                                              0.0326327
nonlinearity.double_exponential---shift                                               0.825605
"""