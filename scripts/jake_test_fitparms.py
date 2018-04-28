from nems_db.params import fitted_params_per_batch, plot_all_params


batch = 271
limit = None
modelname = 'ozgf100ch18_wcg18x2_fir2x15_lvl1_dexp1_fit01'

#batch = 303
#limit = None
#modelname = 'nostim20pupbeh_stategain3_basic-nf'

# Can use mod_key='fn', mod_key='id', etc to display more info in index.
# Formatted as: '<mspec_index--mod_key--parameter_name>'
# So mod_key='id' gives something like: '0--wc15x1--coefficients'.
df = fitted_params_per_batch(batch, modelname, include_stats=False, mod_key='',
                             limit=limit)
print(df)

# Not handling arrays yet, just scalar params
scalar_df = df.iloc[6:]
plot_all_params(scalar_df, only_scalars=True)
#print(df.loc['fir.basic---coefficients'].loc['std'])

# example output (truncated)
"""
                                                              mean  \
0--mean                   [0.6428007712025126, 1.0079999163612767]
0--sd                    [0.4607000818990278, 0.41440913122937123]
1--coefficients  [[0.22104853498428548, 0.3174402233420055, 0.0...
2--level                                                 -0.267411
3--amplitude                                              0.466702
3--base                                                  0.0105623
3--kappa                                                  0.316573
3--shift                                                  0.363319

                                                               std  \
0--mean                   [1.6975881781786677, 1.1959587701620937]
0--sd                    [0.38278921895876744, 0.6927154706654831]
1--coefficients  [[2.06565150846871, 2.262147292283313, 2.05974...
2--level                                                  0.839406
3--amplitude                                                1.4866
3--base                                                   0.207179
3--kappa                                                  0.849237
3--shift                                                   0.83023

                                                      bbl086b-02-1  \
0--mean                  [0.23013036492763214, 0.6304225736574065]
0--sd                     [0.5907674068753405, 0.5820608771096261]
1--coefficients  [[-0.014322736264307473, -0.003411946901872087...
2--level                                               -0.00398745
3--amplitude                                              0.147603
3--base                                                 -0.0497636
3--kappa                                               -0.00249152
3--shift                                                0.00799814

                                                      bbl086b-03-1  \
0--mean                   [0.4346427550345195, 0.7570562046631056]
0--sd                    [0.40228958193793546, 0.9470540791882479]
1--coefficients  [[0.013146256809296559, -0.012661030896089427,...
2--level                                                0.00369642
3--amplitude                                             0.0887719
3--base                                                 -0.0147672
3--kappa                                                 -0.012191
3--shift                                                 0.0145183
"""