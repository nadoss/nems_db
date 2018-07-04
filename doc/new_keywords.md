# KEYWORD TRANSLATIONS CHEATSHEET
Also see: nems/docs/keywords.md

## Loaders / Preprocessors

Coming soon! Pending separation of loading from preprocessing functionality.
But in the meantime, most loaders are the same except for the insertion
of periods between options and the addition of 'fs' preceeding the sampling rate.
For example:
```
old_loader = 'ozgf100ch18n'
new_loader = 'ozgf.fs100.ch18.n'
```


## Modules

Most modules will just need a period inserted before the first numeral.

Ex:
```
old_wc = 'wc15x2'
new_wc = 'wc.15x2'

old_dexp = 'dexp1'
new_dexp = 'dexp.1'
```

But modules with more complicated options have to be moved around a litle more,
and may be combined into a single keyword.

In general, they should adhere to the format:
```
kwhead.requiredOption.otherOption(s)
```

Ex:
```
old_stpz = 'stpz2'
old_stpb = 'stp2b'

new_stpzb = 'stp.2.z.b'
```


## Fitters

Fitter keywords look the most different. They follow the same format as the loaders
and modules, but a lot of options were combined into just two fitters: basic and iter.
As with the loaders, some of the initialization functionality may be broken out into
separate prefitting/postprocessing functions in the future.

Some examples:
```
old = 'fit01'
new = 'basic'

old = 'fit01a'  # or 'basicqk'
new = 'basic.mi1000.t5'  # for max_iterations = 1000 and tolerance = 10^-5

old = 'state01-jkm'
new = 'basic.st.nf5.m'  # for state, nfold w/ 5 folds, and use split_for_jackknife

old = 'iter-T3-T5-S1x2x3-ti50-fi10'
new = 'iter.T3,5.S1,2,3.ti50.fi10'
```

If you are having trouble translating any keywords you've been using into the new system,
just ask Jacob! Or take a look at the function definitions in nems.plugins.default_keywords
(or nems_lbhb.plugins.lbhb_keywords once custom ones are added) if you want to figure out
the parsing for yourself. None of the regex was too complicated so hopefully most of it
is straightforward.