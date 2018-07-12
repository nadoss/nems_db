#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 11 17:27:49 2018

@author: hellerc
"""

"""
Example implementation of loading multiple cells in xforms 
"""

import nems_db.db as nd
import nems_db.xform_wrappers as xfw
import nems.xform_helper as xhelp
import nems.xforms as xforms
from nems import get_setting
from nems.plugins import (default_keywords, default_loaders, default_fitters,
                          default_initializers)
from nems.registry import KeywordRegistry

## single cell example using keyword ldb
batch = 289
cellid = 'bbl086b-02-1'
# simple state gain model
load_keywords = 'nostim.fs5.pup-ldb-st.pup-psthfr'

recording_uri = xfw.generate_recording_uri(cellid, batch, load_keywords)

xforms_lib = KeywordRegistry(recording_uri=recording_uri, cellid=[cellid])
xforms_lib.register_modules([default_loaders, default_fitters,
                             default_initializers])
xforms_lib.register_plugins(get_setting('XFORMS_PLUGINS'))

keyword_lib = KeywordRegistry()
keyword_lib.register_module(default_keywords)
keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

xfspec = []

# 1) Load the data
xfspec.extend(xhelp._parse_kw_string(load_keywords, xforms_lib))

ctx, log_xf = xforms.evaluate(xfspec)
print("Single cell loaded, shape of rec['resp']: {0}".format(ctx['rec']['resp'].shape))
print('\n')
# multichannel example using keyword ldb

batch = 289
cellid = ['bbl086b-02-1', 'bbl086b-03-1']
# simple state gain model
load_keywords = 'nostim.fs5.pup-ldb-st.pup-psthfr'

recording_uri = xfw.generate_recording_uri(cellid, batch, load_keywords)

xforms_lib = KeywordRegistry(recording_uri=recording_uri, cellid=cellid)
xforms_lib.register_modules([default_loaders, default_fitters,
                             default_initializers])
xforms_lib.register_plugins(get_setting('XFORMS_PLUGINS'))

keyword_lib = KeywordRegistry()
keyword_lib.register_module(default_keywords)
keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

xfspec = []

# 1) Load the data
xfspec.extend(xhelp._parse_kw_string(load_keywords, xforms_lib))

ctx, log_xf = xforms.evaluate(xfspec)
print("Mutiple cells loaded, shape of rec['resp']: {0}".format(ctx['rec']['resp'].shape))
print('\n')

# load entire site by providing only the siteid in place of the cellid
batch = 289
cellid = 'bbl086b'
# simple state gain model
load_keywords = 'nostim.fs5.pup-ldb-st.pup-psthfr'

recording_uri = xfw.generate_recording_uri(cellid, batch, load_keywords)

xforms_lib = KeywordRegistry(recording_uri=recording_uri, cellid=[cellid])
xforms_lib.register_modules([default_loaders, default_fitters,
                             default_initializers])
xforms_lib.register_plugins(get_setting('XFORMS_PLUGINS'))

keyword_lib = KeywordRegistry()
keyword_lib.register_module(default_keywords)
keyword_lib.register_plugins(get_setting('KEYWORD_PLUGINS'))

xfspec = []

# 1) Load the data
xfspec.extend(xhelp._parse_kw_string(load_keywords, xforms_lib))

ctx, log_xf = xforms.evaluate(xfspec)
print("Entrie site loaded, shape of rec['resp']: {0}".format(ctx['rec']['resp'].shape))
