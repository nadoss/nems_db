import pytest

from nems import get_setting as ngs
from nems.registry import KeywordRegistry
from nems.plugins.keywords import default_keywords
from nems.plugins.loaders import default_loaders
from nems.plugins.fitters import default_fitters


@pytest.fixture
def loader_registry():
    loader_lib = KeywordRegistry('dummy_recording_uri')
    loader_lib.register_module(default_loaders)
    loader_lib.register_plugins(ngs('XF_LOADER_PLUGINS'))
    return loader_lib


@pytest.fixture
def fitter_registry():
    fitter_lib = KeywordRegistry()
    fitter_lib.register_module(default_fitters)
    fitter_lib.register_plugins(ngs('XF_FITTER_PLUGINS'))
    return fitter_lib


@pytest.fixture
def keyword_registry():
    keyword_lib = KeywordRegistry()
    keyword_lib.register_module(default_keywords)
    keyword_lib.register_plugins(ngs('XF_LOADER_PLUGINS'))
    return keyword_lib

def test_loaders_exist(loader_registry):
    xf = loader_registry['ozgf.18ch100']
    xf = loader_registry['env.100']
    xf = loader_registry['psth']
    xf = loader_registry['nostim']
    xf = loader_registry['evt.pupbehtarlic0']

# TODO: Test aliasing system once backwards-compatibility shims are in place.
# TODO: Test fitters and keywords once there's something to register
#       (both empty at the moment)
