#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 17:05:34 2018

@author: svd
"""

import matplotlib
matplotlib.use("Qt5Agg")
import matplotlib.pyplot as plt
import numpy as np

import PyQt5.QtCore as qc
import PyQt5.QtGui as qg
import PyQt5.QtWidgets as qw

import nems_db.xform_wrappers as nw
import nems.plots.api as nplt
import nems.xforms as xforms
import nems.epoch as ep
from nems.utils import find_module
import pandas as pd
import scipy.ndimage.filters as sf
import nems_lbhb.plots as lplt
import nems_db.db as nd
from nems.plots.recording_browser import (browse_recording)


class model_browser(qw.QWidget):
    """
    For a given batch, list all cellids and modelnames matching in
    NarfResults. Clicking view will call view_model_recording for the
    currently selected model.

    """
    def __init__(self, batch=289,
                 cell_search_string="%",
                 model_search_string="ozgf%",
                 parent=None):
        qw.QWidget.__init__(self, parent=None)

        self.batch = batch

        hLayout = qw.QHBoxLayout(self)

        self.cells = qw.QListWidget(self)
        self.models = qw.QListWidget(self)

        vLayout = qw.QVBoxLayout(self)

        self.batchLE = qw.QLineEdit(self)
        self.batchLE.setText(str(batch))
        vLayout.addWidget(self.batchLE)

        self.cellLE = qw.QLineEdit(self)
        self.cellLE.setText(cell_search_string)
        vLayout.addWidget(self.cellLE)

        self.modelLE = qw.QLineEdit(self)
        self.modelLE.setText(model_search_string)
        vLayout.addWidget(self.modelLE)

        self.loadBtn = qw.QPushButton("Update lists", self)
        self.loadBtn.clicked.connect(self.update_widgets)
        vLayout.addWidget(self.loadBtn)

        self.viewBtn = qw.QPushButton("View recording", self)
        self.viewBtn.clicked.connect(self.view_recording)
        vLayout.addWidget(self.viewBtn)

        hLayout.addLayout(vLayout)
        hLayout.addWidget(self.cells)
        hLayout.addWidget(self.models)

        self.update_widgets()

        self.show()
        self.raise_()

    def update_widgets(self):

        batch = int(self.batchLE.text())
        cellmask = self.cellLE.text()
        modelmask = self.modelLE.text()

        if batch > 0:
            self.batch = batch
        else:
            self.batchLE.setText(str(self.batch))

        self.d_cells = nd.get_batch_cells(self.batch, cellid=cellmask)
        self.d_models = nd.pd_query("SELECT DISTINCT modelname FROM NarfResults" +
                               " WHERE batch=%s AND modelname like %s",
                               (self.batch, modelmask))

        self.cells.clear()
        for c in list(self.d_cells['cellid']):
            list_item = qw.QListWidgetItem(c, self.cells)

        self.models.clear()
        for m in list(self.d_models['modelname']):
            list_item = qw.QListWidgetItem(m, self.models)

        print('updated list widgets')

    def view_recording(self):

        w = self

        batch = w.batch
        cellid = w.cells.currentItem().text()
        modelname = w.models.currentItem().text()

        print("Viewing {},{},{}".format(batch,cellid,modelname))
        aw = view_model_recording(cellid, batch, modelname)

        return aw


def view_model_recording(cellid="TAR010c-18-2", batch=289,
                         modelname="ozgf.fs100.ch18-ld-sev_dlog-wc.18x1.g-fir.1x15-lvl.1-dexp.1_init-basic",
                         recname='val'):

    xf, ctx = nw.load_model_baphy_xform(cellid, batch, modelname, eval_model=True)

    aw = browse_recording(ctx[recname][0], signals=['stim','resp','pred'],
                           cellid=cellid, modelname=modelname)

    return aw
