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


def pandas_table_test():

    data = {'a': [1, 2, 3], 'b': ['dog','cat','ferret']}
    df = pd.DataFrame.from_dict(data)
    w = qw.QWidget()

    def loadFile(self):
        fileName, _ = qw.QFileDialog.getOpenFileName(w, "Open File", "", "CSV Files (*.csv)");
        pathLE.setText(fileName)
        df = pd.read_csv(fileName)
        model = PandasModel(df)
        pandasTv.setModel(model)

    hLayout = qw.QHBoxLayout()
    pathLE = qw.QLineEdit(w)
    hLayout.addWidget(pathLE)
    loadBtn = qw.QPushButton("Select File", w)
    hLayout.addWidget(loadBtn)
    loadBtn.clicked.connect(loadFile)

    vLayout = qw.QVBoxLayout(w)
    vLayout.addLayout(hLayout)

    pandasTv = qw.QTableView()
    model = PandasModel(df)
    pandasTv.setModel(model)
    vLayout.addWidget(pandasTv)

    w.show()
    w.raise_()
    return w


class model_browser(qw.QWidget):

    def __init__(self, batch=289, search_string="ozgf%", parent=None):
        qw.QWidget.__init__(self, parent=None)

        self.batch = batch
        self.search_string = search_string

        hLayout = qw.QHBoxLayout()
        self.batchLE = qw.QLineEdit(self)
        self.batchLE.setText(str(batch))
        hLayout.addWidget(self.batchLE)
        self.loadBtn = qw.QPushButton("Update batch", self)
        self.loadBtn.clicked.connect(self.update_widgets)
        hLayout.addWidget(self.loadBtn)

        self.cells = qw.QListWidget(self)
        self.models = qw.QListWidget(self)

        hLayout2 = qw.QHBoxLayout()
        hLayout2.addWidget(self.cells)
        hLayout2.addWidget(self.models)

        hLayout3 = qw.QHBoxLayout()
        self.viewBtn = qw.QPushButton("View recording", self)
        self.viewBtn.clicked.connect(self.view_recording)
        hLayout3.addWidget(self.viewBtn)

        vLayout = qw.QVBoxLayout(self)
        vLayout.addLayout(hLayout)
        vLayout.addLayout(hLayout2)
        vLayout.addLayout(hLayout3)

        self.update_widgets()

        self.show()
        self.raise_()

    def update_widgets(self):

        batch = int(self.batchLE.text())
        if batch > 0:
            self.batch = batch
        else:
            self.batchLE.setText(str(self.batch))

        self.d_cells = nd.get_batch_cells(self.batch)
        self.d_models = nd.pd_query("SELECT DISTINCT modelname FROM NarfResults" +
                               " WHERE batch=%s AND modelname like %s",
                               (self.batch, self.search_string))

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
