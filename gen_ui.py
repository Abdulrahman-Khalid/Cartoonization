# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ui.ui'
#
# Created by: PyQt5 UI code generator 5.13.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.setEnabled(True)
        MainWindow.resize(832, 359)
        MainWindow.setMinimumSize(QtCore.QSize(832, 326))
        MainWindow.setMaximumSize(QtCore.QSize(832, 359))
        MainWindow.setMouseTracking(False)
        MainWindow.setAcceptDrops(False)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget2D = QtWidgets.QWidget(self.centralwidget)
        self.widget2D.setGeometry(QtCore.QRect(419, 0, 411, 301))
        self.widget2D.setObjectName("widget2D")
        self.widgetFrame = QtWidgets.QWidget(self.centralwidget)
        self.widgetFrame.setGeometry(QtCore.QRect(0, 0, 411, 301))
        self.widgetFrame.setObjectName("widgetFrame")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(440, 310, 225, 25))
        self.widget.setObjectName("widget")
        self.gridLayout = QtWidgets.QGridLayout(self.widget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.hat = QtWidgets.QCheckBox(self.widget)
        self.hat.setObjectName("hat")
        self.gridLayout.addWidget(self.hat, 0, 0, 1, 1)
        self.glasses = QtWidgets.QCheckBox(self.widget)
        self.glasses.setObjectName("glasses")
        self.gridLayout.addWidget(self.glasses, 0, 1, 1, 1)
        self.mustache = QtWidgets.QCheckBox(self.widget)
        self.mustache.setObjectName("mustache")
        self.gridLayout.addWidget(self.mustache, 0, 2, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "Computer Vision Project"))
        self.hat.setText(_translate("MainWindow", "Hat"))
        self.glasses.setText(_translate("MainWindow", "Glasses"))
        self.mustache.setText(_translate("MainWindow", "Mustache"))
