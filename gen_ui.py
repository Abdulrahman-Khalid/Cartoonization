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
        MainWindow.resize(832, 635)
        MainWindow.setMinimumSize(QtCore.QSize(832, 635))
        MainWindow.setMaximumSize(QtCore.QSize(832, 635))
        MainWindow.setMouseTracking(False)
        MainWindow.setAcceptDrops(False)
        MainWindow.setAutoFillBackground(False)
        MainWindow.setToolButtonStyle(QtCore.Qt.ToolButtonIconOnly)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.widget3D = QtWidgets.QWidget(self.centralwidget)
        self.widget3D.setGeometry(QtCore.QRect(-1, -1, 411, 301))
        self.widget3D.setObjectName("widget3D")
        self.widget2D = QtWidgets.QWidget(self.centralwidget)
        self.widget2D.setGeometry(QtCore.QRect(419, 0, 411, 301))
        self.widget2D.setObjectName("widget2D")
        self.widgetFrame = QtWidgets.QWidget(self.centralwidget)
        self.widgetFrame.setGeometry(QtCore.QRect(200, 310, 411, 301))
        self.widgetFrame.setObjectName("widgetFrame")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate(
            "MainWindow", "Computer Vision Project"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
