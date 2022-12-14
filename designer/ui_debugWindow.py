# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'debugWindow.ui'
#
# Created by: PyQt5 UI code generator 5.15.5
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_Form(object):
    def setupUi(self, Form):
        Form.setObjectName("Form")
        Form.resize(860, 433)
        Form.setMinimumSize(QtCore.QSize(860, 433))
        Form.setMaximumSize(QtCore.QSize(16777215, 16777215))
        self.horizontalLayout = QtWidgets.QHBoxLayout(Form)
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.pitchValue = QtWidgets.QLineEdit(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pitchValue.sizePolicy().hasHeightForWidth())
        self.pitchValue.setSizePolicy(sizePolicy)
        self.pitchValue.setObjectName("pitchValue")
        self.gridLayout.addWidget(self.pitchValue, 1, 1, 1, 1)
        self.pitchLabel = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pitchLabel.sizePolicy().hasHeightForWidth())
        self.pitchLabel.setSizePolicy(sizePolicy)
        self.pitchLabel.setObjectName("pitchLabel")
        self.gridLayout.addWidget(self.pitchLabel, 1, 0, 1, 1)
        self.yawLabel = QtWidgets.QLabel(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.yawLabel.sizePolicy().hasHeightForWidth())
        self.yawLabel.setSizePolicy(sizePolicy)
        self.yawLabel.setObjectName("yawLabel")
        self.gridLayout.addWidget(self.yawLabel, 0, 0, 1, 1)
        self.yawValue = QtWidgets.QLineEdit(Form)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.yawValue.sizePolicy().hasHeightForWidth())
        self.yawValue.setSizePolicy(sizePolicy)
        self.yawValue.setObjectName("yawValue")
        self.gridLayout.addWidget(self.yawValue, 0, 1, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.resetGyroBtn = QtWidgets.QPushButton(Form)
        self.resetGyroBtn.setObjectName("resetGyroBtn")
        self.verticalLayout.addWidget(self.resetGyroBtn)
        spacerItem = QtWidgets.QSpacerItem(20, 40, QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        self.verticalLayout.addItem(spacerItem)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        self.frameWidget = QtWidgets.QTabWidget(Form)
        self.frameWidget.setObjectName("frameWidget")
        self.monoRightView = QtWidgets.QWidget()
        self.monoRightView.setObjectName("monoRightView")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.monoRightView)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.monoFrame = QtWidgets.QOpenGLWidget(self.monoRightView)
        self.monoFrame.setMinimumSize(QtCore.QSize(640, 360))
        self.monoFrame.setObjectName("monoFrame")
        self.gridLayout_3.addWidget(self.monoFrame, 0, 0, 1, 1)
        self.frameWidget.addTab(self.monoRightView, "")
        self.depthView = QtWidgets.QWidget()
        self.depthView.setObjectName("depthView")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.depthView)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.depthFrame = QtWidgets.QOpenGLWidget(self.depthView)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.depthFrame.sizePolicy().hasHeightForWidth())
        self.depthFrame.setSizePolicy(sizePolicy)
        self.depthFrame.setMinimumSize(QtCore.QSize(640, 360))
        self.depthFrame.setObjectName("depthFrame")
        self.gridLayout_2.addWidget(self.depthFrame, 0, 0, 1, 1)
        self.frameWidget.addTab(self.depthView, "")
        self.stackedView = QtWidgets.QWidget()
        self.stackedView.setObjectName("stackedView")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.stackedView)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.monoFrame2 = QtWidgets.QOpenGLWidget(self.stackedView)
        self.monoFrame2.setEnabled(True)
        self.monoFrame2.setMinimumSize(QtCore.QSize(320, 180))
        self.monoFrame2.setObjectName("monoFrame2")
        self.verticalLayout_2.addWidget(self.monoFrame2)
        self.depthFrame2 = QtWidgets.QOpenGLWidget(self.stackedView)
        self.depthFrame2.setMinimumSize(QtCore.QSize(320, 180))
        self.depthFrame2.setObjectName("depthFrame2")
        self.verticalLayout_2.addWidget(self.depthFrame2)
        self.verticalLayout_3.addLayout(self.verticalLayout_2)
        self.frameWidget.addTab(self.stackedView, "")
        self.horizontalLayout_2.addWidget(self.frameWidget)
        self.horizontalLayout.addLayout(self.horizontalLayout_2)

        self.retranslateUi(Form)
        self.frameWidget.setCurrentIndex(2)
        QtCore.QMetaObject.connectSlotsByName(Form)

    def retranslateUi(self, Form):
        _translate = QtCore.QCoreApplication.translate
        Form.setWindowTitle(_translate("Form", "Form"))
        self.pitchLabel.setText(_translate("Form", "Pitch:"))
        self.yawLabel.setText(_translate("Form", "Yaw:"))
        self.resetGyroBtn.setText(_translate("Form", "Reset Gyro"))
        self.frameWidget.setTabText(self.frameWidget.indexOf(self.monoRightView), _translate("Form", "Mono"))
        self.frameWidget.setTabText(self.frameWidget.indexOf(self.depthView), _translate("Form", "Depth"))
        self.frameWidget.setTabText(self.frameWidget.indexOf(self.stackedView), _translate("Form", "Page"))
