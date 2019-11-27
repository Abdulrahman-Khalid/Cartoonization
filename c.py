from QPanda3D.Panda3DWorld import Panda3DWorld
from QPanda3D.QPanda3DWidget import QPanda3DWidget

from PyQt5.QtWidgets import QApplication
from PyQt5 import QtCore, QtGui, QtWidgets

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from direct.actor.Actor import Actor
from direct.interval.IntervalGlobal import Sequence
from panda3d.core import Point3

import sys
from math import pi, sin, cos


class MyWorld(Panda3DWorld):
    def __init__(self):
        Panda3DWorld.__init__(self, 411, 301)

        # from this point, act as if you are defining a classic panda3D environment
        self.cam.setPos(0, -28, 6)
        self.testModel = self.loader.loadModel('panda')
        self.testModel.reparentTo(self.render)

        # Disable the camera trackball controls.
        self.disableMouse()

        # Load the environment model.
        self.scene = self.loader.loadModel("models/environment")
        # Reparent the model to render.
        self.scene.reparentTo(self.render)
        # Apply scale and position transforms on the model.
        self.scene.setScale(0.25, 0.25, 0.25)
        self.scene.setPos(-8, 42, 0)

        # Add the spinCameraTask procedure to the task manager.
        self.taskMgr.add(self.spinCameraTask, "SpinCameraTask")

        # Load and transform the panda actor.
        self.pandaActor = Actor("models/panda-model",
                                {"walk": "models/panda-walk4"})
        self.pandaActor.setScale(0.005, 0.005, 0.005)
        self.pandaActor.reparentTo(self.render)
        # Loop its animation.
        self.pandaActor.loop("walk")

        # Create the four lerp intervals needed for the panda to
        # walk back and forth.
        pandaPosInterval1 = self.pandaActor.posInterval(13,
                                                        Point3(0, -10, 0),
                                                        startPos=Point3(0, 10, 0))
        pandaPosInterval2 = self.pandaActor.posInterval(13,
                                                        Point3(0, 10, 0),
                                                        startPos=Point3(0, -10, 0))
        pandaHprInterval1 = self.pandaActor.hprInterval(3,
                                                        Point3(180, 0, 0),
                                                        startHpr=Point3(0, 0, 0))

        pandaHprInterval2 = self.pandaActor.hprInterval(3,
                                                        Point3(0, 0, 0),
                                                        startHpr=Point3(180, 0, 0))

        # Create and play the sequence that coordinates the intervals.
        self.pandaPace = Sequence(pandaPosInterval1,
                                  pandaHprInterval1,
                                  pandaPosInterval2,
                                  pandaHprInterval2,
                                  name="pandaPace")
        self.pandaPace.loop()

    # Define a procedure to move the camera.
    def spinCameraTask(self, task):
        angleDegrees = task.time * 6.0
        angleRadians = angleDegrees * (pi / 180.0)
        self.camera.setPos(20 * sin(angleRadians), -
                           20.0 * cos(angleRadians), 3)
        self.camera.setHpr(angleDegrees, 0, 0)
        return Task.cont


if __name__ == "__main__":
    app = QApplication(sys.argv)

    appw = QtWidgets.QMainWindow()
    appw.setGeometry(50, 50, 800, 600)
    world = MyWorld()
    pandaWidget = QPanda3DWidget(world)
    appw.setCentralWidget(pandaWidget)
    appw.show()
    
    sys.exit(app.exec_())
