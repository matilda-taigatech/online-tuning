# Log simulation with configurable levels of jitter and delay

import numpy as np


class Log:
    def __init__(self, diameter_range, length_range):
        self.maxAcc = 128
        
        self.diameter = np.random.uniform(diameter_range[0], diameter_range[1])
        self.length = np.random.uniform(length_range[0], length_range[1])
        self.angle = 0
        self.angularVelocity = 0
        self.traveledDistance = 0

    def turn(self, logTurnerAngularVelocity, dt, conveyorSpeed):
        angularAcceleration = np.max([np.min([(logTurnerAngularVelocity - self.angularVelocity)/dt, self.maxAcc]), -self.maxAcc])
        self.angularVelocity += angularAcceleration*dt
        self.angle += self.angularVelocity*dt
        self.traveledDistance += conveyorSpeed*dt

        if self.traveledDistance >= self.length:
            return True
        else:
            return False

    def getRotationError(self, goalAngle):
        return goalAngle - self.angle

class LogTurner:
    def __init__(self, diameter_range, length_range, conveyorSpeed, dt):
        self.maxAcc = 128       # 0.08/logDiameter/dt*4 
        self.maxVel = 3840      # 1.20/logDiameter/dt*8
        self.maxSignal = 3840

        self.diameter_range = diameter_range
        self.length_range = length_range
        self.conveyorSpeed = conveyorSpeed
        self.dt = dt

    def generate_log(self):
        self.log = Log(self.diameter_range, self.length_range)
        self.targetAngle = np.random.uniform(-180, 180)

        self.P = self.targetAngle
        self.I = 0
        self.D = 0

        self.angularVelocity = 0

    def turn_log_one_frame(self, kP, kI, kD):
        signal = np.max([np.min([self.P*kP + self.I*kI + self.D*kD, self.maxSignal]), -self.maxSignal])
        angularAcceleration = np.max([np.min([(signal/self.log.diameter - self.angularVelocity)/self.dt, self.maxAcc]), -self.maxAcc])
        self.angularVelocity += angularAcceleration*self.dt
        logEnded = self.log.turn(self.angularVelocity, self.dt, self.conveyorSpeed)

        rotationError = self.log.getRotationError(self.targetAngle)
        self.D = (rotationError - self.P)/self.dt
        self.I += rotationError*self.dt
        self.P = rotationError

        return logEnded, rotationError
    
