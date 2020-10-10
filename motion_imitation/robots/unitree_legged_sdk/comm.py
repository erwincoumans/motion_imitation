"""Defines message classes to communicate with the robot.

Classes defined in this file corresponds to comm.h in unitree_legged_sdk:
https://github.com/unitreerobotics/unitree_legged_sdk/blob/master/include/unitree_legged_sdk/comm.h

Note that since Unitree used a custom version of lcm, standard lcm message
encoding/decoding will not work, and a custom ctypes-based method will be
required.
"""

import ctypes


class Cartesian(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('x', ctypes.c_float), ('y', ctypes.c_float),
              ('z', ctypes.c_float)]


class IMU(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('quaternion', ctypes.c_float * 4),
              ('gyroscope', ctypes.c_float * 3),
              ('accelerometer', ctypes.c_float * 3),
              ('rpy', ctypes.c_float * 3), ('temperature', ctypes.c_int8)]


class LED(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('r', ctypes.c_uint8), ('g', ctypes.c_uint8),
              ('b', ctypes.c_uint8)]


class MotorState(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('mode', ctypes.c_uint8), ('q', ctypes.c_float),
              ('dq', ctypes.c_float), ('ddq', ctypes.c_float),
              ('tauEst', ctypes.c_float), ('q_raw', ctypes.c_float),
              ('dq_raw', ctypes.c_float), ('ddq_raw', ctypes.c_float),
              ('temperature', ctypes.c_int8), ('reserve', ctypes.c_uint32 * 2)]


class MotorCmd(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('mode', ctypes.c_uint8), ('q', ctypes.c_float),
              ('dq', ctypes.c_float), ('tau', ctypes.c_float),
              ('Kp', ctypes.c_float), ('Kd', ctypes.c_float),
              ('reserve', ctypes.c_uint32 * 3)]


class LowState(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('levelFlag', ctypes.c_uint8), ('commVersion', ctypes.c_uint16),
              ('robotID', ctypes.c_uint16), ('SN', ctypes.c_uint32),
              ('bandWidth', ctypes.c_uint8), ('imu', IMU),
              ('motorState', MotorState * 20),
              ('footForce', ctypes.c_int16 * 4),
              ('footForceEst', ctypes.c_int16 * 4), ('tick', ctypes.c_uint32),
              ('wirelessRemote', ctypes.c_uint8 * 40),
              ('reserve', ctypes.c_uint32), ('crc', ctypes.c_uint32)]


class LowCmd(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('levelFlag', ctypes.c_uint8), ('commVersion', ctypes.c_uint16),
              ('robotID', ctypes.c_uint16), ('SN', ctypes.c_uint32),
              ('bandWidth', ctypes.c_uint8), ('motorCmd', MotorCmd * 20),
              ('led', LED * 4), ('wirelessRemote', ctypes.c_uint8 * 40),
              ('reserve', ctypes.c_uint32), ('crc', ctypes.c_uint32)]


class HighState(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('levelFlag', ctypes.c_uint8), ('commVersion', ctypes.c_uint16),
              ('robotID', ctypes.c_uint16), ('SN', ctypes.c_uint32),
              ('bandWidth', ctypes.c_uint8), ('mode', ctypes.c_uint8),
              ('imu', IMU), ('forwardSpeed', ctypes.c_float),
              ('sideSpeed', ctypes.c_float), ('rotateSpeed', ctypes.c_float),
              ('bodyHeight', ctypes.c_float), ('updownSpeed', ctypes.c_float),
              ('forwardPosition', ctypes.c_float),
              ('sidePosition', ctypes.c_float),
              ('footPosition2Body', Cartesian * 4),
              ('footSpeed2Body', Cartesian * 4),
              ('footForce', ctypes.c_int16 * 4),
              ('footForceEst', ctypes.c_int16 * 4), ('tick', ctypes.c_uint32),
              ('wirelessRemote', ctypes.c_uint8 * 40),
              ('reserve', ctypes.c_uint32), ('crc', ctypes.c_uint32)]


class HighCmd(ctypes.Structure):
  _pack_ = 1
  _fields_ = [('levelFlag', ctypes.c_uint8), ('commVersion', ctypes.c_uint16),
              ('robotID', ctypes.c_uint16), ('SN', ctypes.c_uint32),
              ('bandWidth', ctypes.c_uint8), ('mode', ctypes.c_uint8),
              ('forwardSpeed', ctypes.c_float), ('sideSpeed', ctypes.c_float),
              ('rotateSpeed', ctypes.c_float), ('bodyHeight', ctypes.c_float),
              ('yaw', ctypes.c_float), ('pitch', ctypes.c_float),
              ('roll', ctypes.c_float), ('led', LED * 4),
              ('wirelessRemote', ctypes.c_uint8 * 40),
              ('AppRemote', ctypes.c_uint8 * 40), ('reserve', ctypes.c_uint32),
              ('crc', ctypes.c_uint32)]
