"""Test the C++ robot interface.

Follow the
"""

from robot_interface import RobotInterface # pytype: disable=import-error

i = RobotInterface()
o = i.receive_observation()
