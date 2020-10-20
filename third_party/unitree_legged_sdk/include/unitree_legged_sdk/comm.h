/************************************************************************
Copyright (c) 2020, Unitree Robotics.Co.Ltd. All rights reserved.
Use of this source code is governed by the MPL-2.0 license, see LICENSE.
************************************************************************/

#ifndef _UNITREE_LEGGED_COMM_H_
#define _UNITREE_LEGGED_COMM_H_

#include <stdint.h>
#include <array>

namespace UNITREE_LEGGED_SDK
{

	constexpr int HIGHLEVEL = 0x00;
	constexpr int LOWLEVEL  = 0xff;
	constexpr double PosStopF = (2.146E+9f);
	constexpr double VelStopF = (16000.0f);

#pragma pack(1)

	typedef struct
	{
		float x;
		float y;
		float z;
	} Cartesian;

	typedef struct
	{
		std::array<float, 4> quaternion;               // quaternion, normalized, (w,x,y,z)
		std::array<float, 3> gyroscope;                // angular velocity （unit: rad/s)
		std::array<float, 3> accelerometer;            // m/(s2)
		std::array<float, 3> rpy;                      // euler angle（unit: rad)
		int8_t temperature;
	} IMU;                                 // when under accelerated motion, the attitude of the robot calculated by IMU will drift.

	typedef struct
	{
		uint8_t r;
		uint8_t g;
		uint8_t b;
	} LED;                                 // foot led brightness: 0~255

	typedef struct
	{
		uint8_t mode;                      // motor working mode
		float q;                           // current angle (unit: radian)
		float dq;                          // current velocity (unit: radian/second)
		float ddq;                         // current acc (unit: radian/second*second)
		float tauEst;                      // current estimated output torque (unit: N.m)
		float q_raw;                       // current angle (unit: radian)
		float dq_raw;                      // current velocity (unit: radian/second)
		float ddq_raw;
		int8_t temperature;                // current temperature (temperature conduction is slow that leads to lag)
		std::array<uint32_t, 2> reserve;
	} MotorState;                          // motor feedback

	typedef struct
	{
		uint8_t mode;                      // desired working mode
		float q;                           // desired angle (unit: radian)
		float dq;                          // desired velocity (unit: radian/second)
		float tau;                         // desired output torque (unit: N.m)
		float Kp;                          // desired position stiffness (unit: N.m/rad )
		float Kd;                          // desired velocity stiffness (unit: N.m/(rad/s) )
		std::array<uint32_t, 3> reserve;
	} MotorCmd;                            // motor control

	typedef struct
	{
		uint8_t levelFlag;                 // flag to distinguish high level or low level
		uint16_t commVersion;
		uint16_t robotID;
		uint32_t SN;
		uint8_t bandWidth;
		IMU imu;
		std::array<MotorState, 20> motorState;
		std::array<int16_t, 4> footForce;              // force sensors
		std::array<int16_t, 4> footForceEst;           // force sensors
		uint32_t tick;                     // reference real-time from motion controller (unit: us)
		std::array<uint8_t, 40> wirelessRemote;        // wireless commands
		uint32_t reserve;
		uint32_t crc;
	} LowState;                            // low level feedback

	typedef struct
	{
		uint8_t levelFlag;
		uint16_t commVersion;
		uint16_t robotID;
		uint32_t SN;
		uint8_t bandWidth;
		std::array<MotorCmd, 20> motorCmd;
		std::array<LED, 4> led;
		std::array<uint8_t, 40> wirelessRemote;
		uint32_t reserve;
		uint32_t crc;
	} LowCmd;                              // low level control

	typedef struct
	{
		uint8_t levelFlag;
		uint16_t commVersion;
		uint16_t robotID;
		uint32_t SN;
		uint8_t bandWidth;
		uint8_t mode;
		IMU imu;
		float forwardSpeed;
		float sideSpeed;
		float rotateSpeed;
		float bodyHeight;
		float updownSpeed;                 // speed of stand up or squat down
		float forwardPosition;             // front or rear displacement, an integrated number form kinematics function, usually drift
		float sidePosition;                // left or right displacement, an integrated number form kinematics function, usually drift
		std::array<Cartesian, 4> footPosition2Body;    // foot position relative to body
		std::array<Cartesian, 4> footSpeed2Body;       // foot speed relative to body
		std::array<int16_t, 4> footForce;
		std::array<int16_t, 4> footForceEst;
		uint32_t tick;                     // reference real-time from motion controller (unit: us)
		std::array<uint8_t, 40> wirelessRemote;
		uint32_t reserve;
		uint32_t crc;
	} HighState;                           // high level feedback

	typedef struct
	{
		uint8_t levelFlag;
		uint16_t commVersion;
		uint16_t robotID;
		uint32_t SN;
		uint8_t bandWidth;
		uint8_t mode;                      // 0:idle, default stand      1:forced stand     2:walk continuously
		float forwardSpeed;                // speed of move forward or backward, scale: -1~1
		float sideSpeed;                   // speed of move left or right, scale: -1~1
		float rotateSpeed;	               // speed of spin left or right, scale: -1~1
		float bodyHeight;                  // body height, scale: -1~1
		float footRaiseHeight;             // foot up height while walking (unavailable now)
		float yaw;                         // unit: radian, scale: -1~1
		float pitch;                       // unit: radian, scale: -1~1
		float roll;                        // unit: radian, scale: -1~1
		std::array<LED, 4> led;
		std::array<uint8_t, 40> wirelessRemote;
		std::array<uint8_t, 40> AppRemote;
		uint32_t reserve;
		uint32_t crc;
	} HighCmd;                             // high level control

#pragma pack()

	typedef struct
	{
		unsigned long long TotalCount;     // total loop count
		unsigned long long SendCount;      // total send count
		unsigned long long RecvCount;      // total receive count
		unsigned long long SendError;      // total send error
		unsigned long long FlagError;      // total flag error
		unsigned long long RecvCRCError;   // total reveive CRC error
		unsigned long long RecvLoseError;  // total lose package count
	} UDPState;                            // UDP communication state

	constexpr int HIGH_CMD_LENGTH   = (sizeof(HighCmd));
	constexpr int HIGH_STATE_LENGTH = (sizeof(HighState));

}

#endif
