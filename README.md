# Strapdown Inertial Navigation implemented in Python

This repository contains a Python implementation of an Inertial Navigation System(INS) in North-East-Down (NED) frame. The original version is developed in MATLAB by [rodralez/NaveGo](https://github.com/rodralez/NaveGo/tree/master). 

# Conversion Details
* All functions and logic were ported with minimal modification, but adjustments were made to accommodate the differences between the two languages (e.g., handling matrices, functions, indexing). 
* Basic functions (e.g., `lla2ecef`, `radius`) which are part of MATLAB aerospace toolbox, are implemented in `utils.py`.


# Requirements
```
pip install numpy
```

# Usage
```
from deadreckoning import INS

ins = INS(Att_0=INITIAL_ATTITUDE, Vel_0=INITIAL_VELOCITY, Pos_0=INITIAL_POSITION)
for t in time:
    ins.predict_step(gyro[t], accel[t], t)
```
After execution, the following attributes will contain the corresponding trajectory data:

* ins.Pos: Position (NED frame)
* ins.Pos_ecef: Position (ECEF frame)
* ins.Vel: Velocity (NED frame)
* ins.Att: Attitude (NED frame)


# Important Notes

* The `lla2ecef` function is different from MATLAB's aerospace toolbox version. Specifically in MATLAB implementation, latitude and longitude are in **DEGREES**, while our function inputs these coordinates in **RADIANS**.

* The INS is sensitive to noise, hence it may be beneficial to apply denoising algorithms to raw IMU signals prior to feeding them into INS.

