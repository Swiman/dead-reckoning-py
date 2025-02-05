
from utils import *


class INS(object):
    def __init__(self, Att_0, Vel_0, Pos_0):
        self.Att = [Att_0]  # Att->Roll(Phi), Pitch(Theta), Yaw(Psi)
        self.Pos = [Pos_0]  # Pos->Lat, Lon, Alt (aka h)
        self.Vel = [Vel_0]  # Vel->Vn, Ve, Vd
        # our lla3ecef inputs rad contrary to matlab lla2ecef which inputs deg
        self.Pos_ecef = [lla2ecef(Pos_0)]
        self.gravity = INS.gravity_update(lat=Pos_0[0], h=Pos_0[2])
        C_n_b = euler_to_dcm(Att_0)
        self.C_b_n = np.transpose(C_n_b)
        self.qua = euler_to_qua(Att_0)

    def velocity_update(self, fn, omega_ie_n, omega_en_n, dt):
        # INPUT:
        #       fn, specific forces in the nav-frame (m/s^2).
        #       omega_ie_n, 3x3 previous skew-symmetric Earth rate matrix in the nav-frame (rad/s).
        #       omega_en_n, 3x3 previous skew-symmetric transport rate matrix in the nav-frame (rad/s).
        #       dt, integration time step (s).

        # OUTPUT:
        #        vel_n, updated velocity vector in the nav-frame (m/s).
        V = np.array(self.Vel[-1])
        g = np.transpose(self.gravity)
        coriolis = skew_inv(omega_en_n + 2 * omega_ie_n)
        fn_c = fn + g - skew(V) @ coriolis
        V_new = V + np.squeeze(fn_c * dt)
        self.Vel.append(V_new)

    @classmethod
    def gravity_update(cls, lat, h):
        # RN = 6378137               # WGS84 Equatorial radius in meters
        # RM = 6356752.31425         # WGS84 Polar radius in meters
        # e = 0.0818191908425        # WGS84 eccentricity
        # f = 1 / 298.257223563      # WGS84 flattening
        # WGS84 Earth gravitational constant (m^3 s^-2)
        # mu = 3.986004418E14
        # omega_ie_n = 7.292115e-5   # Earth rotation rate (rad/s)
        h = np.abs(h)
        gn = np.zeros((1, 3))
        # Calculate surface gravity using the Somigliana model, (2.134)
        sinl2, sin2l2 = np.sin(lat) ** 2, np.sin(2 * lat) ** 2
        g0 = 9.780318 * (1 + 5.3024e-03 * (sinl2) - 5.9e-06 * (sin2l2))
        Ro = geocradius(lat)

        # Calculate north gravity using (2.140)
        # gn[:, 0] = -8.08e-9 * h * np.sin(2 * lat)
        gn[:, 0] = 0
        # East gravity is zero
        gn[:, 1] = 0
        # Calculate down gravity using (2.139)
        gn[:, 2] = g0 / (1 + (h / Ro)) ** 2
        return gn

    def position_update(self, dt):
        # INPUT:
        #      pos,  3x1 position vector [lat lon h] (rad, rad, m).
        #      vel,  3x1 NED velocities [n e d] (m/s).
        #      dt,   sampling interval (s).

        # OUTPUT:
        #      pos,  3x1 updated position vector [lat lon h] (rad, rad, m).
        lat, lon, alt = self.Pos[-1]
        vn, ve, vd = self.Vel[-1]

        alt_n = abs(alt - vd * dt)
        RM, _ = radius(lat)
        vn_c = vn / (RM + alt_n)
        lat_n = lat + vn_c * dt

        _, RN = radius(lat_n)
        ve_c = ve / ((RN + alt_n) * np.cos(lat_n))
        lon_n = lon + ve_c * dt

        self.Pos.append([lat_n, lon_n, alt_n])

        # our lla2ecef accepts radians (unlike matlab ecef which accepts deg)
        pos_ecef = lla2ecef([lat_n, lon_n, alt_n])
        self.Pos_ecef.append(pos_ecef)

    def attitude_update(self, omega_b, omega_ie_n, omega_en_n, dt):
        # Update attitude using quaternion or DCM.
        # INPUT:
        #       wb,         3x1 incremental turn-rates in body-frame (rad/s).
        #       omega_ie_n, 3x3 skew-symmetric Earth rate matrix (rad/s).
        #       omega_en_n, 3x3 skew-symmetric transport rate (rad/s).
        #       dt,         1x1 IMU sampling interval (s).
        #
        # OUTPUT:
        #        qua,      4x1 updated quaternion.
        #        DCMbn,    3x3 updated body-to-nav DCM.
        #        euler,    3x1 updated Euler angles (rad).

        om_en_n = skew_inv(omega_en_n)
        om_ie_n = skew_inv(omega_ie_n)

        omega_b_n = omega_b - np.matmul(self.C_b_n.T, om_en_n + om_ie_n)

        ########## update_mode=="dcm" ##########
        # euler_i = omega_b_n * dt
        # self.C_b_n = dcm_update(self.C_b_n, euler_i)
        # euler_new = dcm_to_euler(self.C_b_n)
        # self.qua = euler_to_qua(euler_new) # convert and normalize qua

        ########## update_mode=="qua" ##########
        self.qua = qua_update(self.qua, omega_b_n, dt)
        self.C_b_n = qua_to_dcm(self.qua)
        euler_new = dcm_to_euler(self.C_b_n)
        self.Att.append(euler_new)

    def predict_step(self, omega_b, f_b, dt):
        lat, _, alt = self.Pos[-1]
        vn, ve, _ = self.Vel[-1]

        omega_ie_n = earthrate(lat)
        omega_en_n = transportrate(lat, vn, ve, alt)
        self.attitude_update(omega_b, omega_ie_n, omega_en_n, dt)
        self.gravity = INS.gravity_update(lat, alt)
        fn = np.matmul(self.C_b_n, f_b)
        self.velocity_update(fn, omega_ie_n, omega_en_n, dt)
        self.position_update(dt)
