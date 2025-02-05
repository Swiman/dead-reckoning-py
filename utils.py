import numpy as np


def skew(vec):
    # Create skew-symmetric matrix of a vector
    # print("vvv", vec)
    vec = np.squeeze(vec)
    S = [[0, -vec[2], vec[1]], [vec[2], 0, -vec[0]], [-vec[1], vec[0], 0]]
    return np.array(S)


def skew_inv(mat):
    # extract a vector from corresponding skew-symmetric matrix
    mat = np.squeeze(mat)
    vec = [mat[2, 1], mat[0, 2], mat[1, 0]]
    return np.reshape(vec, (3, 1))


def radius(lat):
    # Calculates meridian and normal radii of curvature
    # INPUT:
    #       lat, Nx1 latitude (rad).
    # OUTPUT:
    #       RM, Nx1 meridian radius of curvature (North-South)(m).
    #       RN, Nx1 normal radius of curvature (East-West) (m).

    a = 6378137.0
    e = 0.0818191908426
    den = 1 - (e * np.sin(lat)) ** 2
    #  Meridian radius of curvature: radius of curvature for north-south motion.
    RM = a * (1 - e**2) / (den ** (3 / 2))
    #  Normal radius of curvature: radius of curvature for east-west motion. aka transverse radius.
    RN = a / np.sqrt(den)
    return RM, RN


def geocradius(lat):
    a = 6378137.0  # Semi-major axis in meters (equatorial radius)
    b = 6356752.314245  # Semi-minor axis in meters (polar radius)
    c, s = np.cos(lat), np.sin(lat)
    radius = np.sqrt((a**2 * c) ** 2 + (b**2 * s) ** 2) / np.sqrt(
        (a * c) ** 2 + (b * s) ** 2
    )
    return radius


def lla2ecef(pos):
    lat, lon, alt = np.squeeze(pos)
    a = 6378137
    e = 8.1819190842622e-2

    N = a / np.sqrt(1 - (e * np.sin(lat)) ** 2)

    x = (N + alt) * np.cos(lat) * np.cos(lon)
    y = (N + alt) * np.cos(lat) * np.sin(lon)
    z = ((1 - e**2) * N + alt) * np.sin(lat)
    xyz = np.squeeze([x, y, z]).tolist()
    return xyz


def ecef2lla(ecef):
    # ecef2llh:	converts from ECEF coordinates to navigation coordinates
    #  (latitude, longitude and height).
    #
    # INPUTS
    #   ecef: Nx3 ECEF coordinates [X Y Z] (m, m, m).
    #
    # OUTPUTS
    #   llh, Nx3 [latitude longitude height] (rad, rad, m)
    #
    # References:
    # H. Vermeille. Direct transformation from geocentric
    # coordinates to geodetic coordinates. Journal of Geodesy, 2002.
    #
    #  	R. Gonzalez, J. Giribet, and H. Patiño. NaveGo: a
    # simulation framework for low-cost integrated navigation systems,
    # Journal of Control Engineering and Applied Informatics}, vol. 17,
    # issue 2, pp. 110-120, 2015. Eq. 17.

    x, y, z = np.squeeze(ecef)

    # WGS84 model
    a = 6378137.0000  # Earth radius in meters
    b = 6356752.3142  # Earth semiminor in meters
    e = 0.0818191908426  # Eccentricity

    p = (x**2 + y**2) / a**2
    q = ((1 - e**2) * z**2) / a**2
    r = (p + q - e**4) / 6
    s = (e**4 * p * q) / (4 * r**3)
    t = (1 + s + np.sqrt(s * (2 + s))) ** (1 / 3)
    u = r * (1 + t + 1 / t)
    v = np.sqrt(u**2 + q * e**4)
    w = e**2 * (u + v - q) / (2 * v)
    k = np.sqrt(u + v + w**2) - w

    D = k * np.sqrt(x**2 + y**2) / (k + e**2)

    # Latitude
    # lat = 2 * np.arctan((z / (D+np.sqrt(D**2 + z**2))))
    lat = np.arctan2(z, D)

    # Longitude
    # lon = 2 * np.arctan((y / (x+np.sqrt(x**2 + y**2))))
    lon = np.arctan2(y, x)

    # Altitude
    h = (k + e**2 - 1) / k * (np.sqrt(D**2 + z**2))

    return [lat, lon, h]


# def gerocradious(lat):
#     RM, RN = radius(lat)
#     R0 = np.sqrt(RM * RN)
#     return R0


def earthrate(lat):
    # Turn earthrate of the Earth to the navigation frame
    # INPUT: lat, 1x1 latitude (rad).
    # OUTPUT: omega_ie_n, 3x3 skew-symmetric Earth rate matrix (rad/s).
    lat = np.squeeze(lat)
    temp = skew([np.cos(lat), 0, -np.sin(lat)])
    omega_ie_n = (7.2921155e-5) * temp
    return np.reshape(omega_ie_n, (3, 3))


def transportrate(lat, vn, ve, h):
    # Calculate the transport rate in the navigation frame.
    # INPUT:
    #       lat, 1x1 latitude (rad).
    #       Vn, 1x1 North velocity (m/s).
    #       Ve, 1x1 East velocity (m/s).
    #       h, altitude (m).
    # OUTPUT:
    #       omega_en_n, 3x3 skew-symmetric transport rate matrix (rad/s).
    h = np.abs(h)
    rm, rn = radius(lat)
    omega_en_n = np.zeros((3, 1), dtype=np.float32)
    omega_en_n[0, 0] = ve / (rn + h)  # North
    omega_en_n[1, 0] = -vn / (rm + h)  # East
    omega_en_n[2, 0] = (-ve * np.tan(lat)) / (rn + h)  # Down
    return skew(omega_en_n)


def euler_to_dcm(angles):
    # Convert a set of Euler angles to the corresponding coordinate transformation matrix
    phi, theta, psi = np.squeeze(angles)
    sin_phi, cos_phi = np.sin(phi), np.cos(phi)
    sin_theta, cos_theta = np.sin(theta), np.cos(theta)
    sin_psi, cos_psi = np.sin(psi), np.cos(psi)
    C = np.empty((3, 3), dtype=np.float32)
    C[0, 0] = cos_theta * cos_psi
    C[0, 1] = cos_theta * sin_psi
    C[0, 2] = -sin_theta
    C[1, 0] = -cos_phi * sin_psi + sin_phi * sin_theta * cos_psi
    C[1, 1] = cos_phi * cos_psi + sin_phi * sin_theta * sin_psi
    C[1, 2] = sin_phi * cos_theta
    C[2, 0] = sin_phi * sin_psi + cos_phi * sin_theta * cos_psi
    C[2, 1] = -sin_phi * cos_psi + cos_phi * sin_theta * sin_psi
    C[2, 2] = cos_phi * cos_theta
    return C


def qua_to_dcm(qua):
    # Transform quaternions to body-to-nav direction cosine matrix
    b, c, d, a = np.squeeze(qua)
    C = np.empty((3, 3), dtype=np.float32)
    C[0, 0] = a * a + b * b - c * c - d * d
    C[0, 1] = 2 * (b * c - a * d)
    C[0, 2] = 2 * (a * c + b * d)
    C[1, 0] = 2 * (a * d + b * c)
    C[1, 1] = a * a - b * b + c * c - d * d
    C[1, 2] = 2 * (c * d - a * b)
    C[2, 0] = 2 * (b * d - a * c)
    C[2, 1] = 2 * (c * d + a * b)
    C[2, 2] = a * a - b * b - c * c + d * d
    return C


def dcm_to_euler(C_b_n):
    # Convert from body-to-nav DCM to Euler angles.
    # INPUT:
    #       DCMbn, 3x3 body-to-nav DCM.
    # OUTPUT:
    #        euler, 3x1 Euler angles [roll pitch yaw] (rad, rad, rad).
    phi = np.arctan(C_b_n[2, 1] / C_b_n[2, 2])
    theta = np.arcsin(-C_b_n[2, 0])
    psi = np.arctan2(C_b_n[1, 0], C_b_n[0, 0])  # without -
    euler = [phi, theta, psi]
    return euler


def euler_to_qua(angles):
    # converts from euler angles to quaternions
    # Rearrange verctor for zyx rotation sequences
    temp = np.array(angles[::-1])

    c_eul = np.cos(temp / 2)
    s_eul = np.sin(temp / 2)
    # ZYX rotation sequence
    q0 = c_eul[0] * c_eul[1] * c_eul[2] + s_eul[0] * s_eul[1] * s_eul[2]
    q1 = c_eul[0] * c_eul[1] * s_eul[2] - s_eul[0] * s_eul[1] * c_eul[2]
    q2 = c_eul[0] * s_eul[1] * c_eul[2] + s_eul[0] * c_eul[1] * s_eul[2]
    q3 = s_eul[0] * c_eul[1] * c_eul[2] - c_eul[0] * s_eul[1] * s_eul[2]
    qua = np.array([q1, q2, q3, q0])
    qua /= np.linalg.norm(qua)

    return qua


def qua_to_euler(qua):
    # Transforms quaternion to Euler angles
    dcm = qua_to_dcm(qua)
    phi = np.arctan2(dcm[2, 1], dcm[2, 2])  # roll
    theta = np.arcsin(-dcm[2, 0])  # pitch
    psi = np.arctan2(dcm[1, 0], dcm[0, 0])  # yaw
    euler = [phi, theta, psi]
    return euler


def dcm_update(C_b_n, angles):
    S = skew(angles)
    magn = np.linalg.norm(angles)

    A = np.eye(3)
    if magn < 1e-8:
        A += S
    C_b_n = np.matmul(C_b_n, A)
    return C_b_n


def qua_update(qua, omega_b_n, dt):
    # update Quaternions
    # INPUT:
    #       qua,    4x1 quaternion.
    #       wb,     3x1 incremental turn rates in body-frame [X Y Z] (rad/s).
    #       dt,     1x1 IMU sampling interval (s).
    #
    # OUTPUT:
    #        qua,    4x1 updated quaternion.
    # References:
    #            Crassidis, J.L. and Junkins, J.L. (2011). Optimal Esti-
    #            mation of Dynamic Systems, 2nd Ed. Chapman and Hall/CRC, USA.
    #            Eq. 7.39 to 7.41, p. 458.
    omega_norm = np.linalg.norm(omega_b_n)
    if omega_norm < 1e-8:
        return qua
    s = np.sin(0.5 * omega_norm * dt)
    # Eq. 7.41
    qw1, qw2, qw3 = np.squeeze(omega_b_n / omega_norm) * s
    qw4 = np.cos(0.5 * omega_norm * dt)

    Om = [
        [qw4, qw3, -qw2, qw1],
        [-qw3, qw4, qw1, qw2],
        [qw2, -qw1, qw4, qw3],
        [-qw1, -qw2, -qw3, qw4],
    ]
    # print(f"psi={omega_b_n.shape}")
    # Eq. 7.40
    # Om1 = np.concatenate((c * np.eye(3)-skew(psi), psi.reshape(3, 1)), axis=1)
    # Om2 = np.hstack((-psi, c))
    # Om = np.vstack((Om1, Om2))
    # Eq. 7.39
    # print(Om.shape)
    q = np.matmul(Om, qua)
    q = q / np.linalg.norm(q)

    return q


def initialize_att(f):
    # initialize attitude using accelerometer forces
    fx_0, fy_0, fz_0 = f[:, 0] / -9.81
    Roll0 = np.arctan2(-fy_0, -fz_0)
    Pitch0 = np.arctan(fx_0 / np.sqrt(fy_0**2 + fz_0**2))
    # yaw = np.arctan(fy_0/fx_0)
    # print(yaw)
    att = np.radians([Roll0, Pitch0, 1.79])
    return att


def get_fusion_att(fusion_att, gnss_t1, gnss_t2=None):
    # fusion_att: a dict with keys > (['fuse_time', 'RollF', 'PitchF', 'YawF'])
    # gnss_t1 and gnss_t2 : start and end time for INS
    t = np.squeeze(fusion_att["fuse_time"]) / 1e6
    idx1 = np.argmin(np.abs(t - gnss_t1))
    idx2 = np.argmin(np.abs(t - gnss_t2)) if gnss_t2 is not None else idx1 + 1
    att = np.hstack([fusion_att["RollF"], fusion_att["PitchF"], fusion_att["YawF"]])
    att = np.radians(att[idx1:idx2, :]).squeeze()
    t = t[idx1:idx2]
    return t, att


def ins_gnss_qua_update(qua_n, x):
    # Inputs:
    #       qua: Previous system att. in quaternion format
    #       x:   es_EKF attitude (xp[:3])
    # Outputs:
    #       Corrected system attitude in quaternion format
    # Crassidis, Eq. A.174a

    antm = np.array(
        [[0, qua_n[2], -qua_n[1]], [-qua_n[2], 0, qua_n[0]], [qua_n[1], -qua_n[0], 0]]
    )

    # Compute quaternion update
    eye3 = np.eye(3)
    qua = (
        qua_n
        + 0.5
        * np.concatenate(
            (qua_n[3] * eye3 + antm, -np.array([[qua_n[0], qua_n[1], qua_n[2]]]))
        )
        @ x[:3]
    )

    # Crassidis, Eq. 7.34
    qua = qua / np.linalg.norm(qua)  # Brute-force normalization
    return qua


def pos_vel_mse(gnss_pos_ecef, gnss_vel, ins_pos_ecef, ins_vel):
    v_err = np.sqrt(np.mean(np.subtract(ins_vel, gnss_vel) ** 2))
    p_err = np.sqrt(np.mean(np.subtract(ins_pos_ecef, gnss_pos_ecef) ** 2))
    xy_err = np.sqrt(np.mean(np.subtract(ins_pos_ecef[:2], gnss_pos_ecef[:2]) ** 2))
    return p_err, xy_err, v_err


def approx_pos_vel_mse(gnss_pos, gnss_vel, ins_pos, ins_vel):
    v_err = np.sqrt(np.mean(np.subtract(ins_vel, gnss_vel) ** 2))
    lat_diff_m = (
        gnss_pos[0] - ins_pos[0]
    ) * 111000  # 1 degree latitude ≈ 111,000 meters
    lon_diff_m = (
        (gnss_pos[1] - ins_pos[1]) * 111000 * np.cos(np.radians(gnss_pos[0]))
    )  # 1 degree longitude ≈ 111,000 meters * cos(latitude)
    alt_diff_m = abs(gnss_pos[2] - ins_pos[2])
    # p_err = np.sqrt(np.mean(np.subtract(ins_pos_ecef, gnss_pos_ecef)**2))
    p_err = np.sqrt(np.mean(np.square([lat_diff_m, lon_diff_m, alt_diff_m])))
    xy_err = np.sqrt(np.mean(np.square([lat_diff_m, lon_diff_m])))
    return p_err, xy_err, alt_diff_m, v_err


# def pos_vel_mae(gnss_pos_ecef, gnss_vel, ins_pos_ecef, ins_vel):
#     v_err = np.absolute(np.subtract(ins_vel, gnss_vel))
#     p_err = np.absolute(np.subtract(ins_pos_ecef, gnss_pos_ecef))
#     return p_err, v_err
