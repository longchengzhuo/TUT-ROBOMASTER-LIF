from scipy.linalg import block_diag
import yaml
import numpy as np
# import CRC
with open("config.yaml", 'r', encoding='utf-8') as f:
    yaml = yaml.safe_load(f)

dt = yaml["dt"]
Q_xyz_scale = yaml["Q_xyz_scale"]
Q_vxyz_scale = yaml["Q_vxyz_scale"]
R_scale = yaml["R_scale"]


def f(state, dt):
    x, y, z, vx, vy, vz = state
    return np.array([x + vx * dt, y + vy * dt, z + vz * dt, vx, vy, vz])


def h(state):
    x, y, z, _, _, _ = state
    return np.array([x, y, z])


class EKF:
    def __init__(self, Q, R, initial_state, initial_covariance):
        self.Q = Q
        self.R = R
        self.state = initial_state
        self.covariance = initial_covariance

    def predict(self, dt):
        F = np.array([[1, 0, 0, dt, 0, 0],
                      [0, 1, 0, 0, dt, 0],
                      [0, 0, 1, 0, 0, dt],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0, 1]])

        self.state = f(self.state, dt)
        self.covariance = F @ self.covariance @ F.T + self.Q

    def update(self, r):
        H = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0]])

        y = r - h(self.state) #y是测量值和真实值之差
        S = H @ self.covariance @ H.T + self.R
        K = self.covariance @ H.T @ np.linalg.inv(S)
        self.state += K @ y
        self.covariance = (np.identity(6) - K @ H) @ self.covariance


def Kal_predict(ekf, r, dt):
    ekf.predict(dt)
    ekf.update(np.array(r))
    T = 0.05  #总误差时间，单位 秒 记得算上子弹飞行时间
    x, y, z = float(ekf.state[0] + T * ekf.state[3]), \
              float(ekf.state[1] + T * ekf.state[4]), \
              float(ekf.state[2] + T * ekf.state[5])
    return x, y, z


def coor_to_yaw_pitch(predicted_X, predicted_Y, predicted_Z):
    yaw = np.arctan2(predicted_X, predicted_Z)
    pitch = np.arctan2(predicted_Y, np.sqrt(predicted_X * predicted_X + predicted_Z * predicted_Z))
    return yaw, pitch


def Serial_communication(yaw, pitch, fps, is_autoaim):
    if is_autoaim == 1:
        f1 = bytes("$", encoding='utf8')
        f2 = 10
        f3 = float(yaw)
        f4 = float(pitch)
        f5 = fps
    else:
        f1 = bytes("$", encoding='utf8')
        f2 = 10
        f3 = 0
        f4 = 0
        f5 = 1
    pch_Message1 = get_Bytes(f1, is_datalen_or_fps=0)
    pch_Message2 = get_Bytes(f2, is_datalen_or_fps=1)
    pch_Message3 = get_Bytes(f3, is_datalen_or_fps=0)
    pch_Message4 = get_Bytes(f4, is_datalen_or_fps=0)
    pch_Message5 = get_Bytes(f5, is_datalen_or_fps=2)
    pch_Message = pch_Message1 + pch_Message2 + pch_Message3 + pch_Message4 + pch_Message5

    wCRC = get_CRC16_check_sum(pch_Message, CRC16_INIT)
    # ser.write(struct.pack("=cBffHi", f1, f2, f3, f4, f5, wCRC))  #分别是帧头，长度，数据，数据，fps，校验
    # print("wCRC", hex(wCRC))
    # print("串口玄学--------------------", struct.pack("=cBffHH", f1, f2, f3, f4, f5, wCRC))
    # print(struct.pack("=H", wCRC))