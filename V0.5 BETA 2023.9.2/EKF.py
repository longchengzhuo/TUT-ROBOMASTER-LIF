import numpy as np
from utils_for_EKF import *


initial_state = np.array([0, 0, 0, 0, 0, 0])
initial_covariance = np.identity(6)
Q = block_diag(np.identity(3) * Q_xyz_scale, np.identity(3) * Q_vxyz_scale)  # Q 表示预测状态的高斯噪声的协方差阵,是预测误差的组成部分之一
R = np.identity(3) * R_scale  # 测量误差
ekf = EKF(Q, R, initial_state, initial_covariance)



def main():
    mear = [[0, 2, 0]]
    for i in range(200):
        mear = np.append(mear, [[int(i), 2, int(i)]], axis=0)
    for r in mear:
        # ekf.predict(dt)
        # ekf.update(np.array(r))
        x,y,z = Kal_predict(ekf, r, dt)
        print("pridict", x,y,z)
        # print("State:", ekf.state)

main()
