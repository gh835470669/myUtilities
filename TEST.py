# import numpy as np
# # import matplotlib.pyplot as plt
# #
# # # 生成 0，1 之间等距的 N 个 数
# # N = 10
# # x_tr = np.linspace(0, 1, N)
# # print("x_tr")
# # print(x_tr)
# # # 计算 t
# # t_tr = np.sin(2 * np.pi * x_tr) + 0.25 * np.random.randn(N)
# # t_tr = np.array([-0.05398935,  0.71200393,  0.83381408,  1.0895105,  -0.05698175, -0.45617091,
# #  -0.90464125, -0.82139883, -0.52081802,  0.07393297])
# # print("t_tr")
# # print(t_tr)
# #
# #
# # def phi(x, M):
# #     return x[:,None] ** np.arange(M + 1)
# #
# # # 加正则项的解
# # M = 9
# # lam = 0.0001
# #
# # phi_x_tr = phi(x_tr, M)
# # print(np.arange(M + 1))
# # print("phi")
# # print(phi_x_tr.shape)
# # print(phi_x_tr)
# # S_0 = phi_x_tr.T.dot(phi_x_tr) + lam * np.eye(M+1)
# # print("S")
# # print(S_0.shape)
# # print(S_0)
# # y_0 = t_tr.dot(phi_x_tr)
# # print("y")
# # print(y_0)
# # print(y_0.shape)
# #
# # coeff = np.linalg.solve(S_0, y_0)[::-1]
# # print(coeff)
# #
# # f = np.poly1d(coeff)
# # print(f)
# #
# # # 绘图
# #
# # xx = np.linspace(0, 1, 500)
# #
# # fig, ax = plt.subplots()
# # ax.plot(x_tr, t_tr, 'co')
# # ax.plot(xx, np.sin(2 * np.pi * xx), 'g')
# # ax.plot(xx, f(xx), 'r')
# # ax.set_xlim(-0.1, 1.1)
# # ax.set_ylim(-1.5, 1.5)
# # ax.set_xticks([0, 1])
# # ax.set_yticks([-1, 0, 1])
# # ax.set_xlabel("$x$", fontsize="x-large")
# # ax.set_ylabel("$t$", fontsize="x-large")
# #
# # plt.show()


print("Hello")