import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.cross_decomposition import PLSRegression
import numpy as np

# Load the data
df1 = pd.read_excel(r"C:\Users\86185\Desktop\芒果干物质\train.xlsx", header=1)
df2 = pd.read_excel(r"C:\Users\86185\Desktop\芒果干物质\test.xlsx", header=1)
df3 = pd.read_excel(r"C:\Users\86185\Desktop\芒果干物质\val.xlsx", header=1)

# df1 = pd.read_excel(r"C:\Users\86185\Desktop\小麦-248条\ManufacturerA\A3\Cal_ManufacturerA.xlsx", header=1)
# df2 = pd.read_excel(r"C:\Users\86185\Desktop\小麦-248条\ManufacturerA\A3\Test_ManufacturerA.xlsx", header=1)
# df3 = pd.read_excel(r"C:\Users\86185\Desktop\小麦-248条\ManufacturerA\A3\Val_ManufacturerA.xlsx", header=1)

# df1 = pd.read_excel(r"C:\Users\86185\Desktop\小麦-248条\ManufacturerB\B1\Cal_ManufacturerB.xlsx", header=1)
# df2 = pd.read_excel(r"C:\Users\86185\Desktop\小麦-248条\ManufacturerB\B1\Test_ManufacturerB.xlsx", header=1)
# df3 = pd.read_excel(r"C:\Users\86185\Desktop\小麦-248条\ManufacturerB\B1\Val_ManufacturerB.xlsx", header=1)

# df1 = pd.read_excel(r"C:\Users\86185\Desktop\小麦-248条\ManufacturerC\Cal_ManufacturerC.xlsx", header=1)
# df2 = pd.read_excel(r"C:\Users\86185\Desktop\小麦-248条\ManufacturerC\Test_ManufacturerC.xlsx", header=1)
# df3 = pd.read_excel(r"C:\Users\86185\Desktop\小麦-248条\ManufacturerC\Val_ManufacturerC.xlsx", header=1)

# df1 = pd.read_excel(r"C:\Users\86185\Desktop\芒果干物质\竞赛数据2023\train.xlsx", header=1)
# df2 = pd.read_excel(r"C:\Users\86185\Desktop\芒果干物质\竞赛数据2023\test.xlsx", header=1)
# df3 = pd.read_excel(r"C:\Users\86185\Desktop\芒果干物质\竞赛数据2023\val.xlsx", header=1)

# Extract features and labels
data_train = df1.iloc[:, 2:].values
label_train = df1.iloc[:, 1].values
data_test = df2.iloc[:, 2:].values
data_test1 = df2.iloc[:, 2:].values
label_test = df2.iloc[:, 1].values
data_val = df3.iloc[:, 2:].values
label_val = df3.iloc[:, 1].values


def normalize(data):
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

data_train = normalize(data_train)
data_test = normalize(data_test)
data_val = normalize(data_val)
label_train = normalize(label_train)
label_test = normalize(label_test)
label_val = normalize(label_val)

# # # 初始化参数
# # max_components = 4  # 最大主成分数量
# # components_range = range(1, max_components + 1)
# # cv_scores = []
# #
# # # 进行交叉验证
# # for n_components in components_range:
# #     pls = PLSRegression(n_components=n_components)
# #     scores = cross_val_score(pls, data_train, data_train, cv=5, scoring='neg_mean_squared_error')
# #     cv_scores.append(-np.mean(scores))  # 负均方误差转为正值
# #
# # # 选择最佳主成分数量
# # best_n_components = components_range[np.argmin(cv_scores)]
# # print(f"最佳主成分数量: {best_n_components}")

# 初始化模型
n_components = 1
pls = PLSRegression(n_components=n_components)


train_losses = []
test_losses = []
val_losses = []

pls.fit(data_train, label_train)

train_predictions = pls.predict(data_train)
train_loss = mean_squared_error(label_train, train_predictions)
train_losses.append(train_loss)

val_predictions = pls.predict(data_val)
val_loss = mean_squared_error(label_val, val_predictions)
val_losses.append(val_loss)

test_predictions = pls.predict(data_test)
test_loss = mean_squared_error(label_test, test_predictions)
test_losses.append(test_loss)

print(f'Train MSE: {train_loss:.4f}, Val MSE: {val_loss:.4f}, Test MSE: {test_loss:.4f}')

plt.figure()  # 预测散点图
data_test = np.array(data_test).flatten()
# 生成对应的 X 坐标
x_coords = np.arange(len(test_predictions))
plt.scatter(x_coords,test_predictions, s=10, alpha=0.5, label='Predicted')
plt.scatter(x_coords,label_test, s=10, alpha=0.5, color='red',label='Label')
plt.legend()
plt.ylabel('Values')
plt.xlabel('Mango DMC')
plt.title(f'PLS')
plt.grid(True)
plt.savefig('PLS_MangoDMC5')

# plt.plot(train_losses, label='Train MSE')
# plt.plot( val_losses, label='Val MSE')
# plt.plot(test_losses, label='Test MSE')
# plt.xlabel('Epoch')
# plt.ylabel('MSE')
# plt.legend()
# plt.savefig('wheatC_PLS_MSE_plot.png')





# import numpy as np
# # import matplotlib.pyplot as plt
# from sklearn.cross_decomposition import PLSRegression
# from sklearn.model_selection import train_test_split
# from sklearn.decomposition import PCA
#
#
#
# # 划分数据集为训练集和测试集
# X_train = data_train
# X_test = data_test
# Y_train = label_train
# Y_test = label_test
#
#
# # 计算每个潜在变量解释的方差比例
# def plot_explained_variance(X_train, max_components):
#     # PLS回归模型
#     pls = PLSRegression()
#
#     # 存储每个潜在变量的解释方差
#     explained_variances = []
#
#     # 遍历不同的潜在变量数量
#     for n_components in range(1, max_components + 1):
#         pls.n_components = n_components
#         pls.fit(X_train, Y_train)
#
#         # 通过PCA计算每个成分的解释方差
#         pca = PCA(n_components=n_components)
#         X_train_pca = pca.fit_transform(X_train)
#         explained_variance = np.cumsum(pca.explained_variance_ratio_)
#
#         explained_variances.append(explained_variance[-1])
#
#     # 绘制方差解释图
#     plt.plot(range(1, max_components + 1), explained_variances, marker='o')
#     plt.xlabel('Number of Components')
#     plt.ylabel('Cumulative Explained Variance')
#     plt.title('Explained Variance by Number of Components')
#     plt.grid()
#     plt.savefig('PLS_plot_wheatA.png')
#     plt.show()
#
#
# # 绘制解释方差图
# plot_explained_variance(X_train, max_components=10)



# Example: Assuming you have your data in X and y#
# # import numpy as np
# # from numpy import dot, outer, concatenate, mean, tile, zeros, ones
# # from numpy.linalg import inv, norm
# #
# # # Data Preprocessing Function
# # def pretreat(X):
# #     Xs = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
# #     Xp1 = np.mean(X, axis=0)
# #     Xp2 = np.std(X, axis=0)
# #     return Xs, Xp1, Xp2
# #
# # # PLS Model Training (NIPALS algorithm)
# # def pls1_nipals(X, y, a):
# #     T = zeros((X.shape[0], a))
# #     P = zeros((X.shape[1], a))
# #     Q = zeros((1, a))
# #     W = zeros((X.shape[1], a))
# #     for i in range(a):
# #         v = dot(X.T, y[:, 0])
# #         W[:, i] = v / norm(v)
# #         T[:, i] = dot(X, W[:, i])
# #         P[:, i] = dot(X.T, T[:, i]) / dot(T[:, i].T, T[:, i])
# #         Q[0, i] = dot(T[:, i].T, y[:, 0]) / dot(T[:, i].T, T[:, i])
# #         X = X - outer(T[:, i], P[:, i])
# #     W = dot(W, inv(dot(P.T, W)))
# #     B = dot(W[:, 0:a], Q[:, 0:a].T)
# #     return {'B': B, 'T': T, 'P': P, 'Q': Q, 'W': W}
# #
# # # PLS Cross-Validation
# # def plscvfold(X, y, A, K):
# #     sort_index = np.argsort(y, axis=0)
# #     y = np.sort(y, axis=0)
# #     X = X[sort_index[:, 0]]
# #     M = X.shape[0]
# #     yytest = zeros([M, 1])
# #     YR = zeros([M, A])
# #     groups = np.asarray([i % K + 1 for i in range(0, M)])
# #     group = np.arange(1, K + 1)
# #     for i in group:
# #         Xtest = X[groups == i]
# #         ytest = y[groups == i]
# #         Xcal = X[groups != i]
# #         ycal = y[groups != i]
# #         index_Xtest = np.nonzero(groups == i)
# #         index_Xcal = np.nonzero(groups != i)
# #
# #         (Xs, Xp1, Xp2) = pretreat(Xcal)
# #         (ys, yp1, yp2) = pretreat(ycal)
# #         PLS1 = pls1_nipals(Xs, ys, A)
# #         W, T, P, Q = PLS1['W'], PLS1['T'], PLS1['P'], PLS1['Q']
# #         yp = zeros([ytest.shape[0], A])
# #         for j in range(1, A + 1):
# #             B = dot(W[:, 0:j], Q.T[0:j])
# #             C = dot(B, yp2) / Xp2
# #             coef = concatenate((C, yp1 - dot(C.T, Xp1)), axis=0)
# #             Xteste = concatenate((Xtest, ones([Xtest.shape[0], 1])), axis=1)
# #             ypred = dot(Xteste, coef)
# #             yp[:, j - 1:j] = ypred
# #
# #         YR[index_Xtest, :] = yp
# #         yytest[index_Xtest, :] = ytest
# #         print(f"The {i}th group finished")
# #
# #     error = YR - tile(y, A)
# #     errs = error * error
# #     PRESS = np.sum(errs, axis=0)
# #     RMSECV_ALL = np.sqrt(PRESS / M)
# #     index_A = np.nonzero(RMSECV_ALL == min(RMSECV_ALL))
# #     RMSECV_MIN = min(RMSECV_ALL)
# #     SST = np.sum((yytest - mean(y)) ** 2)
# #     Q2_all = 1 - PRESS / SST
# #     return {'index_A': index_A[0][0] + 1, 'RMSECV_ALL': RMSECV_ALL, 'Q2_all': Q2_all}
# #
# # # PLS Prediction
# # def plspredtest(B, Xtest, xp1, xp2, yp1, yp2):
# #     C = dot(B, yp2) / xp2
# #     coef = concatenate((C, yp1 - dot(C.T, xp1)), axis=0)
# #     Xteste = concatenate((Xtest, ones([Xtest.shape[0], 1])), axis=1)
# #     ypred = dot(Xteste, coef)
# #     return ypred
# #
# # # RMSEP Calculation
# # def RMSEP(ypred, Ytest):
# #     error = ypred - Ytest
# #     errs = error ** 2
# #     PRESS = np.sum(errs)
# #     RMSEP = np.sqrt(PRESS / Ytest.shape[0])
# #     SST = np.sum((Ytest - np.mean(Ytest)) ** 2)
# #     Q2 = 1 - PRESS / SST
# #     return RMSEP, Q2
# #
# # # Usage with your data
# # def pls_model(X, y, A=5, K=5):
# #     # Cross-validation
# #     cv_results = plscvfold(X, y, A, K)
# #     optimal_A = cv_results['index_A']
# #     print(f"Optimal number of components: {optimal_A}")
# #
# #     # Train final PLS model with optimal number of components
# #     (Xs, Xp1, Xp2) = pretreat(X)
# #     (ys, yp1, yp2) = pretreat(y)
# #     PLS_model = pls1_nipals(Xs, ys, optimal_A)
# #
# #     return PLS_model
# X = np.array(...)  # Feature matrix
# y = np.array(...)  # Target variable

# model = pls_model(X, y)
