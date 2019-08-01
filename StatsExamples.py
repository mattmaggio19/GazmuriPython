import numpy as np
from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt
import SA1DataLoader
import statsmodels.api as sm
from statsmodels.formula.api import ols
from scipy import stats

def Type1ErrorRateSim():
    # doing to understand a bit better, why lots of tests inflates the error rate.
    simNum, Mu, Sigma, N, alpha, NGroups = 1000, 20, 2, 40, 0.05, 3

    PosResults, NegResults, fStat, pValues, RejectNull = [], [], [], [], []

    for i in range(simNum):
        data1 = np.random.normal(loc=Mu, scale=Sigma, size=N) #Data is drawn from the same pop, each sample is independant
        data2 = np.random.normal(loc=Mu, scale=Sigma, size=N)
        data3 = np.random.normal(loc=Mu, scale=Sigma, size=N)
        data4 = np.random.normal(loc=Mu, scale=Sigma, size=N)
        (F, p) = stats.f_oneway(data1, data2, data3, data4)
        fStat.append(F)
        pValues.append(p)
        if p >= alpha: #Accept the null hypothesis.
            PosResults.append(0)
            NegResults.append(1)
        else:
            PosResults.append(1)
            NegResults.append(0)
        RejectNull.append(sum(PosResults)/(sum(PosResults)+sum(NegResults)))
    # print(pValues)
    plt.plot(RejectNull)
    plt.show()



def SimpleLinearRegression(x= None,y = None):
    # x = np.array([5, 15, 25, 35, 45, 55]).reshape((-1, 1))
    # y = np.array([5, 20, 14, 32, 22, 38])
    if x is None and y is None:
        x = np.arange(0, 10, 0.1).reshape(-1, 1)
        noise = np.random.normal(0, 2.0, len(x)).reshape(-1, 1)
        y = 2 * x + 15 + noise
    else:
        pass

    model = LinearRegression()
    model.fit(x, y)
    r_sq = model.score(x, y)
    print('coefficient of determination, R^2:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)
    y_pred = model.predict(x)

    ax1 = plt.subplot(1, 2, 1)
    plt.title('simple linear regression')
    plt.scatter(x,y)
    plt.plot(x,y_pred)
    plt.subplot(1, 2, 2)
    plt.title('Residuals')
    plt.scatter(x, y-y_pred)
    plt.show()


if __name__ == '__main__':

    Dataset = SA1DataLoader.StandardLoadingFunction(useCashe= True)
    Ao = SA1DataLoader.selectData(Dataset)
    # SimpleLinearRegression(x = Dataset[15]['Time'], y = Ao['POV'] )
    Type1ErrorRateSim()