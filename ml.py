import yfinance as yf
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
import numpy as np
import ta_py as ta


def get_stock_history(ticker):
    dat = yf.Ticker(ticker)

    #data = dat.history(period="60d", interval="5m")
    data = dat.history(period="252d")
    data = data.drop(columns=["Dividends", "Stock Splits"])

    return data

def get_overnight_percent_change():
    data = get_stock_history("TQQQ")
    data = data.drop(columns=["Capital Gains"])
    data["Percent Change"] = (data["Close"] - data["Open"]) / data["Open"]
    data["Overnight Percent Change"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)

    # volume the day before
    data["Volume"] = data["Volume"].shift(1)

    #data.columns = ['open', 'high', 'low', 'close', 'volume', 'percent change', 'overnight percent change']
    
    #data = data.reset_index()
    #print(data)
    
    #data['KAMA'] = ta.kama(data) # add Kaufman's Adaptive Moving Average
    #data['MOM'] = ta.mom(data)

    # normalize volume column
    #data["Volume"] = (data["Volume"] - data["Volume"].mean()) / data["Volume"].std()
    #data["Volume"] = (data["Volume"] - data["Volume"].min()) / (data["Volume"].max() - data["Volume"].min())  # Min-Max normalization

    
    data = data.dropna()
    print(data)

    return data


data = get_overnight_percent_change()

clf1 = LogisticRegression(random_state=1)
clf2 = RandomForestClassifier(random_state=1)
clf3 = GaussianNB()
eclf = EnsembleVoteClassifier(clfs=[clf1, clf2, clf3], weights=[1,1,1])

X = data[["Overnight Percent Change","Volume"]]
y = np.where(data["Percent Change"] > 0, 1, -1)


'''
labels = ['Logistic Regression', 'Random Forest', 'Naive Bayes']

for clf, label in zip([clf1, clf2, clf3], labels):
    
    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=5, 
                                              scoring='accuracy')
    print("Accuracy: %0.2f (+/- %0.2f) [%s]"
          % (scores.mean(), scores.std(), label))
'''

labels = ['Logistic_Regression', 'Random_Forest', 'Naive_Bayes', 'Ensemble']

buy_price = data["Open"].iloc[0]
sell_price = data["Close"].iloc[len(data)-1]

# print buy and hold strategy percent
print("Stock Percent Change: ", (sell_price - buy_price) / buy_price)
print()

for clf, label in zip([clf1, clf2, clf3, eclf], labels):

    scores = model_selection.cross_val_score(clf, X, y, 
                                              cv=5, 
                                              scoring='accuracy')
    print("Test Accuracy: %0.2f (+/- %0.2f) [%s]" 
          % (scores.mean(), scores.std(), label))
    
    # split data into training and testing sets
    #X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.4, random_state=42)
    # split data in half to create train and test sets
    X_train = X.iloc[:int(len(X)/2)]
    X_test = X.iloc[int(len(X)/2):]
    y_train = y[:int(len(y)/2)]
    y_test = y[int(len(y)/2):]

    # fit model
    clf.fit(X_train, y_train)

    # pickle the model
    import pickle
    filename = label+'_model.sav'
    pickle.dump(clf, open('./models/'+filename, 'wb'))

    # predict bins
    pred = clf.predict(X_test)
    output_df = X_test.copy()
    output_df["Actual"] = y_test
    output_df["Predicted"] = pred
    # add column of percent change
    output_df["Percent Change"] = data["Percent Change"]
    #print(output_df)
    
    # determine accuracy of predictions
    correct = 0
    for i in range(len(pred)):
        if pred[i] == y_test[i]:
            correct += 1
    accuracy = correct / len(pred)
    print("Prediction Accuracy: ", accuracy)
    

    # simulate buying at open and selling at clcose if prediction is 1, selling at open and buying at close if prediction is -1
    money = 42000
    for i in range(len(pred)):
        if pred[i] == 1:
            money *= (1 + data["Percent Change"].iloc[i])
        elif pred[i] == -1:
            money *= (1 - data["Percent Change"].iloc[i])
    print("Money: ", money)
    # print money percent change
    print("Money Percent Change: ", (money - 42000) / 42000)    
    input()