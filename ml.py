import yfinance as yf
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import EnsembleVoteClassifier
import numpy as np
import ta


def get_stock_history(ticker):
    stock = yf.Ticker(ticker)

    #data = dat.history(period="60d", interval="5m")
    data = stock.history(period="252d")
    data = data.drop(columns=["Dividends", "Stock Splits"])

    return data

def get_overnight_percent_change():
    data = get_stock_history("TQQQ")
    data = data.drop(columns=["Capital Gains"])
    data["Percent Change"] = (data["Close"] - data["Open"]) / data["Open"]
    data["Overnight Percent Change"] = (data["Open"] - data["Close"].shift(1)) / data["Close"].shift(1)

    # volume the day before
    data["Volume"] = data["Volume"].shift(1)

    data = ta.add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume", fillna=True)

    del data['others_dr']
    del data['others_dlr']

    print(data)
    
    print(data)
    print("done")
    input()
    


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

X = data.copy()

y = np.where(data["Percent Change"] > 0, 1, -1)
del X['Percent Change']

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)


forest = RandomForestClassifier(n_estimators=250,
                                random_state=0)

forest.fit(X_train, y_train)

print('Training accuracy:', np.mean(forest.predict(X_train) == y_train)*100)
print('Test accuracy:', np.mean(forest.predict(X_test) == y_test)*100)

importance_vals = forest.feature_importances_
print(importance_vals)

std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importance_vals)[::-1]

# Plot the feature importances of the forest
plt.figure()
plt.title("Random Forest feature importance")
plt.bar(range(X.shape[1]), importance_vals[indices],
        yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.ylim([0, 0.5])
#plt.show()

feature_names = X.columns

# make a sorted list of the feature importances
sorted_importances = sorted(zip(importance_vals, feature_names), reverse=True)
print("Feature importances:")
for importance, name in sorted_importances:
    print(f"{name}: {importance:.4f}")

# use the top features
top_features = [name for importance, name in sorted_importances[:10]]

# write the top features to disk
with open('top_features.txt', 'w') as f:
    for feature in top_features:
        f.write(feature + '\n')


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

# print buy and hold strategy percent
buy_price = data["Open"].iloc[0]
sell_price = data["Close"].iloc[len(data)-1]
print("Stock Percent Change: ", (sell_price - buy_price) / buy_price)
print()

X = X[top_features]

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
    print("predicted numbers")
    print(pred)
    
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