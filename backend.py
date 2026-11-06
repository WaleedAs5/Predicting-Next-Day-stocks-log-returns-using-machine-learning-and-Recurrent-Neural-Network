import pandas as pd
import numpy as np
import yfinance as y
import tensorflow as tf


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import RandomForestRegressor

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, Input
from tensorflow.keras.callbacks import EarlyStopping


np.random.seed(42)
tf.random.set_seed(42)


###################
stocks = [ "AAPL", "GOOGL", "NVDA", "TSLA"]

#ٌRaw columns
RawColumns = {"Date","Open","Close","High","Low","Volume"}
#add Stock name to it
storedColumns =  ["Stock", "Date", "Open", "High", "Low", "Close", "Volume"]


#returns features
returns =  [
    "dailyReturns",
    "lagDailyReturn1","lagDailyReturn2","lagDailyReturn3","lagDailyReturn5"
]



technicalindcitaors = [
    "SMA5day","EMA5day","momentum5day","volatility5day","SMA20day","EMA20day","momentum20day", "volatility20day","RSI"
]


volume = [
    "volumeChanges", "relativeVolume"
]


OpenHLC = [
    "dailyRange", "openToCloseReturn", "nightGap"
]


marketindicators = [
    "QQQDailyReturn", "SPYDailyReturn","marketVolatility","QQQReturnLag1", "QQQReturnLag2","SPYReturnLag1", "SPYReturnLag2"
]

stockFeatures = {

    "AAPL": returns,
    "TSLA": returns + technicalindcitaors + volume + OpenHLC,

    "NVDA": returns + technicalindcitaors + volume,

    "GOOGL": returns + technicalindcitaors + volume + OpenHLC + marketindicators
}

#best RNN parameters after tuning
RNNParameters = {
    "AAPL": {"sequenceDays": 20, "rnnUnits": 32, "learnRate": 0.0005},
    "GOOGL": {"sequenceDays": 10, "rnnUnits": 8, "learnRate": 0.001},
    "NVDA": {"sequenceDays": 20, "rnnUnits": 8, "learnRate": 0.001},
    "TSLA": {"sequenceDays": 20, "rnnUnits": 8, "learnRate": 0.001}
}

rfParameters = {
    "AAPL": {"depth": 2, "features": 0.5, "leaf": 60, "trees": 800},
    "GOOGL": {"depth": 2, "features": 0.5, "leaf": 40, "trees": 200},
    "NVDA": {"depth": 2, "features": 0.5, "leaf": 10, "trees": 200},
    "TSLA": {"depth": 2, "features": 0.75, "leaf": 40, "trees": 200}
}


#target
target = "nextDayReturns"




#check if the data has a missing column
def ValidateData(df,Stock = ""):

    Missing = RawColumns - set(df.columns)
    if Missing:
        raise ValueError(f" {Stock} is missing the columns: {Missing} ")
    
    



def downloadStockData(stock, From, To, Filepath):
    df = y.download(
        stock,
        start = From,
        end = To,
        interval="1d"
    )
    #check if the dataframe is downloaded correctly
    if df.empty:
        raise ValueError(f"the dataframe of stock {stock} is empty")
    
        #drop the extra columns, from ('Open', 'AAPL') To just ('Open')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
  
        #Make the date as normal column and create an integer index for each trading day
    df = df.reset_index()
     #sort values based on dates ascendingly (oldest first)
    df = df.sort_values("Date") 
    ValidateData(df,stock)
        
            

    #add the stock name in the first column
    df.insert(0,"Stock",stock)
    #keep only the colummns in storedColumns and save 
    df = df[storedColumns]
    df.to_csv(Filepath, index = False)
    return df 

#############
def downloadMultipleStock(s,From,To):

    stockData = {}
    #download the required stocks data(AAPL,NVDA,TSLA)
    for stk in s:
        path = f"{stk}.csv"
        #download each stock data and add it to the dictonary
        df = downloadStockData(stk, From, To, path)
        stockData[stk] = df
    return stockData

#download a stock and calculate its  (QQQ,SPY) returns for each day
def GetMarketReturns(stock, columnName, From, To):

    df = y.download(
        stock,
        start=From,
        end=To,
        interval="1d"
    )
    #drop the extra columns, from ('Open', 'AAPL') To just ('Open')
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    #create an index
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    df[columnName] = ( df["Close"] / df["Close"].shift(1) )

    return df[["Date", columnName]]

def getMarketFeatures(From, To):

   
    #get market daily returns
    spy = GetMarketReturns( "SPY", "SPYDailyReturn", From, To)
    qqq = GetMarketReturns( "QQQ", "QQQDailyReturn", From, To)
    #merge into one dataframe
    marketInformation = pd.merge(spy, qqq, on="Date")
    #calculate the market volatilty with 20 day window
    marketInformation["marketVolatility"] =( marketInformation["SPYDailyReturn"].rolling(20).std()* np.sqrt(252))

    return marketInformation


#read the downloaded data
def load_data(path, stock):

    df = pd.read_csv(path)
    ValidateData(df,stock)

    return df

def clean(df):
    output = df.copy()
    #convert dates into pandas date/time
    output["Date"] = pd.to_datetime(errors="coerce" , arg = output["Date"])
    col = ["High","Low","Close","Open","Volume"]
    
    i = 0
    while i < len(col): 
        tmp = col[i]
        #convert from string to numeric value
        output[tmp] = pd.to_numeric(errors = "coerce", arg = output[tmp])
        i +=1
    #drop invalid values
    output = output.dropna(subset=col) 
    output = output.dropna(subset=["Date"])
    #drop duplicate dates
    output = output.drop_duplicates(keep = "last", subset =["Date"])
    #sort by date
    output = output.sort_values(by="Date")
    #remove not reasonable values
    output = output.query("Open > 0 and High > 0 and Low > 0 and Close > 0 and Volume >= 0")
    output.reset_index(drop=True, inplace=True)
    return output



def rsi(cls):
    #calculate the difference in closing price between each 2 days
    difference = cls.diff()
    #calculate profit and loss and remove negative values
    #1/14 is the smoothing factor for 14-day RSI
    profit = difference.clip(lower=0).ewm( 1/14, adjust=False).mean()
    loss = (-difference.clip(upper=0)).ewm( 1/14, adjust=False).mean()

    #calculate average (gains / losses) 
    rs = profit / loss.replace(0, np.nan)
    return 100 - (100/(1 + rs))


###################
def newFeatures(df):
    inputt = df.copy()
    cls = inputt["Close"]
    #calculate the current day returns by ln (todays closing / yesterday closing)
    inputt["dailyReturns"] = np.log(cls / cls.shift(1))
    #create the next day return 
    inputt["nextDayReturns"] = inputt["dailyReturns"].shift(-1)
    inputt["TargetDate"] = (inputt["Date"].shift(-1))

    #volume change between previuos and current days
    inputt["volumeChanges"] = np.log( inputt["Volume"] / inputt["Volume"].shift(1))
    #compare todays volume with avgerage previous 20 days volume
    inputt["relativeVolume"] = (inputt["Volume"] / inputt["Volume"].rolling(20).mean())

#price range in a day
    inputt["dailyRange"] = ((inputt["High"] - inputt["Low"]) /inputt["Close"])

#difference between open and close return
    inputt["openToCloseReturn"] = ( (inputt["Close"] - inputt["Open"]) / inputt["Open"])
    #difference betwen todays opening and yesterdays closing price
    inputt["nightGap"] = (inputt["Open"] /inputt["Close"].shift(1)) - 1

#previous market returns
    inputt["QQQReturnLag1"] = inputt["QQQDailyReturn"].shift(1)
    inputt["QQQReturnLag2"] = inputt["QQQDailyReturn"].shift(2)

    inputt["SPYReturnLag1"] = inputt["SPYDailyReturn"].shift(1)
    inputt["SPYReturnLag2"] = inputt["SPYDailyReturn"].shift(2)





    #create a lag returns for (1,2,3,5) days ago
    for lag in [1, 2, 3, 5]:
        inputt[f"lagDailyReturn{lag}"] = inputt["dailyReturns"].shift(lag)

    #calculate the technical indicators
    for period in [5, 20]:
        #average closing price 
        inputt[f"SMA{period}day"] = ( cls / cls.rolling(period).mean() - 1)
        #exponential moving average
        inputt[f"EMA{period}day"] = ( cls / cls.ewm( span=period,adjust=False).mean() - 1 )
        #compare current price with the price from the (5,20) days ago
        inputt[f"momentum{period}day"] = (cls / cls.shift(period) - 1 )
        #calculate the variation in price 
        inputt[f"volatility{period}day"] = ( inputt["dailyReturns"].rolling(period).std() * np.sqrt(252))

    inputt["RSI"] = rsi(cls)


    # Remove incomplete rows 
    inputt = inputt.replace( [np.inf, -np.inf],  np.nan ).dropna().reset_index(drop=True)

    
    inputt.insert(0, "TradingDay", range(1, len(inputt) + 1))

    return inputt



def splitData(input, features):

    input = input.copy()

    input["Date"] = pd.to_datetime( input["Date"] )
    input = input.sort_values("Date").reset_index(drop=True)

    splitIndex = int( len(input) * 0.8)
    #split the data into (80% training, 20% testing)
    training = input.head(splitIndex).copy()
    testing = input.tail( len(input) - splitIndex ).copy()

    xTraining = training[features]
    yTraining = training[target]

    xTesting = testing[features]
    yTesting = testing[target]

    return xTraining,xTesting,yTraining,yTesting
    

def evaluate(realReturn, predictedReturn):
    meanAbsErr = mean_absolute_error(realReturn, predictedReturn)
    RmeanSqErr = np.sqrt(mean_squared_error(realReturn, predictedReturn))
    R2 = r2_score(realReturn, predictedReturn)

    return meanAbsErr, RmeanSqErr, R2

def createRNNData(inputData,features,days=30):

    data = inputData.copy()

    data["Date"] = pd.to_datetime(data["Date"])
    data = data.sort_values("Date").reset_index(drop=True)

    #split the training and testing data 80/20
    splitIndex = int(len(data) * 0.8)
    #then split the training and validation 80/20
    validationStart = int(splitIndex * 0.8)

    xScaler = StandardScaler()
    yScaler = StandardScaler()

    #learn mean and standard deviation of (features & target)
    xScaler.fit(data.loc[:validationStart - 1,features])
    yScaler.fit(data.loc[:validationStart - 1,[target]])

    #apply
    scaledFeature = xScaler.transform(data[features])
    scaledTarget = yScaler.transform(data[[target]]).flatten()

    xTraining, yTraining, xValidation, yValidation, xTesting, yTesting = [], [], [], [], [], []

    #create a feature sequence based on the number of days
    for i in range(days - 1,len(data)):
        sequence = scaledFeature[ i - days + 1:i + 1]
        targetValue = scaledTarget[i]


        #decide where the sequence belongs
        if i < validationStart:
            xTraining.append(sequence)
            yTraining.append(targetValue)

        elif i < splitIndex:

            xValidation.append(sequence)

            yValidation.append(targetValue)

        else:

            xTesting.append(sequence)

            yTesting.append(targetValue)

    return (
        np.array(xTraining),
        np.array(yTraining),

        np.array(xValidation),
        np.array(yValidation),

        np.array(xTesting),
        np.array(yTesting),

        yScaler
    )







if __name__ == "__main__":

    stockData = downloadMultipleStock(
        stocks,
         "2010-01-01",
    "2026-01-01"
    )
    marketInformation = getMarketFeatures(
       "2010-01-01",
    "2026-01-01"
    )

    cleanedStk = {}
    featureStk = {}
    baseline = {}
    split = {}
    i = 0

    while i < len(stocks):
        #AAPL, NVDA, TSLA
        stk = stocks[i]
        #remove invalid or duplicate
        cleanedStk[stk] = clean(load_data( f"{stk}.csv",stk))
        
        cleanedStk[stk] = cleanedStk[stk].merge(marketInformation, on="Date")
        #create new features
        featureStk[stk] = newFeatures(cleanedStk[stk])
        
        
        df = featureStk[stk]
        featureCol = stockFeatures[stk]


        mainData = featureStk[stk][
    featureStk[stk]["TargetDate"] <
    pd.Timestamp("2025-01-01")
].copy()
        

        print("Stock name:", stk)
        #split the data 80% training, 20% for testing 
        
        split[stk] = splitData(mainData,featureCol)
        xTraining, xTesting, yTraining, yTesting = split[stk]

        #baseline for direction
        up = np.mean( np.asarray(yTesting) > 0)
        down = np.mean(np.asarray(yTesting) <= 0)

        directionBaseline = max(up,down)




        #Zero baseline
        forcast = np.zeros(len(yTesting))



        
        meanAbsErr, meanSqErr, R2 = evaluate(yTesting,forcast)
        baselineMSE = mean_squared_error(yTesting,forcast)

        baseline[stk] = { "MAE" : meanAbsErr,"RMSE" : meanSqErr, "R2" : R2}
      
   
        
        print("Baseline Results:", "MAE: ", meanAbsErr, "MSE: ", baselineMSE, "RMSE: ", meanSqErr,"R2: ", R2, " Direction: ", directionBaseline)




        # Create a Linear Regression model
        linear = Pipeline([("feature_scaler", StandardScaler()), ("regression_model", LinearRegression())])
        # Train only using training data
        linear.fit(xTraining, yTraining)
        # Predict the testing data
        modelForecast = linear.predict(xTesting)
        # Evaluate model predictions
        modelMeanAbsErr, modelMeanSqErr, modelR2 = evaluate(yTesting, modelForecast)

        
        modelMSE = mean_squared_error(yTesting,modelForecast)
        linearDirection = np.mean(np.sign(modelForecast) ==np.sign(yTesting))
        linearImprovement = ((meanSqErr - modelMeanSqErr) /meanSqErr) * 100

        print("Linear Regression Results:", " MAE:", modelMeanAbsErr, " MSE:", modelMSE, " RMSE:", modelMeanSqErr,
      " R2:", modelR2," Direction:", linearDirection)


        
        
        

        
        # Ridge Regression

        timeSplit = TimeSeriesSplit(n_splits=5, gap = 1)
  
        ridgeModel = Pipeline([("scaler", StandardScaler()),("ridge", Ridge())])

        ridgeValues = {"ridge__alpha": [0.01, 0.1, 1, 10, 100,500,1000,5000,10000,50000,100000,250000]}

        ridgeSearch = GridSearchCV(ridgeModel,ridgeValues,cv=timeSplit, scoring="neg_root_mean_squared_error")

        ridgeSearch.fit(xTraining, yTraining)

        ridgeForecast = ridgeSearch.predict(xTesting)

        ridgeMeanAbsErr, ridgeMeanSqErr, ridgeR2 = evaluate(yTesting,ridgeForecast)
        ridgeMSE = mean_squared_error(yTesting,ridgeForecast)
        ridgeDirection = np.mean(np.sign(ridgeForecast) == np.sign(yTesting))

        print("Ridge Regression Results:","Best alpha:", ridgeSearch.best_params_["ridge__alpha"],
      " MAE:", ridgeMeanAbsErr, " MSE:", ridgeMSE, " RMSE:", ridgeMeanSqErr,
      " R2:", ridgeR2," Direction:", ridgeDirection, "%")



        

       
        
        # Random Forest
        

        rfTuned = rfParameters[stk]

        rfModel = RandomForestRegressor( n_estimators=rfTuned["trees"],
                                        max_depth=rfTuned["depth"],
                                        min_samples_leaf=rfTuned["leaf"],
                                        max_features=rfTuned["features"],
)

        rfModel.fit(xTraining,yTraining)

        rfForecast = rfModel.predict(xTesting)

        rfMeanAbsErr, rfMeanSqErr, rfR2 = evaluate( yTesting,rfForecast)

        rfDirection = np.mean(np.sign(rfForecast) == np.sign(yTesting))
        rfMSE = mean_squared_error( yTesting,rfForecast)
        

        print("Random Forest Results:",
      " MAE:", rfMeanAbsErr, " MSE:", rfMSE, " RMSE:", rfMeanSqErr, " R2:", rfR2,
      "Direction:", rfDirection,"%")






#RNN
    
        rnnTuned = RNNParameters[stk]
        (xRTraining,yRTraining,xRValidation,yRValidation,xRTesting,yRTesting,yRScaler) = createRNNData( mainData,featureCol,days=rnnTuned["sequenceDays"])

       


# create RNN model
        rnn = Sequential([
    Input(shape=(xRTraining.shape[1],xRTraining.shape[2])),
    SimpleRNN(rnnTuned["rnnUnits"]),
    Dense(1)    ])


        #optimize
        rnnOpt = tf.keras.optimizers.Adam( learning_rate=rnnTuned["learnRate"])

        rnn.compile(optimizer= rnnOpt ,loss="mean_squared_error")


# stop training if no improvement happened
        rnnStop = EarlyStopping( monitor="val_loss",patience=8,restore_best_weights=True)


#training
        rnn.fit(xRTraining,yRTraining,

    validation_data=(xRValidation,yRValidation),
    epochs=60,
    shuffle=False,
    callbacks=[rnnStop],
    verbose=0)
        

        

        # calculate predictions
        rnnForecastScaled = rnn.predict(xRTesting,verbose=0)


        # get the predictions to the original scale
        rnnForecast = yRScaler.inverse_transform(rnnForecastScaled).flatten()


        # get real values to original scale
        rnnReal = yRScaler.inverse_transform(yRTesting.reshape(-1, 1)).flatten()



        rnnMAE, rnnRMSE, rnnR2 = evaluate(rnnReal,rnnForecast)
        rnnMSE = mean_squared_error(rnnReal,rnnForecast)


# direction accuracy
        rnnDirection = np.mean(np.sign(rnnForecast) ==np.sign(rnnReal))

# zero return baseline
        rnnBaseline = np.zeros(len(rnnReal))

        (rnnBaselineMAE,rnnBaselineRMSE,rnnBaselineR2) = evaluate( rnnReal,rnnBaseline)
        rnnBaselineMSE = mean_squared_error(rnnReal,rnnBaseline)


        rnnAlwaysUp = np.mean(rnnReal > 0)
        rnnAlwaysDown = np.mean(rnnReal <= 0)
        rnnDirectionBaseline = max(rnnAlwaysUp,rnnAlwaysDown)
      


        print(
    "RNN Results:",
    "RMSE:", rnnRMSE,
    "Baseline RMSE:", rnnBaselineRMSE,
    "R2:", rnnR2,
    "Direction:", rnnDirection,
    "Direction baseline:", rnnDirectionBaseline,
    "Days:", rnnTuned["sequenceDays"],
    "Units:", rnnTuned["rnnUnits"],
    "Learning rate:", rnnTuned["learnRate"]
)
  

        i += 1

    
    






    

