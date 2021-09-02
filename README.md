# Quant.ify
Working on developing a profitable trade bot, initially forex, and eventually moving onto the stock market.

# Project Intuition
Since people are able to day-trade forex using purely technical analysis and consistently making money over time, there must be some recognizable patterns that can allow us to apply machine learning and capitalize on these patterns.

# Methods for Classifying and Labeling Data
## Method 1
Define a long or short position as when the price moves up 30 pips before moving down 10 pips, or vice versa. Don’t trade if neither condition is satisfied within the next 10 hours of candles (600 candles if we use minute candles). Defined as a multi-class classification problem with 3 classes [0, 0, 0], [go short, don’t trade, go long]

## Method 2
Multi-class classification: [go short, don't trade, go long]. Define neutral as price movement of under X pips (ex. 5 pips) at the end of T time horizon (ex. 5 hours). Long would be a price movement more than X pips up, and short would be a price movement of more than X pips down (within the time horizon T).

## Method 3
Price prediction using a linear regression model to predict the exact future price after some time horizon. Trades will be based on a “big enough” expected price movement (ex. At least 10 pips predicted price movement)

## Using these methods on the candle data, we can create labels to teach our model through supervised learning when to trade vs. when not to

# Macro-level Itererations of this Project
## Iteration 1
* Used method 1 of defining data
* Used 1 year of data
  * 8 months for training data
  * 4 months for testing data
* Input: 10 hours worth of minute candles

###### When tested, this iteration performed equivalent to random guessing

## Iteration 2
* Used method 1 of defining data
* Used 1 year of data
  * 8 months for training data
  * 4 months for testing data
* Input: 10 hours worth of 5 minute candles (120 candles)
* Added two features:
  * change (close - open) 
  * intraday range (high - low)
###### This iteration performed slightly better but nowhere near the desired results, overall lost -20% per year

### Realized that lower level time frames often respect the support and resistance levels created at higher timeframes, thus added a second LSTM input to the model. So now the inputs are 10 hours of 5 minutes candles (120 candles) and 90 daily candles with change and intraday range, this way the model can make decisions based on the long term trends (long term up trend/down trend)

This iteration also performed slightly better again, but there was no clear pattern emerging that it was able to capitalize on to profit consistently

### After this realization, I played around with multiple different combinations of timeframes in conjuction with Iteration 3

## Iteration 3
Decided to add in more technical indicators that day-traders often use:
* CCI
* RSI
* Stochastic Oscillator
* multiple EMAs
* ATR

I only kept these additional features on the lower level time frame as part of the LSTM input (so part of every candle as input)
This resulted in some level of overfitting and thus I moved these technical indicators to a 3rd input with a dense layer, representing only the technical indicators at the time of the last candle (the most recent candle)

This model produced a net loss of -15% per year.

## Iteration 4
I Kept playing around with timeframes and decided on hourly candles (H1) mixed with 4 hour candles (H4). Analyzing hourly candles revealed that they seemed to create enough stability to find patterns and still fit the support and resistance levels created by the 4 hour chart. I finally settled on using 120 hourly candles and 180 H4 candles plus the technical indicators of the last hour candle as a third input.

This model produced a net loss of -10% per year.

## Iteration 5
Brainstorming for even more features led to: day of week and time of day (intuition: the forex markets trade 24 hours a day, but the pair I am focusing on EUR/USD is traded through different markets based on the time of day. The three sessions are: north American, European, and Asian. Thus there may be patterns in the way the pair is traded based on the time of day which determines which of the 3 session the pair is currently trading in.)

This model produced a net loss of -6% per year.

## Iteration 6
Further brainstorming led to binary features to show whether the current price is over/under the 50 ema, 100 ma, and 200 ema. This can be seen as up/down over the short-term, mid-term, and long-term. These 3 features could further help. The model to predict the current price.

This model produced a net loss of -6% per year.

## Iteration 7
Even further brainstroming led to adding heiken ashi candles as further features, and a binary feature for whether a candle is green or red. Thus both LSTM inputs had 10 features, 4 from the original candle, green/red original candle, 4 from heiken ashi candle, green/red heiken ashi candle. This iteration alone added 6 new features per candle, which we have 120 + 180 of so we added 1800 inputs to the model in this iteration alone!

This model produced a profit of 50% per year.

### After also testing the 3 different methods in conjunction with most of these overall model iterations, I finally settled on method number 2, define up or down after 5 hours. The way this would trade is at the end of every hour input the live data into the model, if it predicts up or down -> open a trade in that direction, if neutral -> do nothing. Exactly 5 hours later close the trade, hopefully we profit if the model was right, or we lose money if we were wrong.

## Iteration 8
Also added more and more data slowly, settled on 10 years worth of data with 7 for training and 3 for testing. This seemed very promising with some models creating a 110% return per year on the entire 3 years worth of test data, which was the last 3 years.


### Implementing this, I ran it locally off of my computer for a few weeks. However, everytime my laptop closed or lost internet connection, the program had to be restarted and messed with my overall understanding of the model’s performance on live data.

### Thus, I learned how to use docker, and ran it locally first through a container, so that my program would be independent. Then I was able to deploy my container to AWS, and there it currently resides running 24/7. 

### I also learned how to use google sheets api to document the overall model performances: [spreadsheet](https://docs.google.com/spreadsheets/d/1g1ghv8778tTjkO5G4p1kad8dGqkHZAfIY0B-Mh5o0o0/edit#gid=0)

### It has now been running on live data for about 2 months, and it is still seemingly random, most of the models are currently green but not by large margins. I believe that as of now, the project is unable to definitively work on live data, thus not to be used with real money. It is possible that the current markets are too complex for such patterns to even exist and be profitable.

# Getting Started With Your First Forex Trading Model
## Part 1 - Set Up
1. Make a demo trading account at [Oanda](https://www.oanda.com/apply/demo)
2. Request an api access key from your account settings
3. Use pip to install this [library](https://github.com/hootnot/oanda-api-v20)
4. Familiarize yourself with the endpoints in the [oanda api](http://developer.oanda.com/rest-live-v20/instrument-ep/)
5. Look through examples in the github api wrapper (number 3) as well as number 4 ^

## Part 2 - Building Your First Model
Make a new notebook locally and try to build your first model using:
1. Use the oanda api to get historic EUR_USD data (try at least 2 years), this is the primary currency we will attempt to trade because it has the lowest [spreads](https://www.dailyforex.com/forex-articles/2009/05/forex-spreads-the-basics/1030) and the highest allowed [leverage](https://www.investopedia.com/articles/forex/07/forex_leverage.asp), 1:50
2. 288 M15 Candles with [open, high, low, close] values as an LSTM layer input
3. Compute indicators such as [RSI](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjXpIrZl8LvAhUDNH0KHYL5B6wQFjAAegQIBBAD&url=https%3A%2F%2Fwww.investopedia.com%2Fterms%2Fr%2Frsi.asp&usg=AOvVaw03QeCun2Y2fpO4fA_ZaFMm), [CCI](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&ved=2ahUKEwjZtZbll8LvAhW1FTQIHdATCwUQFjABegQIAxAD&url=https%3A%2F%2Fwww.investopedia.com%2Fterms%2Fc%2Fcommoditychannelindex.asp&usg=AOvVaw15zI6NFRobwc814_uvGqXS), [%K Stochastic](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwj1yK7yl8LvAhV2JzQIHehSDJ8QFjAAegQICRAD&url=https%3A%2F%2Fwww.investopedia.com%2Fterms%2Fs%2Fstochasticoscillator.asp&usg=AOvVaw2a-tDvXJx9MnyamqWRHSgj), [50 EMA](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwjhqKD_l8LvAhUOFTQIHQ5cCVUQFjAAegQICBAD&url=https%3A%2F%2Fwww.investopedia.com%2Fterms%2Fe%2Fema.asp&usg=AOvVaw1ZcdUTJqie7SIDRGp6dvDH), 100 EMA, 200 EMA, and use them as a secondary dense input
4. Scale values however you see fit
5. Define output as [down, neutral, up], based on either whether price closed above/below a certain threshold above/below the intial value after X candles (you decide X), or whether it would have been a succeful trade after setting a stop-loss and take-profit. For example, a 10 pip stop-loss and 30 pip take-profit has a risk:reward ratio of 1:3, aim for at least 1:2. 
6. Train your model and evaluate its ability to trade profitably
