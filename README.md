# Quant.ify
Working on developing a profitable trade bot, initially forex, and eventually moving onto the stock market.


# Getting Started
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
6. Train your model, and evaluate its ability to trade profitably, then we can all discuss further and truly get started.
