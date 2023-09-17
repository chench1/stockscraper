import yfinance as yf

obj = yf.Ticker('goog')

print(obj.get_news())
