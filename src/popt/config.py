FIRST_DATE     = "1954-01-03"  # first date
FINAL_DATE     = "2026-03-25"  # final date
FIRST_DATE_BTC = "2017-09-07"  # according to Kasper's paper
AUTO_ADJUST = True

D0 = "2005-01-03"
D1 = "2024-12-31"

# "IS3S.DE"
indx = ["SPY", "QQQ", "IWM"]  # market
bond = ["AGG", "TLT", "LQD", "TIP"]  # bonds
metl = ["GLD", "SLV", "CPER", "DBB"]  # precious metals
sect = ["XLK", "XLV", "XLF", "XLY", "XLI", "XLP", "XLE", "XLU", "XLB"]  # sectors
extr = ["IBB", "IYR"]  # extra sectors
intn = ["EWJ", "EWG", "EWU", "EWA", "EWH", "EWS", "EWZ", "EWT", "EWY", "EWP", "EWW", "EWD", "EWL", "EWC"]  # international "EEM"
comd = ["DBC", "DBA", "CORN", "SOYB", "USO", "WEAT", "CANE"]  # commodities COTN.L
crpt = ["BTC-USD"]  # crypto
universe = [*indx, *bond, *metl, *sect, *extr, *intn, *comd, *crpt]

# Time periods
_1W  = 5
_2W  = 10
_4W  = _1M  = 21
_12W = _3M  = _1Q = 63
_26W = _6M  = _2Q = 126
_52W = _12M = _4Q = _1Y = 252
