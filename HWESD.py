from sklearn.metrics import mean_squared_error
from scipy.optimize import differential_evolution

# Use this for using with python=2.7
# def mean_absolute_percentage_error(true, pred):
#     assert len(true) == len(pred)
#     return (sum([abs(i-j) for i,j in zip(true, pred)])/sum(true))*100

class CustomHWES:
    
    def __init__(self, endog, exog, seasonal=7):
        self.series = endog
        self.exog = exog
        self.slen = seasonal
        
    def __setHyp(self, hyperparams):
        self.alpha, self.beta, self.gamma, self.disc, self.damp = hyperparams
    
    def initial_trend(self):
        sumc = 0.0
        for i in range(self.slen):
            sumc += float(self.series[i+self.slen] - self.series[i]) / self.slen
        return sumc / self.slen

    def initial_seasonal_components(self):
        seasonals = {}
        season_averages = []
        n_seasons = int(len(self.series)/self.slen)
#         print(n_seasons)
        # compute season averages
        for j in range(n_seasons):
            season_averages.append(sum(self.series[self.slen*j:self.slen*j+self.slen])/float(self.slen))
        # compute initial values
        for i in range(self.slen):
            sum_of_vals_over_avg = 0.0
            for j in range(n_seasons):
                sum_of_vals_over_avg += (self.series[self.slen*j+i]/season_averages[j])
            seasonals[i] = sum_of_vals_over_avg/n_seasons
        return seasonals

    def __triple_exponential_smoothing_multi(self, n_preds, discount):
        result = []
        seasonals = self.initial_seasonal_components()
        self.exog.extend(discount)
        
        for i in range(len(self.series)+n_preds):
            if i == 0: # initial values
                smooth = self.series[0]
                trend = self.initial_trend()
                result.append(self.series[0])
                continue
#             if i == len(self.series):
#                 print(seasonals)
            if i >= len(self.series): # we are forecasting
                m = i - len(self.series) + 1
#                 print(smooth, trend, seasonals[i%self.slen])
                result.append((smooth*(1+self.disc*discount[m-1]) + (self.damp**(m-1))*trend) * seasonals[i%self.slen])
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha*(val/seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                smooth = (1+self.disc*self.exog[i]) * smooth
                trend = (self.beta * (smooth-last_smooth) + (1-self.beta)*trend)*self.damp
                seasonals[i%self.slen] = self.gamma*(val/smooth) + (1-self.gamma)*seasonals[i%self.slen]
                result.append((smooth*(1+self.disc*self.exog[i])+trend)*seasonals[i%self.slen])
        return result[-n_preds:]
    
    def __getErr(self):
        
        result = []
        seasonals = self.initial_seasonal_components()
        for i in range(len(self.series)):
            if i == 0: # initial values
                smooth = self.series[0]
                trend = self.initial_trend()
                result.append(self.series[0])
                continue
            else:
                val = self.series[i]
                last_smooth, smooth = smooth, self.alpha*(val/seasonals[i%self.slen]) + (1-self.alpha)*(smooth+trend)
                smooth = (1+self.disc*self.exog[i]) * smooth
                trend = (self.beta * (smooth-last_smooth) + (1-self.beta)*trend)*self.damp
                seasonals[i%self.slen] = self.gamma*(val/smooth) + (1-self.gamma)*seasonals[i%self.slen]
                result.append((smooth*(1+self.disc*self.exog[i])+trend)*seasonals[i%self.slen])
#         print(result)
        return mean_absolute_percentage_error(self.series, result)
    
    def __fit(self, hyper):
#         print(hyper)
        self.__setHyp(hyperparams=hyper)
        return self.__getErr()
    
    def fit(self):
        HWES_alpha = (0.00001, 1.0)
        HWES_beta = (0.00001, 1.0)
        HWES_gamma = (0.00001, 1.0)
        HWES_disc = (0.00001, 1.0)
        HWES_damp = (0.00001, 1.0)
        boundaries = [HWES_alpha] + [HWES_beta] + [HWES_gamma] + [HWES_disc] + [HWES_damp]

        solver = differential_evolution(self.__fit, bounds=boundaries)
        
        self.__setHyp(solver.x)
     
    def forecast(self, n_preds, discount):
        return self.__triple_exponential_smoothing_multi(n_preds, discount)
