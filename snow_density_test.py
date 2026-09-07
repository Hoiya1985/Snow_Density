import pandas as pd
import numpy as np
import datetime as dt
from pygam import LinearGAM, l, predict
from scipy import stats
from sklearn.metrics import root_mean_squared_error as rmse
from sklearn.metrics import mean_absolute_error as maer
from sklearn.linear_model import LinearRegression as lr

def density_fun(data, sf_tau = 1.07, sf_mu = 4.1, sigma = 3, k_sf = 6, k_t = -1.24, k_age = 20, age_tau = 3.18, k_td = -2.38):
    
    doy_fun = 127.7516 - (0.3035548 * data["DOY"]) + (0.003896515 * (data["DOY"] ** 2))
    sf_a = -108.9076 * data["SnowFrac"] * np.exp(-data["SnowFrac"] / sf_tau) 
    sf_b = 27.19235 * np.exp(-((data["data$SnowFrac"] - sf_mu) ** 2) / (2 * (sigma ** 2)))
    sf_c = 23.22282 * np.maximum(data["SnowFrac"] - k_sf, 0)
    sf_fun = sf_a + sf_b - sf_c
    ros_fun = 11.60323 * data["ROS"]
	t_fun = (1.294826 * data["T"]) + (6.981445 * np.log(1 + np.exp(data["T"] - k_t)))
	hs_fun = 13.85482 * np.log(1 + data["HS_HN"])
	age_fun = (4.716678 * data["Age"]) - (16.09779 * np.log(1 + np.exp((data["Age"] - k_age) / age_tau)))
	td_fun = (1.672076 * data["TD_HN"]) + (7.844527 * np.log(1 + np.exp(data["TD_HN"] - k_td)))

	main_fun = doy_fun + sf_fun + ros_fun + t_fun + hs_fun + age_fun + td_fun
	
	return main_fun
    
def marty_fun(data):
    
    coeff = np.full((len(data), 2), np.nan)
    m = data["Date"].dt.month
    hgt = data["Elevation"]
    
    coeff[(m == 11) & (hgt >= 2000), :] = [47, 206]
    coeff[(m == 11) & ((hgt >= 1400) & (hgt < 2000)), :] = [35, 183]
	coeff[(m == 11) & (hgt < 1400), :] = [37, 149]
	
	coeff[(m == 12) & (hgt >= 2000), :] = [52, 203]
	coeff[(m == 12) & ((hgt >= 1400) & (hgt < 2000)), :] = [47, 190]
	coeff[(m == 12) & (hgt < 1400), :] = [26, 201]
	
	coeff[(m == 1) & (hgt >= 2000), :] = [52, 206]
	coeff[(m == 1) & ((hgt >= 1400) & (hgt < 2000)), :] = [47, 208]
	coeff[(m == 1) & (hgt < 1400), :] = [31, 235]
	
	coeff[(m == 2) & (hgt >= 2000), :] = [46, 217]
	coeff[(m == 2) & ((hgt >= 1400) & (hgt < 2000)), :] = [52, 218]
	coeff[(m == 2) & (hgt < 1400), :] = [9, 279]
	
	coeff[(m == 3) & (hgt >= 2000), :] = [26, 272]
	coeff[(m == 3) & ((hgt >= 1400) & (hgt < 2000)), :] = [31, 281]
	coeff[(m == 3) & (hgt < 1400), :] = [3, 333]
	
	coeff[(m == 4) & (hgt >= 2000), :] = [9, 331]
	coeff[(m == 4) & ((hgt >= 1400) & (hgt < 2000)), :] = [15, 354]
	coeff[(m == 4) & (hgt < 1400), :] = [25, 347]
	
	coeff[(m == 5) & (hgt >= 2000), :] = [21, 378]
	coeff[(m == 5) & ((hgt >= 1400) & (hgt < 2000)), :] = [29, 409]
	coeff[(m == 5) & (hgt < 1400), :] = [19, 413]
    
    a = coeff[:, 0]
	b = coeff[:, 1]

	rho = (a * (data["HS"]/100)) + b
	
	return rho

def pistocchi_fun(data, rho0 = 200, k = 1):
    
    main_out = rho0 + (k * data["DOY"])
	
	return main_out
    
def sturm_fun(data, rmax = 0.5975, r0 = 0.2237, k1 = 0.0012, k2 = 0.0038):

	snow_fun = 1000 * ((rmax - r0) * (1 - np.exp(-(k1 * data["HS"]) - (k2 * (data["DOY"] + 61)))) + r0)
	
	return snow_fun
    
def metrics_fun(x, y, model, i):
    
    xy = pd.DataFrame({"x": x, "y": y}).dropna()
    obs = x[~np.isnan(x)]
    mod = y[~np.isnan(y)]
    r = stats.pearsonr(xy["x"], xy["y"]).statistic
    nrmse = rmse(xy["x"], xy["y"]) / np.std(obs, ddof = 1)
    mae = maer(xy["x"], xy["y"])
    pbias = 100 * np.sum(mod - obs) / sum(obs)
    a = np.std(mod, ddof = 1) / np.std(obs, ddof = 1)
    b = np.mean(mod) / np.mean(obs)
    kge = 1 - np.sqrt(((r - 1) ** 2) + ((a - 1) ** 2) + ((b - 1) ** 2))
    
    df_out = pd.DataFrame({"Ens": f"{i:03d}",
                           "Model": model,
                           "R": r,
                           "NRMSE": nrmse,
                           "MAE": mae,
                           "PBIAS": pbias,
                           "KGE": kge})
                           
    return df_out
    
data_in = pd.read_csv("./Snow_Density/Dataset_Density.txt")
data_in["Date"] = pd.to_datetime(data_in["Date"])
data_in = data_in.dropna(subset = ["Rho"])
data_in = data_in[data_in["Elevation"] < 2500]
m = data_in["Date"].dt.month
data_in = data_in[(m <= 5) | (m >= 11)]

data_fit = pd.DataFrame({"Y": data_in["Rho"],
						  "Code": data_in["Code"],
						  "DOY": data_in["DOY"],
						  "T": data_in["T"],
						  "ROS": data_in["ROS"],
						  "SnowFrac": data_in["SnowFrac"],
						  "Age": data_in["Age"],
						  "TD_HN": data_in["TD_HN"],
						  "HS_HN": data_in["HS_HN"]}).dropna()
                          
data_test = pd.DataFrame({"Y": data_in["Rho"],
                           "Date": data_in["Date"],
						   "Code": data_in["Code"],
                           "Elevation": data_in["Elevation"],
						   "DOY": data_in["DOY"],
						   "T": data_in["T"],
						   "ROS": data_in["ROS"],
						   "SnowFrac": data_in["SnowFrac"],
						   "Age": data_in["Age"],
						   "TD_HN": data_in["TD_HN"],
						   "HS_HN": data_in["HS_HN"],
                           "HS": data_in["HS"]}).dropna()


stats = pd.unique(data_fit["Code"])                  

model_df = pd.DataFrame({"DOY1": data_fit["DOY"],
                         "DOY2": data_fit["DOY"] ** 2,
                         "SF1": data_fit["SnowFrac"] * np.exp(-data_fit["SnowFrac"] / 1.07),
                         "SF2": np.exp(-((data_fit["SnowFrac"] - 4.1) ** 2) / 18),
                         "SF3": np.maximum(data_fit["SnowFrac"] - 6, 0),
                         "ROS": data_fit["ROS"],
                         "T1": data_fit["T"],
                         "T2": np.log(1 + np.exp(data_fit["T"] + 1.24)),
                         "HS_HN": np.log(1 + data_fit["HS_HN"]),
                         "Age1": data_fit["Age"],
                         "Age2": np.log(1 + np.exp((data_fit["Age"] - 20) / 3.18)),
                         "TD_HN1": data_fit["TD_HN"],
                         "TD_HN2": np.log(1 + np.exp(data_fit["TD_HN"] + 2.38)),
                         "Code": data_fit["Code"]})

model = l(0) + l(1) + l(2) + l(3) + l(4) + l(5) + l(6) + l(7) + l(8) + l(9) + l(10) + l(11) + l(12)

#Fold-Validation
fold_list = []

for i in range(0,100):
    
    groups = np.array_split(np.random.permutation(stats), 3)
    group1, group2, group3 = groups

    y = data_fit[data_fit["Code"].isin(np.concatenate((group2, group3)))]
    y = y["Y"].to_numpy()
    X = model_df[model_df["Code"].isin(np.concatenate((group2, group3)))].drop(columns = "Code").to_numpy()
    fit = lr().fit(X, y)
    x_test = model_df[model_df["Code"].isin(group1)].drop(columns = "Code")
    y_test = data_fit[data_fit["Code"].isin(group1)]
    fit_out = fit.predict(x_test.to_numpy())
    
    data_out = pd.DataFrame({"Obs": y_test["Y"], "Mod": fit_out}).dropna()
    data_rmse = rmse(data_out["Obs"], data_out["Mod"])
    data_mae = maer(data_out["Obs"], data_out["Mod"])
    data_bias = np.sum(data_out["Mod"] - data_out["Obs"])
    
    fold_tmp = pd.DataFrame({"Ens": f"{i:03d}",
                             "Group": "Group 1",
                             "RMSE": data_rmse,
                             "MAE": data_mae,
                             "BIAS": data_bias})
                             
    fold_list.append(fold_tmp)
    
    y = data_fit[data_fit["Code"].isin(np.concatenate((group1, group3)))]
    y = y["Y"].to_numpy()
    X = model_df[model_df["Code"].isin(np.concatenate((group1, group3)))].drop(columns = "Code").to_numpy()
    fit = lr().fit(X, y)
    x_test = model_df[model_df["Code"].isin(group2)].drop(columns = "Code")
    y_test = data_fit[data_fit["Code"].isin(group2)]
    fit_out = fit.predict(x_test.to_numpy())
    
    data_out = pd.DataFrame({"Obs": y_test["Y"], "Mod": fit_out}).dropna()
    data_rmse = rmse(data_out["Obs"], data_out["Mod"])
    data_mae = maer(data_out["Obs"], data_out["Mod"])
    data_bias = np.sum(data_out["Mod"] - data_out["Obs"])
    
    fold_tmp = pd.DataFrame({"Ens": f"{i:03d}",
                             "Group": "Group 2",
                             "RMSE": data_rmse,
                             "MAE": data_mae,
                             "BIAS": data_bias})
                             
    fold_list.append(fold_tmp)
    
    y = data_fit[data_fit["Code"].isin(np.concatenate((group1, group2)))]
    y = y["Y"].to_numpy()
    X = model_df[model_df["Code"].isin(np.concatenate((group1, group2)))].drop(columns = "Code").to_numpy()
    fit = lr().fit(X, y)
    x_test = model_df[model_df["Code"].isin(group3)].drop(columns = "Code")
    y_test = data_fit[data_fit["Code"].isin(group3)]
    fit_out = fit.predict(x_test.to_numpy())
    
    data_out = pd.DataFrame({"Obs": y_test["Y"], "Mod": fit_out}).dropna()
    data_rmse = rmse(data_out["Obs"], data_out["Mod"])
    data_mae = maer(data_out["Obs"], data_out["Mod"])
    data_bias = np.sum(data_out["Mod"] - data_out["Obs"])
    
    fold_tmp = pd.DataFrame({"Ens": f"{i:03d}",
                             "Group": "Group 3",
                             "RMSE": data_rmse,
                             "MAE": data_mae,
                             "BIAS": data_bias})
                             
    fold_list.append(fold_tmp)
    
y = data_fit["Y"].to_numpy()
X = model_df.drop(columns = "Code").to_numpy()
fit = lr().fit(X, y)
fit_out = fit.predict(X.to_numpy())
    
data_out = pd.DataFrame({"Obs": data_fit["Y"], "Mod": fit_out}).dropna()
data_rmse = rmse(data_out["Obs"], data_out["Mod"])
data_mae = maer(data_out["Obs"], data_out["Mod"])
data_bias = np.sum(data_out["Mod"] - data_out["Obs"])
    
fold_tmp = pd.DataFrame({"Ens": f"{i:03d}",
                         "Group": "Group 3",
                         "RMSE": data_rmse,
                         "MAE": data_mae,
                         "BIAS": data_bias})
                             
fold_list.append(fold_tmp)   
    
fold_out = pd.concat(fold_list)
file_out =  "./Snow_Density/Parameter_Validation.txt"
fold_out.to_csv(file_out)

#Multi-method Validation
metrics_list = []

n = int(np.round(len(stats) * 0.8, 0))

for i in range(0,1000):

    stats_sub = np.random.choice(a = stats, size = n, replace = False)
    test_sub = data_test[data_test["Code"].isin(stats_sub)]
    x = test_sub["Rho"]
    
    y = density_fun(test_sub)
    met = metrics_fun(x, y, "EEAR", i)
    met_tmp = pd.DataFrame(met)
    metrics_list.append(met_tmp)
    
    y = marty_fun(test_sub)
    met = metrics_fun(x, y, "Jonas-Marty", i)
    met_tmp = pd.DataFrame(met)
    metrics_list.append(met_tmp)
    
    y = pistocchi_fun(test_sub)
    met = metrics_fun(x, y, "Pistocchi", i)
    met_tmp = pd.DataFrame(met)
    metrics_list.append(met_tmp)
    
    y = sturm_fun(test_sub)
    met = metrics_fun(x, y, "Sturm", i)
    met_tmp = pd.DataFrame(met)
    metrics_list.append(met_tmp)
    
metrics_out = pd.concat(metrics_list)
file_out = "./Snow_Density/Test.txt"
metrics_out.to_csv(file_out)	
