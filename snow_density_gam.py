import pandas as pd
import numpy as np
import datetime as dt
from pygam import LinearGAM, s

data_in = pd.read_csv("./Snow_Density/Dataset_Density.txt")
data_in["Date"] = pd.to_datetime(data_in["Date"])
data_in = data_in.dropna(subset = ["Rho"])
data_in = data_in[data_in["Elevation"] < 2500]
m = data_in["Date"].dt.month
data_in = data_in[(m <= 5) | (m >= 11)]

data_gam = data_in.dropna()                  
y = data_gam["Rho"].to_numpy()
X = data_gam.to_numpy()                    
gam_out = LinearGAM(s(0), n_splines = 15).fit(X, y)
gam_out.summary()

drop_param = [col for col in data_in.columns if col not in ["TMIN", "TMAX", "WS", "G", "WS_HN", "DTR"]]
data_sub = data_in.drop(columns = drop_param)
data_gam = data_sub.dropna()
y = data_gam["Rho"].to_numpy()
X = data_gam.to_numpy()                      
gam_out = LinearGAM(s(0), n_splines = 15).fit(X, y)
gam_out.summary()

drop_param = [col for col in data_sub.columns if col not in ["P", "RH", "RH_HN", "TD", "T_HN"]]
data_sub = data_in.drop(columns = drop_param)
data_gam = data_sub.dropna()
y = data_gam["Rho"].to_numpy()
X = data_gam.to_numpy()                      
gam_out = LinearGAM(s(0), n_splines = 15).fit(X, y)
gam_out.summary()

drop_param = [col for col in data_sub.columns if col not in ["HS", "HN", "PDD", "FT", "HS_Loss", "ROS_P"]]
data_sub = data_in.drop(columns = drop_param)
data_gam = data_sub.dropna()
y = data_gam["Rho"].to_numpy()
X = data_gam.to_numpy()                      
gam_out = LinearGAM(s(0), n_splines = 15).fit(X, y)
gam_out.summary()

#Rho ~ s(DOY, k = 15) + s(SnowFrac) + s(ROS) + s(T, k = 15) + s(HS_HN, k = 15) + s(Age, k = 15) + s(TD_HN, k = 15)
