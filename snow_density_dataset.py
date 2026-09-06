import glob
from pathlib import Path
import pandas as pd
import numpy as np
from pyproj import Geod

def get_variable(data_in, data_aux, variable, meta, meta_sub):
    
    if meta[variable] == "x":

        data_sub = pd.merge(data_in, 
                            pd.DataFrame({"Date": pd.to_datetime(data_aux["Date"]), "Val": data_aux[f"{variable}_QC"]}), 
                            on = "Date", 
                            how = "left"
                           )
                           
        var = data_sub["Val"]
        
    else:
        
        ref_sample = meta_sub[meta_sub[variable] == "x"]
        
        if len(ref_sample) > 0:
        
            ref = ref_sample.iloc[0]
            
            file_ref = glob.glob(f'./Dataset_v2/Data/{ref["Provider"]}/{ref["Code"]}_*.txt')
            data_ref = pd.read_csv(file_ref[0])
            data_ref = pd.merge(data_in, 
                                pd.DataFrame({"Date": pd.to_datetime(data_ref["Date"]), "Val": data_ref[f"{variable}_QC"]}), 
                                on = "Date", 
                                how = "left"
                               )
            var = data_ref["Val"]
            
        else:
            
            var = np.full(len(data_in), np.nan)
            
    return var

list_out = []

file_in = glob.glob("./Snow_Density/Data/*.txt")

meta_list = []

meta_in = glob.glob("./Dataset_v2/Metadata/*_meta.txt")

for file in meta_in:
    
    meta_tmp = pd.read_csv(file)
    meta_tmp["Provider"] = file.replace("./Dataset_v2/Metadata/", "").split("_meta.txt")[0]
    
    meta_list.append(meta_tmp)
    
meta_all = pd.concat(meta_list)

for file in file_in:
    
    data_in = pd.read_csv(file)
    code = Path(file).name.split("_")[0]
    pr = "".join(list(code)[:3])
    
    if pr == "GER":
        provider = "DWD"
    if pr == "MMO":
        provider = "MeteoMont"
    if pr == "SLF":
        provider = "SLF"
        
    g = Geod(ellps = "WGS84")
    
    meta = meta_all[meta_all["Code"] == code].iloc[0]
    elevation = meta["Elevation"]
    longitude = meta["Longitude"]
    latitude = meta["Latitude"]
    
    _, _, stat_geo = g.inv(np.full(len(meta_all), longitude), 
                     np.full(len(meta_all), latitude), 
                     meta_all["Longitude"].to_numpy(), 
                     meta_all["Latitude"].to_numpy()
                    )
    
    stat_hgt = np.abs(meta_all["Elevation"].to_numpy() - elevation)
    ref_stat = np.where((stat_geo <= 50000) & (stat_hgt <= 300))[0]
    stat_geo = stat_geo[(stat_geo <= 50000) & (stat_hgt <= 300)]
    ref_stat = ref_stat[np.argsort(stat_geo)]
    ref_stat = ref_stat[ref_stat != np.where(meta_all["Code"] == code)[0][0]]

    if len(ref_stat) == 0:
        ref_stat = np.where(meta_all["Code"] == code)[0]

    meta_sub = meta_all.iloc[ref_stat]

    aux_file = glob.glob(f"./Dataset_v2/Data/{provider}/{code}_*.txt")
    data_aux = pd.read_csv(aux_file[0])

    data_in["Date"] = pd.to_datetime(data_in["Date"])
    year = data_in["Date"].dt.year
    y = np.where(data_in["Date"].dt.month < 11, year - 1, year)
    ref_date = pd.to_datetime({"year": y, "month": 11, "day": 1})
    doy = (data_in["Date"] - ref_date).dt.days
    
    hs = get_variable(data_in, data_aux, "HS", meta, meta_sub)
    hn = get_variable(data_in, data_aux, "HN", meta, meta_sub)
    tas = get_variable(data_in, data_aux, "T", meta, meta_sub)
    tmin = get_variable(data_in, data_aux, "TMIN", meta, meta_sub)
    tmax = get_variable(data_in, data_aux, "TMAX", meta, meta_sub)
    p = get_variable(data_in, data_aux, "P", meta, meta_sub)
    rh = get_variable(data_in, data_aux, "RH", meta, meta_sub)
    ws = get_variable(data_in, data_aux, "WS", meta, meta_sub)
    gs = get_variable(data_in, data_aux, "G", meta, meta_sub)
    
    rat = 17.625 * tas / (243.04 + tas)
    td =(243.04 * (np.log(rh / 100) + rat))/ (17.625 - np.log(rh / 100) - rat)
    
    pdd = tas.clip(lower = 0).rolling(window = 15).sum()
    ft = ((tmin < 0) & (tmax > 0)).astype(int).rolling(window = 15).sum()

    snowfrac = hn.rolling(window = 3).sum() / hs
    snowfrac[~np.isfinite(snowfrac)] = np.nan
    
    snowdays = np.where(hn >= 3, 1, 0)
    delta_hs = hs - hs.shift()
    snowdays[(snowdays == 0) & (np.isnan(hn)) & (delta_hs > 0)] = 1
    snowdays[np.isnan(hn) & np.isnan(hs)] = np.nan
    
    age = pd.DataFrame({"Date": data_in["Date"], "Hn": hn, "Snow": snowdays})
    
    age = np.full(len(data_in), np.nan)
    age_sum = 0
    
    for j in range(len(snowdays)):
        
        if snowdays[j] == 1:
            
            age_sum = 0
            
        else:
            
            if not np.isnan(snowdays[j]):
                
                age_sum += 1
                
            else:
                
                age_sum = np.nan
                
        age[j] = age_sum
            
    age = pd.DataFrame({"Date": data_in["Date"], "Hn": hn, "Hs": hs, "Snow": snowdays, "Age": age})

    snow_gain = (delta_hs > 0).astype(int).rolling(window = 15).sum()
    snow_loss = (delta_hs < 0).astype(int).rolling(window = 15).sum()
    
    ros = ((p > 0) & (hs >= 5) & (hn == 0)).astype(int).rolling(window = 15).sum()
    ros_p = p.where((p > 0) & (hs >= 5) & (hn == 0), 0).rolling(window = 15).sum()
    
    rr = hn >= 3
    event_id = ((rr & ~rr.shift(fill_value = False)).cumsum().where(rr, 0))
    
    snow_td = np.full(len(data_in), np.nan)
    snow_t = np.full(len(data_in), np.nan)
    snow_rh = np.full(len(data_in), np.nan)
    snow_ws = np.full(len(data_in), np.nan)
    snow_hs = np.full(len(data_in), np.nan)
    
    for j in range(len(data_in)):
        
        if event_id[j] != 0:
            
            last_snow = np.where(event_id[: j+1] == event_id[j])[0]
            
        else:
            
            past_event = event_id[: j+1]
            last_event = past_event[past_event != 0]
            if len(last_event) == 0:
            
                continue
                
            last_snow = np.where(event_id[: j+1] == last_event[-1])[0]
            
        if len(last_snow) == 0:
            
            continue
            
        if last_snow[0] > 0:
            
            snow_hs[j] = hs[last_snow[0] - 1]
            
        hn_sub = hn[: j+1][last_snow]
        td_sub = td[: j+1][last_snow]
        tas_sub = tas[: j+1][last_snow]
        rh_sub = rh[: j+1][last_snow]
        ws_sub = ws[: j+1][last_snow]
        
        snow_td[j] = (td_sub * hn_sub).sum() / hn_sub[~np.isnan(td_sub)].sum()
        snow_t[j] = (tas_sub * hn_sub).sum() / hn_sub[~np.isnan(tas_sub)].sum()
        snow_rh[j] = (rh_sub * hn_sub).sum() / hn_sub[~np.isnan(rh_sub)].sum()
        snow_ws[j] = (ws_sub * hn_sub).sum() / hn_sub[~np.isnan(ws_sub)].sum()
    
    out_tmp = pd.DataFrame({"Date" : data_in["Date"], 
                            "Rho" : data_in["Value"], 
                            "HS" : hs,						  
                            "T" : tas, 
                            "TMIN" : tmin, 
                            "TMAX" : tmax, 
                            "P" : p,
                            "RH" : rh,
                            "WS" : ws,
                            "G" : gs,
                            "HN" : hn,
                            "TD" : td,
                            "PDD" : pdd,
                            "FT" : ft,
                            "SnowFrac" : snowfrac,
                            "Age" : age["Age"],
                            "HS_Gain" : snow_gain,
                            "HS_Loss" : snow_loss,
                            "ROS" : ros,
                            "ROS_P" : ros_p,
                            "WS_HN" : snow_ws,
                            "RH_HN" : snow_rh,
                            "T_HN" : snow_t,
                            "TD_HN" : snow_td,
                            "DTR" : tmax - tmin,
                            "Elevation" : elevation,
                            "Longitude" : longitude,
                            "Latitude" : latitude,
                            "DOY" : doy,
                            "Code" : code
                           }
                          )
                  
    list_out.append(out_tmp)
    
data_out = pd.concat(list_out)
file_out =  "./Snow_Density/Dataset_density_py.txt"
data_out.to_csv(file_out)
  