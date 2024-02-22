import os
import time
import json
import yaml
import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB


def get_load_data(data_file: str, mode: str) -> dict:
    """
    Get load data from excel.

    Returns:
        dict: multi-energy load profile
    """
    datas = pd.read_excel(data_file)
    P_DE = list(datas['电负荷kW'].fillna(0))
    G_DE = list(datas['供暖热负荷(kW)'].fillna(0))
    Q_DE = list(datas['冷负荷(kW)'].fillna(0))
    H_DE = list(datas['生活热水负荷kW'].fillna(0))
    R_PV = list(datas['pv'].fillna(0))

    if mode == "ele":
        for i in range(365):
            for j in range(8, 18):
                P_DE[i + j] += 50
    elif mode == "cool":
        for i in range(365):
            for j in range(8, 18):
                Q_DE[i + j] += 50

    load_data = {
        'P_DE': P_DE,
        'G_DE': G_DE,
        'Q_DE': Q_DE,
        'H_DE': H_DE,
        'R_PV': R_PV
    }
    return load_data


def crf(year: int) -> float:
    """
    Calculate capital recovery factor.

    Args:
        year (int): device life

    Returns:
        float: capital recovery factor
    """
    i = 0.04
    a_e = ((1 + i) ** year) * i / ((1 + i) ** year - 1)
    return a_e


def my_opt(param_cfg, load_data, time_scale, **kwargs):
    """
    Optimize multi-energy system.

    Args:
        param_cfg (dict): model parameters config
        load_data (dict): load data
        time_scale (int): variable time scales

    Returns:
        optimal device installed capacity and optimal operation results (dict)
    """
    # ------ settings ------
    # --- set result directory ---
    if "res_dir" in kwargs:
        res_dir = kwargs["res_dir"]
    # default result directory
    else:
        res_dir = "./Output/"
    # create result directory
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    # --- set mode ---
    if "mode" in kwargs:
        mode = kwargs["mode"]
    # default mode
    else:
        mode = "opt"
    if mode not in ["opt", "ele", "cool"]:
        raise ValueError("Unsupported mode!")

    # ------ init parameters ------
    # --- device life ---
    if "device_life" in kwargs:
        device_life = kwargs["device_life"]
    # default device life
    else:
        device_life = 20
    # --- unit cost ---
    c_PV = param_cfg['device']['pv']['capital_cost']
    c_EL = param_cfg['device']['el']['capital_cost']
    c_FC = param_cfg['device']['fc']['capital_cost']
    c_HS = param_cfg['device']['hs']['capital_cost']
    c_HW = param_cfg['device']['hw']['capital_cost']
    c_CW = param_cfg['device']['ct']['capital_cost']
    c_HP = param_cfg['device']['hp']['capital_cost']
    c_EB = param_cfg['device']['eb']['capital_cost']
    # --- price ---
    daily_ele_price = param_cfg['price']['ele_TOU_price']
    hydrogen_price = param_cfg['price']['hydrogen_price']
    days = int(time_scale / len(daily_ele_price))
    ele_price = [daily_ele_price[i] for _ in range(days) for i in range(len(daily_ele_price))]
    # --- device parameters ---
    # electrolyzer
    beta_el = param_cfg['device']['el']['beta_el']
    # fuel cell
    eta_fc_p = param_cfg['device']['fc']['eta_fc_p']
    eta_fc_g = param_cfg['device']['fc']['eta_fc_g']
    theta_ex = param_cfg['device']['fc']['theta_ex']
    # hydrogen storage
    sto_min = param_cfg['device']['hs']['sto_min']
    # hot & cold water storage
    c_water = 4.2 / 3600  # kW·h/(kg·°C)
    t_hw_min = param_cfg['device']['hw']['t_min']
    t_hw_max = param_cfg['device']['hw']['t_max']
    t_cw_min = param_cfg['device']['ct']['t_min']
    t_cw_max = param_cfg['device']['ct']['t_max']
    # heat pump
    beta_hpg = param_cfg['device']['hp']['beta_hpg']
    beta_hpq = param_cfg['device']['hp']['beta_hpq']
    # electric boiler
    beta_eb = param_cfg['device']['eb']['beta_eb']
    # --- others ---
    eta_ele_g = beta_eb
    eta_ele_q = 3
    heat_price = 5
    cool_price = 5
    heat_area = 12.03e4
    cool_area = 2.18e4
    c_p = 0.5703  # kg CO2/kWh
    c_g = 9.45  # kg CO2/kWh

    # ------ create model ------
    model = gp.Model("HIES")

    # ------ create variables ------
    # --- device capacity ---
    P_PV = model.addVar(name='P_PV')  # photovoltaic
    P_EL = model.addVar(name='P_EL')  # electrolyzer
    P_FC = model.addVar(name='P_FC')  # fuel cells
    H_STO = model.addVar(name='P_HS')  # hydrogen storage
    G_HW = model.addVar(name='G_HW')  # hot water storage
    Q_CW = model.addVar(name='Q_CW')  # cold water storage
    P_HP = model.addVar(name='P_HP')  # heat pump
    P_EB = model.addVar(name='P_EB')  # electric boiler
    # --- investment cost ---
    c_IN = model.addVar(name='c_IN')  # investment cost (CAPEX)
    # ---operation cost ---
    c_OP = model.addVar(name='c_OP')  # operation cost (OPEX)
    # --- device operation ---
    # photovoltaic
    p_pv = model.addVars(time_scale, name='p_pv')
    # electrolyzer
    p_el = model.addVars(time_scale, name='p_el')
    h_el = model.addVars(time_scale, name='h_el')
    # fuel cell
    p_fc = model.addVars(time_scale, name='p_fc')
    h_fc = model.addVars(time_scale, name='h_fc')
    g_fc = model.addVars(time_scale, name='g_fc')
    # hydrogen storage
    h_sto = model.addVars(time_scale, lb=sto_min, name='h_sto')
    h_sto_next = model.addVars(time_scale, lb=sto_min, name='h_sto_next')
    # hot water storage
    g_hw_io = model.addVars(time_scale, name='g_hw_io')
    g_hw = model.addVars(time_scale, name='g_hw')
    g_hw_next = model.addVars(time_scale, name='g_hw_next')
    # cold water storage
    q_cw_io = model.addVars(time_scale, name='q_cw_io')
    q_cw = model.addVars(time_scale, name='q_cw')
    q_cw_next = model.addVars(time_scale, name='q_cw_next')
    # heat pump
    p_hp = model.addVars(time_scale, name='p_hp')
    g_hp = model.addVars(time_scale, name='g_hp')
    q_hp = model.addVars(time_scale, name='q_hp')
    z_hp_g = model.addVars(time_scale, vtype=GRB.BINARY, name='z_hp_g')
    z_hp_q = model.addVars(time_scale, vtype=GRB.BINARY, name='z_hp_q')
    # electric boiler
    p_eb = model.addVars(time_scale, name='p_eb')
    g_eb = model.addVars(time_scale, name='g_eb')
    # --- purchase electricity & hydrogen ---
    p_pur = model.addVars(time_scale, name='p_pur')
    h_pur = model.addVars(time_scale, name='h_pur')

    # ------ update Gurobi model ------
    model.update()

    # ------ set objective function ------
    obj = crf(device_life) * c_IN + c_OP
    model.setObjective(obj, GRB.MINIMIZE)

    # ------ add constraints ------
    # --- set investment cost ---
    model.addConstr(c_IN == c_PV * P_PV + c_EL * P_EL + c_FC * P_FC
                    + c_HS * H_STO + c_HW * G_HW + c_CW * Q_CW
                    + c_HP * P_HP + c_EB * P_EB)
    # --- set operation cost ---
    model.addConstr(c_OP == gp.quicksum(ele_price[i] * p_pur[i] for i in range(time_scale))
                    + hydrogen_price * gp.quicksum(h_pur))
    # --- state constraints ---
    model.addConstrs(h_sto_next[i] == h_sto[i + 1] for i in range(time_scale - 1))
    model.addConstrs(g_hw_next[i] == g_hw[i + 1] for i in range(time_scale - 1))
    model.addConstrs(q_cw_next[i] == q_cw[i + 1] for i in range(time_scale - 1))
    # final state constraint
    model.addConstr(h_sto_next[time_scale - 1] == h_sto[0])
    model.addConstr(g_hw_next[time_scale - 1] == g_hw[0])
    model.addConstr(q_cw_next[time_scale - 1] == q_cw[0])

    for i in range(time_scale):
        # --- device constraints ---
        model.addConstr(p_el[i] <= P_EL)
        model.addConstr(p_fc[i] <= P_FC)
        model.addConstr(p_hp[i] <= P_HP)
        model.addConstr(p_eb[i] <= P_EB)
        # photovoltaic
        model.addConstr(p_pv[i] == P_PV * load_data['R_PV'][i])
        # electrolyzer
        model.addConstr(h_el[i] == beta_el * p_el[i])
        # fuel cell
        model.addConstr(p_fc[i] == eta_fc_p * h_fc[i])
        model.addConstr(g_fc[i] == theta_ex * eta_fc_g * h_fc[i])
        # hydrogen storage
        model.addConstr(h_sto_next[i] - h_sto[i] == h_el[i] + h_pur[i] - h_fc[i])
        model.addConstr(h_sto[i] <= H_STO)
        model.addConstr(h_sto_next[i] - h_sto[i] >= -H_STO * 0.5)
        model.addConstr(h_sto_next[i] - h_sto[i] <= H_STO * 0.5)
        # hot water storage
        model.addConstr(g_hw_io[i] == g_hw_next[i] - g_hw[i])
        model.addConstr(g_hw[i] <= G_HW)
        # cold water storage
        model.addConstr(q_cw_io[i] == q_cw_next[i] - q_cw[i])
        model.addConstr(q_cw[i] <= Q_CW)
        # heat pump
        model.addConstr(g_hp[i] == p_hp[i] * beta_hpg * z_hp_g[i])
        model.addConstr(q_hp[i] == p_hp[i] * beta_hpq * (1 - z_hp_q[i]))
        model.addConstr(z_hp_g[i] + z_hp_q[i] == 1)
        # electric boiler
        model.addConstr(g_eb[i] == beta_eb * p_eb[i])
        # --- energy balance ---
        model.addConstr(p_pv[i] + p_fc[i] + p_pur[i] - p_el[i] - p_hp[i] - p_eb[i] - load_data["P_DE"][i] == 0)
        model.addConstr(g_fc[i] + g_hp[i] + g_eb[i] - g_hw[i] - load_data["G_DE"][i] == 0)
        model.addConstr(q_cw[i] + q_hp[i] - load_data["Q_DE"][i] == 0)

    # ------ optimize ------
    model.Params.NonConvex = 2
    model.Params.MIPGap = 0.01  # 百分比界差
    model.Params.TimeLimit = 6000  # 限制求解时间为 6000s

    tick = time.time()
    print("Start Opt...")
    model.optimize()
    tock = time.time()
    opt_time = tock - tick

    # ------ get results ------
    if model.status == GRB.OPTIMAL:
        print('Optimal Objective: %g' % model.objVal)
        model.write(os.path.join(res_dir, 'HIES.sol'))

        c_OP_c = 0
        revenue = cool_area * cool_price + heat_area * heat_price
        c_sys = 0
        c_contrast = 0
        for i in range(time_scale):
            c_OP_c += ele_price[i] * (load_data["P_DE"][i]
                                      + load_data["G_DE"][i] / eta_ele_g
                                      + load_data["Q_DE"][i] / eta_ele_q)
            revenue += ele_price[i] * (load_data["P_DE"][i])
            c_sys += p_pur[i].x * c_p
            c_contrast += load_data["P_DE"][i] * c_p + load_data["G_DE"][i] * c_g + load_data["Q_DE"][i] * c_p / eta_ele_q

        receive = c_IN.x / (revenue - c_OP.x)
        rel_receive = c_IN.x / (c_OP_c - c_OP.x)
        c_reduce = c_contrast - c_sys

        dict_planning = {
            'obj': model.objVal,
            'P_PV': P_PV.x,
            'P_EL': P_EL.x,
            'P_FC': P_FC.x,
            'H_STO': H_STO.x,
            'G_HW': G_HW.x,
            'Q_CW': Q_CW.x,
            'P_HP': P_HP.x,
            'P_EB': P_EB.x,
            'E_cost': c_IN.x,
            'OPEX': c_OP.x,
            'OPEX_contrast': c_OP_c,
            'revenue': revenue,
            'receive': receive,
            'rel_receive': rel_receive,
            'c_sys': c_sys,
            'c_contrast': c_contrast,
            'c_reduce': c_reduce
        }
        dict_operation = {
            'p_pv': [p_pv[i].x for i in range(time_scale)],
            'p_el': [p_el[i].x for i in range(time_scale)],
            'h_el': [h_el[i].x for i in range(time_scale)],
            'p_fc': [p_fc[i].x for i in range(time_scale)],
            'h_fc': [h_fc[i].x for i in range(time_scale)],
            'g_fc': [g_fc[i].x for i in range(time_scale)],
            'h_sto': [h_sto[i].x for i in range(time_scale)],
            'h_sto_next': [h_sto_next[i].x for i in range(time_scale)],
            'g_hw_io': [g_hw_io[i].x for i in range(time_scale)],
            'g_hw': [g_hw[i].x for i in range(time_scale)],
            'g_hw_next': [g_hw_next[i].x for i in range(time_scale)],
            'q_cw_io': [q_cw_io[i].x for i in range(time_scale)],
            'q_cw': [q_cw[i].x for i in range(time_scale)],
            'q_cw_next': [q_cw_next[i].x for i in range(time_scale)],
            'p_hp': [p_hp[i].x for i in range(time_scale)],
            'g_hp': [g_hp[i].x for i in range(time_scale)],
            'q_hp': [q_hp[i].x for i in range(time_scale)],
            'z_hp_g': [z_hp_g[i].x for i in range(time_scale)],
            'z_hp_q': [z_hp_q[i].x for i in range(time_scale)],
            'p_eb': [p_eb[i].x for i in range(time_scale)],
            'g_eb': [g_eb[i].x for i in range(time_scale)],
            'p_pur': [p_pur[i].x for i in range(time_scale)],
            'h_pur': [h_pur[i].x for i in range(time_scale)],
        }
        df_plan = pd.DataFrame(dict_planning, index=['planning'])
        df_opt = pd.DataFrame(dict_operation)
        if mode == "opt":
            df_plan.to_csv(os.path.join(res_dir, 'dict_planning.csv'))
            df_opt.to_csv(os.path.join(res_dir, 'dict_operation.csv'))
        elif mode == "ele":
            df_plan.to_csv(os.path.join(res_dir, 'ele_dict_planning.csv'))
            df_opt.to_csv(os.path.join(res_dir, 'ele_dict_operation.csv'))
        elif mode == "cool":
            df_plan.to_csv(os.path.join(res_dir, 'cool_dict_planning.csv'))
            df_opt.to_csv(os.path.join(res_dir, 'cool_dict_operation.csv'))
        return dict_planning, dict_operation

    elif model.status != GRB.INFEASIBLE:
        print('Optimization stopped with status %d' % model.status)
        return None, None

    elif model.status == GRB.INFEASIBLE:
        print('Model is infeasible')
        model.computeIIS()
        model.write(os.path.join(res_dir, 'model.ilp'))
        print("Irreducible inconsistent subsystem is written to file 'model.ilp'")
        return None, None

    else:
        print('Optimization stopped with status %d' % model.status)
        return None, None


def main():
    load_file = "./data_source/yulin_water_load.xlsx"
    param_config = "./config.json"
    res_dir = "./Output/"
    time_scale = 24 * 365
    device_life = 20
    mode = "cool"

    if param_config.endswith(".yaml") or param_config.endswith(".yml"):
        with open(param_config, "r", encoding="utf-8") as file:
            param_cfg = yaml.safe_load(file)
    elif param_config.endswith(".json"):
        with open(param_config, "r", encoding="utf-8") as file:
            param_cfg = json.load(file)
    else:
        raise ValueError("Unsupported param_config file format!")
    load_data = get_load_data(load_file, mode=mode)

    dict_planning, dict_operation = my_opt(
        param_cfg=param_cfg,
        load_data=load_data,
        time_scale=time_scale,
        device_life=device_life,
        res_dir=res_dir,
        mode=mode
    )


if __name__ == "__main__":
    main()
