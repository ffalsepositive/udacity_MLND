import sys
sys.path.append("../")

import pandas as pd

index_cols = [
    "dataset", "customer_id", "offer_id", "order"
]

feat_cols = [
    'recurrence', 'time', 'amount', 'total_trans_amount', 'total_trans_count', 
    'avg_trans_amount', 'time_after_last_trans', 'web', 'mobile', 'social', 
    'was_null_profile', 'oft_bogo', 'oft_discount', 'dfc_5', 'dfc_10', 'delta_3', 
    'delta_4', 'delta_5', 'delta_7', 'delta_10', 'rw_0', 'rw_2', 'rw_3', 'rw_5', 'rw_10', 
    'member_new', 'member_old', 'gen_F', 'gen_O', 'age_18_23', 'age_24_29', 'age_30_35', 
    'age_36_41', 'age_42_47', 'age_48_53', 'age_53_59', 'age_60_65', 'age_66_71', 'age_72_77', 
    'age_78_83', 'age_84_101', 'income_30k_40k', 'income_40k_50k', 'income_50k_60k', 
    'income_60k_75k', 'income_75k_100k', 'income_100k_120k'
]

target_cols = ["target"]

def load_customer_data(customer_id, dataset="train"):
    data = pd.read_csv("../data/processed/{}.csv".format(dataset), index_col=0)
    data.index.name = "index"
    data = data.reset_index()
    
    cols = {0:"customer_id", 1:"offer_id", 2:"order"}
    df = data["index"].str.split("-",expand=True).rename(columns=cols)
    
    data = data.join(df).drop(columns=["index"])
    data["dataset"] = dataset
    
    return  data[data["customer_id"] == customer_id].reset_index(drop=True)

def run_model(model, customer_id):
    train = load_customer_data(customer_id, "train")
    test = load_customer_data(customer_id, "test")

    df = pd.concat([train, test], axis=0)
    
    if len(df) == 0:
        return True, None
    else:
        pred_data = df[feat_cols].values
        df["prob_c0"] = model.predict_proba(pred_data)[:, 0]
        df["prob_c1"] = model.predict_proba(pred_data)[:, 1]
        df["prob_c2"] = model.predict_proba(pred_data)[:, 2]

        prob_cols = ["prob_c" + str(i) for i in range(3)]

        df = df[index_cols + feat_cols + prob_cols + target_cols].reset_index()
        df.index = df["dataset"] + "_sample_no: " + df["index"].astype(str)
        df = df.drop(columns=["index", "dataset"], axis=1)
    
    return False, df.T