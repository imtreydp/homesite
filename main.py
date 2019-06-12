import calendar
import pandas as pd
import numpy as np
from sklearn import compose, preprocessing
import xgboost as xgb


def add_date_features(df_in, date_col_list):
    df = df_in
    for col_name in date_col_list:
        df['{}_weekday'.format(col_name)] = df[col_name].apply(lambda x: calendar.day_abbr[x.dayofweek])
        df['{}_month'.format(col_name)] = df[col_name].apply(lambda x: calendar.month_name[x.month])
        df[col_name] = df[col_name].astype('int64')//1e9
    return df


def remove_fields(df_in, target_col, others_list=None):
    col_list = [target_col]
    if others_list:
        col_list = col_list + others_list
    df = df_in.drop(col_list, axis=1, errors='ignore')
    return df


def prepare_training_data(df_dict, target, remove, date_fields):
    df_dict_target = {}
    for df_name, df in df_dict.items():
        try:
            df_dict_target[df_name] = df[target]
        except KeyError:
            df_dict_target[df_name] = None
        df = remove_fields(add_date_features(df, date_col_list=date_fields), target, remove)
        df['df_id'] = df_name
        df_dict[df_name] = df

    df_master = pd.concat(df_dict.values(), axis=0, sort=False)

    one_hot_list = []
    scaler_list = []
    for idx, col in enumerate(df_master.columns):
        d_type = type(df_master[col].iloc[0])
        if col not in ['df_id']:
            if d_type in [str]:
                df_master[col] = df_master[col].fillna('N/A')
                one_hot_list.append(idx)
            elif d_type not in [np.datetime64, pd.Timestamp]:
                df_master[col] = df_master[col].fillna(df_master[col].median())
                scaler_list.append(idx)

    for df_name, df in df_dict.items():
        df_dict[df_name] = df_master.loc[df_master['df_id'] == df_name].drop('df_id', axis=1)

    df_master = df_master.drop('df_id', axis=1)

    ct = compose.ColumnTransformer(
        transformers=[
            ('one_hot_1', preprocessing.OneHotEncoder(sparse=False), one_hot_list),
            ('scaler_1', preprocessing.StandardScaler(), scaler_list)
        ],
        sparse_threshold=0
    ).fit(df_master, y=target)

    for df_name, df in df_dict.items():
        df_dict[df_name] = xgb.DMatrix(
            data=ct.transform(df),
            missing=np.nan
        )
        if df_dict_target[df_name] is not None:
            df_dict[df_name].set_label(df_dict_target[df_name])
    return df_dict


def train_model(model_name, training_data, params, boosting_rounds=3000, save=True):
    model = xgb.train(
        params=params,
        dtrain=training_data,
        num_boost_round=boosting_rounds,
        evals=[(training_data, model_name)],
        early_stopping_rounds=10
    )
    if save:
        model.save_model('models\\{}.model'.format(model_name))

    return model


def get_predictions(model_name, prediction_data, predict_df, predict_field_name, save=True):
    bst = xgb.Booster(model_file="models\\{}.model".format(model_name))
    df = predict_df.to_frame()
    df[predict_field_name] = pd.Series(bst.predict(prediction_data), name=predict_field_name)
    if save:
        df.to_csv('data\\{}_predictions.csv'.format(model_name), index=False)
    return df


def main():
    # name for the model/session
    name_model = 'homesite_2'

    # target field
    target_field = 'QuoteConversion_Flag'

    # record id field
    id_field = 'QuoteNumber'

    # date fields
    date_list = ['Original_Quote_Date']

    training_data_dict = prepare_training_data(
        df_dict=dict(
            train=pd.read_csv(
                filepath_or_buffer='data\\train.csv',
                parse_dates=date_list
            ),
            test=pd.read_csv(
                filepath_or_buffer='data\\test.csv',
                parse_dates=date_list
            )
        ),
        target=target_field,
        remove=[id_field, 'SalesField8'],
        date_fields=date_list
    )

    train_model(
        model_name=name_model,
        training_data=training_data_dict['train'],
        params={
            'eta': .03,
            'subsample': .45,
            'lambda': 11,
            'gamma': 11,
            'min_child_weight': 6,
            'max_depth': 10,
            'objective': 'binary:logistic',
            'eval_metric': 'auc',
            'silent': 1,
            'seed': 42
        },
        boosting_rounds=3000
    )

    get_predictions(
        model_name=name_model,
        prediction_data=training_data_dict['test'],
        predict_df=pd.read_csv(
            filepath_or_buffer='data\\test.csv',
            header=0
        )[id_field],
        predict_field_name=target_field
    )

    print("done")


if __name__ == '__main__':
    main()
