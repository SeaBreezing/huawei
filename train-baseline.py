# -*- coding:utf-8 -*-

import os
import json
from collections import defaultdict

import argparse
import tensorflow as tf
import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import torch
from deepctr_torch.inputs import (DenseFeat, SparseFeat, VarLenSparseFeat,
                                  get_feature_names)
from deepctr_torch.models.dcn import DCN
# from deepctr.feature_column import SparseFeat, VarLenSparseFeat, DenseFeat, get_feature_names
#from deepctr.metrics import evaluation
from const import Const

from torch.nn import functional as F

def gen_seq_data(dataframe, fc, config):
    """
    sequence feature process, padding feature list to seq max len

    Args:
        dataframe: dataframe of the feature
        fc：feature name

    Returns:
        out: feature list after padding

    """
    temp_data = dataframe[fc.name].str.split('^').values
    out = []
    # 遍历temp_data的每一行，比如第0行是30157^30648^14278^31706
    for data in temp_data:
        data = data + [0] * (config[Const.SEQ_MAX_LEN] - len(data)) # config[Const.SEQ_MAX_LEN] = 5
        out.append(data)
    return out


def get_seq_dim(dataframe, fc): #pad填充
    """
    get max index of sequence feature

    Args:
        dataframe: dataframe of the feature
        fc：feature name

    Returns:
        dim：max index of sequence feature

    """
    data = dataframe[fc].str.split('^').values  
    out = [i for item in data for i in item]  #将fc(ad_click_list_v00)展平为列表
    return np.array(out).astype(int).max() + 1 #
 

def read_data_from_csv(input_dir):
    """
    read csv file

    Args:
        input_dir: input file dir

    Returns:
        train_data_ads_d: train dataframe
        test_data_df: test dataframe

    """
    train_data_ads_df = pd.read_csv(os.path.join(input_dir, Const.MERGE_TRAIN_DATA_FILE), dtype=str)
    test_data_df = pd.read_csv(os.path.join(input_dir, Const.MERGE_TEST_DATA_FILE), dtype=str)
    return train_data_ads_df, test_data_df


def get_input_data(train_df, test_df, config):
    """
    get input data for model

    Args:
        train_df： train dataframe
        test_df：test dataframe

    Returns:
        train_input：train input dict
        test_input： test input dict

    """
    sparse_feature_name = Const.MERGE_ADS_FEATURE_NAME # 合并后所有稀疏特征
    sequence_feature_name = Const.MERGE_SEQUENCE_FEATURE_NAME  #合并后所有序列特征

    print("start init feature...")
    all_data_df = pd.concat([train_df, test_df])  #合并训练集与测试集
    feature_columns = [] # 是个列表，往里面加具名元组SparseFeat
    dim_dict = {}
    #对所有稀疏特征构建embedding，feature_columns储存29个稀疏特征的最大值
    for feature in sparse_feature_name:  
        all_data_df[feature].fillna(-1, inplace=True) #用-1填补缺失值,需要修改为更合理的填充方法
        dim = np.array(all_data_df[feature]).astype(int).max() + 1 #dim使用all_data_df[feature].nunique()是否更好？
        # 取出当前feature的最大值并+1
        dim_dict[feature] = dim
        if dim > 0:
            feature_columns.append(
                SparseFeat(feature, dim + 1, config[Const.EMBEDDING_SIZE],
                           use_hash=config[Const.HASH_FLG], dtype=tf.int32))
    #对所有序列特征构建embedding
    for feature in sequence_feature_name:
        all_data_df[feature].fillna('-1', inplace=True) # 用'-1'填补缺失值,需要修改为更合理的填充方法
        dim = get_seq_dim(all_data_df, feature) # 得到特征序列维度
        dim_dict[feature] = dim # 2:36354 3:376 1.1:36354 2.1:2001 3.1:376 ca2:221
        if dim > 0:
            feature_columns.append(
                VarLenSparseFeat(SparseFeat(feature, vocabulary_size=dim + 1,
                                            embedding_dim=config[Const.EMBEDDING_SIZE]),
                                 maxlen=config[Const.SEQ_MAX_LEN], combiner=config[Const.POOLING_MODE])) #

    print("start feature fillna...")
    #使用统计的最大dim在训练集和测试集上填充，可修改
    for feature in sparse_feature_name:
        train_df[feature].fillna(dim_dict[feature], inplace=True)
        test_df[feature].fillna(dim_dict[feature], inplace=True)
    for feature in sequence_feature_name:
        train_df[feature].fillna(str(dim_dict[feature]), inplace=True)
        test_df[feature].fillna(str(dim_dict[feature]), inplace=True)

    print("start gen input...")
    train_input = {}
    test_input = {}
    #对于稀疏数据统一转int,对于序列数据进行pad填充 可以参考鱼佬的baseline对数据进行压缩
    for fc in feature_columns:
        if isinstance(fc, SparseFeat):
            train_input[fc.name] = train_df[fc.name].values.astype(int)
            test_input[fc.name] = test_df[fc.name].values.astype(int)
        elif isinstance(fc, VarLenSparseFeat):
            train_input[fc.name] = np.array(gen_seq_data(train_df, fc, config)).astype(int)
            test_input[fc.name] = np.array(gen_seq_data(test_df, fc, config)).astype(int)
    train_input['label'] = train_df['label'].values.astype(int) # train_input每个key对应的值长度均为7675517
    test_input['log_id'] = test_df['log_id'].values.astype(int) # test_input每个key对应的值长度均为976058
    return train_input, test_input, feature_columns


def get_model(feature_columns, config):
    """
    define model

    Args:
        feature_columns：feature list, define type of feature

    Returns:
        model：model

    """
    #未将两种feature分开，分开可能效果更好（可以看看DCN模型）
    return DCN(feature_columns, feature_columns, cross_num=config[Const.CROSS_NUM],
               cross_parameterization=config[Const.CROSS_PARAMETERIZATION],
               dnn_hidden_units=config[Const.HIDDEN_SIZE], dnn_dropout=config[Const.DNN_DROPOUT])


def get_early_stop(config):
    """
    get early stop of model

    Args:
        config: config

    Returns:
        early stop callback

    """
    mode = 'min' if config[Const.EARLY_STOP_DEPENDENCY] == 'loss' else 'max'
    return EarlyStopping(monitor=config[Const.EARLY_STOP_DEPENDENCY],
                         mode=mode,
                         patience=config[Const.EARLY_STOP_PATIENCE])


def get_checkpoint(ckptpath, config):
    """
    get checkpoint of model

    Args:
        ckptpath: checkpoint dir
        config: config

    Returns:
        model_check_point： checkpoint

    """
    model_check_point = ModelCheckpoint(
        os.path.join(ckptpath, "cp-{epoch:04d}.ckpt"),
        save_best_only=True,
        monitor=config[Const.EARLY_STOP_DEPENDENCY],
        mode='min'
        if config[Const.EARLY_STOP_DEPENDENCY] == 'loss' else 'max',
        save_weights_only=True)

    return model_check_point


def get_callback_list(input_dir, config):
    """
    get callback list
    Args:
        input_dir: input file dir
        config: config

    Returns:
        callback list

    """
    callback_list = []
    callback_list.append(get_early_stop(config))
    callback_list.append(get_checkpoint(os.path.join(input_dir, Const.CHECK_POINT_PATH), config))

    return callback_list


def train(model, train_data, input_dir, config):
    """
    train model
    Args:
        model: model
        train_data: train data
        input_dir: input file dir
        config: config

    Returns:

    """
    print("start model compile!")
    model.compile(config[Const.OPTIMIZER], config[Const.LOSS], metrics=config[Const.METRICS])
    print("start model fit!")
    data = dict((key, value) for key, value in train_data.items() if key != 'label')
    model.fit(data,
              train_data['label'],
              batch_size=config[Const.BATCH_SIZE],
              verbose=config[Const.VERBOSE],
              epochs=config[Const.EPOCHS],
              shuffle=config[Const.SHUFFLE],
              validation_split=config[Const.VALIDATION_SPLIT],
              callbacks=get_callback_list(input_dir, config))

    
def predict(model, test_data, output_dir, config):
    """
    predict to get pctr score

    Args:
        model: model
        test_data: test data
        output_dir: output dir
        config: config

    Returns:

    """
    print("start model predict!")
    data = dict((key, value) for key, value in test_data.items() if key != 'log_id')
    pred_ans = model.predict(data, batch_size=config[Const.BATCH_SIZE])
    data = np.concatenate((test_data['log_id'].reshape(-1, 1), pred_ans.reshape(-1, 1)),
                          axis=1)
    result_df = pd.DataFrame(data, columns=['log_id', 'pctr'])
    print("save pctr to submission.csv")
    result_df.to_csv(path_or_buf=os.path.join(output_dir, Const.PREDICT_OUT_FILE), index=False)


def get_hyperparam(model_config_path):
    """
    get_hyperparam

    Args:
        model_config_path: config file path
 
    Returns:
        config: config

    """
    config = defaultdict()
    model_config_dict = json.loads(open(os.path.join(model_config_path, Const.CONFIG_FILE), 'r').read())

    def set_default_val(key, default):
        nonlocal config
        nonlocal model_config_dict
        if key in model_config_dict.keys():
            config[key] = model_config_dict[key]
        else:
            config[key] = default

    set_default_val(Const.EMBEDDING_SIZE, 16)
    set_default_val(Const.BATCH_SIZE, 5120)
    set_default_val(Const.EPOCHS, 10)
    set_default_val(Const.VALIDATION_SPLIT, 0.1)
    set_default_val(Const.VERBOSE, 2)
    set_default_val(Const.SHUFFLE, True)
    set_default_val(Const.SEQ_MAX_LEN, 5)
    set_default_val(Const.HASH_FLG, False)
    set_default_val(Const.POOLING_MODE, 'sum')
    set_default_val(Const.CROSS_NUM, 2)
    set_default_val(Const.CROSS_PARAMETERIZATION, 'vector')
    set_default_val(Const.HIDDEN_SIZE, (256, 128, 64))
    set_default_val(Const.EARLY_STOP_PATIENCE, 4)
    set_default_val(Const.EARLY_STOP_DEPENDENCY, 'val_loss')
    set_default_val(Const.DNN_DROPOUT, '0.5')
    set_default_val(Const.OPTIMIZER, 'adam')
    set_default_val(Const.LOSS, 'binary_crossentropy')
    set_default_val(Const.AUC_THRESHOLDS, 200)
    set_default_val(Const.METRICS, ['binary_crossentropy'])

    return config


def train_predict(input_dir, output_dir, model_config_path):
    """
    train_predict
    Args:
        input_dir: input file dir
        output_dir: output file dir

    Returns:
    """
    epoch = 5
    for i in range(epoch):
        
        print(f"start reading {input_dir}")
        train_df, test_df = read_data_from_csv(input_dir) #读取数据
        print(f"reading {input_dir} successfully")
        
        config = get_hyperparam(model_config_path)  #设置超参数(可调)

        train_data, test_data, feature_columns = get_input_data(train_df, test_df, config)  #处理并加载字典格式数据 

        model = get_model(feature_columns, config)  #加载模型
        model = DCN(feature_columns, feature_columns, cross_num=config[Const.CROSS_NUM],
               cross_parameterization=config[Const.CROSS_PARAMETERIZATION],
               dnn_hidden_units=config[Const.HIDDEN_SIZE], dnn_dropout=config[Const.DNN_DROPOUT])

        
        train_loss = 0
        train_acc = 0
        loop = tqdm(train_data, total = len(train_data))
        for batch in loop:
            model.train()
            output = model()
            outputs = model(data)
            loss = F.binary_cross_entropy_with_logits(outputs, train_data['label'])






    train(model, train_data, input_dir, config) #训练

    predict(model, test_data, output_dir, config) #预测


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='data', type=str, help="input file dir")
    parser.add_argument("--output_dir", default='output', type=str, help="output file dir")
    parser.add_argument("--model_config_path", default='config', type=str, help="config file dir")


    args, unknown = parser.parse_known_args()
    if len(unknown) != 0:
        print("unknown args:%s", unknown)

    input_dir = args.input_dir
    output_dir = args.output_dir
    model_config_path = args.model_config_path

    train_predict(input_dir, output_dir, model_config_path)
