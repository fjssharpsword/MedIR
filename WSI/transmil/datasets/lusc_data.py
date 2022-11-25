
from genericpath import exists
from torch.utils.data import DataLoader
import random
import torch
import pandas as pd
import torch.utils.data as data
import os
import numpy as np

random.seed(2022)

class LuscData(data.Dataset):
    def __init__(self, fold,dataset_cfg=None, state=None,n_bins=4,eps=1e-6):
        self.__dict__.update(locals())
        self.dataset_cfg = dataset_cfg
        self.eps=eps
        self.n_bins=n_bins
        self.state == 'train'

        self.fold = fold
        if self.dataset_cfg.dataset_name == 'brca_data':
            self.feature_dir= '/data_local2/ljjdata/TCGA/CLAM_preprocessing/size_512/BRCA/feat_dir/pt_files/'
            self.fold_dir= 'dataset_csv/brca'
            self.survival_info= 'dataset_csv/tcga_brca_all_clean.csv'
            self.feat_num = 2600
        elif self.dataset_cfg.dataset_name == 'lusc_data':
            self.feature_dir= '/data_local2/ljjdata/TCGA/CLAM_preprocessing/size_512/LUSC/feat_dir/pt_files/'
            self.fold_dir= 'dataset_csv/lusc'
            self.survival_info= 'dataset_csv/tcga_lusc_all_clean.csv'
            self.feat_num = 3000
        elif self.dataset_cfg.dataset_name == 'gbm_data':
            self.feature_dir= '/data_local2/ljjdata/TCGA/CLAM_preprocessing/size_512/GBM/feat_dir/pt_files/'
            self.fold_dir= 'dataset_csv/gbm'
            self.survival_info= 'dataset_csv/tcga_gbm_all_clean.csv'
            self.feat_num = 2500

        # self.feature_dir = self.dataset_cfg.data_dir
        # self.feat_num = self.dataset_cfg.feat_num

        self.csv_dir = self.fold_dir + f'/fold_{self.fold}.csv'

        self.patient_data = pd.read_csv(self.csv_dir)

        self.survival_info = pd.read_csv(self.survival_info)
        
        self.get_time_interval()

        self.survival_info.index = self.survival_info['slide_id']
        self.survival_info['survival_months'] = self.survival_info['survival_months'] / 30

        # self.get_time_interval_60()

        if state == 'train':
            self.case_ids = self.patient_data.loc[:, 'train'].dropna()
        elif state == 'val':
            self.case_ids = self.patient_data.loc[:, 'val'].dropna()
        elif state == 'test':
            self.case_ids = self.patient_data.loc[:, 'test'].dropna()

        # 一个case_id可能对应多个slide_id
        self.survival_data = pd.DataFrame()
        for slide_id in self.survival_info['slide_id']:
            case_id = self.survival_info.loc[slide_id,'case_id']
            exists_pt_path = os.path.exists(os.path.join(self.feature_dir, slide_id.replace('.svs','.pt')))
            if case_id in self.case_ids.values and exists_pt_path:
                self.survival_data = self.survival_data.append(self.survival_info.loc[slide_id])

        self.shuffle = self.dataset_cfg.data_shuffle

    def get_time_interval(self):
        '''
        划分区间
        '''
        
        patients_df = self.survival_info.drop_duplicates(['case_id']).copy()
        uncensored_df = patients_df[patients_df['censorship'] < 1]

        # 按照数据出现频率百分比划分，比如要把数据分为四份，则四段分别是数据的0-25%，25%-50%，50%-75%，75%-100%，每个间隔段里的元素个数都是相同的
        interval_labels, q_bins = pd.qcut(uncensored_df['survival_months'], q=self.n_bins, retbins=True, labels=False)
        q_bins[-1] = self.survival_info['survival_months'].max() + self.eps
        q_bins[0] = self.survival_info['survival_months'].min() - self.eps

        # 保存病人所属区间信息
        interval_labels, q_bins = pd.cut(patients_df['survival_months'], bins=q_bins, retbins=True, labels=False, right=False, include_lowest=True)
        patients_df.insert(2, 'interval_label', interval_labels.values.astype(int))

        # 相同的病人也保留区间信息
        self.survival_info.index = self.survival_info['case_id']
        patients_df.index = patients_df['case_id']
        for p_id in patients_df['case_id']:
            interval_label = patients_df.loc[p_id,'interval_label']
            self.survival_info.loc[p_id,'interval_label'] = interval_label

    def get_time_interval_60(self):
        '''
        划分区间
        '''
        interval_label = np.ceil(self.survival_info['survival_months'].values)
        interval_label = np.clip(interval_label, 1, 60)-1
        self.survival_info['interval_label'] = interval_label


    def __len__(self):
        return len(self.survival_data)

    def __getitem__(self, idx):
        label_series = self.survival_data.iloc[[idx]]
        survival_time = label_series['survival_months'].values[0]#  / 30
        state_label = 1 - label_series['censorship'].values[0]
        interval_label = label_series['interval_label'].values[0]
        slide_id = os.path.splitext(','.join(label_series['slide_id'].values))[0]  # array --> str --> 去后缀

        feat_file = os.path.join(self.feature_dir, slide_id) + '.pt'
        # features = pd.read_csv(feat_file).values
        features = torch.load(feat_file)

        # ----> shuffle
        if self.shuffle == True and self.state == 'train':
            index = [x for x in range(features.shape[0])]
            random.shuffle(index)
            features = features[index]

        if not self.feat_num is None:
            if features.shape[0] >= self.feat_num:
                features_vec = features[:self.feat_num, :]
            else:
                features_vec = features
                cha = self.feat_num - features.shape[0]
                shang = cha // features.shape[0]
                yu = cha % features.shape[0]
                for sh in range(shang):
                    features_vec = torch.cat([features_vec,features],dim=0)
                if yu > 0:
                    features_vec = torch.cat([features_vec,features[:yu,:]],dim=0)
        else:
            features_vec = torch.from_numpy(features)

        return features_vec, survival_time, state_label, interval_label,slide_id


if __name__ == '__main__':
    import argparse
    from utils.utils import read_yaml

    def make_parse():
        parser = argparse.ArgumentParser()
        parser.add_argument('--stage', default='train', type=str)
        parser.add_argument('--config', default='../LUSC/TransMIL.yaml', type=str)
        parser.add_argument('--gpus', default=[0])
        parser.add_argument('--fold', default=0)
        args = parser.parse_args()
        return args

    args = make_parse()
    cfg = read_yaml(args.config)


    LuscDataset = LuscData(dataset_cfg=cfg.Data, state='train')
    # dataloader打乱的是wsi的顺序，data_shuffle打乱的是patches的顺序
    dataloader = DataLoader(LuscDataset, batch_size=8, num_workers=8, shuffle=False)
    for idx, data in enumerate(dataloader):
        print('feature:', data[0].shape)
        print('feature:', data[0].dtype)
        print('survival_label:', data[1])
        print('censorship_label:', data[2])
        print('slide_id:', data[3])
        break