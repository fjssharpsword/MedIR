from radiomics import featureextractor
import pandas as pd
import os,joblib
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score

#https://github.com/AIM-Harvard/pyradiomics
def features_select_lasso(x,y):
    '''
    去掉冗余特征，需要很长时间，建议把结果index保存出来
    '''
    print('features_select_lasso...')
    model_lassoCV = LassoCV(alphas=np.logspace(-3,1,50), cv=10, max_iter=1000).fit(x,y)# np.logspace(-3,1,50) 正则项前面的lambda系数,1e-3,1e1,范围内50个,目的是看哪个最优,cv=10数据划分的份数,其中一份为测试集,max_iter迭代次数
    print(model_lassoCV.alpha_) # 输出最后的alpha,如果接近np.logspace的两端,要适当调参
    print('\n')
    coef = pd.Series(model_lassoCV.coef_, index=x.columns) 
    print('coef {}'.format(coef))
    print('\n')
    index = coef[coef!=0].index # 挑选出的特征
    print('features_select_lasso Finish')

    return index


def get_calculate(x,y,model):
    predict_label = model.predict(x)
    positive_predict_proba = model.predict_proba(x)
    precison = precision_score(y, predict_label)
    recall = recall_score(y, predict_label)
    f1 = f1_score(y, predict_label)
    auc_score = roc_auc_score(y, positive_predict_proba[:,1])

    return precison, recall, f1, auc_score


def train(X, Y):
    print('SVM train...')
    # SVM model
    X = StandardScaler().fit_transform(X) # 尺度标准化

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
    
    model = svm.SVC(C=1, kernel='rbf', gamma='auto', probability=True).fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    train_precison, train_recall, train_f1, train_auc_score = get_calculate(X_train,y_train,model)
    test_precison, test_recall, test_f1, test_auc_score = get_calculate(X_test,y_test,model)

    print('train acc:{:.6f}, train precison:{:.6f}, train recall:{:.6f}, train f1:{:.6f}, train auc:{:.6f}'.format(train_acc,train_precison,train_recall,train_f1,train_auc_score))
    print('test acc:{:.6f}, test precison:{:.6f}, test recall:{:.6f}, test f1:{:.6f}, test auc:{:.6f}'.format(test_acc,test_precison,test_recall,test_f1,test_auc_score))

    print('train finish!')

    joblib.dump(model, 'train_model.m')


def read_features_csv(path):
    df_train = pd.read_csv(path)
    ImageID_train = list(df_train['ImageID'])
    df_train.index = ImageID_train

    X_train = df_train.iloc[:,39:]  # 从这一列（39）开始：original_shape_Elongation
    y_train = df_train['label'].astype("int")

    return X_train, y_train


def svae_feature(df,featureVector,label,ImageID):
    '''
    function:把所有特征保存在一个表格
    '''
    df_add = pd.DataFrame.from_dict(featureVector.values()).T
    df_add.columns = featureVector.keys()
    df_add.insert(0,'label',label) # 增加一列，训练需要用到
    df_add.insert(0,'ImageID',ImageID)
    df = pd.concat([df, df_add])

    return df


def extractor_features(paramsFile,img_path,mask_path):
    '''
    function:提取影像组学特征
    paramsFile:参数配置文件
    img_path:image Filepath or SimpleITK object
    mask_path:image Filepath or SimpleITK object
    '''
    extractor = featureextractor.RadiomicsFeatureExtractor(paramsFile)
    try:
        featureVector = extractor.execute(img_path, mask_path)
        # print(featureVector.items())
        return featureVector
    except Exception as e:
        print(e)
        return None


if __name__ =="__main__":
    paramsFile = r'exampleSettings/exampleMR_5mm.yaml'
    img_path = r'/data_local/2021_Data/Lymph/LN_Origin_Data_Annoy/LN_MRI/LN_MRI_NII_before_Annoy/'
    mask_path = r'/data_local/2021_Data/Lymph/LN_Origin_Data_Annoy/LN_mha/LN_mha_before/'
    img_label_info = r'/data_local/2021_Data/Lymph/LN_Origin_Data_Annoy/2021-2-21 LN-before.xlsx'
    df_img_info = pd.read_excel(img_label_info)
    df_img_info.index = df_img_info['ImageID']

    # 提取影像组学特征
    df = pd.DataFrame()
    for f in tqdm(os.listdir(img_path)):
        img_file_path = os.path.join(img_path,f)
        mask_file_path = os.path.join(mask_path,f)
        ImageID = f.replace('.nii.gz','')
        label = df_img_info.loc[ImageID,'label']
        featureVector = extractor_features(paramsFile,img_file_path,mask_file_path)
        if featureVector is not None:
            df = svae_feature(df,featureVector,label,ImageID)


    df.to_csv(r'featureVector.csv',index=False)

    # 特征降维
    x, y = read_features_csv(r'featureVector.csv')
    index = features_select_lasso(x,y)
    # 训练
    train(x[index],y)
    