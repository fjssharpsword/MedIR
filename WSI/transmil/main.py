from torch.utils.tensorboard import SummaryWriter
import argparse
from torch.utils.data import DataLoader
import os
import numpy as np
import random
from utils import losses
from utils.utils import *
from datasets.lusc_data import LuscData
from MyOptimizer.optim_factory import create_optimizer
from models.TransMIL import TransMIL
from train import train
from eval import val_test
import pandas as pd
import warnings
from utils.logger import get_logger
warnings.filterwarnings("ignore")

def mkfile(file):
    if not os.path.exists(file):
        os.makedirs(file)


def write_csv(total_data,save_path, headers = ['id','time','state','interval_1','interval_2','interval_3','interval_4','risk']):
    
    df_feature = pd.DataFrame(total_data, columns=headers)
    df_feature.to_csv(save_path,index=False)

    

def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', default='train', type=str)
    parser.add_argument('--config', default='LUSC/TransMIL.yaml', type=str)
    # parser.add_argument('--gpus', type=str, default="5")
    parser.add_argument('--fold', default=0)
    args = parser.parse_args()
    return args

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def main(cfg,k):
    # ---->main
    if cfg.General.loss_name == 'CELoss':
        criterion = losses.CELoss()
    elif cfg.General.loss_name == 'CE+Loss':
        criterion = losses.CEPlusLoss()
    elif cfg.General.loss_name == 'CoxLoss':
        criterion = losses.CoxLoss()
    elif cfg.General.loss_name == 'Cox+Loss':
        criterion = losses.CoxPlusLoss()
    elif cfg.General.loss_name == 'OursLoss':
        criterion = losses.TMCLoss()
    elif cfg.General.loss_name == 'NLLLoss':
        criterion = losses.NLLLoss()
    print('load dataset')
    train_dataset = LuscData(fold=k,dataset_cfg=cfg.Data, state='train',n_bins=cfg.Model.n_classes)
    train_loader = DataLoader(train_dataset, batch_size=cfg.Data.train_dataloader.batch_size,
                              num_workers=cfg.Data.train_dataloader.num_workers, shuffle=True, drop_last=False)
    val_dataset = LuscData(fold=k,dataset_cfg=cfg.Data, state='val',n_bins=cfg.Model.n_classes)
    val_loader = DataLoader(val_dataset, batch_size=cfg.Data.test_dataloader.batch_size,
                            num_workers=cfg.Data.test_dataloader.num_workers, shuffle=False, drop_last=False)
    print('finish load dataset!')

    model = TransMIL(n_classes=cfg.Model.n_classes).cuda()
    optimizer = create_optimizer(cfg.Optimizer, model)

    # save_dir = cfg.General.output_path + '/' + cfg.Version + "/" + cfg.Date
    save_dir = os.path.join(cfg.General.output_path,cfg.Data.dataset_name,cfg.General.loss_name,str(k))
    mkfile(save_dir)
    

    writer_train = SummaryWriter(os.path.join(save_dir, 'run_tensorboard/Train'))
    writer_val = SummaryWriter(os.path.join(save_dir, 'run_tensorboard/Val'))
    print('init SummaryWriter')

    best_score = 0.0
    best_epoch = 0
    last_kappa = 0
    last_auc = 0
    last_acc = 0
    for epoch in range(1, cfg.General.epochs + 1):
        print('\nEpoch [k={} {}/{}]'.format(k,epoch, cfg.General.epochs))
        logger.info('\nEpoch [k={} {}/{}]'.format(k,epoch, cfg.General.epochs))

        # train
        train_loss, train_score,model,train_total_time,train_total_label = train(train_loader, optimizer, model,criterion,logger,cfg)
        writer_train.add_scalar('loss/total', train_loss, epoch)
        writer_train.add_scalar('score/c_index', train_score, epoch)

        # val
        val_loss, val_score,total_data,mean_auc_5year,kappa_5_year,tpr,fpr,optimal_threshold,acc = val_test(val_loader, model,criterion,cfg,train_total_time,train_total_label)
        writer_val.add_scalar('loss/total', val_loss, epoch)
        writer_val.add_scalar('score/c_index', val_score, epoch)
        writer_val.add_scalar('score/auc', mean_auc_5year, epoch)
        writer_val.add_scalar('score/kappa', kappa_5_year, epoch)
        writer_val.add_scalar('score/acc', acc, epoch)

        print('[TRAIN]\nloss: {:.4f} | cindex: {:.4f}\n[VAL]\nloss: {:.4f} | cindex: {:.4f} auc={:.4f} kappa={:.4f} acc={:.4f} lr={:.4f}'.format(train_loss,
                                                                                                  train_score,
                                                                                                  val_loss,
                                                                                                  val_score,mean_auc_5year,kappa_5_year,acc,optimizer.param_groups[0]['lr']))
        logger.info('[TRAIN]\nloss: {:.4f} | cindex: {:.4f}\n[VAL]\nloss: {:.4f} | cindex: {:.4f} auc={:.4f} kappa={:.4f} acc={:.4f} lr={:.4f}'.format(train_loss,
                                                                                                  train_score,
                                                                                                  val_loss,
                                                                                                  val_score,mean_auc_5year,kappa_5_year,acc,optimizer.param_groups[0]['lr']))
        if val_score >= best_score:
            best_score = val_score
            last_kappa = kappa_5_year
            last_auc = mean_auc_5year
            last_acc = acc
            best_epoch = epoch
            torch.save(model.state_dict(), save_dir + '/model_weight.pth')

            save_d = os.path.join(save_dir,'csv')
            if not os.path.exists(save_d):
                os.makedirs(save_d)
            save_csv_path = os.path.join(save_d,str(k)+'.csv')
            save_tpr_fpr_path = os.path.join(save_d,str(k)+'_tpr_fpr.csv')
        
            write_csv(total_data,save_csv_path, headers = ['slide_id','time','state','interval','interval_1','interval_2','interval_3','interval_4'])  
            write_csv(np.stack([tpr,fpr,[optimal_threshold for x in range(len(fpr))]],axis=1),save_tpr_fpr_path, headers = ['tpr','fpr','optimal_threshold'])    

        print('[BEST] cindex: {:.4f} its auc:{:.4f} kappa:{:.4f} acc:{:.4f} in epoch {}'.format(best_score,last_auc,last_kappa,last_acc,best_epoch))
        logger.info('[BEST] cindex: {:.4f} its auc:{:.4f} kappa:{:.4f} acc:{:.4f} in epoch {}'.format(best_score,last_auc,last_kappa,last_acc,best_epoch))

    writer_train.close()
    writer_val.close()

    return best_score,last_kappa,last_auc,last_acc


if __name__ == '__main__':
    args = make_parse()
    cfg = read_yaml(args.config)

    # ---->update
    cfg.config = args.config
    # cfg.General.gpus = args.gpus
    cfg.General.server = args.stage
    cfg.Data.fold = args.fold

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.General.gpus
    setup_seed(cfg.General.seed)

    save_dir = os.path.join(cfg.General.output_path,cfg.Data.dataset_name,cfg.General.loss_name)
    mkfile(save_dir)
    logger = get_logger(save_dir)

    best_scores = []
    last_kappas,last_aucs,last_accs = [],[],[]
    for k in range(cfg.Data.nfold):
        # if k != 1:
        #     continue
        best_score,last_kappa,last_auc,last_acc = main(cfg,k)
        best_scores.append(best_score)
        last_kappas.append(last_kappa)
        last_aucs.append(last_auc)
        last_accs.append(last_acc)

        break
    
    print('best_cindex',best_scores) 
    logger.info('best_cindex '+str(best_scores))

    print('last auc',last_aucs) 
    logger.info('last auc '+str(last_aucs))

    print('last kappa',last_kappas) 
    logger.info('last kappa '+str(last_kappas))

    print('last acc',last_accs) 
    logger.info('last acc '+str(last_accs))
    
    print('cindex Mean {:.4f}'.format(np.mean(best_scores)))
    print('cindex STD {:.4f}'.format(np.std(best_scores)))

    print('auc Mean {:.4f}'.format(np.mean(last_aucs)))
    print('auc STD {:.4f}'.format(np.std(last_aucs)))

    print('kappa Mean {:.4f}'.format(np.mean(last_kappas)))
    print('kappa STD {:.4f}'.format(np.std(last_kappas)))

    print('acc Mean {:.4f}'.format(np.mean(last_accs)))
    print('acc STD {:.4f}'.format(np.std(last_accs)))

    print("END")
    logger.info('cindex Mean {:.4f}'.format(np.mean(best_scores)))
    logger.info('cindex STD {:.4f}'.format(np.std(best_scores)))

    logger.info('auc Mean {:.4f}'.format(np.mean(last_aucs)))
    logger.info('auc STD {:.4f}'.format(np.std(last_aucs)))

    logger.info('kappa Mean {:.4f}'.format(np.mean(last_kappas)))
    logger.info('kappa STD {:.4f}'.format(np.std(last_kappas)))

    logger.info('acc Mean {:.4f}'.format(np.mean(last_accs)))
    logger.info('acc STD {:.4f}'.format(np.std(last_accs)))

    logger.info("END")
