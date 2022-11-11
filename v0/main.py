from model.DGI import * 

from gh import * 
import genghao_lightning as gl 

G = pickle_load('/Dataset/PyG/PubMed/Processed/PubMed.dglg.pkl')

HYPER_PARAM = dict(
    embed_dim = 512, 
    gcn_activation = nn.PReLU(),
    readout_activation = nn.Sigmoid(), 

    num_epochs = 500, 
    lr = 0.001,
    weight_decay = 0., 
    eval_interval = 10, 
)


def train_func(model, g, feat, **useless):
    loss = model(g=g, feat=feat)
    
    return dict(loss=loss)


def val_test_func(model, g, feat, label, train_mask, val_mask, test_mask, **useless):
    with torch.no_grad():
        embed = model.embed(g=g, feat=feat).detach() 
        
    result = gl.linear_classify(
        train_feat = embed[train_mask],
        train_label = label[train_mask],
        val_feat = embed[val_mask],
        val_label = label[val_mask],
        test_feat = embed[test_mask],
        test_label = label[test_mask],
    )
    
    val_f1_micro = result['best_val_f1_micro']
    val_f1_macro = result['best_val_f1_macro']
    test_f1_micro = result['best_test_f1_micro']
    test_f1_macro = result['best_test_f1_macro']

    return dict(
        val = dict(
            val_f1_micro = val_f1_micro,
            val_f1_macro = val_f1_macro,
        ),
        test = dict(
            test_f1_micro = test_f1_micro,
            test_f1_macro = test_f1_macro,
        ),
    )


def main():
    set_cwd(__file__)
    
    g = G 
    feat = g.ndata.pop('feat')
    feat_dim = feat.shape[-1]
    label = g.ndata.pop('label')
    num_classes = len(label.unique())
    train_mask = g.ndata.pop('train_mask')
    val_mask = g.ndata.pop('val_mask')
    test_mask = g.ndata.pop('test_mask')

    model = DGI(
        in_dim = feat_dim,
        out_dim = HYPER_PARAM['embed_dim'],
        gcn_activation = HYPER_PARAM['gcn_activation'],
        readout_activation = HYPER_PARAM['readout_activation'], 
    )
    
    trainer = gl.FullBatchTrainer(
        model = model,
        use_wandb = True, 
        project_name = 'DGI',
        param_dict = HYPER_PARAM, 
    )
    
    trainer.train_and_eval(
        dataset = dict(
            g = g,
            feat = feat, 
            label = label, 
            train_mask = train_mask, 
            val_mask = val_mask, 
            test_mask = test_mask, 
        ),
        train_func = train_func,
        val_test_func = val_test_func,
        evaluator = gl.UnsupervisedMultiClassClassificationEvaluator(), 
        optimizer_type = 'Adam',
        optimizer_param = dict(lr=HYPER_PARAM['lr'], weight_decay=HYPER_PARAM['weight_decay']), 
        num_epochs = HYPER_PARAM['num_epochs'], 
        eval_interval = HYPER_PARAM['eval_interval'], 
    )


if __name__ == '__main__':
    main() 
