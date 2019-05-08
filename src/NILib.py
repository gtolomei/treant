import os
import numpy as np
import pandas as pd
import json


def label_encode(dataset, categorical_features):
    dataset_le = dataset.copy()
    for column in dataset_le.columns:
        if column in categorical_features:
            dataset_le[column] = dataset_le[column].astype('category')
            dataset_le[column] = dataset_le[column].cat.codes.astype(np.int32)
    return dataset_le

def load_atk_train_valid_test(atk_train_file, atk_valid_file, atk_test_file, 
                              train_split=0.6, valid_split=0.2, force=False):
    
    
    if  (force or 
          not os.path.exists(atk_train_file+".cat.bz2") or
          not os.path.exists(atk_valid_file+".cat.bz2") or
          not os.path.exists(atk_test_file+".cat.bz2") or 
          not os.path.exists(atk_train_file+".cat.json") ):
    
        print ("Pre-processing original files...")

        print ("Loading:", atk_train_file)
        print ("Loading:", atk_valid_file)
        print ("Loading:", atk_test_file)

        train = pd.read_csv(atk_train_file)
        valid = pd.read_csv(atk_valid_file)
        test  = pd.read_csv(atk_test_file)
        
        print ("Train/Valid/Test sizes:", train.shape, valid.shape, test.shape)
        print ("Train/Valid/Test split: {:.2f} {:.2f} {:.2f}"
                   .format( train.shape[0]/(train.shape[0]+valid.shape[0]+test.shape[0]),
                            valid.shape[0]/(train.shape[0]+valid.shape[0]+test.shape[0]),
                            test.shape[0] /(train.shape[0]+valid.shape[0]+test.shape[0]) ) )


        # split-back into train valid test
        if 'instance_id' in train.columns.values:
            print ('   ... with instance ids')
            valid['instance_id'] += train.iloc[-1,0]
            test['instance_id']  += valid.iloc[-1,0]
            assert max(train['instance_id'])<min(valid['instance_id']), "Instance ID mismatch"
            assert max(valid['instance_id'])<min(test['instance_id']), "Instance ID mismatch"
            
            groups = np.concatenate( [ train['instance_id'].value_counts().sort_index().values,
                                       valid['instance_id'].value_counts().sort_index().values,
                                       test['instance_id'].value_counts().sort_index().values ] )
            
            num_train_groups = int( len(groups)*train_split )
            train_size = sum(groups[:num_train_groups])
            num_valid_groups = int( len(groups)*valid_split )
            valid_size = sum(groups[num_train_groups:num_train_groups+num_valid_groups])
        else:
            full_size = len(train) + len(valid) + len(test)
            train_size = int( full_size*train_split )
            valid_size = int( full_size*valid_split )
        
        # concat to process correctly label encoding
        full = pd.concat( [train, valid, test] )

        # get index of categorical features (-1 because of instance_id)
        cat_fx = full.columns.values[np.where(full.dtypes=='object')[0]]
        cat_fx = list(cat_fx)    
        full = label_encode(full, cat_fx)
        with open(atk_train_file+".cat.json", 'w') as fp:
            json.dump(cat_fx, fp)
        print ("CatFX:", cat_fx)

        train_cat = full.iloc[0:train_size,:]
        valid_cat = full.iloc[train_size:train_size+valid_size,:]
        test_cat  = full.iloc[train_size+valid_size:,:]
        
        assert len(train_cat)+len(valid_cat)+len(test_cat)==len(full), "Split sizes mismatch"
        

        print ("Train/Valid/Test sizes:", train_cat.shape, valid_cat.shape, test_cat.shape)
        print ("Train/Valid/Test split: {:.2f} {:.2f} {:.2f}"
                   .format( train_cat.shape[0]/(train_cat.shape[0]+valid_cat.shape[0]+test_cat.shape[0]),
                            valid_cat.shape[0]/(train_cat.shape[0]+valid_cat.shape[0]+test_cat.shape[0]),
                            test_cat.shape[0] /(train_cat.shape[0]+valid_cat.shape[0]+test_cat.shape[0]) ) )

        # save to file
        print ("Saving processed files *.cat.bz2")
        train_cat.to_csv(atk_train_file+".cat.bz2", compression="bz2", index=False)
        valid_cat.to_csv(atk_valid_file+".cat.bz2", compression="bz2", index=False)
        test_cat.to_csv (atk_test_file+".cat.bz2",  compression="bz2", index=False)
        
    else:
        print ("Loading pre-processed files...")

        train_cat = pd.read_csv(atk_train_file+".cat.bz2")
        valid_cat = pd.read_csv(atk_valid_file+".cat.bz2")
        test_cat  = pd.read_csv(atk_test_file+".cat.bz2")
        
        with open(atk_train_file+".cat.json", 'r') as fp:
            cat_fx = json.load(fp)
    
    # return data
    return train_cat, valid_cat, test_cat, cat_fx


# # Objective Functions

# ## Standard
# 
# The following function, called <code>optimize_log_loss</code>, is the one that should be optimized (i.e., minimized) for learning _standard_ and _baseline_ approaches. More specifically, this is the standard binary log loss which is used to train any _standard_ or _baseline_ model.

# # $L$ = <code>optimize_log_loss</code>
# 
# $$
# L = \frac{1}{|\mathcal{D}|} \cdot \sum_{(\mathbf{x},y) \in \mathcal{D}}\ell(h(\mathbf{x}), y)
# $$
# 
# where:
# 
# $$
# \ell(h(\mathbf{x}), y) = log(1+e^{(-yh(\mathbf{x}))})
# $$

# In[ ]:


def optimize_log_loss(preds, train_data):
    labels = train_data.get_label()
    exp_pl = np.exp(preds * labels)
    # http://www.wolframalpha.com/input/?i=differentiate+log(1+%2B+exp(-kx)+)
    grads = -labels / (1.0 +  exp_pl)  
    # http://www.wolframalpha.com/input/?i=d%5E2%2Fdx%5E2+log(1+%2B+exp(-kx)+)
    hess = labels**2 * exp_pl / (1.0 + exp_pl)**2 

    # this is to optimize average logloss
    norm = 1.0/len(preds)
    grads *= norm
    hess *= norm
    
    return grads, hess


# ## Custom
# 
# In addition to the standard binary log loss used to train a model, we introduce our custom <code>optimize_non_interferent_log_loss</code>, which is computed as the weighted combination of two objective functions, as follows:
# 
# -  $L$ = <code>optimize_log_loss</code> (standard, already seen above);
# -  $L^A$ = <code>optimize_log_loss_uma</code> (custom, defined below).

# # $L^A$ = <code>optimize_log_loss_uma</code>
# 
# This function is used to train a **full** _non-interferent_ model; in other words, full non-interferent models are learned by optimizing (i.e., minimizing) the function which measures the binary log loss **under the maximal attack** possible.
# 
# $$
# L^A = \frac{1}{|\mathcal{D}|} \cdot \sum_{(\mathbf{x},y) \in \mathcal{D}} \log  \left( \sum_{\mathbf{x}' \in \mathit{MaxAtk}({\mathbf{x}},{A})} e^{\ell(h(\mathbf{x}'), y)} \right).
# $$
# 
# where still:
# 
# $$
# \ell(h(\mathbf{x}), y) = log(1+e^{(-yh(\mathbf{x}))})
# $$

# In[ ]:


def optimize_log_loss_uma(preds, train_data):
    labels = train_data.get_label()
    attack_lens = train_data.get_group()
    
    grads = np.zeros_like(labels, dtype=np.float64)
    hess = np.zeros_like(grads)
    
    if attack_lens is not None:

        norm = 1.0 / float(len(attack_lens))

        offset = 0
        for atk in attack_lens:
            exp_pl = np.exp(- preds[offset:offset+atk] * labels[offset:offset+atk])

            inv_sum = 1.0 / np.sum(1.0 + exp_pl)

            x_grad = inv_sum * exp_pl

            grads[offset:offset+atk] = norm * x_grad * (- labels[offset:offset+atk])
            hess[offset:offset+atk]  = norm * x_grad * (1.0 - x_grad)

            offset += atk    
    
    return grads, hess

def optimize_log_loss_uma_ext(preds, train_data):
    labels = train_data.get_label()
    weights = train_data.get_weight()
    
    grads = np.zeros_like(labels, dtype=np.float64)
    hess = np.zeros_like(grads)
    
    norm = 1.0 / float(len(labels))

    exp_pl = np.exp(- preds * labels)

    x_grad = weights * exp_pl

    grads = norm * x_grad * (- labels)
    hess  = norm * x_grad * (1.0 - x_grad)

    return grads, hess

# # <code>optimize_non_interferent_log_loss</code>
# 
# $$
# \alpha\cdot L^A + (1-\alpha)\cdot L
# $$
# 
# $$
# \alpha \cdot \underbrace{\Bigg[\frac{1}{|\mathcal{D}|} \cdot \sum_{(\mathbf{x},y) \in \mathcal{D}} \log  \left( \sum_{\mathbf{x}' \in \mathit{MaxAtk}({\mathbf{x}},{A})} e^{\ell(h(\mathbf{x}'), y)} \right)\Bigg]}_{L^A} + (1-\alpha) \cdot \underbrace{\Bigg[\frac{1}{|\mathcal{D}|} \cdot \sum_{(\mathbf{x},y) \in \mathcal{D}} \ell(h(\mathbf{x}, y))\Bigg]}_{L}
# $$

# In[ ]:


def optimize_non_interferent_log_loss(preds, train_data, alpha=1.0):
    # binary logloss under maximal attack
    # grads_uma, hess_uma = optimize_log_loss_uma(preds, train_data)
    grads_uma, hess_uma = optimize_log_loss_uma_ext(preds, train_data)
    
    # binary logloss (plain)
    grads_plain, hess_plain = optimize_log_loss(preds, train_data)
    
    #print ("uma:   ", grads_uma.min(), grads_uma.max(), hess_uma.min(), hess_uma.max())
    #print ("plain: ", grads_plain.min(), grads_plain.max(), hess_plain.min(), hess_plain.max())
    #print ("uma:   ", np.quantile(grads_uma,[.25, .75]), np.quantile( hess_uma, [.25, .75]) )
    #print ("plain: ", np.quantile(grads_plain,[.25, .75]), np.quantile( hess_plain, [.25, .75]) )
    
    # combine the above two losses together
    k=1
    grads = alpha*grads_uma + (1.0-alpha)*grads_plain
    hess  = alpha*hess_uma  + (1.0-alpha)*hess_plain
#     grads *= k
#     hess *= k
    
    return grads, hess


# ## Using one objective function for both _standard_ and _non-interferent_ learning
# 
# The advantage of the <code>optimize_non_interferent_log_loss</code> function defined above is that we can wrap it so that we can use it as the only objective function (<code>fobj</code>) passed in to LightGBM. 
# 
# In other words, if we call <code>fobj=optimize_non_interferent_log_loss</code> with <code>alpha=0.0</code>, this will end up optimizing (i.e., minimizing) the "vanilla" objective function (i.e., the standard binary log loss, defined by the function <code>optimize_log_loss</code> above).
# 
# Conversely, calling <code>fobj=optimize_non_interferent_log_loss</code> with <code>alpha=1.0</code> turns into optimizing (i.e., minimizing) the full non-interferent objective function (i.e., the custom binary log loss under max attack, defined by the function <code>optimize_log_loss_uma</code> above).
# 
# Anything that sits in between (i.e., <code>0 < alpha < 1</code>) optimizes an objective function that trades off between the standard and the full non-interferent term.

# # Evaluation Metrics

# ## Standard
# 
# The following function is the one used for evaluating the quality of the learned model (either _standard_, _adversarial-boosting_, or _non-interferent_). This is the standard <code>avg_log_loss</code>.

# In[ ]:


def logistic(x):
    return 1.0/(1.0 + np.exp(-x))


# In[ ]:


def logit(p):
    return np.log(p/(1-p))


# # <code>avg_log_loss</code>

# In[ ]:


# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, value: array, is_higher_better: bool
def avg_log_loss(preds, train_data):
    
    labels = train_data.get_label()
    losses = np.log(1.0 + np.exp(-preds*labels))
    avg_loss = np.mean(losses)
    
    return 'avg_binary_log_loss', avg_loss, False


# ## Custom
# 
# Similarly to what we have done for <code>fobj</code>, <code>feval</code> can be computed from a weighted combination of two evaluation metrics:
# 
# -  <code>avg_log_loss</code> (standard, defined above);
# -  <code>avg_log_loss_uma</code> (custom, defined below).

# # <code>avg_log_loss_uma</code>
# 
# This is the binary log loss yet modified to operate on groups of perturbed instances.

# In[ ]:


# Our custom metrics
def binary_log_loss(pred, true_label):

    return np.log(1.0 + np.exp(-pred * true_label))

# self-defined eval metric
# f(preds: array, train_data: Dataset) -> name: string, value: array, is_higher_better: bool
def avg_log_loss_uma(preds, train_data):
    labels = train_data.get_label()
    attack_lens = train_data.get_group()
    
    offset = 0
    max_logloss = []
    avg_max_logloss = 0.0
    
    if attack_lens is not None:
    
        for atk in attack_lens:
            losses = [binary_log_loss(h,t) for h,t in zip(preds[offset:offset+atk], labels[offset:offset+atk])]
            max_logloss.append(max(losses))

            offset += atk
        
        avg_max_logloss = np.mean(max_logloss)  

    return 'avg_binary_log_loss_under_max_attack', avg_max_logloss, False


# # <code>feval=avg_non_interferent_log_loss</code>
# 
# Used for measuring the validity of any model (either _standard_, _baseline_, or _non-interferent_). More precisely, <code>avg_non_interferent_log_loss</code> is the weighted sum of the binary log loss and the binary log loss under maximal attack.

# In[ ]:


def avg_non_interferent_log_loss(preds, train_data, alpha=1.0):
    
    # binary logloss under maximal attack
    _, loss_uma, _    = avg_log_loss_uma(preds, train_data)
    
    # binary logloss (plain)
    #_, loss_plain, _  = avg_log_loss(preds, train_data)
    ids = []
    attack_lens = train_data.get_group()
    if attack_lens is not None:
        offset=0
        for atk in attack_lens:
            ids += [offset]
            offset += atk      
            
    ids = np.array(ids)
    labels = train_data.get_label()
    losses = binary_log_loss(pred=preds[ids], true_label=labels[ids])
    loss_plain = np.mean(losses)

    # combine the above two losses together
    weighted_loss = alpha*loss_uma + (1.0-alpha)*loss_plain

    return 'avg_non_interferent_log_loss [alpha={:.2f}]'.format(alpha), weighted_loss, False


