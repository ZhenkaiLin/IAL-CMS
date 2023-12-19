import sys

import torch
from torch.nn import functional
from copy import deepcopy
from itertools import permutations
from mir_eval.separation import bss_eval_sources
from torch_mir_eval.separation import bss_eval_sources as gpu_bss_eval_sources
import pytorch_lightning as pl
from os.path import join
import librosa
from torch import nn

def SAR_calculation(source,estimate_source):
    # B*C*T B*C*T
    # 每段音频长度都为T
    # 且source 和 estimate_source 排序相同相同
    max_sir=30
    EPS = 1e-8
    C=source.size(1)
    device=source.device
    # Step 1. Zero_Norm
    mean_target = torch.mean(source, dim=2, keepdim=True)
    mean_estimate = torch.mean(estimate_source, dim=2, keepdim=True)
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate
    # zero_mean_target=zero_mean_target/(torch.norm(zero_mean_target,dim=2)+1e-8)
    # zero_mean_estimate=zero_mean_estimate/(torch.norm(zero_mean_estimate,dim=2)+1e-8)

    # B*C*T
    # Step 2. SIR
    # calculate dots(signal,interference) B*C*(C-1)
    artifacts=zero_mean_estimate-zero_mean_target #(B,C,T)

    artifacts_energy=torch.sum(artifacts ** 2, dim=2) + EPS #(B,C)
    signal_energy=torch.sum(zero_mean_estimate ** 2, dim=2) + EPS #(B,C)

    th = 10 ** (-max_sir / 10) * signal_energy
    sar = 10 * torch.log10(signal_energy.squeeze() / (artifacts_energy + th.squeeze() + EPS) + EPS)  # (B,C)

    return sar

def SIR_calculation(source,estimate_source):
    # B*C*T B*C*T
    # 每段音频长度都为T
    # 且source 和 estimate_source 排序相同相同
    max_sir=30
    EPS = 1e-8
    C=source.size(1)
    device=source.device
    # Step 1. Zero_Norm
    mean_target = torch.mean(source, dim=2, keepdim=True)
    mean_estimate = torch.mean(estimate_source, dim=2, keepdim=True)
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate

    # B*C*T
    # Step 2. SIR
    # calculate dots(signal,interference) B*C*(C-1)
    s_i_pair_dots = (zero_mean_estimate.unsqueeze(2) * zero_mean_target.unsqueeze(1)).sum(-1)
    signal_energy = torch.sum(zero_mean_estimate ** 2, dim=2, keepdim=True) + EPS  # [B,C,1]

    # calculating interference energy=(||s||*cos<s,s_inter>)**2
    # ||s||*cos(s,s_inter)=<s,s_inter>/||s_inter||
    # inter_energy=<s,s_inter>**2/||s_inter||**2
    target_energy = torch.sum(zero_mean_target ** 2, dim=2).unsqueeze(1) + EPS  # [B,1,C]
    inter_energy = s_i_pair_dots ** 2 / (target_energy + EPS)  # (B,C,C)
    off_diagonal_mask = (1 - torch.eye(C)).to(device).unsqueeze(0)  # (1,C,C)

    total_inter_energy = (inter_energy * off_diagonal_mask).sum(2)  # (B,C)
    th = 10 ** (-max_sir / 10) * signal_energy
    sir = 10 * torch.log10(signal_energy.squeeze() / (total_inter_energy + th.squeeze() + EPS) + EPS)  # (B,C)

    return sir

def add_sn(m):
    for name, layer in m.named_children():
        m.add_module(name, add_sn(layer))
    if isinstance(m, (nn.Conv2d, nn.Linear,nn.Conv1d,nn.Conv3d)):
        return nn.utils.spectral_norm(m)
    else:
        return m

def organize_batch(valid_feature,valid_nums):
    #valid_feature: (N,...) valid_nums: (B,NumMix*Objects) ->batch_feature(B,NumMix*Objects,.....)
    batch_feature=[]
    count=0
    B,NumMixObjects=valid_nums.size()
    valid_nums=valid_nums.flatten()
    for i in range(B*NumMixObjects):
        if valid_nums[i]:
            batch_feature.append(valid_feature[count])
            count+=1
        else:
            batch_feature.append(torch.zeros_like(valid_feature[0]))
    assert count==valid_feature.size(0) and len(batch_feature)==B*NumMixObjects
    return torch.stack(batch_feature).view(B,NumMixObjects,*(valid_feature.size()[1:]))
def combine_dictionaries(dicts):
    d={}
    for dict in dicts:
        d.update(dict)
    return d
def flatten_batch(batch_feature,valid_nums):
    # batch_feature: (B,NumMix*Objects) valid_nums: (B,NumMix*Objects) ->(N,.....)
    return batch_feature[valid_nums.bool()]

def get_normalization_for_G(C,normalization):
    if normalization == "batchnorm":
        return [nn.BatchNorm2d(C)]
    elif normalization == "spectralnorm":
        return []
    else:
        raise AttributeError("Unkown Normalization Layer Type")


def cal_and_visualize_feauture_cos_array(f, name):
    # f:(N,D)
    N, D = f.size()
    f1 = f[None, :, :]
    f2 = f1.view(N, 1, D)
    cos_array = functional.cosine_similarity(f1, f2, dim=2)
    vis_mask(cos_array, name)
    return


def cal_and_visualize_av_MIL_cos_array(fa, fv, name):
    # (N1,D) (N2,D,H,W)
    fa = fa[None, :, :, None, None]
    fv = fv[:, None, :, :, :]
    va_cos_array = functional.cosine_similarity(fv, fa, dim=2).max(-1)[0].max(-1)[0]
    vis_mask(va_cos_array, name)
    return

def random_sample_from_est_mel_mag( est_mel_mag):
    B, NumMix, C, F, T = est_mel_mag.size()


    est_mel_mag1 = est_mel_mag.flatten(end_dim=2)
    ind_sampled_components = est_mel_mag1[torch.randint(est_mel_mag1.size(0), (est_mel_mag1.size(0),))].view(B,
                                                                                                                  NumMix,
                                                                                                                  C, F,T)

    return ind_sampled_components

def random_sample_components_from_bank(N,bank,device):
    bank_size= bank.size(0)

    ind_sampled_components = bank[torch.randint(bank_size, (N,))]

    return ind_sampled_components.to(device)
def dependent_sample_from_est_mel_mag( est_mel_mag):
    N1, N2, F, T = est_mel_mag.size()
    device=est_mel_mag.device
    index=torch.randint(N2-1, (N1,N2)).to(device) #(N1,N2)
    index[index>=(torch.arange(N2).to(device)[None,:]) ]+=1
    index=index[:,:,None,None].repeat(1,1,F,T)
    dependent_sampled_components=est_mel_mag.gather(index=index,dim=1)

    return dependent_sampled_components

# WRONG
# def load_checkpoint_ignore_size_mismatch(checkpoint,map_location,lm_class,**kwargs):
#     checkpoint = pl.utilities.cloud_io.load(checkpoint, map_location=map_location)
#     lm=lm_class(**kwargs)
#     state_dict=checkpoint["state_dict"]
#
#     local_state = {k: v for k, v in lm.named_parameters() if v is not None}
#
#     for name, param in local_state.items():
#         if name in state_dict:
#             input_param=state_dict[name]
#             try:
#                 with torch.no_grad():
#                     param.copy_(input_param)
#             except Exception as ex:
#                 print('While copying the parameter named "{}", '
#                                   'whose dimensions in the model are {} and '
#                                   'whose dimensions in the checkpoint are {}, '
#                                   'an exception occurred : {}.'
#                                   .format(name, param.size(), input_param.size(), ex.args))
#         else:
#             print('{} in module is not in the state_dict'.format(name))
#     return lm

def vis_loc_or_segment(img,cam,b,nm,sub1,sub2,vis_dir):
    vis_cam(img, cam,
            join(vis_dir ,"%03d_v%03d_%010s_%15s"%(b,nm,sub1,sub2)))
def vis_name(vis_dir,b,nm,sub1,sub2):
    return join(vis_dir, "%03d_v%03d_%010s_%15s" % (b, nm, sub1, sub2))
def pathf(b,nm,sub1,sub2):
    return "./%03d_v%03d_%010s_%15s" % (b, nm, sub1, sub2)
def vis_feat_dist_by_cos_mean_and_std_matrix(f1,f2,vis_dir,name):
    # (C1,N,D),(C2,N,D)->(C1,C2,N,N)->(C1,C2)
    f1=f1.unsqueeze(1).unsqueeze(3)
    f2=f2.unsqueeze(0).unsqueeze(2)
    cos=functional.cosine_similarity(f1,f2,dim=4)#(C1,C2,N,N)
    mean_cos=cos.mean(dim=[-1,-2])
    std_cos=((cos-mean_cos[:,:,None,None])**2).mean(dim=[-1,-2])**0.5
    vis_cos_matrix(mean_cos,join(vis_dir,name+"mean"))
    vis_cos_matrix(std_cos, join(vis_dir, name + "std"))

def binaryzation_loc(loc,p=0.5):
    return loc > loc.max(-1)[0].max(-1)[0].unsqueeze(-1).unsqueeze(-1)*p

def get_idx( B, NumMix, MaxObject, device):
    bidx = torch.arange(B).to(device)
    nmidx = torch.arange(NumMix).to(device)
    bnmidx = [torch.stack([bidx] * NumMix, dim=1), torch.stack([nmidx] * B, dim=0)]
    bnmidx = [torch.stack([i] * MaxObject, dim=2) for i in bnmidx]
    return bnmidx
def entropy_estimate(f):
    #(N,D)
    f=functional.normalize(f,dim=1)
    f=f-f.mean(0,keepdim=True)
    cov=torch.matmul(f.permute(1,0),f)/(f.size(0)-1)
    est_entropy=cov.sum()/f.size(1)
    return est_entropy
def sep_val(audios,est_wavs,B):
    print("start_val_sep")
    result = []
    record_result = []
    device=audios.device
    audios = audios.detach().cpu().numpy()
    NumMix=audios.shape[1]
    val_video_audios = est_wavs.detach().cpu().numpy()
    for i in range(B):
        valid = (np.sum(np.abs(audios[i]), axis=1) > 1e-5).all()
        if valid:
            if (np.sum(np.abs(val_video_audios[i]),axis=1)>1e-5).all():
                try:
                    sdr, sir, sar, _ = bss_eval_sources(audios[i], val_video_audios[i], False)
                    result.append(np.array([sdr.mean(), sir.mean(), sar.mean()]))

                    record_result.append(np.stack([sdr, sir, sar], axis=1))
                except:
                    print("bss_eval_sources WRONG",file=sys.stderr)
                    record_result.append(np.zeros((NumMix,3)))
                    pass
            else:
                print("zero_est_video_src",file=sys.stderr)
                result.append(np.array([0., 0., 0.]))
                record_result.append(np.zeros((NumMix,3)))
        else:
            record_result.append(np.zeros((NumMix,3)))
            print("zero target src",file=sys.stderr)
    if len(result) == 0:
        result.append(np.array([0, 0, 0]))
    result = np.stack(result).mean(axis=0)
    print("sdr: ",result[0],"sir: ",result[1],"sar: ",result[2])
    record_result=torch.tensor(np.stack(record_result)).to(device)
    return result,record_result

def sep_val_gpu(audios,est_wavs,B):
    B,NumMix,_=audios.size()
    print("start_val_sep")
    result = []
    record_result=[]
    audios = audios
    val_video_audios = est_wavs
    for i in range(B):
        valid = (torch.abs(audios[i]).sum(1) > 1e-5).all()
        if valid:
            if (torch.abs(val_video_audios[i]).sum(1) > 1e-5).all():
                try:
                    sdr, sir, sar, _ = gpu_bss_eval_sources(audios[i], val_video_audios[i], False)
                    if i==0:
                        print("sdr: ", sdr, "sir: ", sir, "sar: ", sar)
                    if sdr.isinf().any() or sir.isinf().any() or sar.isinf().any() or \
                            sdr.isnan().any() or sir.isnan().any() or sar.isnan().any():
                        print("-----------------inf or nan value occur---------------------")
                        print(audios[i])
                        print(val_video_audios[i])
                        # result.append(torch.tensor([0., 0., 0.]).to(audios.device))
                        record_result.append(torch.zeros(NumMix, 3).to(audios.device))
                    else:
                        result.append(torch.stack([sdr, sir, sar],dim=1).to(audios.device))
                        record_result.append(torch.stack([sdr, sir, sar],dim=1).to(audios.device))
                        #result.append(torch.tensor([sdr.mean(), sir.mean(), sar.mean()]).to(audios.device))
                except:
                    print("gpu_bss_eval_sources WRONG",file=sys.stderr)
                    record_result.append(torch.zeros(NumMix, 3).to(audios.device))
                    pass
            else:
                print("zero_est_video_src")
                result.append(torch.zeros(NumMix,3).to(audios.device))
                record_result.append(torch.zeros(NumMix,3).to(audios.device))
        else:
            record_result.append(torch.zeros(NumMix,3).to(audios.device))
            print("zero target src")
    if len(result) == 0:
        result.append(torch.zeros(NumMix,3).to(audios.device))

    return torch.stack(result).mean([0,1]),torch.stack(record_result)

def l1_loss(v):
    #(B,...)
    v=torch.flatten(v,start_dim=1)
    return torch.abs(v).sum(1).mean()
def spacial_minmaxnormalize_3D(x):
    min=x.min(-1,keepdim=True)[0].min(-2,keepdim=True)[0].min(-3,keepdim=True)[0]
    max=x.max(-1,keepdim=True)[0].max(-2,keepdim=True)[0].max(-3,keepdim=True)[0]
    x=(x-min)/(max-min+1e-8)
    return x

def spacial_minmaxnormalize_2D(x):
    min=x.min(-1,keepdim=True)[0].min(-2,keepdim=True)[0]
    max=x.max(-1,keepdim=True)[0].max(-2,keepdim=True)[0]
    x=(x-min)/(max-min+1e-8)
    return x

def cal_ID_Semantic_Tolerence_DML(cos_array,opts):
    assert cos_array.size(0)==cos_array.size(1)
    B=cos_array.size(0)
    O_pos=torch.zeros_like(cos_array)
    O_pos[torch.arange(B),torch.arange(B)]=1
    O_neg=torch.logical_not(O_pos)
    return cal_Semantic_Tolerence_DML(cos_array,O_pos,O_neg,opts)

def cal_JSD_MI_ESTIMATION(cos_array, O_pos, O_neg, opts):
    N=O_pos.size(0)
    device=O_pos.device
    # nidx=torch.arange(N).to(device)
    # O_pos[nidx,nidx]=0
    # assert (O_pos.sum(1)>0).all()

    pos_score_array = torch.log(1+torch.exp(- cos_array / opts.t ))
    pos_ml=((pos_score_array*O_pos).sum(1))/(O_pos.sum(1)+1e-8)

    neg_score_array= torch.log(1+torch.exp(cos_array / opts.t ))
    neg_ml = ((neg_score_array * O_neg).sum(1)) / (O_neg.sum(1) + 1e-8)

    return (pos_ml+neg_ml).mean()


def cal_Semantic_Tolerence_DML(cos_array, O_pos, O_neg, opts):
    score_array = torch.exp(1 / opts.t * cos_array)

    O_mining=O_neg

    value, idx = torch.topk(score_array * O_mining, k=opts.hard_sampling.discard_K, largest=True, sorted=True,
                            dim=1)

    O_topK = torch.zeros(score_array.size()).bool().to(O_neg.device)
    Nfeature= score_array.size(0)
    o_idx = torch.stack([torch.arange(Nfeature)] * opts.hard_sampling.discard_K, dim=1)
    O_topK[o_idx, idx] = 1

    O_neg=torch.logical_and(O_mining,torch.logical_not(O_topK))
    neg_score = torch.sum(score_array * O_neg, dim=1, keepdim=True)

    if opts.positive_weight.turn_on:
        weight = torch.exp(-1 / opts.positive_weight.t * cos_array) * O_pos
        Z = weight.sum(dim=1, keepdim=True)
        weight = (weight / (Z + 1e-8)).detach()
        tuplet_loss = torch.mean(
            torch.sum(weight * O_pos *
                      -torch.log(score_array / (score_array + neg_score + 1e-8) + 1e-8), dim=1)
        )
    else:
        tuplet_loss = torch.mean(
            torch.sum(O_pos *
                      -torch.log(score_array / (score_array + neg_score + 1e-8) + 1e-8), dim=1)
            / (1e-8 + O_pos.sum(dim=1))
        )

    dml_loss = tuplet_loss
    return dml_loss

def cal_Semantic_Tolerence_DML_w_assigned_neg(cos_array, O_pos, O_neg,O_mining, opts):
    score_array = torch.exp(1 / opts.t * cos_array)

    value, idx = torch.topk(score_array * O_mining, k=opts.hard_sampling.discard_K, largest=True, sorted=True,
                            dim=1)

    O_topK = torch.zeros(score_array.size()).bool().to(O_neg.device)
    Nfeature= score_array.size(0)
    o_idx = torch.stack([torch.arange(Nfeature)] * opts.hard_sampling.discard_K, dim=1)
    O_topK[o_idx, idx] = 1

    O_neg_append=torch.logical_or(O_neg,torch.logical_and(O_mining,torch.logical_not(O_topK)))
    neg_score = torch.sum(score_array * O_neg_append, dim=1, keepdim=True)

    if opts.positive_weight.turn_on:
        weight = torch.exp(-1 / opts.positive_weight.t * cos_array) * O_pos
        Z = weight.sum(dim=1, keepdim=True)
        weight = (weight / (Z + 1e-8)).detach()
        tuplet_loss = torch.mean(
            torch.sum(weight * O_pos *
                      -torch.log(score_array / (score_array + neg_score + 1e-8) + 1e-8), dim=1)
        )
    else:
        tuplet_loss = torch.mean(
            torch.sum(O_pos *
                      -torch.log(score_array / (score_array + neg_score + 1e-8) + 1e-8), dim=1)
            / (1e-8 + O_pos.sum(dim=1))
        )
    dml_loss = tuplet_loss
    return dml_loss

def cal_cycle_walk_loss(cos_array,opts):
    score_array = 1 / opts.t * cos_array
    A2B_random_walk_matrix=torch.softmax(score_array,dim=1)
    B2A_random_walk_matrix=torch.softmax(score_array,dim=0).permute(1,0)
    ABA_matrix=torch.matmul(A2B_random_walk_matrix,B2A_random_walk_matrix)#(A,A)
    A=A2B_random_walk_matrix.size(0)
    O_pos=torch.eye(A).bool().to(score_array.device)
    cycle_walk_loss = torch.sum(-torch.log( ABA_matrix[O_pos]+1e-8))/A

    return cycle_walk_loss

def constant_softmax(x,dim,constant):
    exp=torch.exp(x)
    return exp/(exp.sum(dim=dim,keepdim=True)+constant)

def spacial_minmaxnormalize(x):
    min=x.min(-1,keepdim=True)[0].min(-2,keepdim=True)[0]
    max=x.max(-1,keepdim=True)[0].max(-2,keepdim=True)[0]
    x=(x-min)/(max-min+1e-8)
    return x
def cal_sisnr_loss(source, estimate_source):
    pair_wise_si_snr = cal_video_sisnr(source, estimate_source)

    mean_snr = pair_wise_si_snr.mean()
    return -(mean_snr)

def MixIT_Mask(NumMix,C,device):
    mask=torch.zeros([NumMix, C]).to(device)
    for i in range(C):
        torch.ones_like(mask)
        mask=torch.stack([mask]*(NumMix+1),dim=2)
        for nm in range(NumMix):
            mask[nm,i,nm]=1
            mask[nm,i,nm]=1
    mask=mask.flatten(start_dim=2).permute(2,0,1)
    mask=mask[(mask.sum(dim=2)>0).all(dim=1)]
    return mask

def cal_mixIT_video_sisnr(source, estimate_source):
    #(B,NumMix,AudLen) (B,NumMix,C,AudLen) 测试时为（B,1,C,AudLen）
    B,NumMix,C,AudLen=estimate_source.size()
    mask=MixIT_Mask(NumMix,C,source.device)
    A = mask.size(0)
    est_source1=estimate_source[:,None,:,:,:]
    mask1 = mask[None,:,:,:,None] # (1,A,NumMix,C,1)
    proposal_estimate_sources=(est_source1*mask1).sum(dim=3)#(B,A,NumMix,AudLen)
    sisnr_array = cal_video_sisnr(torch.stack([source] * A, dim=1).flatten(start_dim=1, end_dim=2),
                                 proposal_estimate_sources.flatten(start_dim=1, end_dim=2)).view(B,A,NumMix)  # (B,A*NumMix,AudLen) (B,A*NumMix,AudLen) ->(B,A,NumMix)
    _,idx=sisnr_array.mean(dim=2).max(dim=1)
    select_mask=mask[idx]#(B,NumMix,C)
    bidx=torch.arange(B).to(source.device)
    mixit_sisnr=sisnr_array[bidx,idx]#(B,NumMix)
    return mixit_sisnr,select_mask

def MixIT_Maskv2(NumMix,C,device):
    mask=torch.zeros([NumMix, C]).to(device)
    for i in range(C):
        torch.ones_like(mask)
        mask=torch.stack([mask]*(NumMix+1),dim=2)
        for nm in range(NumMix):
            mask[nm,i,nm]=1
            mask[nm,i,nm]=1
    mask=mask.flatten(start_dim=2).permute(2,0,1)
    mask = mask[(mask.sum(dim=2) > 0).all(dim=1)]
    mask=mask[(mask.sum([-1,-2])==C)]
    return mask


def cal_mixIT_video_sisnrv2(source, estimate_source):
    #(B,NumMix,AudLen) (B,NumMix,C,AudLen) 测试时为（B,1,C,AudLen）
    B,NumMix,C,AudLen=estimate_source.size()
    mask=MixIT_Maskv2(NumMix,C,source.device)
    A = mask.size(0)
    est_source1=estimate_source[:,None,:,:,:]
    mask1 = mask[None,:,:,:,None] # (1,A,NumMix,C,1)
    proposal_estimate_sources=(est_source1*mask1).sum(dim=3)#(B,A,NumMix,AudLen)
    sisnr_array = cal_video_sisnr(torch.stack([source] * A, dim=1).flatten(start_dim=1, end_dim=2),
                                 proposal_estimate_sources.flatten(start_dim=1, end_dim=2)).view(B,A,NumMix)  # (B,A*NumMix,AudLen) (B,A*NumMix,AudLen) ->(B,A,NumMix)
    _,idx=sisnr_array.mean(dim=2).max(dim=1)
    select_mask=mask[idx]#(B,NumMix,C)
    bidx=torch.arange(B).to(source.device)
    mixit_sisnr=sisnr_array[bidx,idx]#(B,NumMix)
    return mixit_sisnr,select_mask


def cal_Duet_mixIT_video_sisnr(source, estimate_source):
    #(B,NumMix,AudLen) (B,NumMix,C,AudLen) 测试时为（B,1,C,AudLen）
    B,NumMix,C,AudLen=estimate_source.size()
    mask=MixIT_Mask(NumMix,C,source.device)
    mask = mask[(mask.sum(dim=2)==2).all(dim=1)]
    A = mask.size(0)
    est_source1=estimate_source[:,None,:,:,:]
    mask1 = mask[None,:,:,:,None] # (1,A,NumMix,C,1)
    proposal_estimate_sources=(est_source1*mask1).sum(dim=3)#(B,A,NumMix,AudLen)
    sisnr_array = cal_video_sisnr(torch.stack([source] * A, dim=1).flatten(start_dim=1, end_dim=2),
                                 proposal_estimate_sources.flatten(start_dim=1, end_dim=2)).view(B,A,NumMix)  # (B,A*NumMix,AudLen) (B,A*NumMix,AudLen) ->(B,A,NumMix)
    _,idx=sisnr_array.mean(dim=2).max(dim=1)
    select_mask=mask[idx]#(B,NumMix,C)
    bidx=torch.arange(B).to(source.device)
    mixit_sisnr=sisnr_array[bidx,idx]#(B,NumMix)
    return mixit_sisnr,select_mask

def cal_pi_video_sisnr(source, estimate_source):
    #(B,NumMix,AudLen) (B,NumMix,C,AudLen) 测试时为（B,1,C,AudLen）
    B,NumMix,C,AudLen=estimate_source.size()
    sisnr_array = cal_video_sisnr(torch.stack([source] * C, dim=2).flatten(start_dim=1, end_dim=2),
                                 estimate_source.flatten(start_dim=1, end_dim=2)).view(B, NumMix,
                                                                               C)  # (B,NumMix*C,AudLen) (B,NumMix*C,AudLen) ->(B,NumMix,C)


    permu = torch.tensor(list(permutations(range(C), NumMix)), dtype=torch.int64).to(source.device)  # (A,NumMix)
    A = permu.size(0)
    one_hot = torch.zeros(1, A, NumMix, C).to(source.device)\
        .scatter_(3, permu[None, :, :, None], 1)  # (1,A,NumMix,C)
    sisnr_array1=sisnr_array.unsqueeze(1) #(B,1,NumMix,C)
    _,idx=(sisnr_array1*one_hot).sum(dim=-1).mean(-1).max(-1)#(B)
    select_idx=permu[idx]#(B,NumMix)
    pi_sisnr=sisnr_array.gather(dim=2,index=select_idx.unsqueeze(-1))[:,:,0]
    return pi_sisnr,one_hot[0][idx].detach()


def cal_video_sisnr(source, estimate_source,max_sisnr=None):
    # B*C*T B*C*T
    # 每段音频长度都为T
    # 且source 和 estimate_source 排序相同相同
    EPS = 1e-8
    # Step 1. Zero_Norm
    mean_target = torch.mean(source, dim=2, keepdim=True)
    mean_estimate = torch.mean(estimate_source, dim=2, keepdim=True)
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate

    # B*C*T
    # Step 2. SI-SNR
    # 向量间点积 B*C
    pair_wise_dot = torch.einsum("bit,bit->bi", zero_mean_estimate, zero_mean_target)
    pair_wise_dot = torch.unsqueeze(pair_wise_dot, dim=2)
    # s_target = <s', s>s / ||s||^2
    s_target_energy = torch.sum(zero_mean_target ** 2, dim=2, keepdim=True) + 1e-8  # [B,C,1]

    s_target = zero_mean_target
    # ( B*C(tar)*1) *(B*C*T)/(B*C*1) ->B*C*T
    pair_wise_proj = pair_wise_dot * s_target / (s_target_energy+1e-8)  # [B, C, T]
    # B,C1,C2,T  B1,1,C2,T
    # e_noise = s' - s_target
    s_estimate = zero_mean_estimate
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]
    # SI-SNR = 10 * log_10(||s_target||^2 / ||e_noise||^2) 计算(estimated=包含信息部分+噪声)能量比(db尺度)
    if max_sisnr==None:
        pair_wise_si_snr = torch.sum(pair_wise_proj ** 2, dim=2) / (torch.sum(e_noise ** 2, dim=2) + EPS)
    else:
        proj_energy=torch.sum(pair_wise_proj ** 2, dim=2)
        noise_energy=torch.sum(e_noise ** 2, dim=2)
        th=10**(-max_sisnr/10)*proj_energy
        pair_wise_si_snr = proj_energy / (noise_energy + th+1e-8)
    pair_wise_si_snr = 10 * torch.log10(pair_wise_si_snr + EPS)  # [B, C]

    return pair_wise_si_snr

def cal_modified_sisnr_loss(source, estimate_source):
    # B*C*T B*C*T
    # 每段音频长度都为T
    # 且source 和 estimate_source 排序相同相同
    EPS = 1e-8
    # Step 1. Zero_Norm
    mean_target = torch.mean(source, dim=2, keepdim=True)
    mean_estimate = torch.mean(estimate_source, dim=2, keepdim=True)
    zero_mean_target = source - mean_target
    zero_mean_estimate = estimate_source - mean_estimate

    # B*C*T
    # Step 2. Proj
    # 向量间点积 B*C
    pair_wise_dot = torch.einsum("bit,bit->bi", zero_mean_estimate, zero_mean_target)
    pair_wise_dot = torch.unsqueeze(pair_wise_dot, dim=2)
    # s_target = <s', s>s / ||s||^2
    s_target_energy = torch.sum(zero_mean_target ** 2, dim=2, keepdim=True) + 1e-8  # [B,C,1]

    s_target = zero_mean_target
    # ( B*C(tar)*1) *(B*C*T)/(B*C*1) ->B*C*T
    pair_wise_proj = pair_wise_dot * s_target / s_target_energy  # [B, C, T]
    # B,C1,C2,T  B1,1,C2,T
    # e_noise = s' - s_target
    s_estimate = zero_mean_estimate
    e_noise = s_estimate - pair_wise_proj  # [B, C, T]

    weight=(1/((pair_wise_proj*pair_wise_proj).sum(-1)+1e-8)).detach()
    modified_sisnr_loss=(e_noise*e_noise).sum(-1)-weight*(pair_wise_proj*pair_wise_proj).sum(-1)

    return modified_sisnr_loss.mean()

def cal_cos_array(f1, f2):
    # (M,C)(N,C)->(M,N)
    f1 = f1.unsqueeze(1)  # (M,1,C)
    f2 = f2.unsqueeze(0)  # (1,N,C)
    cos_array = functional.cosine_similarity(f1, f2, dim=2)
    # 为什么
    # assert cos_array.min()>=-1-1e-8 and cos_array.max()<=1+1e-8
    return cos_array


def cal_cos(f1, f2):
    # (M,C)(M,C)->(M)
    cos = functional.cosine_similarity(f1, f2, dim=1)
    return cos


def sum_object_audios(est_obj_audios, valid_nums):
    # (B,NumMix,MaxObject,AudLen) , (B,NumMix)
    B, NumMix, MaxObject, _ = est_obj_audios.size()
    # TODO: inplace 问题 为什么有时候梯度计算会报错有时候不会
    valid_mask = torch.zeros(B, NumMix, MaxObject, 1).bool().to(est_obj_audios.device)
    for i in range(B):
        for j in range(NumMix):
            valid_mask[i, j, :valid_nums[i, j]] = 1
    valid_mask = valid_mask.detach()
    est_video_audios = torch.sum(est_obj_audios * valid_mask, dim=2)
    return est_video_audios


def sum_object_masks(est_obj_mel_masks, valid_nums):
    # (B,NumMix,MaxObject,F,T) , (B,NumMix)
    B, NumMix, MaxObject, F, T = est_obj_mel_masks.size()

    valid_mask = torch.zeros(B, NumMix, MaxObject, 1, 1).type_as(est_obj_mel_masks)
    for i in range(B):
        for j in range(NumMix):
            valid_mask[i, j, :valid_nums[i, j]] = 1
    valid_mask = valid_mask.detach()
    est_video_mel_masks = torch.sum(est_obj_mel_masks * valid_mask, dim=2)
    return est_video_mel_masks


def logical_or_object_masks(est_obj_mel_masks, valid_nums):
    # (B,NumMix,MaxObject,F,T) , (B,NumMix)
    B, NumMix, MaxObject, F, T = est_obj_mel_masks.size()
    assert MaxObject == 2 or MaxObject == 1
    if MaxObject == 1:
        return est_obj_mel_masks[:, :, 0, :, :]
    valid_mask = torch.zeros(B, NumMix, MaxObject, 1, 1).type_as(est_obj_mel_masks)
    for i in range(B):
        for j in range(NumMix):
            valid_mask[i, j, :valid_nums[i, j]] = 1
    valid_mask = valid_mask.detach()
    masked_est_obj_mel_masks = (est_obj_mel_masks * valid_mask)
    est_video_mel_masks = 1 - (1 - masked_est_obj_mel_masks[:, :, 0, :, :]) * (
                1 - masked_est_obj_mel_masks[:, :, 1, :, :])
    return est_video_mel_masks


def extend(f, validnum):
    # (B, NumMix, MaxObject, C),(B,NumMix) -> (ObjectNum,C)
    feautures = []
    B, NumMix, MaxObject, C = f.size()
    for i in range(B):
        for j in range(NumMix):
            feautures.append(f[i, j, :validnum[i, j], :])
    obj_features = torch.cat(feautures, dim=0)
    return obj_features


def get_valid_mask(validnum, Maxobject):
    # (B,NumMix)->(B*NumMix*MaxObject)
    B, NumMix = validnum.size()
    mask = torch.zeros(B, NumMix, Maxobject).bool().to(validnum.device)
    for i in range(B):
        for j in range(NumMix):
            mask[i, j, :validnum[i, j]] = 1
    return mask.detach()


def get_valid_mask_array(mask):
    # (B,NumMix)->(B*NumMix*MaxObject)
    mask = torch.flatten(mask)
    m1 = mask.unsqueeze(dim=1)
    m2 = mask.unsqueeze(dim=0)
    mask_array = m1 * m2
    return mask_array.detach()


def get_video_mask_by_avc(obj_avc, B, NumMix, MaxObject):
    # NumObject
    obj_avc_mask = (obj_avc > 0.5).view(B, NumMix, MaxObject)
    video_avc_mask = torch.ones(B, NumMix).bool().to(obj_avc.device)
    for i in range(MaxObject):
        video_avc_mask *= obj_avc_mask[:, :, i]
    return video_avc_mask.detach()


def get_object_mask_array_by_video_snr(video_snr, MaxObject,snr_threhold):
    # (B,NumMix) ->(NumObject,NumObject)
    B, NumMix = video_snr.size()
    valid_video_by_snr = video_snr > snr_threhold
    valid_object_mask_by_snr = torch.zeros(B, NumMix, MaxObject).bool().to(video_snr.device)
    valid_object_mask_by_snr[valid_video_by_snr] = 1
    snr_obj_mask_array = get_valid_mask_array(valid_object_mask_by_snr)
    return snr_obj_mask_array


def get_yij_by_threhold(cos_array, pos_threhold,neg_threhold):
    O_pos = cos_array > pos_threhold  # (NumObject,NumObject)
    O_neg = cos_array < neg_threhold  # (NumObject,NumObject)
    return O_pos.detach(), O_neg.detach()


def get_yij_by_avc(cos_array):
    O_pos = cos_array > 0.5
    return O_pos.detach()


def get_yij_by_topN(cos_array, opt):
    # examine
    O_pos = torch.zeros(cos_array.size()).bool().to(cos_array.device)
    O_neg = torch.zeros(cos_array.size()).bool().to(cos_array.device)
    Nobject_idx = torch.arange(cos_array.size(0)).unsqueeze(dim=1)

    _, idx = torch.topk(cos_array, k=opt.pos_topN, dim=-1, largest=True)
    O_pos[Nobject_idx, idx] = 1

    if opt.neg_topN + opt.pos_topN > cos_array.size(0):
        k = cos_array.size(0) - opt.pos_topN
    else:
        k = opt.neg_topN
    _, idx = torch.topk(cos_array, k=k, dim=-1, largest=False)
    O_neg[Nobject_idx, idx] = 1
    return O_pos.detach(), O_neg.detach()


def get_yij_by_labels(pseudo_labels):
    # (NumObject) ->(NumObject,NumObject)
    l1 = pseudo_labels.unsqueeze(dim=1)
    l2 = pseudo_labels.unsqueeze(dim=0)
    O_pos = (l1 == l2)
    O_neg = (l1 != l2)
    return O_pos.detach(), O_neg.detach()


def cal_fv_sep_irrelevant_loss(obj_fv_sep, O_pos, O_neg):
    v_sep_cos_array = cal_cos_array(obj_fv_sep, obj_fv_sep)
    # assert v_sep_cos_array.min()>=0-1e-8 and v_sep_cos_array.max()<=1+1e-8 #obj_fv_sep>=0
    # may result nan value 1-1+eps
    fv_sep_cos_loss = torch.sum(-torch.log(1 - v_sep_cos_array + 1e-6) * O_neg) / (O_neg.sum() + 1e-8) \
        # + torch.sum(-torch.log(v_sep_cos_array + 1e-6) * O_pos) / O_pos.sum()
    # assert  not torch.isnan(fv_sep_cos_loss)
    return fv_sep_cos_loss


# 计算量太大
def cal_score_map_irrelevant_loss(score_maps):
    # (B,K,F,T) 计算量大 B*K*K*(F*T) 反馈是内存要求太大
    dist = torch.flatten(score_maps, start_dim=2)
    d_mean = torch.mean(dist, dim=2, keepdim=True)  # (B,K,1)
    delta = torch.abs(dist - d_mean)  # (B,K,-1)
    # M1=delta.mean(dim=2)#(B,K)
    # M2=((delta)**2).mean(dim=2)#(B,K)

    delta1 = delta.unsqueeze(dim=1)
    delta2 = delta.unsqueeze(dim=2)
    cov = torch.mean(delta1 * delta2, dim=3)  # (B,K,K)
    for i in range(cov.size(1)):
        cov[:, i, i] = 0
    loss = torch.abs(cov).mean()
    return loss


def cal_score_map_sparse_loss(score_maps):
    v = torch.flatten(score_maps, start_dim=2)
    rms = torch.sqrt((v * v).mean(dim=2) + 1e-6)  # (B,K)
    l1 = torch.sum(rms, dim=1)  # (B)
    l2 = torch.sqrt(torch.sum(rms * rms, dim=1) + 1e-6)
    loss = (l1 / l2 / score_maps.size(1)).mean()
    return loss


# 数值不稳定 需要数学推导 证明online learning在batch中计算的KL散度能近似整体的KL散度 /修改为控制n阶矩相同 /
def KL_divergence_loss(miu0, logvar0, miu1, logvar1):
    # 用分布2近似分布1的KL散度
    var0 = torch.exp(logvar0)  # 0~1
    var1 = torch.exp(logvar1)  # 0~1
    kld = 0.5*(
        torch.clip(logvar0 - logvar1, max=30).exp() + (miu1 - miu0) ** 2 / (var1 + 1e-8) - 1 + logvar1 - logvar0)

    return kld


def label_pro(map,label_table,i, j, mask ,con_th):
    #(C,H,W) (H,W)
    H,W=label_table.size()
    ds=[[-1,-1],[-1,0],[-1,1],
       [0,-1],[0,1],
       [1,-1],[1,0],[1,1]]
    label_table[i][j] = mask
    ret = 1
    for d in ds:
        next_i=i+d[0]
        next_j=j+d[1]
        if next_i >= 0 and next_i < H and next_j >= 0 and next_j < W and label_table[next_i][next_j] == 0 :
            # print(functional.cosine_similarity(map[:,i,j],map[:,next_i,next_j],dim=0))
            if functional.cosine_similarity(map[:,i,j],map[:,next_i,next_j],dim=0)>con_th:
                ret += label_pro(map, label_table, next_i, next_j, mask, con_th)

    return ret
#获得最大的K个连通分量 当两个区域相邻且特征cos大于联通阈值认为联通 设为八连通/四连通connective 小于fg_th的不计算连通分量
def get_connective_component(maps,K,fg_th,con_th,connective=8):
    # return: B,K,H,W
    B,C,H,W=maps.size()

    #每个map返回K个联通分量的0-1图
    results=torch.zeros(B,K,H,W).bool().to(maps.device)
    for b in range(B):
        map=maps[b]
        bg=map.max(dim=0)[0]<fg_th
        label_table = torch.zeros(H,W).to(map.device)
        label_table[bg]=-1

        mask = 1
        label2area = {}

        for i in range(H):
            for j in range(W):
                if label_table[i][j] == 0:
                    area = label_pro(map,label_table,i, j, mask,con_th)
                    label2area[mask] = area
                    mask += 1

        sorted_table = [(k, label2area[k]) for k in sorted(label2area, key=label2area.get, reverse=True)]
        # print(len(sorted_table))
        for k in range(min(K,len(sorted_table))):
            results[b,k]=label_table==sorted_table[k][0]
    return results

def find(x,pnodes):

    if pnodes[x] != x:
        pnodes[x] = find(pnodes[x], pnodes)
    return pnodes[x]
def first_pass(g,fg_th,con_th) :
    C,height,width = g.size()
    graph=torch.zeros([C,height+2,width+2]).to(g.device)
    graph[:,1:-1,1:-1]=g
    bg=graph.max(dim=0)[0] < fg_th

    label_map=torch.zeros([height+2,width+2]).to(g.device)
    label = 1
    pnodes = {}

    dhws = [[-1, -1], [-1, 0], [-1, 1], [0, -1]]
    for h in range(1,height+1):
        for w in range(1,width+1):
            if h==14 and w==4:
                a=1
            if bg[h,w]:
                continue
            neighbors = []
            for dhw in dhws:
                if label_map[h+dhw[0],w+dhw[1]]>0 and functional.cosine_similarity(graph[:,h,w],graph[:,h+dhw[0],w+dhw[1]],dim=0)>=con_th:
                    neighbors.append(label_map[h+dhw[0],w+dhw[1]])
            neighbors=list(set(neighbors))
            if len(neighbors) > 0:
                label_map[h][w] = min(neighbors)
                root=neighbors[0].item()
                for n in neighbors:
                    assert n.item() in pnodes
                    root=min(root,find(n.item(),pnodes))
                for n in neighbors:
                    if find(n.item(),pnodes)==15:
                        a=1
                    pnodes[find(n.item(),pnodes)]=root
            else:
                label_map[h][w] = label
                pnodes[label]=label
                label += 1
    return  label_map[1:-1,1:-1], pnodes

def remap(idx_dict) -> dict:
    index_dict = deepcopy(idx_dict)
    for id in idx_dict:
        idv = idx_dict[id]
        while idv in idx_dict:
            if idv == idx_dict[idv]:
                break
            idv = idx_dict[idv]
        index_dict[id] = idv
    return index_dict


def second_pass(g, index_dict) -> list:
    graph = g.clone()
    height = len(graph)
    width = len(graph[0])
    for h in range(height):
        for w in range(width):
            if graph[h][w] == 0:
                continue
            if graph[h][w].item() in index_dict:
                graph[h][w] = index_dict[graph[h][w].item()]
    return graph

def flatten(g):
    graph = g.clone()
    fgraph = sorted(set(list(graph.flatten().cpu().numpy())))
    flatten_dict = {}
    for i in range(len(fgraph)):
        flatten_dict[fgraph[i]] = i
    graph = second_pass(graph, flatten_dict)
    return graph

def get_connnective_componet_two_pass(maps,K,fg_th,con_th):
    #maps:(B,C,H,W)
    B,C,H,W=maps.size()
    results = torch.zeros([B,K, H, W]).to(maps.device)

    for b in range(B):
        labels_1, pnodes = first_pass(maps[b], fg_th, con_th)  # (H,W),dict
        pnodes = remap(pnodes)
        labels_2 = second_pass(labels_1, pnodes)
        labels_3 = flatten(labels_2)

        label2area = {}
        for i in range(1, int(labels_3.max()) + 1):
            label2area[i] = (labels_3 == i).sum()
        sorted_table = [(k, label2area[k]) for k in sorted(label2area, key=label2area.get, reverse=True)]
        for k in range(min(K, len(sorted_table))):
            results[b,k] = labels_3 == sorted_table[k][0]

    return results


def cal_iou(mask1,mask2):
    #(*,H,W)(*,H,W)->*
    iou = torch.sum(mask1 * mask2,dim=[-1,-2]) / (torch.sum(mask1,dim=[-1,-2]) + torch.sum(mask2 * (mask1 == 0),dim=[-1,-2])+1e-8)
    return iou
import soundfile as sf
import matplotlib
from matplotlib import pyplot as plt
import numpy as np

def vis_cos_matrix(matrix,name):
    C = matrix.size(0)
    matrix = matrix.detach().cpu().numpy()
    plt.matshow(matrix)

    for i in range(C):
        for j in range(C):
            plt.annotate("%.2f" % (matrix[j, i]), xy=(i, j), horizontalalignment='center',
                         verticalalignment='center', fontsize=4)  # 小心坐标轴 和matrix坐标轴不同
    plt.colorbar()
    plt.savefig(name, dpi=300, bbox_inches='tight')
    plt.close()

def vis_mask(mask, name,vmax=None,vmin=None,binary=False):
    mask = mask.detach().cpu().numpy()
    if vmax==None:
        vmax=np.max(mask)
    if vmin==None:
        vmin=np.min(mask)
    if binary:
        plt.imshow(mask, vmax=vmax, vmin=vmin,cmap="gray")
    else:
        plt.imshow(mask,vmax=vmax,vmin=vmin)
    plt.colorbar()
    plt.savefig(name)
    plt.close()

def vis_spectrogram(spectrogram, name,vmax=40,vmin=-20,vis_axis=False):
    #
    db_spec = 20 * torch.log10(spectrogram+1e-8)
    db_spec = db_spec.detach().cpu().numpy()
    if vis_axis:
        plt.xlabel('Time')
        plt.ylabel('Frequency')
    plt.imshow(db_spec, vmax=vmax, vmin=vmin)
    plt.gca().invert_yaxis()
    if vis_axis:
        cbar = plt.colorbar()
        cbar.set_label('Magnitude')
    else:
        plt.axis('off')
    plt.savefig(name,bbox_inches='tight',pad_inches=0)
    plt.close()


def vis_audio(audio, name):
    audio = audio.detach().cpu().numpy()
    sf.write(name+".wav", audio, samplerate=11025, format="wav")


def vis_image(img, name):
    img = img.detach().cpu().numpy()  #
    img = np.transpose(img, (1, 2, 0))
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img=np.clip(img,0,1)
    plt.imsave(name+".jpg",img)

import cv2
def vis_cam(img,cam, name):
    # cam : 要求0~1之间 -0.0x 会被映射到0.9x

    _,H,W=img.size()
    img = img.detach().cpu().numpy()  #
    img = np.transpose(img, (1, 2, 0))
    img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])

    cam = cam.detach().cpu().numpy()
    cam =np.clip(cam,0,1)
    cam = cam * 255
    cam = cv2.applyColorMap(cam.astype(np.uint8), cv2.COLORMAP_JET)
    cam = cv2.resize(cam,[W,H])
    cam = cam[:, :, ::-1] / 255

    vis=0.5*img+0.5*cam
    vis=np.clip(vis,a_min=0,a_max=1)
    plt.imsave(name+".jpg",vis)
    plt.close()

def vis_matrix(matrix, name, ins, p=True):
    C = matrix.size(0)
    matrix = matrix.cpu().numpy()
    plt.matshow(matrix)
    plt.xlabel('ground truth labels')
    plt.ylabel('predicted labels')
    plt.yticks(np.arange(C), ins, fontsize=5)
    plt.xticks(np.arange(C), ins, fontsize=5, rotation=90)
    plt.colorbar()
    for i in range(C):
        for j in range(C):
            if p:
                plt.annotate(("%.2f" % (matrix[j, i] * 100)) + r"%", xy=(i, j), horizontalalignment='center',
                             verticalalignment='center', fontsize=4)  # 小心坐标轴 和matrix坐标轴不同
            else:
                plt.annotate(("%d" % (matrix[j, i])), xy=(i, j), horizontalalignment='center',
                             verticalalignment='center', fontsize=4)  # 小心坐标轴 和matrix坐标轴不同
    plt.savefig(name, dpi=300, bbox_inches='tight')

def vis_confusion_matrix(matrix, name, ins, p=True):
    C = matrix.size(0)
    matrix = matrix.cpu().numpy()
    plt.matshow(matrix)
    plt.xlabel('ground truth labels')
    plt.ylabel('predicted labels')
    plt.yticks(np.arange(C), ins, fontsize=5)
    plt.xticks(np.arange(C), ins, fontsize=5, rotation=90)
    plt.colorbar()
    for i in range(C):
        for j in range(C):
            if p:
                plt.annotate(("%.2f" % (matrix[j, i] * 100)) + r"%", xy=(i, j), horizontalalignment='center',
                             verticalalignment='center', fontsize=4)  # 小心坐标轴 和matrix坐标轴不同
            else:
                plt.annotate(("%d" % (matrix[j, i])), xy=(i, j), horizontalalignment='center',
                             verticalalignment='center', fontsize=4)  # 小心坐标轴 和matrix坐标轴不同
    plt.savefig(name, dpi=300, bbox_inches='tight')

def vis_two_mix_ins_separation_array(matrix, name, ins,vis_dir):
    # matrix axis is different from plt.plot
    # matrix row correspond to x axis
    # origin is at the top-left point
    C = matrix.size(0)
    matrix = matrix.cpu().numpy()
    plt.matshow(matrix)
    plt.title(name)
    plt.xlabel('B instrument')
    plt.ylabel('A instrument')
    plt.yticks(np.arange(C), ins, fontsize=6)
    plt.xticks(np.arange(C), ins, fontsize=6,rotation=90)
    plt.colorbar()
    for i in range(C):
        for j in range(C):
            plt.annotate(("%.1f" % (matrix[i, j])), xy=(j, i), horizontalalignment='center',
                         verticalalignment='center', fontsize=3)  # 小心坐标轴 和matrix坐标轴不同
    plt.savefig(join(vis_dir,name),figsize=(1,1.5), dpi=450, bbox_inches='tight')
    plt.close()

def vis_class_similarity_array(matrix, name, ins,vis_dir):
    # matrix axis is different from plt.plot
    # matrix row correspond to x axis
    # origin is at the top-left point
    C = matrix.size(0)
    matrix = matrix.cpu().numpy()
    plt.matshow(matrix)
    plt.title(name,y=1.3)
    plt.xlabel('B instrument')
    plt.ylabel('A instrument')
    plt.yticks(np.arange(C), ins, fontsize=6)
    plt.xticks(np.arange(C), ins, fontsize=6,rotation=90)
    plt.colorbar()
    for i in range(C):
        for j in range(C):
            plt.annotate(("%.2f" % (matrix[i, j])), xy=(j, i), horizontalalignment='center',
                         verticalalignment='center', fontsize=3)  # 小心坐标轴 和matrix坐标轴不同
    plt.savefig(join(vis_dir,name),figsize=(1,1.5), dpi=450, bbox_inches='tight')
    plt.close()

def vis_sdr_volumn_scatter_plot(sdr, volumn,vis_dir):
    # sdr:N volumn:N
    N = sdr.size(0)
    sdr = sdr.cpu().numpy()
    volumn=volumn.cpu().numpy()

    # plt.xlim(0, 10)
    plt.ylim(-10,30)

    plt.xlabel('volumn')
    plt.ylabel('sdr')

    plt.scatter(volumn,sdr,s=3)
    plt.savefig(join(vis_dir,"sdr_volumn_scatter_plot"))
    plt.close()

def vis_channel_category(channel_count, ins, name, p=True):
    # (K,Numlabels)
    K, Numlabels = channel_count.size()
    channel_count = channel_count.cpu().numpy().T
    plt.matshow(channel_count)
    plt.ylabel('category')
    plt.xlabel('chnnels')
    plt.xticks(np.arange(K), np.arange(K), fontsize=5)
    plt.yticks(np.arange(Numlabels), ins, fontsize=3, rotation=90, horizontalalignment='center',
               verticalalignment='center')
    plt.colorbar()
    for j in range(Numlabels):
        for i in range(K):
            if p:
                plt.annotate(("%.2f" % (channel_count[j, i] * 100)) + r"%", xy=(i, j),
                             horizontalalignment='center', verticalalignment='center', fontsize=4)  # 小心坐标轴 和matrix坐标轴不同
            else:
                plt.annotate(("%d" % (channel_count[j, i])), xy=(i, j),
                             horizontalalignment='center', verticalalignment='center', fontsize=4)  # 小心坐标轴 和matrix坐标轴不同
    plt.savefig(name, dpi=300, bbox_inches='tight')
    return

def get_batch_size(dm_config):
    import yaml
    return yaml.full_load(open(dm_config,"r"))["batch_size"]



def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv+1)/2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


def istft_reconstruction(mag, phase, hop_length=256):
    spec = mag.astype(np.complex) * np.exp(1j*phase)
    wav = librosa.istft(spec, hop_length=hop_length)
    return np.clip(wav, -1., 1.)


def RealComplexSpec2MagPhase(spec):
    #spec: Tensor B*C*F*T*2 -> B*C*F*T,B*C*F*T
    spec=(1+0j)*spec[:,:,:,:,0]+1j*spec[:,:,:,:,1]
    return torch.abs(spec),torch.angle(spec)

def MagPhase2RealComplexSpec(mag,phase):
    #spec:B*C*F*T,B*C*F*T -> B*C*F*T*2
    Complex = (1 + 0j) * mag * torch.exp(1j * phase)
    RealComplex = torch.stack([Complex.real, Complex.imag], dim=4)
    return RealComplex


from torch.nn import functional
def warpMagSpec(mag,warp,target_F):
    #mag: Tensor B*C*F*T -> B*C*F*T
    B,C,_,T=mag.size()
    grid = warpgrid(B, target_F, T, warp=warp)
    grid=torch.from_numpy(grid).to(mag.device)
    warpmag = functional.grid_sample(mag,grid , align_corners=True)
    return warpmag
if __name__=="__main__":
    torch.manual_seed(4)
    graph = torch.rand(1, 3, 20, 20)
    graph[graph > 0.5] = 1
    graph[graph <= 0.5] = 0
    graph[0, 0] = 0
    graph[0, 1] = 0
    results = get_connnective_componet_two_pass(graph, 3, 0.8, 0.5)
    vis_image(graph[0])
    vis_image(results[0])