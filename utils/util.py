import os
import math
import time
import copy
import random
import warnings
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
    elif torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def setup_ddp(args):
    from datetime import timedelta
    if torch.cuda.is_available():
        torch.cuda.set_device(args.rank)
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        torch.distributed.init_process_group(
            backend="nccl" if torch.distributed.is_nccl_available() else "gloo",
            timeout=timedelta(seconds=10800),
            rank=args.rank, world_size=args.world_size, )


def freeze_layer(model):
    layer_names = [".dfl"]
    freeze_layer_names = [f"model.{x}." for x in []] + layer_names
    for k, v in model.named_parameters():
        if any(x in k for x in freeze_layer_names):
            print(f"Freezing layer '{k}'")
            v.requires_grad = False


def wh2xy(x):
    assert x.shape[-1] == 4, f"expected 4 but input shape is {x.shape}"
    if isinstance(x, torch.Tensor):
        y = torch.empty_like(x, dtype=torch.float32)
    else:
        y = np.empty_like(x, dtype=np.float32)
    xy = x[..., :2]
    wh = x[..., 2:] / 2
    y[..., :2] = xy - wh
    y[..., 2:] = xy + wh
    return y


def make_anchors(feats, strides, offset=0.5):
    anchor_points, stride_tensor = [], []
    assert feats is not None
    dtype, device = feats[0].dtype, feats[0].device
    for i, stride in enumerate(strides):
        h, w = feats[i].shape[2:] if isinstance(feats, list) else (
            int(feats[i][0]), int(feats[i][1]))
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        sy, sx = torch.meshgrid(sy, sx, indexing="ij")
        anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(
            torch.full((h * w, 1), stride, dtype=dtype, device=device))
    return torch.cat(anchor_points), torch.cat(stride_tensor)


def smart_optimizer(args, model, decay=1e-5):
    g = [], [], []
    nc = args.num_cls

    lr_fit = round(0.002 * 5 / (4 + nc), 6)
    name, lr, momentum = "AdamW", lr_fit, 0.9
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):
            if module_name:
                fullname = f"{module_name}.{param_name}"
            else:
                fullname = f"{param_name}"
            if "bias" in fullname:
                g[2].append(param)
            elif isinstance(module, bn):
                g[1].append(param)
            else:
                g[0].append(param)

    optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(0.9, 0.999),
                                  weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})
    return optimizer


# ------------------------- MODEL EMA Start---------------------------------#
def is_parallel(model):
    return isinstance(model, (
        nn.parallel.DataParallel, nn.parallel.DistributedDataParallel))


def de_parallel(model):
    return model.module if is_parallel(model) else model


def copy_attr(a, b, include=(), exclude=()):
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith(
                "_") or k in exclude:
            continue
        else:
            setattr(a, k, v)


class EMA:
    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        self.ema = copy.deepcopy(de_parallel(model)).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = de_parallel(model).state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:
                    v *= d
                    v += (1 - d) * msd[k].detach()

    def update_attr(self, model, include=(),
                    exclude=("process_group", "reducer")):
        if self.enabled:
            copy_attr(self.ema, model, include, exclude)


# ------------------------- MODEL EMA End ---------------------------------#


# ----------------------- Detection Loss Start --------------
def bbox_iou(box1, box2, eps=1e-7):
    b1_x1, b1_y1, b1_x2, b1_y2 = box1.chunk(4, -1)
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.chunk(4, -1)
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    x = (b1_x2.minimum(b2_x2) - b1_x1.maximum(b2_x1)).clamp_(0)
    y = (b1_y2.minimum(b2_y2) - b1_y1.maximum(b2_y1)).clamp_(0)
    inter = x * y
    union = w1 * h1 + w2 * h2 - inter + eps
    iou = inter / union

    cw = b1_x2.maximum(b2_x2) - b1_x1.minimum(b2_x1)
    ch = b1_y2.maximum(b2_y2) - b1_y1.minimum(b2_y1)
    c2 = cw.pow(2) + ch.pow(2) + eps
    a = (b2_x1 + b2_x2 - b1_x1 - b1_x2)
    b = (b2_y1 + b2_y2 - b1_y1 - b1_y2)
    rho2 = (a.pow(2) + b.pow(2)) / 4

    v = (4 / math.pi ** 2) * ((w2 / h2).atan() - (w1 / h1).atan()).pow(2)
    with torch.no_grad():
        alpha = v / (v - iou + (1 + eps))
    return iou - (rho2 / c2 + v * alpha)


class DFLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.reg_max = reg_max

    def __call__(self, pred_dist, target):
        target = target.clamp_(0, self.reg_max - 1 - 0.01)
        tl, tr = target.long(), target.long() + 1
        wl, wr = tr - target, 1 - (tr - target)
        loss_l = F.cross_entropy(pred_dist, tl.view(-1), reduction="none")
        loss_r = F.cross_entropy(pred_dist, tr.view(-1), reduction="none")
        loss = (loss_l.view(tl.shape) * wl + loss_r.view(tl.shape) * wr)
        return loss.mean(-1, keepdim=True)


class BoxLoss(nn.Module):
    def __init__(self, reg_max=16):
        super().__init__()
        self.dfl_loss = DFLoss(reg_max)

    def forward(self, p_dist, p_box, anchors, gt_box, scores, scores_sum,
                mask):
        reg_max = self.dfl_loss.reg_max
        weight = scores.sum(-1)[mask].unsqueeze(-1)
        iou = bbox_iou(p_box[mask], gt_box[mask])
        loss_box = ((1.0 - iou) * weight).sum() / scores_sum

        a, b = gt_box.chunk(2, -1)
        distance = torch.cat((anchors - a, b - anchors), -1)
        target = distance.clamp_(0, (reg_max - 1) - 0.01)
        pred = p_dist[mask].view(-1, reg_max)
        loss_dfl = self.dfl_loss(pred, target[mask])
        loss_dfl = (loss_dfl * weight).sum() / scores_sum

        return loss_box, loss_dfl


class Assigner(nn.Module):
    def __init__(self, top_k=10, nc=80, alpha=0.5, beta=6.0, eps=1e-9):
        super().__init__()
        self.nc = nc
        self.eps = eps
        self.beta = beta
        self.top_k = top_k
        self.alpha = alpha

    @torch.no_grad()
    def forward(self, score, p_box, anchors, gt_labels, gt_box, mask):
        bs = score.shape[0]
        na = p_box.shape[-2]
        n_max_boxes = gt_box.shape[1]

        if n_max_boxes == 0:
            return (
                torch.full_like(score[..., 0], self.nc),
                torch.zeros_like(p_box),
                torch.zeros_like(score),
                torch.zeros_like(score[..., 0]),
                torch.zeros_like(score[..., 0]))

        lt, rb = gt_box.view(-1, 1, 4).chunk(2, 2)
        box_delta = torch.cat((anchors[None] - lt, rb - anchors[None]), dim=2)
        mask_in_gts = box_delta.view(gt_box.shape[0], gt_box.shape[1],
                                     anchors.shape[0], -1)
        mask_in_gts = mask_in_gts.amin(3).gt_(1e-9)
        mask_gts = (mask_in_gts * mask).bool()
        overlaps = torch.zeros([bs, n_max_boxes, na], dtype=p_box.dtype,
                               device=p_box.device)
        bbox_scores = torch.zeros([bs, n_max_boxes, na], dtype=score.dtype,
                                  device=score.device)

        ind = torch.zeros([2, bs, n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=bs).view(-1, 1).expand(-1, n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores[mask_gts] = score[ind[0], :, ind[1]][mask_gts]
        pd_boxes = p_box.unsqueeze(1).expand(-1, n_max_boxes, -1, -1)[mask_gts]
        gt_boxes = gt_box.unsqueeze(2).expand(-1, -1, na, -1)[mask_gts]
        overlaps[mask_gts] = bbox_iou(gt_boxes, pd_boxes).squeeze(-1).clamp_(0)

        metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        top_mask = mask.expand(-1, -1, self.top_k).bool()
        top_metrics, top_id = torch.topk(metric, self.top_k, dim=-1,
                                         largest=True)
        if top_mask is None:
            top_mask = (top_metrics.max(-1, keepdim=True)[
                            0] > self.eps).expand_as(top_id)
        top_id.masked_fill_(~top_mask, 0)

        count_tensor = torch.zeros(metric.shape, dtype=torch.int8,
                                   device=top_id.device)
        ones = torch.ones_like(top_id[:, :, :1], dtype=torch.int8,
                               device=top_id.device)
        for k in range(self.top_k):
            count_tensor.scatter_add_(-1, top_id[:, :, k: k + 1], ones)

        count_tensor.masked_fill_(count_tensor > 1, 0)
        mask_pos = count_tensor.to(metric.dtype) * mask_in_gts * mask
        fg_mask = mask_pos.sum(-2)
        if fg_mask.max() > 1:
            mask_multi_gts = (fg_mask.unsqueeze(1) > 1).expand(-1, n_max_boxes,
                                                               -1)

            max_over = torch.zeros(mask_pos.shape, dtype=mask_pos.dtype,
                                   device=mask_pos.device)
            max_over.scatter_(1, overlaps.argmax(1).unsqueeze(1), 1)

            mask_pos = torch.where(mask_multi_gts, max_over, mask_pos).float()
            fg_mask = mask_pos.sum(-2)
        gt_idx = mask_pos.argmax(-2)

        batch_ind = \
            torch.arange(end=bs, dtype=torch.int64, device=gt_labels.device)[
                ..., None]
        gt_idx = gt_idx + batch_ind * n_max_boxes
        target_labels = gt_labels.long().flatten()[gt_idx]

        target_bboxes = gt_box.view(-1, gt_box.shape[-1])[gt_idx]
        target_labels.clamp_(0)
        sc = (target_labels.shape[0], target_labels.shape[1], self.nc)
        target_scores = torch.zeros(sc, dtype=torch.int64,
                                    device=target_labels.device)
        target_scores.scatter_(2, target_labels.unsqueeze(-1), 1)
        scores_mask = fg_mask[:, :, None].repeat(1, 1, self.nc)
        target_scores = torch.where(scores_mask > 0, target_scores, 0)

        # Normalize
        metric *= mask_pos
        pos_metrics = metric.amax(dim=-1, keepdim=True)
        pos_overlaps = (overlaps * mask_pos).amax(dim=-1, keepdim=True)
        norm_metric = (metric * pos_overlaps / (pos_metrics + self.eps))
        target_scores = target_scores * (norm_metric.amax(-2).unsqueeze(-1))
        return target_bboxes, target_scores, fg_mask.bool()


class DetectionLoss:
    def __init__(self, model):
        device = next(model.parameters()).device

        m = model.detect
        self.nc = m.nc
        self.device = device
        self.stride = m.stride
        self.reg_max = m.reg_max
        self.no = m.nc + m.reg_max * 4

        self.assigner = Assigner(nc=self.nc)
        self.bbox_loss = BoxLoss(m.reg_max).to(device)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.proj = torch.arange(m.reg_max, dtype=torch.float, device=device)

    def preprocess(self, gt, bs, scale):
        nl, ne = gt.shape
        if nl == 0:
            out = torch.zeros(bs, 0, ne - 1, device=self.device)
        else:
            i = gt[:, 0]  # image index
            _, counts = i.unique(return_counts=True)
            counts = counts.to(dtype=torch.int32)
            out = torch.zeros(bs, counts.max(), ne - 1, device=self.device)
            for j in range(bs):
                matches = i == j
                n = matches.sum()
                if n:
                    out[j, :n] = gt[matches, 1:]
            out[..., 1:5] = wh2xy(out[..., 1:5].mul_(scale))
        return out

    def bbox_decode(self, anchor, pred_dist):
        b, a, c = pred_dist.shape
        pred_dist = pred_dist.view(b, a, 4, c // 4).softmax(3)
        pred_dist = pred_dist.matmul(self.proj.type(pred_dist.dtype))
        lt, rb = pred_dist.chunk(2, -1)
        x1y1, x2y2 = anchor - lt, anchor + rb
        return torch.cat((x1y1, x2y2), -1)

    def __call__(self, pred, batch):
        loss = torch.zeros(3, device=self.device)
        feats = pred[1] if isinstance(pred, tuple) else pred
        x = torch.cat([f.view(feats[0].shape[0], self.no, -1) for f in feats],
                      2)
        pred_distri, pred_scores = x.split((self.reg_max * 4, self.nc), 1)

        pred_scores = pred_scores.permute(0, 2, 1).contiguous()
        pred_distri = pred_distri.permute(0, 2, 1).contiguous()

        dtype, bs = pred_scores.dtype, pred_scores.shape[0]
        img_size = torch.tensor(feats[0].shape[2:], device=self.device,
                                dtype=dtype)
        img_size = img_size * self.stride[0]
        anchor_points, stride_tensor = make_anchors(feats, self.stride, 0.5)

        idx, cls, box = batch["idx"].view(-1, 1), batch["cls"].view(-1, 1), \
            batch["box"]
        targets = torch.cat((idx, cls, box), 1).to(self.device)
        targets = self.preprocess(targets, bs, img_size[[1, 0, 1, 0]])

        gt_labels, gt_bboxes = targets.split((1, 4), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)
        target_bboxes, target_scores, fg_mask = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor, gt_labels, gt_bboxes, mask_gt)

        _loss = self.bce(pred_scores, target_scores.to(dtype))
        loss[1] = _loss.sum() / max(target_scores.sum(), 1)

        # Bbox loss
        if fg_mask.sum():
            target_bboxes /= stride_tensor
            loss[0], loss[2] = self.bbox_loss(pred_distri,
                                              pred_bboxes,
                                              anchor_points,
                                              target_bboxes,
                                              target_scores,
                                              max(target_scores.sum(), 1),
                                              fg_mask)

        loss[0] *= 7.5
        loss[1] *= 0.5
        loss[2] *= 1.5
        return loss.sum() * bs, loss.detach()


# ----------------------- Detection Loss End --------------

# ----------------------- Compute AP Start -----------------
def non_max_suppression(pred, conf_th=0.001, iou_th=0.7):
    import torchvision
    max_det = 300
    max_wh = 7680
    max_nms = 30000

    pred = pred[0] if isinstance(pred, (list, tuple)) else pred

    bs = pred.shape[0]  # batch size
    nc = pred.shape[1] - 4  # number of classes
    xc = pred[:, 4:(4 + nc)].amax(1) > conf_th

    start_time = time.time()
    time_limit = 4.0 + 0.05 * bs

    pred = pred.transpose(-1, -2)
    pred[..., :4] = wh2xy(pred[..., :4])

    output = [torch.zeros((0, 6), device=pred.device)] * bs
    for xi, x in enumerate(pred):
        x = x[xc[xi]]
        if not x.shape[0]:
            continue

        box, cls = x.split((4, nc), 1)

        if nc > 1:
            i, j = torch.where(cls > conf_th)
            x = torch.cat((box[i], x[i, 4 + j, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = cls.max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_th]

        n = x.shape[0]  # number of boxes
        if not n:
            continue
        if n > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]

        c = x[:, 5:6] * max_wh  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]

        idx = torchvision.ops.nms(boxes, scores, iou_th)  # NMS
        idx = idx[:max_det]

        output[xi] = x[idx]
        if (time.time() - start_time) > time_limit:
            print(f"WARNING ⚠️ NMS time limit {time_limit:.3f}s exceeded")
            break

    return output


def smooth(y, f=0.05):
    nf = round(
        len(y) * f * 2) // 2 + 1
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(tp, conf, pred, target, plot=False, on_plot=None,
               save_dir=Path(), names={}):
    save_dir = Path(save_dir)
    pr_val = []
    i = np.argsort(-conf)
    tp, conf, pred = tp[i], conf[i], pred[i]
    x = np.linspace(0, 1, 1000)
    unique_cls, nt = np.unique(target, return_counts=True)

    p = np.zeros((unique_cls.shape[0], 1000))
    r = np.zeros((unique_cls.shape[0], 1000))
    ap = np.zeros((unique_cls.shape[0], tp.shape[1]))

    for ci, c in enumerate(unique_cls):
        i = pred == c
        nl, no = nt[ci], i.sum()
        if no == 0 or nl == 0: continue

        # Recall
        tpc = tp[i].cumsum(0)
        fpc = (1 - tp[i]).cumsum(0)

        recall = tpc / (nt[ci] + 1e-16)
        r[ci] = np.interp(-x, -conf[i], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-x, -conf[i], precision[:, 0], left=1)

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            mrec = np.concatenate(([0.0], recall[:, j], [1.0]))
            mp_re = np.concatenate(([1.0], precision[:, j], [0.0]))
            mp_re = np.flip(np.maximum.accumulate(np.flip(mp_re)))
            px = np.linspace(start=0, stop=1, num=101)
            ap[ci, j] = np.trapz(np.interp(px, mrec, mp_re), px)

            if j == 0:
                pr_val.append(np.interp(x, mrec, mp_re))

    pr_val = np.array(pr_val)

    f1 = 2 * p * r / (p + r + 1e-16)
    names = dict(enumerate([v for k, v in names.items() if k in unique_cls]))
    if plot:
        plot_pr_curve(x, pr_val, ap, save_dir / f"PR_curve.png", names,
                      plot=on_plot)
        plot_mc_curve(x, f1, save_dir / f"F1_curve.png", names, y="F1",
                      plot=on_plot)
        plot_mc_curve(x, p, save_dir / f"P_curve.png", names, y="Precision",
                      plot=on_plot)
        plot_mc_curve(x, r, save_dir / f"R_curve.png", names, y="Recall",
                      plot=on_plot)

    i = smooth(f1.mean(0), 0.1).argmax()
    p, r, f1 = p[:, i], r[:, i], f1[:, i]

    mean_ap, map50 = ap.mean(), ap[:, 0].mean()
    m_pre, m_rec = p.mean(), r.mean()  # precision, recall

    # tp = (r * nt).round()  # true positive
    # fp = (tp / (p + 1e-16) - tp).round() # false positive
    # weight = [0.0, 0.0, 0.1, 0.9]
    # fitness = (np.array([m_pre, m_rec, map50, mean_ap]) * weight).sum()

    return {"precision": m_pre, "recall": m_rec, "mAP50": map50,
            "mAP50-95": mean_ap}


# ----------------------- Compute AP End ---------------------

# ----------------------------- Metrics & Plotting Start -----
def update_metrics(preds, batch, niou, iou_v, stats, device):
    for i, pred in enumerate(preds):
        stat = dict(conf=torch.zeros(0).to(device), pred_cls=torch.zeros(0).to(device),
                    tp=torch.zeros(len(pred), niou, dtype=torch.bool).to(device))

        idx = batch["idx"] == i
        box = batch["box"][idx]
        cls = batch["cls"][idx]
        cls = cls.squeeze(-1)

        if len(cls):
            img_shape = batch["img"].shape[2:]
            tensor = torch.tensor(img_shape).to(device)[[1, 0, 1, 0]]
            box = wh2xy(box) * tensor
            scale_boxes(box, batch["shape"][i], batch["pad"][i], device)

        stat["target_cls"] = cls
        stat["target_img"] = cls.unique()
        if len(pred) == 0:
            if len(cls):
                for k in stats.keys():
                    stats[k].append(stat[k])
            continue

        output = pred.clone()
        scale_boxes(output[:, :4], batch["shape"][i], batch["pad"][i])

        stat["conf"] = output[:, 4]
        stat["pred_cls"] = output[:, 5]

        # Evaluate
        if len(cls):
            iou = box_iou(box, output[:, :4])
            stat["tp"] = match_predictions(iou_v, output, cls, iou)

        for k in stats.keys():
            stats[k].append(stat[k])
    return stats


def box_iou(box1, box2, eps=1e-7):
    (a1, a2) = box1.float().unsqueeze(1).chunk(2, 2)
    (b1, b2) = box2.float().unsqueeze(0).chunk(2, 2)
    inter = (torch.min(a2, b2) - torch.max(a1, b1))
    inter = inter.clamp_(0).prod(2)
    union = (a2 - a1).prod(2) + (b2 - b1).prod(2) - inter
    return inter / (union + eps)


def scale_boxes(boxes, shape, r_pad, device):
    gain, pad = r_pad[0][0], r_pad[1]
    boxes[..., :4] -= torch.tensor([pad[0], pad[1], pad[0], pad[1]]).to(device)
    boxes[..., :4] /= gain

    boxes[..., [0, 2]] = boxes[..., [0, 2]].clamp(0, shape[1])
    boxes[..., [1, 3]] = boxes[..., [1, 3]].clamp(0, shape[0])
    return boxes


def plot_pr_curve(px, py, ap, save_dir, names, plot):
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py = np.stack(py, axis=1)

    if 0 < len(names) < 21:
        [ax.plot(px, y, linewidth=1, label=f"{names[i]} {ap[i, 0]:.3f}") for
         i, y in enumerate(py.T)]
    else:
        ax.plot(px, py, linewidth=1, color="grey")

    ax.plot(px, py.mean(1), linewidth=3, color="blue",
            label=f"all classes {ap[:, 0].mean():.3f} mAP@0.5")
    ax.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1),
           title="Precision-Recall Curve")
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")

    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if plot:
        plot(save_dir)


def plot_mc_curve(px, py, save_dir, names, y, plot):
    names = names or {}
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:
        [ax.plot(px, y, linewidth=1, label=f"{names[i]}") for i, y in
         enumerate(py)]
    else:
        ax.plot(px, py.T, linewidth=1, color="grey")

    y_smooth = smooth(py.mean(0), 0.05)
    max_val, max_id = y_smooth.max(), px[y_smooth.argmax()]

    ax.plot(px, y_smooth, linewidth=3, color="blue",
            label=f"all classes {max_val:.2f} at {max_id:.3f}")
    ax.set(xlabel="Confidence", ylabel=y, xlim=(0, 1), ylim=(0, 1),
           title=f"{y}-Confidence Curve")

    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.savefig(save_dir, dpi=250)
    plt.close(fig)
    if plot:
        plot(save_dir)


def match_predictions(iou_v, pred, cls, iou):
    # Use PyTorch tensors throughout, avoid numpy conversion
    device = pred.device
    correct = torch.zeros((len(pred[:, 5]), len(iou_v)), dtype=torch.bool, device=device)
    
    # Keep everything on the same device, avoid cpu().numpy() conversion
    iou = iou * (cls[:, None] == pred[:, 5])
    
    for i, th in enumerate(iou_v.tolist()):
        # Find matches above threshold
        match_matrix = iou >= th
        
        if match_matrix.any():
            # Get all IoU values and coordinates where match_matrix is True
            iou_values = iou[match_matrix]
            row_idx, col_idx = torch.where(match_matrix)
            
            # Sort by IoU values in descending order
            sorted_indices = torch.argsort(iou_values, descending=True)
            row_idx = row_idx[sorted_indices]
            col_idx = col_idx[sorted_indices]
            
            # Track used ground truth and prediction indices
            used_gt = torch.zeros(iou.size(0), dtype=torch.bool, device=device)
            used_pred = torch.zeros(iou.size(1), dtype=torch.bool, device=device)
            
            # Assign matches greedily (highest IoU first)
            for j in range(len(row_idx)):
                gt_idx = row_idx[j]
                pred_idx = col_idx[j]
                
                # If neither ground truth nor prediction has been used
                if not used_gt[gt_idx] and not used_pred[pred_idx]:
                    correct[pred_idx, i] = True
                    used_gt[gt_idx] = True
                    used_pred[pred_idx] = True
    
    return correct


# ----------------------------- Metrics & Plotting End -----------------------

# ----------------------------- inference visualization-----------------------
class Colors:
    def __init__(self):
        hexs = (
            "042AFF", "0BDBEB", "F3F3F3", "00DFB7", "111F68", "FF6FDD",
            "FF444F",
            "CCED00", "00F344", "BD00FF", "00B4FF", "DD00BA", "00FFFF",
            "26C000",
            "01FFB3", "7D24FF", "7B0068", "FF1B6C", "FC6D2F", "A2FF0B"
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([
            [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
            [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
            [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
            [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
            [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
        ], dtype=np.uint8)

    def __call__(self, i, bgr=False):
        c = self.palette[int(i) % self.n]
        return c[::-1] if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

def draw_box(im, box, index, label=""):
    import cv2
    color = Colors()(index, True)
    x1, y1, x2, y2 = map(int, box[:4])
    cv2.rectangle(im, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)

    if label:
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        h += 3
        outside = y1 < h + 3
        if x1 + w > im.shape[1]:
            x1 = im.shape[1] - w
        y2_label = y1 + h if outside else y1 - h
        cv2.rectangle(im, (x1, y1), (x1 + w, y2_label), color, -1)

        # Draw corner markers
        corners = [
            ((x1, y1), (x1 + 15, y1), (x1, y1 + 15)),  # Top-left
            ((x2, y2), (x2 - 15, y2), (x2, y2 - 15)),  # Bottom-right
            ((x2, y1), (x2 - 15, y1), (x2, y1 + 15)),  # Top-right
            ((x1, y2), (x1, y2 - 15), (x1 + 15, y2)),  # Bottom-left
        ]
        for center, *lines in corners:
            for pt in lines:
                cv2.line(im, center, pt, (0, 255, 255), 3)

        cv2.putText(im, label, (x1, y1 + h - 3 if outside else y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return im


def load_ultralytics_weight(model, ckpt_path, verbose=False):
    """
    Load Ultralytics format weight file (like yolo11n.pt)
    
    Args:
        model: The model architecture to load weights into
        ckpt_path: Path to the Ultralytics format checkpoint file
    
    Returns:
        model: The model with loaded weights
    """
    # Load the checkpoint
    # Try loading with weights_only=True first (PyTorch 2.6+ default)
    try:
        checkpoint = torch.load(ckpt_path, weights_only=True)
    except:
        # If that fails, try with weights_only=False
        checkpoint = torch.load(ckpt_path, weights_only=False)

    # Extract the model from checkpoint
    src_model = checkpoint['model'].float()

    # Get state dicts
    dst_state_dict = model.state_dict()
    src_state_dict = src_model.state_dict()

    # Create a new state dict with matching keys
    new_state_dict = {}

    # Create a detailed mapping from Ultralytics keys to our keys
    key_mapping = {
        # Backbone mappings
        '0.conv': 'backbone.p1.0.conv',
        '0.bn': 'backbone.p1.0.norm',
        '1.conv': 'backbone.p2.0.conv',
        '1.bn': 'backbone.p2.0.norm',
        '2.cv1': 'backbone.p2.1.conv1',
        '2.cv2': 'backbone.p2.1.conv2',
        '2.m.0.cv1': 'backbone.p2.1.res_m.0.conv1',
        '2.m.0.cv2': 'backbone.p2.1.res_m.0.conv2',
        '3.conv': 'backbone.p3.0.conv',
        '3.bn': 'backbone.p3.0.norm',
        '4.cv1': 'backbone.p3.1.conv1',
        '4.cv2': 'backbone.p3.1.conv2',
        '4.m.0.cv1': 'backbone.p3.1.res_m.0.conv1',
        '4.m.0.cv2': 'backbone.p3.1.res_m.0.conv2',
        '5.conv': 'backbone.p4.0.conv',
        '5.bn': 'backbone.p4.0.norm',
        '6.cv1': 'backbone.p4.1.conv1',
        '6.cv2': 'backbone.p4.1.conv2',
        '6.m.0.cv1': 'backbone.p4.1.res_m.0.conv1',
        '6.m.0.cv2': 'backbone.p4.1.res_m.0.conv2',
        '6.m.0.cv3': 'backbone.p4.1.res_m.0.conv3',
        '6.m.0.m.0.cv1': 'backbone.p4.1.res_m.0.res_m.0.conv1',
        '6.m.0.m.0.cv2': 'backbone.p4.1.res_m.0.res_m.0.conv2',
        '6.m.0.m.1.cv1': 'backbone.p4.1.res_m.0.res_m.1.conv1',
        '6.m.0.m.1.cv2': 'backbone.p4.1.res_m.0.res_m.1.conv2',
        '7.conv': 'backbone.p5.0.conv',
        '7.bn': 'backbone.p5.0.norm',
        '8.cv1': 'backbone.p5.1.conv1',
        '8.cv2': 'backbone.p5.1.conv2',
        '8.m.0.cv1': 'backbone.p5.1.res_m.0.conv1',
        '8.m.0.cv2': 'backbone.p5.1.res_m.0.conv2',
        '8.m.0.cv3': 'backbone.p5.1.res_m.0.conv3',
        '8.m.0.m.0.cv1': 'backbone.p5.1.res_m.0.res_m.0.conv1',
        '8.m.0.m.0.cv2': 'backbone.p5.1.res_m.0.res_m.0.conv2',
        '8.m.0.m.1.cv1': 'backbone.p5.1.res_m.0.res_m.1.conv1',
        '8.m.0.m.1.cv2': 'backbone.p5.1.res_m.0.res_m.1.conv2',
        '9.cv1': 'backbone.p5.2.conv1',
        '9.cv2': 'backbone.p5.2.conv2',
        '10.cv1': 'backbone.p5.3.conv1',
        '10.cv2': 'backbone.p5.3.conv2',
        '10.m.0.attn.qkv': 'backbone.p5.3.m.0.att.qkv_conv',
        '10.m.0.attn.proj': 'backbone.p5.3.m.0.att.proj_conv',
        '10.m.0.attn.pe': 'backbone.p5.3.m.0.att.pe_conv',
        '10.m.0.ffn.0': 'backbone.p5.3.m.0.ffn.0',
        '10.m.0.ffn.1': 'backbone.p5.3.m.0.ffn.1',

        # FPN mappings
        '13.cv1': 'head.h1.conv1',
        '13.cv2': 'head.h1.conv2',
        '13.m.0.cv1': 'head.h1.res_m.0.conv1',
        '13.m.0.cv2': 'head.h1.res_m.0.conv2',
        '16.cv1': 'head.h2.conv1',
        '16.cv2': 'head.h2.conv2',
        '16.m.0.cv1': 'head.h2.res_m.0.conv1',
        '16.m.0.cv2': 'head.h2.res_m.0.conv2',
        '17.conv': 'head.h3.conv',
        '17.bn': 'head.h3.norm',
        '19.cv1': 'head.h4.conv1',
        '19.cv2': 'head.h4.conv2',
        '19.m.0.cv1': 'head.h4.res_m.0.conv1',
        '19.m.0.cv2': 'head.h4.res_m.0.conv2',
        '20.conv': 'head.h5.conv',
        '20.bn': 'head.h5.norm',
        '22.cv1': 'head.h6.conv1',
        '22.cv2': 'head.h6.conv2',
        '22.m.0.cv1': 'head.h6.res_m.0.conv1',
        '22.m.0.cv2': 'head.h6.res_m.0.conv2',
        '22.m.0.cv3': 'head.h6.res_m.0.conv3',
        '22.m.0.m.0.cv1': 'head.h6.res_m.0.res_m.0.conv1',
        '22.m.0.m.0.cv2': 'head.h6.res_m.0.res_m.0.conv2',
        '22.m.0.m.1.cv1': 'head.h6.res_m.0.res_m.1.conv1',
        '22.m.0.m.1.cv2': 'head.h6.res_m.0.res_m.1.conv2',

        # Head mappings - Box regression layers (cv3 in Ultralytics corresponds to cls in our model due to structural differences)
        '23.cv3.0.0.0': 'detect.cls.0.0.0',
        '23.cv3.0.0.1': 'detect.cls.0.0.1',
        '23.cv3.0.1.0': 'detect.cls.0.1.0',
        '23.cv3.0.1.1': 'detect.cls.0.1.1',
        '23.cv3.0.2': 'detect.cls.0.2',
        '23.cv3.1.0.0': 'detect.cls.1.0.0',
        '23.cv3.1.0.1': 'detect.cls.1.0.1',
        '23.cv3.1.1.0': 'detect.cls.1.1.0',
        '23.cv3.1.1.1': 'detect.cls.1.1.1',
        '23.cv3.1.2': 'detect.cls.1.2',
        '23.cv3.2.0.0': 'detect.cls.2.0.0',
        '23.cv3.2.0.1': 'detect.cls.2.0.1',
        '23.cv3.2.1.0': 'detect.cls.2.1.0',
        '23.cv3.2.1.1': 'detect.cls.2.1.1',
        '23.cv3.2.2': 'detect.cls.2.2',
        
        # Head mappings - Classification layers (cv2 in Ultralytics corresponds to box in our model due to structural differences)
        '23.cv2.0.0': 'detect.box.0.0',
        '23.cv2.0.1': 'detect.box.0.1',
        '23.cv2.0.2': 'detect.box.0.2',
        '23.cv2.1.0': 'detect.box.1.0',
        '23.cv2.1.1': 'detect.box.1.1',
        '23.cv2.1.2': 'detect.box.1.2',
        '23.cv2.2.0': 'detect.box.2.0',
        '23.cv2.2.1': 'detect.box.2.1',
        '23.cv2.2.2': 'detect.box.2.2',
        '23.dfl.conv': 'detect.dfl.conv',
    }

    # Map the keys from Ultralytics format to our format
    for src_key, src_value in src_state_dict.items():
        # Remove the 'model.' prefix that Ultralytics uses
        if src_key.startswith('model.'):
            src_key_no_prefix = src_key[6:]  # Remove 'model.' prefix

            # Try to find a direct mapping
            dst_key = None
            for ultralytics_prefix, our_prefix in key_mapping.items():
                if src_key_no_prefix.startswith(ultralytics_prefix):
                    # Replace the prefix
                    dst_key = src_key_no_prefix.replace(
                        ultralytics_prefix, our_prefix, 1)
                    break

            # If no direct mapping found, try pattern-based conversion
            if dst_key is None:
                # Convert Ultralytics key format to our format
                dst_key = src_key_no_prefix.replace('.bn.', '.norm.').replace(
                    '.cv1.', '.conv1.').replace('.cv2.', '.conv2.').replace('.cv3.', '.conv3.')
            else:
                # Apply additional pattern replacements to the mapped key
                # Handle special case for detection head layers
                if 'detect' in dst_key:
                    # For detection head, we need to be more careful about the mapping
                    # Remove the trailing .conv part if it exists and will be duplicated
                    if dst_key.endswith('.conv') and src_key_no_prefix.endswith('.conv'):
                        # Don't add extra .conv
                        pass
                    else:
                        dst_key = dst_key.replace('.bn.', '.norm.').replace(
                            '.cv1.', '.conv1.').replace('.cv2.', '.conv2.').replace('.cv3.', '.conv3.')
                else:
                    dst_key = dst_key.replace('.bn.', '.norm.').replace(
                        '.cv1.', '.conv1.').replace('.cv2.', '.conv2.').replace('.cv3.', '.conv3.')

            # Check if this key exists in our model and has the same shape
            if dst_key in dst_state_dict and src_value.shape == dst_state_dict[dst_key].shape:
                new_state_dict[dst_key] = src_value
                if verbose:
                    print(f"Successfully mapped {src_key} -> {dst_key}")
            else:
                if verbose:
                    print(
                        f"Skipping {src_key} -> {dst_key} due to shape mismatch or missing key")
        else:
            # For keys that don't start with 'model.', check if they exist directly
            if src_key in dst_state_dict and src_value.shape == dst_state_dict[src_key].shape:
                new_state_dict[src_key] = src_value
            else:
                if verbose:
                    print(
                        f"Skipping {src_key} due to shape mismatch or missing key")

    # Load the new state dict into the model
    model.load_state_dict(new_state_dict, strict=False)

    return model


def set_params(model, decay):
    p1 = []
    p2 = []
    norm = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)
    for m in model.modules():
        for n, p in m.named_parameters(recurse=0):
            if not p.requires_grad:
                continue
            if n == "bias":  # bias (no decay)
                p1.append(p)
            # norm-weight (no decay)
            elif n == "weight" and isinstance(m, norm):
                p1.append(p)
            else:
                p2.append(p)  # weight (with decay)
    return [{'params': p1, 'weight_decay': 0.00},
            {'params': p2, 'weight_decay': decay}]


# ----------------------------- Checkpoint Utilities -------------------------
def save_checkpoint(epoch, model, optimizer, scheduler, scaler, ema, best_map, filename):
    """
    Save complete training state to checkpoint file
    
    Args:
        epoch: Current epoch number
        model: Model object (typically EMA model for best/last checkpoints)
        optimizer: Optimizer object
        scheduler: Learning rate scheduler object
        scaler: Gradient scaler object
        ema: EMA object
        best_map: Best mAP achieved so far
        filename: Path to save checkpoint
    """
    state = {
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict() if optimizer else None,
        'scheduler': scheduler.state_dict() if scheduler else None,
        'scaler': scaler.state_dict() if scaler else None,
        'ema': ema.ema.state_dict() if ema else None,
        'best_map': best_map,
        'model_config': {
            'nc': model.detect.nc if hasattr(model, 'detect') else 0,
            'reg_max': model.detect.reg_max if hasattr(model, 'detect') else 16,
            'stride': model.stride if hasattr(model, 'stride') else None
        }
    }
    
    # Add EMA attributes if available
    if ema:
        state['ema_updates'] = ema.updates
    
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None, scaler=None, ema=None, device=None):
    """
    Load complete training state from checkpoint file
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model object to load weights into
        optimizer: Optimizer object (optional)
        scheduler: Learning rate scheduler object (optional)
        scaler: Gradient scaler object (optional)
        ema: EMA object (optional)
        device: Device to load checkpoint to
        
    Returns:
        checkpoint: Loaded checkpoint dictionary
        start_epoch: Epoch to resume from
        best_map: Best mAP from checkpoint
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    # Try loading with weights_only=True first (PyTorch 2.6+ default)
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except:
        # If that fails, try with weights_only=False
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Handle quantized models
    is_quantized_model = any(
        hasattr(module, '_packed_params') or 
        'Quantized' in type(module).__name__ or
        hasattr(module, 'qconfig') and module.qconfig is not None
        for module in model.modules()
    )
    
    # Load model state
    if 'model' in checkpoint:
        try:
            model.load_state_dict(checkpoint['model'])
        except Exception as e:
            # Handle loading into quantized models
            if is_quantized_model:
                print(f"Warning: Could not load state dict directly into quantized model: {e}")
                print("Attempting to load with strict=False...")
                try:
                    model.load_state_dict(checkpoint['model'], strict=False)
                    print("Successfully loaded state dict with strict=False")
                except Exception as e2:
                    print(f"Failed to load state dict even with strict=False: {e2}")
                    raise e2
            else:
                raise e
    elif is_quantized_model:
        # Handle case where checkpoint is a quantized model itself
        try:
            model.load_state_dict(checkpoint, strict=False)
        except Exception as e:
            print(f"Warning: Could not load quantized checkpoint: {e}")
    
    # Load optimizer state if provided
    if optimizer and 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
        try:
            optimizer.load_state_dict(checkpoint['optimizer'])
        except Exception as e:
            print(f"Warning: Could not load optimizer state: {e}")
    
    # Load scheduler state if provided
    if scheduler and 'scheduler' in checkpoint and checkpoint['scheduler'] is not None:
        try:
            scheduler.load_state_dict(checkpoint['scheduler'])
        except Exception as e:
            print(f"Warning: Could not load scheduler state: {e}")
    
    # Load scaler state if provided
    if scaler and 'scaler' in checkpoint and checkpoint['scaler'] is not None:
        try:
            scaler.load_state_dict(checkpoint['scaler'])
        except Exception as e:
            print(f"Warning: Could not load scaler state: {e}")
    
    # Load EMA state if provided
    if ema and 'ema' in checkpoint and checkpoint['ema'] is not None:
        try:
            ema.ema.load_state_dict(checkpoint['ema'])
            if 'ema_updates' in checkpoint:
                ema.updates = checkpoint['ema_updates']
        except Exception as e:
            print(f"Warning: Could not load EMA state: {e}")
    
    start_epoch = checkpoint.get('epoch', 0) + 1  # Resume from next epoch
    best_map = checkpoint.get('best_map', 0.0)
    
    print(f"Checkpoint loaded. Resuming from epoch {start_epoch}")
    return checkpoint, start_epoch, best_map