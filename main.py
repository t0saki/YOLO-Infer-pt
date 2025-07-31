import os
import csv
import cv2
import copy
import tqdm
import yaml
import torch
import argparse
import warnings
import numpy as np
from torch.utils import data
from torch import distributed as dist
from torch.nn.utils import clip_grad_norm_ as clip
from torch.nn.parallel import DistributedDataParallel

from nets import nn
from utils import util
from utils.util import device
from utils.dataset import Dataset


def train(args, params):
    util.init_seeds()

    model = nn.yolo_v11_n(args.num_cls)

    # Resume from checkpoint or load pretrained weights
    start_epoch = 0
    best = 0
    if hasattr(args, 'resume') and args.resume:
        # Resume from checkpoint
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model'].state_dict() if hasattr(
            checkpoint['model'], 'state_dict') else checkpoint['model'])
        start_epoch = checkpoint.get('epoch', 0)
        best = checkpoint.get('best', 0)
        print(f"Resumed training from epoch {start_epoch}")
    elif hasattr(args, 'weights') and args.weights:
        # Load pretrained weights
        from utils.util import load_ultralytics_weight
        model = load_ultralytics_weight(model, args.weights)
    model.to(device)

    if args.distributed:
        util.setup_ddp(args)

    # Freeze DFL Layer
    util.freeze_layer(model)

    # Fix for MPS device: disable GradScaler on MPS as it doesn't support float64
    use_scaler = device.type == 'cuda'
    scaler = torch.amp.GradScaler(device=device, enabled=use_scaler)
    
    # DDP setup
    if args.distributed:
        model = DistributedDataParallel(module=model,
                                        device_ids=[args.rank],
                                        find_unused_parameters=True)

    ema = util.EMA(model) if args.rank == 0 else None

    sampler = None
    dataset = Dataset(args, params, True)
    if args.distributed:
        sampler = data.distributed.DistributedSampler(dataset)

    batch_size = args.batch_size // max(args.world_size, 1)
    loader = data.DataLoader(dataset, batch_size, sampler is None,
                             sampler, num_workers=8, pin_memory=True,
                             collate_fn=Dataset.collate_fn)

    accumulate = max(round(64 / args.batch_size * args.world_size), 1)
    decay = params['decay'] * args.batch_size * accumulate / 64
    optimizer = util.smart_optimizer(args, model, decay)
    linear = lambda x: (max(1  / args.epochs, 0) * (1.0 - 0.01) + 0.01)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=linear)
    scheduler.last_epoch = - 1
    criterion = util.DetectionLoss(model)

    opt_step = -1
    num_batch = len(loader)
    warm_up = max(round(3 * num_batch), 100)

    best_map = 0.0

    with open('weights/step.csv', 'w') as log:
        if args.rank == 0:
            logger = csv.DictWriter(log, fieldnames=['epoch',
                                                     'box', 'cls', 'dfl',
                                                     'Recall', 'Precision', 'mAP@50', 'mAP'])
            logger.writeheader()
        for epoch in range(args.epochs):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                scheduler.step()

            model.train()
            if args.distributed:
                sampler.set_epoch(epoch)

            p_bar = enumerate(loader)
            if args.epochs - epoch == 10:
                loader.dataset.mosaic = False

            if args.rank == 0:
                print("\n" + "%11s" * 5 % ("Epoch", "GPU", "box", "cls", "dfl"))
                p_bar = tqdm.tqdm(enumerate(loader), total=num_batch)

            t_loss = None
            for i, batch in p_bar:
                glob_step = i + num_batch * epoch
                if glob_step <= warm_up:
                    xi = [0, warm_up]
                    accumulate = max(1, int(np.interp(glob_step, xi, [1, 64 / args.batch_size]).round()))
                    for j, x in enumerate(optimizer.param_groups):
                        x["lr"] = np.interp(glob_step, xi, [0.0 if j == 0 else 0.0,
                                                            x["initial_lr"] * linear(epoch)])

                        if "momentum" in x:
                            x["momentum"] = np.interp(glob_step, xi, [0.8, 0.937])

                images = batch["img"].to(device).float() / 255

                if device.type == 'cuda':
                    with torch.amp.autocast(device):
                        pred = model(images)
                        loss, loss_items = criterion(pred, batch)
                        if args.distributed:
                            loss *= args.world_size

                        t_loss = ((t_loss * i + loss_items) / (
                                    i + 1) if t_loss is not None else loss_items)
                elif device.type == 'mps':
                    # MPS autocast support
                    with torch.amp.autocast('cpu'):  # Use CPU autocast for MPS
                        pred = model(images)
                        loss, loss_items = criterion(pred, batch)
                        if args.distributed:
                            loss *= args.world_size

                        t_loss = ((t_loss * i + loss_items) / (
                                    i + 1) if t_loss is not None else loss_items)
                else:
                    pred = model(images)
                    loss, loss_items = criterion(pred, batch)
                    if args.distributed:
                        loss *= args.world_size

                    t_loss = ((t_loss * i + loss_items) / (
                                i + 1) if t_loss is not None else loss_items)

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()
                    
                if glob_step - opt_step >= accumulate:
                    if use_scaler:
                        scaler.unscale_(optimizer)
                    clip(model.parameters(), max_norm=10.0)
                    if use_scaler:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad()
                    if ema:
                        ema.update(model)
                    opt_step = glob_step

                if args.rank == 0:
                    fmt = "%11s" * 2 + "%11.4g" * 3
                    if device.type == 'cuda':
                        memory = f'{torch.cuda.memory_allocated() / 1e9:.3g}G'
                    elif device.type == 'mps':
                        memory = f'{torch.mps.current_allocated_memory() / 1e9:.3g}G'
                    else:
                        memory = 'CPU'
                    p_bar.set_description(fmt % (f"{epoch + 1}/{args.epochs}", memory, *t_loss))

            if args.rank == 0:
                m_pre, m_rec, map50, mean_map = validate(args, params, ema.ema)
                box, cls, dfl = map(float, t_loss)

                logger.writerow({'epoch': str(epoch + 1).zfill(3),
                                 'box': str(f'{box:.3f}'),
                                 'cls': str(f'{cls:.3f}'),
                                 'dfl': str(f'{dfl:.3f}'),
                                 'mAP': str(f'{mean_map:.3f}'),
                                 'mAP@50': str(f'{map50:.3f}'),
                                 'Recall': str(f'{m_rec:.3f}'),
                                 'Precision': str(f'{m_pre:.3f}')})
                log.flush()

                ckpt = {'epoch': epoch+1, 'model': copy.deepcopy(ema.ema)}
                torch.save(ckpt, 'weights/last.pt')

                if mean_map > best_map:
                    best_map = mean_map
                    torch.save(ckpt, 'weights/best.pt')

                del ckpt

            if args.distributed:
                dist.barrier()

        if args.distributed:
            dist.destroy_process_group()

        print("Training complete.")


def validate(args, params, model=None):
    iou_v = torch.linspace(0.5, 0.95, 10)
    n_iou = iou_v.numel()

    metric = {"tp": [], "conf": [], "pred_cls": [], "target_cls": [], "target_img": []}

    if not model:
        args.plot = True
        model = torch.load(f='weights/best.pt', map_location=device, weights_only=False)
        model = model['model'].float().fuse()

    # model.half()
    model.eval()
    dataset = Dataset(args, params, False)
    loader = data.DataLoader(dataset, batch_size=16,
                             shuffle=False, num_workers=4,
                             pin_memory=True, collate_fn=Dataset.collate_fn)

    for batch in tqdm.tqdm(loader, desc=('%10s' * 5) % (
    '', 'precision', 'recall', 'mAP50', 'mAP')):
        image = (batch["img"].to(device).float()) / 255
        for k in ["idx", "cls", "box"]:
            batch[k] = batch[k].to(device)

        outputs = util.non_max_suppression(model(image))

        metric = util.update_metrics(outputs, batch, n_iou, iou_v, metric)

    stats = {k: torch.cat(v, 0).cpu().numpy() for k, v in metric.items()}
    stats.pop("target_img", None)
    if len(stats) and stats["tp"].any():
        result = util.compute_ap(tp=stats['tp'],
                                 conf=stats['conf'],
                                 pred=stats['pred_cls'],
                                 target=stats['target_cls'],
                                 plot=args.plot,
                                 save_dir='weights/',
                                 names=params['names'])

        m_pre = result['precision']
        m_rec = result['recall']
        map50 = result['mAP50']
        mean_ap = result['mAP50-95']
    else:
        m_pre, m_rec, map50, mean_ap = 0.0, 0.0, 0.0, 0.0

    print(('%10s' + '%10.3g' * 4) % ('', m_pre, m_rec, map50, mean_ap))

    model.float()
    return m_pre, m_rec, map50, mean_ap

@torch.no_grad()
def inference(args, params):
    model = torch.load('./weights/v11_n.pt', map_location=device, weights_only=False)['model'].float()
    model.half()
    model.eval()

    camera = cv2.VideoCapture('input.mp4')

    # Get video properties
    width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = camera.get(cv2.CAP_PROP_FPS)
    
    # Use more reliable codec - try H264 first, fallback to XVID
    try:
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
        # Test if the writer was created successfully
        if not out.isOpened():
            raise Exception("H264 codec failed")
    except:
        # Fallback to XVID codec with .avi extension
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
        print("Using XVID codec, output file will be: output.avi")

    if not camera.isOpened():
        print("Error opening video stream or file")
        return

    if not out.isOpened():
        print("Error creating video writer")
        camera.release()
        return

    while camera.isOpened():
        success, frame = camera.read()
        if success:
            # Ensure frame is in correct format (BGR, uint8)
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            image = frame.copy()
            shape = image.shape[:2]

            r = args.inp_size / max(shape[0], shape[1])
            if r != 1:
                resample = cv2.INTER_LINEAR if r > 1 else cv2.INTER_AREA
                image = cv2.resize(image, dsize=(int(shape[1] * r), int(shape[0] * r)), interpolation=resample)
            height, width = image.shape[:2]

            # Scale ratio (new / old)
            r = min(1.0, args.inp_size / height, args.inp_size / width)

            # Compute padding
            pad = int(round(width * r)), int(round(height * r))
            w = (args.inp_size - pad[0]) / 2
            h = (args.inp_size - pad[1]) / 2

            if (width, height) != pad:  # resize
                image = cv2.resize(image, pad, interpolation=cv2.INTER_LINEAR)
            top, bottom = int(round(h - 0.1)), int(round(h + 0.1))
            left, right = int(round(w - 0.1)), int(round(w + 0.1))
            image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT)

            # Convert HWC to CHW, BGR to RGB
            x = image.transpose((2, 0, 1))[::-1]
            x = np.ascontiguousarray(x)
            x = torch.from_numpy(x)
            x = x.unsqueeze(dim=0)
            x = x.to(device)
            x = x.half()
            x = x / 255
            # Inference
            outputs = model(x)
            # NMS
            outputs = util.non_max_suppression(outputs, 0.15, 0.2)[0]

            if outputs is not None:
                outputs[:, [0, 2]] -= w
                outputs[:, [1, 3]] -= h
                outputs[:, :4] /= min(height / shape[0], width / shape[1])

                outputs[:, 0].clamp_(0, shape[1])
                outputs[:, 1].clamp_(0, shape[0])
                outputs[:, 2].clamp_(0, shape[1])
                outputs[:, 3].clamp_(0, shape[0])

                for box in outputs:
                    box = box.cpu().numpy()
                    x1, y1, x2, y2, score, index = box
                    class_name = params['names'][int(index)]
                    label = f"{class_name} {score:.2f}"
                    util.draw_box(frame, box, index, label)

            # Ensure frame is in correct format before writing
            if frame.dtype != np.uint8:
                frame = frame.astype(np.uint8)
            
            cv2.imshow('Frame', frame)
            out.write(frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    
    camera.release()
    out.release()
    cv2.destroyAllWindows()
    print("Video processing completed successfully!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--epochs', default=2, type=int)
    parser.add_argument('--num-cls', type=int, default=80)
    parser.add_argument('--inp-size', type=int, default=640)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--data-dir', type=str, default='COCO')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--validate', action='store_true')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--weights', type=str, help='Path to weights file')
    parser.add_argument('--resume', type=str,
                        help='Path to checkpoint file for resuming training')

    args = parser.parse_args()

    args.rank = int(os.environ.get("RANK", 0))
    args.world_size = int(os.getenv('WORLD_SIZE', 1))
    args.distributed = int(os.getenv('WORLD_SIZE', 1)) > 1

    with open('utils/args.yaml', errors='ignore') as f:
        params = yaml.safe_load(f)

    if args.train:
        train(args, params)
    if args.validate:
        validate(args, params)
    if args.inference:
        inference(args, params)

if __name__ == "__main__":
    main()