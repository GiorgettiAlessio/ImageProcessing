"""
    Comandi per windows!!
    Per runnare: installre anaconda prompt e scrivere:
    conda activate alphapose
    cd "Dove avete AlfaPose il repository" es: C:\Users\marti \AlphaPose
    python newcoordinateAlphaPose.py --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml --checkpoint pretrained_models/fast_res50_256x192.pth --webcam 0 --gpus 0 --vis --print-fps
"""

#!/usr/bin/env python3

import argparse
import json
import os
import sys
import time

import cv2
import torch


# Root AlphaPose
ALPHAPOSE_ROOT = os.getcwd()
os.chdir(ALPHAPOSE_ROOT)

if ALPHAPOSE_ROOT not in sys.path:
    sys.path.insert(0, ALPHAPOSE_ROOT)


from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.transforms import get_func_heatmap_to_coord
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from alphapose.utils.pPose_nms import pose_nms
from alphapose.utils.vis import vis_frame
from detector.apis import get_detector


def parse_args():

    parser = argparse.ArgumentParser(description="Realtime pose -> Print JSON to stdout")

    # Configurazione modello
    parser.add_argument("--cfg",type=str,required=True)
    parser.add_argument("--checkpoint",type=str,required=True)

    # Webcam / detector
    parser.add_argument("--webcam",type=int,default=0)
    parser.add_argument("--detector",type=str,default="yolo")

    # GPU
    parser.add_argument("--gpus",type=str,default="-1",help="'-1' CPU, '0' GPU 0")

    # Parametri usati da AlphaPose internamente
    parser.add_argument("--detbatch",type=int,default=1)
    parser.add_argument("--posebatch",type=int,default=64)
    parser.add_argument("--qsize",type=int,default=128)

    # Post processing
    parser.add_argument("--min_box_area",type=int,default=0)

    # Opzioni AlphaPose
    parser.add_argument("--flip",action="store_true",default=False)
    parser.add_argument("--debug",action="store_true",default=False)

    # Serve a webcam_detector.py
    parser.add_argument("--sp",action="store_true",default=True)

    # Tracking
    parser.add_argument("--pose_track",action="store_true",default=False)
    parser.add_argument("--pose_flow",action="store_true",default=False)

    # Visualizzazione
    parser.add_argument("--vis",action="store_true",default=False,help="Mostra finestra webcam con scheletro")

    # FPS
    parser.add_argument("--print-fps",action="store_true",default=False)

    return parser.parse_args()


def to_scalar(x, default):

    if x is None:
        return default

    while isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]

    if torch.is_tensor(x):
        x = x.item()

    try:
        return int(x)
    except:
        return default


def main():

    args = parse_args()
    cfg = update_config(args.cfg)

    # ---------------- DEVICE ----------------
    if torch.cuda.device_count() > 0 and args.gpus != "-1":

        args.gpus = [
            int(x)
            for x in args.gpus.split(",")
        ]

        args.device = torch.device("cuda:" + str(args.gpus[0]))

    else:
        args.device = torch.device("cpu")

    args.tracking = (
        args.pose_track
        or args.pose_flow
        or args.detector == "tracker"
    )

    print(f"Device in uso: {args.device}",flush=True)


    # ---------------- DETECTOR ----------------

    print("Creazione detector YOLO", flush=True)
    det_loader = WebCamDetectionLoader(args.webcam,get_detector(args),cfg,args)
    print("Detector creato",flush=True)

    # ---------------- POSE MODEL ----------------


    print("Creazione FastPose",flush=True)
    pose_model = builder.build_sppe(cfg.MODEL,preset_cfg=cfg.DATA_PRESET)
    print("POSE MODEL CREATO",flush=True)

    pose_model = pose_model.to(args.device)
    print("POSE MODEL SU GPU",flush=True)
    print("Caricamento checkpoint",flush=True)

    checkpoint = torch.load(args.checkpoint,map_location=args.device)
    print("Checkpoint letto",flush=True)
    result = pose_model.load_state_dict(checkpoint)
    print("Pesi caricati:",result,flush=True)
    pose_model.eval()
    print("MODELLO IN EVAL",flush=True)

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    norm_type = cfg.LOSS.get("NORM_TYPE",None)
    use_heatmap_loss = (
        cfg.DATA_PRESET.get(
            "LOSS_TYPE",
            "MSELoss"
        ) == "MSELoss")

    # Ora parte webcam + YOLO
    print("Avvio webcam detector",flush=True)
    det_loader.start()
    print("Streaming iniziato",flush=True)

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            with torch.no_grad():
                (inps,orig_img,im_name,boxes,scores,ids,cropped_boxes) = det_loader.read()

                if orig_img is None:
                    break

                if boxes is None or boxes.nelement() == 0:

                    if args.vis:
                        cv2.imshow("AlphaPose",orig_img)

                        if cv2.waitKey(1) & 0xff == ord("q"):
                            break

                    continue

                # -------- POSE --------
                inps = inps.to(args.device)
                hm = pose_model(inps)
                hm = hm.cpu()

                pose_coords = []
                pose_scores = []

                for i in range(hm.shape[0]):

                    bbox = cropped_boxes[i].tolist()
                    pose_coord, pose_score = heatmap_to_coord(
                        hm[i],
                        bbox,
                        hm_shape=hm_size,
                        norm_type=norm_type
                    )
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))

                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))

                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)

                if not args.pose_track:
                    boxes, scores, ids, preds_img, preds_scores, _ = pose_nms(boxes,scores,ids,preds_img,preds_scores,args.min_box_area,use_heatmap_loss=use_heatmap_loss)

                # -------- JSON OUTPUT --------
                for k in range(len(scores)):
                    packet = {
                        "person_id":to_scalar(ids[k], k),
                        "timestamp":time.time(),
                        "keypoints":preds_img[k].numpy().tolist(),
                        "scores":preds_scores[k].numpy().flatten().tolist()
                    }

                    print(json.dumps(packet),flush=True)

                # -------- VIS --------
                if args.vis:
                    results = {"imgname": im_name,"result": []}

                    for k in range(len(scores)):

                        results["result"].append({
                            "keypoints": preds_img[k],
                            "kp_score":preds_scores[k],
                            "proposal_score":torch.mean(preds_scores[k])+ scores[k]+ 1.25 * max(preds_scores[k]),
                            "idx":ids[k]
                        })

                    args.showbox = False
                    vis_img = vis_frame(orig_img,results,args,[0.4]*100)
                    vis_img = cv2.cvtColor(vis_img,cv2.COLOR_RGB2BGR)

                    cv2.imshow("AlphaPose",vis_img)

                    if cv2.waitKey(1) & 0xff == ord("q"):
                        break

                frame_count += 1
                if args.print_fps and frame_count % 10 == 0:

                    fps = frame_count / (time.time() - t_start)

                    print(f"FPS: {fps:.2f}",flush=True)


    except KeyboardInterrupt:

        print("Interrotto",flush=True)
    finally:
        det_loader.stop()

        if args.vis:
            cv2.destroyAllWindows()

        print("Chiuso",flush=True)



if __name__ == "__main__":
    main()