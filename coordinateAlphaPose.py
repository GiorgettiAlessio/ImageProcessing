#!/usr/bin/env python3
"""
    eseguire dentro ambiente conda o con dipendenze corrette
    python3 /home/alessio/Desktop/progettoImage/ImageProcessing/coordinateAlphaPose.py \\
        --cfg configs/coco/resnet/256x192_res50_lr1e-3_1x.yaml \\
        --checkpoint pretrained_models/fast_res50_256x192.pth \\
        --webcam (SCEGLIERE CANALE WEBCAM) \\
        --print-fps
"""

import argparse
import json
import os
import sys
import time

import cv2
import torch


# Forza Python a spostarsi nella cartella AlphaPose all'avvio dello script
ALPHAPOSE_ROOT = "/home/alessio/AlphaPose"
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
    parser = argparse.ArgumentParser(description="Realtime pose -> Print JSON to stdout")    # --- stessi argomenti chiave di demo_inference.py -------------------
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--webcam", type=int, default=0)
    parser.add_argument("--detector", type=str, default="yolo")
    parser.add_argument("--gpus", type=str, default="-1",
                         help="'-1' per CPU, '0' per GPU 0, ecc.")
    parser.add_argument("--detbatch", type=int, default=1)
    parser.add_argument("--posebatch", type=int, default=64)
    parser.add_argument("--qsize", type=int, default=128)
    parser.add_argument("--min_box_area", type=int, default=0)
    parser.add_argument("--flip", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--sp", action="store_true", default=True)
    parser.add_argument("--pose_track", action="store_true", default=False)
    parser.add_argument("--pose_flow", action="store_true", default=False)

    #--- opzione per attivare la finestra di visualizzazione ------------
    parser.add_argument("--vis", action="store_true", default=False,help="Mostra la finestra della webcam con lo scheletro disegnato")

    # --- opzioni nostre, per l'invio UDP ---------------------------------
    parser.add_argument("--print-fps", action="store_true", default=False)
    return parser.parse_args()

def to_scalar(x, default):
    if x is None:
        return default
    # Srotola ricorsivamente liste o tuple annidate (es. [[0]] -> 0)
    while isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]
    if isinstance(x, (list, tuple)) and len(x) == 0:
        return default
    if torch.is_tensor(x):
        x = x.item() if x.numel() == 1 else x.flatten()[0].item()
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

def main():
    args = parse_args()
    cfg = update_config(args.cfg)

    # --- device: CPU o GPU, stessa logica di demo_inference.py ----------
    args.gpus = [int(i) for i in args.gpus.split(",")] if torch.cuda.device_count() >= 1 else [-1]
    args.device = torch.device("cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu")
    args.tracking = args.pose_track or args.pose_flow or args.detector == "tracker"

    print(f"Device in uso: {args.device}")

    # --- detector (YOLO) + caricatore webcam -----------------------------
    det_loader = WebCamDetectionLoader(args.webcam, get_detector(args), cfg, args)
    det_loader.start()

    # --- modello di posa ---------------------------------------------------
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print(f"Loading pose model from {args.checkpoint}...")
    pose_model.load_state_dict(torch.load(args.checkpoint, map_location=args.device))
    pose_model.to(args.device)
    pose_model.eval()

    heatmap_to_coord = get_func_heatmap_to_coord(cfg)
    hm_size = cfg.DATA_PRESET.HEATMAP_SIZE
    norm_type = cfg.LOSS.get("NORM_TYPE", None)
    use_heatmap_loss = (cfg.DATA_PRESET.get("LOSS_TYPE", "MSELoss") == "MSELoss")


    print("Streaming avviato. Output JSON in corso su stdout... (Ctrl+C per uscire)", file=sys.stderr)
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            with torch.no_grad():
                (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes) = det_loader.read()

                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    if args.vis:
                        cv2.imshow("AlphaPose Realtime Stream", orig_img)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            break
                    continue

                # --- pose estimation sul frame corrente ----------------------
                inps = inps.to(args.device)
                hm = pose_model(inps)
                hm = hm.cpu()

                # --- da heatmap a coordinate (x, y) ---------------------------
                pose_coords = []
                pose_scores = []
                for i in range(hm.shape[0]):
                    bbox = cropped_boxes[i].tolist()
                    pose_coord, pose_score = heatmap_to_coord(
                        hm[i], bbox, hm_shape=hm_size, norm_type=norm_type)
                    pose_coords.append(torch.from_numpy(pose_coord).unsqueeze(0))
                    pose_scores.append(torch.from_numpy(pose_score).unsqueeze(0))
                preds_img = torch.cat(pose_coords)
                preds_scores = torch.cat(pose_scores)

                # --- soppressione dei duplicati multi-persona -----------------
                if not args.pose_track:
                    boxes, scores, ids, preds_img, preds_scores, pick_ids = pose_nms(
                        boxes, scores, ids, preds_img, preds_scores,
                        args.min_box_area, use_heatmap_loss=use_heatmap_loss)
                    
                

                # --- stampa del JSON su stdout, una riga per persona ----------
                for k in range(len(scores)):
                    keypoints = preds_img[k].numpy().tolist()      # [[x,y], [x,y], ...]
                    kp_scores = preds_scores[k].numpy().flatten().tolist()
                    packet = {
                        "person_id": to_scalar(ids[k], default=k),
                        "timestamp": time.time(),
                        "keypoints": keypoints,
                        "scores": kp_scores,
                    }
                    # Stampa e forza lo svuotamento del buffer per lettura in tempo reale
                    print(json.dumps(packet), flush=True)


                # --- Visualizzazione a schermo (se attiva l'opzione --vis) ---
                if args.vis:
                    # Prepara la struttura dati richiesta da vis_frame
                    results = {
                        'imgname': im_name,
                        'result': []
                    }
                    for k in range(len(scores)):
                        results['result'].append({
                            'keypoints': preds_img[k],
                            'kp_score': preds_scores[k],
                            'proposal_score': torch.mean(preds_scores[k]) + scores[k] + 1.25 * max(preds_scores[k]),
                            'idx': ids[k]
                        })

                    args.showbox = getattr(args, 'showbox', False)

                    # Disegna gli scheletri sopra la copia del frame originale
                    vis_img = vis_frame(orig_img, results, args, [0.4] * 100)
                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
                    
                    cv2.imshow("AlphaPose Realtime Stream", vis_img)
                    
                    # Premi 'q' per uscire velocemente dalla finestra video
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

                frame_count += 1
                if args.print_fps and frame_count % 10 == 0:
                    elapsed = time.time() - t_start
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(f"\n---------------------FPS medio: {fps:.2f} | persone rilevate questo frame: {len(scores)}---------------------\n")




    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.", file=sys.stderr)
    finally:
        det_loader.stop()
        if args.vis:
            cv2.destroyAllWindows()
        print("Chiuso pulito.", file=sys.stderr)


if __name__ == "__main__":
    main()