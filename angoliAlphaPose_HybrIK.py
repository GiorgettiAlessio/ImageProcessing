"""
    python /home/alessio/Desktop/progettoImage/ImageProcessing/angoliAlphaPose_HybrIK.py \
    --cfg configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml \
    --checkpoint pretrained_models/smpl/pretrained_w_cam.pth \
    --webcam 1 \
    --detector yolo \
    --udp-ip 127.0.0.1 \
    --udp-port 5065 \
    --print-fps
"""
import argparse
import json
import os
import socket
import sys
import time

import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Forza Python a spostarsi nella cartella AlphaPose all'avvio dello script
ALPHAPOSE_ROOT = "/home/alessio/AlphaPose"
os.chdir(ALPHAPOSE_ROOT)
if ALPHAPOSE_ROOT not in sys.path:
    sys.path.insert(0, ALPHAPOSE_ROOT)

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from detector.apis import get_detector

# Mappatura Nomi Joint SMPL (24 articolazioni standard HybrIK)
SMPL_JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime HybrIK Pose & Unity Euler Angles Output"
    )
    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--webcam", type=int, default=0)
    parser.add_argument("--detector", type=str, default="yolo")
    parser.add_argument(
        "--gpus", type=str, default="0", help="'-1' per CPU, '0' per GPU"
    )
    parser.add_argument("--detbatch", type=int, default=1)
    parser.add_argument("--posebatch", type=int, default=64)
    parser.add_argument("--qsize", type=int, default=128)
    parser.add_argument("--min_box_area", type=int, default=0)
    parser.add_argument("--flip", action="store_true", default=False)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--sp", action="store_true", default=True)
    parser.add_argument("--pose_track", action="store_true", default=False)
    parser.add_argument("--pose_flow", action="store_true", default=False)

    parser.add_argument(
        "--vis",
        action="store_true",
        default=False,
        help="Mostra finestra con scheletro",
    )
    parser.add_argument("--print-fps", action="store_true", default=False)

    # --- opzioni per l'invio UDP verso Unity ---
    parser.add_argument("--udp-ip", type=str, default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5065)
    parser.add_argument(
        "--no-stdout",
        action="store_true",
        default=False,
        help="Disattiva la stampa JSON/testuale su stdout (utile in produzione, tiene solo UDP)",
    )
    return parser.parse_args()


def to_scalar(x, default):
    if x is None:
        return default
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


def matrix_to_unity_euler(rot_data_single_person):
    """Converte le rotazioni di HybrIK per UNA singola persona in Angoli di Eulero (XYZ) per Unity."""
    unity_rotations = {}

    if rot_data_single_person is None:
        return unity_rotations

    # Se è un tensore PyTorch, portiamo su CPU/Numpy
    if hasattr(rot_data_single_person, "detach"):
        rot_data_single_person = rot_data_single_person.detach().cpu().numpy()

    rot_data_single_person = np.asarray(rot_data_single_person)

    # Rimuove un'eventuale dimensione batch residua (es. [1, 24, 3, 3] -> [24, 3, 3])
    if rot_data_single_person.ndim == 4 and rot_data_single_person.shape[0] == 1:
        rot_data_single_person = rot_data_single_person[0]

    for i, joint_name in enumerate(SMPL_JOINTS):
        if i >= rot_data_single_person.shape[0]:
            break

        mat = rot_data_single_person[i]

        try:
            # Caso 1: Matrice di rotazione 3x3
            if mat.shape == (3, 3):
                r = R.from_matrix(mat)
            # Caso 2: Vettore Asse-Angolo (Rodrigues) 3D
            elif mat.shape == (3,) or mat.shape == (1, 3):
                r = R.from_rotvec(mat.reshape(3))
            # Caso 3: Quaternione 4D
            elif mat.shape == (4,):
                r = R.from_quat(mat)
            else:
                continue

            euler_deg = r.as_euler("xyz", degrees=True)

            # Conversione coordinate Right-Handed (SMPL) -> Left-Handed (Unity)
            unity_rotations[joint_name] = {
                "x": round(float(euler_deg[0]), 2),
                "y": round(float(-euler_deg[1]), 2),
                "z": round(float(-euler_deg[2]), 2),
            }
        except Exception:
            continue

    return unity_rotations


def main():
    args = parse_args()
    cfg = update_config(args.cfg)

    args.gpus = (
        [int(i) for i in args.gpus.split(",")]
        if torch.cuda.device_count() >= 1
        else [-1]
    )
    args.device = torch.device(
        "cuda:" + str(args.gpus[0]) if args.gpus[0] >= 0 else "cpu"
    )
    args.tracking = args.pose_track or args.pose_flow or args.detector == "tracker"

    print(f"Device in uso: {args.device}")

    # --- Detector e caricatore webcam ---
    det_loader = WebCamDetectionLoader(
        args.webcam, get_detector(args), cfg, args
    )
    det_loader.start()

    # --- Modello di Posa HybrIK ---
    pose_model = builder.build_sppe(cfg.MODEL, preset_cfg=cfg.DATA_PRESET)
    print(f"Loading HybrIK pose model from {args.checkpoint}...")
    pose_model.load_state_dict(
        torch.load(args.checkpoint, map_location=args.device)
    )
    pose_model.to(args.device)
    pose_model.eval()

    # --- Socket UDP verso Unity ---
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_dest = (args.udp_ip, args.udp_port)
    print(f"Invio dati via UDP verso {udp_dest[0]}:{udp_dest[1]}", file=sys.stderr)

    print(
        "Streaming avviato. (Ctrl+C per uscire)",
        file=sys.stderr,
    )
    frame_count = 0
    t_start = time.time()

    try:
        while True:
            with torch.no_grad():
                (
                    inps,
                    orig_img,
                    im_name,
                    boxes,
                    scores,
                    ids,
                    cropped_boxes,
                ) = det_loader.read()

                if orig_img is None:
                    break
                if boxes is None or boxes.nelement() == 0:
                    if args.vis:
                        cv2.imshow("AlphaPose Realtime Stream", orig_img)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

                inps = inps.to(args.device)

                output = pose_model(inps)

                pred_rotations = None
                pred_xyz = None
                pred_root = None

                # 1. Se output è un dizionario o EasyDict (uso di .get() sicuro)
                if isinstance(output, dict) or hasattr(output, "get"):
                    pred_rotations = output.get("pred_theta_mats", None)
                    pred_xyz = output.get("pred_xyz_jts_29", None)
                    # Posizione root: proviamo 'transl' e, se assente, 'cam_trans'
                    pred_root = output.get("transl", None)
                    if pred_root is None:
                        pred_root = output.get("cam_trans", None)

                # 2. Se output è un oggetto custom con attributi
                elif hasattr(output, "__dict__"):
                    pred_rotations = getattr(output, "pred_theta_mats", None)
                    pred_xyz = getattr(output, "pred_xyz_jts_29", None)
                    pred_root = getattr(output, "transl", None)
                    if pred_root is None:
                        pred_root = getattr(output, "cam_trans", None)

                # 3. Fallback se output è una lista/tupla
                if pred_rotations is None and isinstance(output, (tuple, list)):
                    pred_rotations = output[0]
                    pred_xyz = output[1] if len(output) > 1 else None

                # --- DEBUG UNA TANTUM: verifica shape di pred_root al primo frame
                if frame_count == 0 and pred_root is not None:
                    print("SHAPE pred_root (transl/cam_trans):", pred_root.shape, file=sys.stderr)

                # --- FIX: pred_theta_mats arriva appiattito (N, 216) -----------
                # 216 = 24 giunti x 9 valori (matrice 3x3 srotolata). Va
                # riportato a (N, 24, 3, 3) prima di poterlo usare per giunto.
                if pred_rotations is not None and pred_rotations.dim() == 2:
                    n_batch = pred_rotations.shape[0]
                    pred_rotations = pred_rotations.reshape(n_batch, 24, 3, 3)
                # -----------------------------------------------------------------

                # --- Stampa a schermo / stdout per Unity ---
                for k in range(len(scores)):
                    person_id = to_scalar(ids[k], default=k)

                    # Estraiamo rotazioni, coordinate 3D e posizione root per la SINGOLA persona k
                    rot_k = pred_rotations[k] if (pred_rotations is not None and len(pred_rotations) > k) else None
                    xyz_k = pred_xyz[k] if (pred_xyz is not None and len(pred_xyz) > k) else None
                    root_k = pred_root[k] if (pred_root is not None and len(pred_root) > k) else None

                    # Conversione Cinematica Inversa per la persona k
                    unity_degrees = matrix_to_unity_euler(rot_k)

                    xyz_list = xyz_k.cpu().numpy().tolist() if hasattr(xyz_k, "cpu") else []

                    # --- Posizione root (Hips) per il root motion in Unity ---
                    root_position_unity = None
                    if root_k is not None:
                        root_np = root_k.detach().cpu().numpy() if hasattr(root_k, "detach") else np.asarray(root_k)
                        root_np = root_np.reshape(-1)  # appiattisce, es. (1,3)->(3,)
                        if root_np.shape[0] >= 3:
                            # Conversione destrorso (SMPL) -> sinistrorso (Unity):
                            # stessa convenzione di segno usata per le rotazioni (flip Y e Z).
                            # Verifica visivamente in Unity e correggi se l'avatar
                            # risulta specchiato o si muove nel verso sbagliato.
                            root_position_unity = {
                                "x": float(root_np[0]),
                                "y": float(-root_np[1]),
                                "z": float(-root_np[2]),
                            }

                    packet = {
                        "person_id": person_id,
                        "timestamp": time.time(),
                        "unity_rotations_deg": unity_degrees,
                        "joint_xyz_3d": xyz_list,
                        "root_position": root_position_unity,
                    }

                    # --- Invio UDP verso Unity ---
                    data = json.dumps(packet).encode("utf-8")
                    sock.sendto(data, udp_dest)

                    # Stampa JSON su stdout (disattivabile con --no-stdout)
                    if not args.no_stdout:
                        print(json.dumps(packet), flush=True)

                    # Stampa riassuntiva a terminale
                    if unity_degrees and not args.no_stdout:
                        print(f"\n=== [PERSONA {person_id}] ROTAZIONI UNITY (Gradi) ===")
                        if 'L_Elbow' in unity_degrees:
                            print(f"Braccio SX (L_Elbow):    X:{unity_degrees['L_Elbow']['x']}° Y:{unity_degrees['L_Elbow']['y']}° Z:{unity_degrees['L_Elbow']['z']}°")
                        if 'R_Elbow' in unity_degrees:
                            print(f"Braccio DX (R_Elbow):    X:{unity_degrees['R_Elbow']['x']}° Y:{unity_degrees['R_Elbow']['y']}° Z:{unity_degrees['R_Elbow']['z']}°")
                        if 'L_Knee' in unity_degrees:
                            print(f"Ginocchio SX (L_Knee):   X:{unity_degrees['L_Knee']['x']}° Y:{unity_degrees['L_Knee']['y']}° Z:{unity_degrees['L_Knee']['z']}°")
                        if 'R_Knee' in unity_degrees:
                            print(f"Ginocchio DX (R_Knee):   X:{unity_degrees['R_Knee']['x']}° Y:{unity_degrees['R_Knee']['y']}° Z:{unity_degrees['R_Knee']['z']}°")
                        print("=" * 50 + "\n")

                # --- Visualizzazione opzionale ---
                if args.vis:
                    cv2.imshow("AlphaPose Realtime Stream", orig_img)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

                frame_count += 1
                if args.print_fps and frame_count % 10 == 0:
                    elapsed = time.time() - t_start
                    fps = frame_count / elapsed if elapsed > 0 else 0
                    print(
                        f"--- FPS Medio: {fps:.2f} | Rilevati: {len(scores)} ---",
                        file=sys.stderr,
                    )

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.", file=sys.stderr)
    finally:
        det_loader.stop()
        sock.close()
        if args.vis:
            cv2.destroyAllWindows()
        print("Chiuso pulito.", file=sys.stderr)


if __name__ == "__main__":
    main()