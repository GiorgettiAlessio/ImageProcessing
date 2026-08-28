"""
python /home/alessio/Desktop/progettoImage/ImageProcessing/angoliAlphaPose_HybrIK.py \
  --cfg configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml \
  --checkpoint pretrained_models/smpl/pretrained_w_cam.pth \
  --webcam 1 \
  --detector yolo \
  --gpus 0 \
  --detbatch 2 \
  --posebatch 16 \
  --fp16 \
  --udp-ip 127.0.0.1 \
  --udp-port 5065 \
  --print-fps \
  --no-stdout
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

# ---------------------------------------------------------------------------
# AlphaPose root
# ---------------------------------------------------------------------------
ALPHAPOSE_ROOT = "/home/alessio/AlphaPose"
os.chdir(ALPHAPOSE_ROOT)
if ALPHAPOSE_ROOT not in sys.path:
    sys.path.insert(0, ALPHAPOSE_ROOT)

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.webcam_detector import WebCamDetectionLoader
from detector.apis import get_detector
from scipy.spatial.transform import Rotation as R


SMPL_JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime HybrIK Pose & Unity Quaternion Output - GPU optimized"
    )

    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--webcam", type=int, default=0)
    parser.add_argument("--detector", type=str, default="yolo")

    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="'0' per GPU 0; CUDA è obbligatoria"
    )

    parser.add_argument("--detbatch", type=int, default=2)
    parser.add_argument("--posebatch", type=int, default=16)
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
        help="Mostra finestra con scheletro"
    )

    parser.add_argument("--print-fps", action="store_true", default=False)

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Usa AMP FP16 durante l'inferenza HybrIK"
    )

    parser.add_argument(
        "--tf32",
        action="store_true",
        default=True,
        help="Abilita TF32 su GPU NVIDIA compatibili"
    )

    parser.add_argument(
        "--no-stdout",
        action="store_true",
        default=False,
        help="Disattiva la stampa JSON/testuale su stdout"
    )

    parser.add_argument("--udp-ip", type=str, default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5065)

    return parser.parse_args()


def to_scalar(x, default):
    if x is None:
        return default

    while isinstance(x, (list, tuple)) and len(x) > 0:
        x = x[0]

    if isinstance(x, (list, tuple)) and len(x) == 0:
        return default

    if torch.is_tensor(x):
        if x.numel() == 1:
            x = x.item()
        else:
            x = x.flatten()[0].item()

    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def rotmats_to_quat_dict(theta_mats):
    """
    Converte le matrici di rotazione 3x3 dei 24 giunti SMPL in
    quaternioni unitari nel formato x, y, z, w atteso da Unity.

    La conversione viene eseguita direttamente dalla matrice di rotazione
    tramite scipy.spatial.transform.Rotation, evitando la conversione
    intermedia in angoli di Eulero. Questa scelta evita ambiguita' dovute
    all'ordine degli assi, al gimbal lock e all'interpretazione dei valori
    x/y/z come componenti di un quaternion.
    """
    unity_rotations = {}

    if theta_mats is None:
        return unity_rotations

    if torch.is_tensor(theta_mats):
        theta_mats = theta_mats.detach().float().cpu().numpy()

    theta_mats = np.asarray(theta_mats)
    if theta_mats.ndim != 3 or theta_mats.shape[1:] != (3, 3):
        raise RuntimeError(
            f"Matrici di rotazione inattese: shape={theta_mats.shape}; "
            "atteso (N,3,3)"
        )

    quats = R.from_matrix(theta_mats).as_quat()  # x, y, z, w

    for i, joint_name in enumerate(SMPL_JOINTS):
        if i >= quats.shape[0]:
            break

        x, y, z, w = quats[i]
        unity_rotations[joint_name] = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "w": float(w),
        }

    return unity_rotations


def extract_output(output):
    """Estrae gli output HybrIK senza assumere un unico tipo di container."""

    pred_rotations = None
    pred_xyz = None
    pred_root = None

    if isinstance(output, dict) or hasattr(output, "get"):
        pred_rotations = output.get("pred_theta_mats", None)
        pred_xyz = output.get("pred_xyz_jts_29", None)
        pred_root = output.get("transl", None)

        if pred_root is None:
            pred_root = output.get("cam_trans", None)

    elif hasattr(output, "__dict__"):
        pred_rotations = getattr(output, "pred_theta_mats", None)
        pred_xyz = getattr(output, "pred_xyz_jts_29", None)
        pred_root = getattr(output, "transl", None)

        if pred_root is None:
            pred_root = getattr(output, "cam_trans", None)

    elif isinstance(output, (tuple, list)):
        pred_rotations = output[0] if len(output) > 0 else None
        pred_xyz = output[1] if len(output) > 1 else None

    return pred_rotations, pred_xyz, pred_root


def prepare_rotations(pred_rotations):
    """
    Porta pred_theta_mats nella forma (N,24,3,3).
    """

    if pred_rotations is None:
        return None

    if pred_rotations.dim() == 2:
        n_batch = pred_rotations.shape[0]

        if pred_rotations.shape[1] != 216:
            raise RuntimeError(
                f"pred_theta_mats inatteso: shape={tuple(pred_rotations.shape)}; "
                "atteso (N,216)"
            )

        pred_rotations = pred_rotations.reshape(n_batch, 24, 3, 3)

    elif pred_rotations.dim() == 3:
        # Caso singola persona: (24,3,3)
        if pred_rotations.shape == (24, 3, 3):
            pred_rotations = pred_rotations.unsqueeze(0)

    elif pred_rotations.dim() != 4:
        raise RuntimeError(
            f"Dimensione pred_theta_mats non supportata: {pred_rotations.dim()}"
        )

    return pred_rotations


def root_to_unity(root_k):
    """Trasferisce e converte la root position solo quando serve per UDP."""

    if root_k is None:
        return None

    root_np = root_k.detach().float().cpu().numpy().reshape(-1)

    if root_np.shape[0] < 3:
        return None

    return {
        "x": float(root_np[0]),
        "y": float(-root_np[1]),
        "z": float(-root_np[2]),
    }


def main():
    args = parse_args()

    # CUDA: niente fallback silenzioso su CPU
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA non disponibile. Questo script è configurato per usare "
            "la GPU. Controlla driver NVIDIA e installazione PyTorch CUDA."
        )

    gpu_ids = [int(i) for i in args.gpus.split(",")]

    if gpu_ids[0] < 0:
        raise RuntimeError(
            "È stato richiesto --gpus -1, ma questa versione richiede CUDA."
        )

    if gpu_ids[0] >= torch.cuda.device_count():
        raise RuntimeError(
            f"GPU {gpu_ids[0]} non disponibile. "
            f"GPU CUDA rilevate: {torch.cuda.device_count()}"
        )

    torch.cuda.set_device(gpu_ids[0])
    device = torch.device(f"cuda:{gpu_ids[0]}")

    # Ottimizzazioni CUDA.
    torch.backends.cudnn.benchmark = True

    if args.tf32 and torch.cuda.get_device_capability(0)[0] >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    print("=" * 70, file=sys.stderr)
    print("GPU:", torch.cuda.get_device_name(gpu_ids[0]), file=sys.stderr)
    print("CUDA:", torch.version.cuda, file=sys.stderr)
    print("PyTorch:", torch.__version__, file=sys.stderr)
    print("Device:", device, file=sys.stderr)
    print("FP16:", args.fp16, file=sys.stderr)
    print("TF32:", args.tf32, file=sys.stderr)
    print("=" * 70, file=sys.stderr)

    cfg = update_config(args.cfg)

    args.gpus = gpu_ids
    args.device = device

    args.tracking = (
        args.pose_track
        or args.pose_flow
        or args.detector == "tracker"
    )

    # Detector / webcam
    print("Inizializzazione detector...", file=sys.stderr)

    det_loader = WebCamDetectionLoader(
        args.webcam,
        get_detector(args),
        cfg,
        args,
    )

    det_loader.start()

    # HybrIK
    print(
        f"Loading HybrIK pose model from {args.checkpoint}...",
        file=sys.stderr,
    )

    pose_model = builder.build_sppe(
        cfg.MODEL,
        preset_cfg=cfg.DATA_PRESET,
    )

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )

    pose_model.load_state_dict(checkpoint)

    # Libera immediatamente il checkpoint CPU/GPU se non serve più.
    del checkpoint

    pose_model.to(device)
    pose_model.eval()
    # UDP
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_dest = (args.udp_ip, args.udp_port)

    print(
        f"Invio dati via UDP verso {udp_dest[0]}:{udp_dest[1]}",
        file=sys.stderr,
    )

    print(
        "Streaming avviato. Ctrl+C per uscire.",
        file=sys.stderr,
    )

    frame_count = 0
    t_start = time.perf_counter()

    try:
        while True:

            # Detection + pose
            with torch.inference_mode():

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
                        cv2.imshow(
                            "AlphaPose Realtime Stream",
                            orig_img,
                        )

                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break

                    continue

                # Non sincronizzare CPU/GPU inutilmente.
                inps = inps.to(
                    device,
                    non_blocking=True,
                )

                # HybrIK inference
                if args.fp16:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.float16,
                    ):
                        output = pose_model(inps)
                else:
                    output = pose_model(inps)

                (
                    pred_rotations,
                    pred_xyz,
                    pred_root,
                ) = extract_output(output)

                # Rotazioni: matrici SMPL -> quaternioni x,y,z,w
                pred_rotations = prepare_rotations(pred_rotations)

                # Trasferimenti CPU minimizzati
                scores_cpu = (
                    scores.detach().cpu().tolist()
                    if torch.is_tensor(scores)
                    else list(scores)
                )

                n_people = len(scores_cpu)

                # xyz e root vengono convertiti solamente quando dobbiamo
                # creare il packet UDP.
                for k in range(n_people):

                    person_id = to_scalar(
                        ids[k] if ids is not None else None,
                        default=k,
                    )

                    # Rotazioni: conversione diretta matrice -> quaternion
                    unity_rotations = {}

                    if pred_rotations is not None and k < pred_rotations.shape[0]:
                        mats_k = pred_rotations[k].detach().float().cpu().numpy()
                        unity_rotations = rotmats_to_quat_dict(mats_k)

                    # XYZ 3D
                    xyz_list = []

                    if (
                        pred_xyz is not None
                        and k < pred_xyz.shape[0]
                    ):
                        xyz_list = (
                            pred_xyz[k]
                            .detach()
                            .float()
                            .cpu()
                            .tolist()
                        )

                    # Root position
                    root_position_unity = None

                    if (
                        pred_root is not None
                        and k < pred_root.shape[0]
                    ):
                        root_position_unity = root_to_unity(
                            pred_root[k]
                        )

                    # UDP packet
                    packet = {
                        "person_id": person_id,
                        "timestamp": time.time(),
                        "unity_rotations_deg": unity_rotations,
                        "joint_xyz_3d": xyz_list,
                        "root_position": root_position_unity,
                    }

                    data = json.dumps(
                        packet,
                        separators=(",", ":"),
                    ).encode("utf-8")

                    sock.sendto(
                        data,
                        udp_dest,
                    )

                    # stdout è costoso nel realtime:
                    # usarlo solo se esplicitamente richiesto.
                    if not args.no_stdout:
                        print(
                            json.dumps(packet),
                            flush=True,
                        )

                    # Debug rotazioni
                    if unity_rotations and not args.no_stdout:
                        print(
                            f"\n=== [PERSONA {person_id}] "
                            "ROTAZIONI UNITY (Quaternion x,y,z,w) ==="
                        )

                        if "L_Elbow" in unity_rotations:
                            r = unity_rotations["L_Elbow"]
                            print(
                                f"Braccio SX (L_Elbow): "
                                f"x:{r['x']:.4f} y:{r['y']:.4f} z:{r['z']:.4f} w:{r['w']:.4f}"
                            )

                        if "R_Elbow" in unity_rotations:
                            r = unity_rotations["R_Elbow"]
                            print(
                                f"Braccio DX (R_Elbow): "
                                f"x:{r['x']:.4f} y:{r['y']:.4f} z:{r['z']:.4f} w:{r['w']:.4f}"
                            )

                        if "L_Knee" in unity_rotations:
                            r = unity_rotations["L_Knee"]
                            print(
                                f"Ginocchio SX (L_Knee): "
                                f"x:{r['x']:.4f} y:{r['y']:.4f} z:{r['z']:.4f} w:{r['w']:.4f}"
                            )

                        if "R_Knee" in unity_rotations:
                            r = unity_rotations["R_Knee"]
                            print(
                                f"Ginocchio DX (R_Knee): "
                                f"x:{r['x']:.4f} y:{r['y']:.4f} z:{r['z']:.4f} w:{r['w']:.4f}"
                            )

                        print("=" * 50)

                # Visualizzazione
                if args.vis:
                    cv2.imshow(
                        "AlphaPose Realtime Stream",
                        orig_img,
                    )

                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

            # FPS
            frame_count += 1

            if args.print_fps and frame_count % 10 == 0:
                elapsed = time.perf_counter() - t_start

                fps = (
                    frame_count / elapsed
                    if elapsed > 0
                    else 0.0
                )

                print(
                    f"--- FPS Medio: {fps:.2f} | "
                    f"Rilevati: {n_people} ---",
                    file=sys.stderr,
                )

    except KeyboardInterrupt:
        print(
            "\nInterrotto dall'utente.",
            file=sys.stderr,
        )

    finally:
        det_loader.stop()
        sock.close()

        if args.vis:
            cv2.destroyAllWindows()

        print(
            "Chiuso pulito.",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()