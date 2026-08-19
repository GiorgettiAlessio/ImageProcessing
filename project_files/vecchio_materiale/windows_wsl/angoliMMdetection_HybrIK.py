import os

# ============================================================
# CONFIGURAZIONE CUDA
# ============================================================

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["MAGMA_VERBOSE"] = "0"

import warnings

warnings.filterwarnings(
    "ignore",
    message=".*MAGMA.*"
)

# ============================================================
# IMPORT
# ============================================================

import argparse
import json
import time
import cv2
import numpy as np
import torch

import shutil
import sys

from easydict import EasyDict as edict
from contextlib import contextmanager

from scipy.spatial.transform import Rotation as R


# ============================================================
# GESTIONE DIRECTORY
# ============================================================

@contextmanager
def in_directory(path):
    old_dir = os.getcwd()
    os.chdir(path)

    try:
        yield
    finally:
        os.chdir(old_dir)


# ============================================================
# PERCORSI
# ============================================================

MMDET_DIR = "/home/marti/mmdetection"
HYBRIK_DIR = "/home/marti/HybrIK"

# Aggiungiamo i repository al PYTHONPATH

sys.path.insert(0, MMDET_DIR)
sys.path.insert(0, HYBRIK_DIR)


# ============================================================
# MODELLO SMPL-X
# ============================================================

target_smplx_dir = os.path.join(
    HYBRIK_DIR,
    "model_files",
    "smplx"
)

target_smplx_file = os.path.join(
    target_smplx_dir,
    "SMPLX_NEUTRAL.npz"
)

source_smplx_file = "/home/marti/SMPLX_NEUTRAL.npz"


if not os.path.exists(target_smplx_file):

    if os.path.exists(source_smplx_file):

        os.makedirs(
            target_smplx_dir,
            exist_ok=True
        )

        shutil.copy(
            source_smplx_file,
            target_smplx_file
        )

        print(
            f"File SMPL-X copiato con successo in: "
            f"{target_smplx_file}"
        )

    else:

        raise FileNotFoundError(
            f"Non trovo il file SMPLX_NEUTRAL.npz in "
            f"{source_smplx_file}!"
        )


# ============================================================
# IMPORT MMDET
# ============================================================

with in_directory(MMDET_DIR):

    from mmdet.apis import (
        inference_detector,
        init_detector
    )


# ============================================================
# IMPORT HYBRIK
# ============================================================

with in_directory(HYBRIK_DIR):

    from hybrik.models import builder

    from hybrik.utils.config import update_config

    from hybrik.utils.presets import (
        SimpleTransform3DSMPLCam
    )


# ============================================================
# CONFIGURAZIONE MODELLI
# ============================================================

DET_CONFIG = os.path.join(
    MMDET_DIR,
    "checkpoints",
    "faster-rcnn_r50_fpn_1x_coco.py"
)

DET_CKPT = os.path.join(
    MMDET_DIR,
    "checkpoints",
    "faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth"
)

HYBRIK_CFG = os.path.join(
    HYBRIK_DIR,
    "configs",
    "256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml"
)

HYBRIK_CKPT = os.path.join(
    HYBRIK_DIR,
    "pretrained_models",
    "hybrik_hrnet.pth"
)

NUM_JOINTS = 24


# ============================================================
# ORDINE DEI JOINT SMPL
# ============================================================

JOINT_NAMES = [

    "Pelvis",

    "L_Hip",
    "R_Hip",

    "Spine1",

    "L_Knee",
    "R_Knee",

    "Spine2",

    "L_Ankle",
    "R_Ankle",

    "Spine3",

    "L_Foot",
    "R_Foot",

    "Neck",

    "L_Collar",
    "R_Collar",

    "Head",

    "L_Shoulder",
    "R_Shoulder",

    "L_Elbow",
    "R_Elbow",

    "L_Wrist",
    "R_Wrist",

    "L_Hand",
    "R_Hand"
]


# ============================================================
# CONVERSIONE SISTEMA DI COORDINATE
# ============================================================
#
# HybrIK / SMPL e Unity non utilizzano necessariamente
# lo stesso sistema di assi.
#
# Questa matrice viene usata per convertire la base
# della rotazione prima della conversione in Euler.
#
# ATTENZIONE:
# questa è una prima conversione di assi.
# Il retargeting completo SMPL -> X-Bot richiederà
# successivamente anche gli offset del rig.
# ============================================================
AXIS_CONVERSION = np.array([
    [1.0,  0.0,  0.0],
    [0.0,  1.0,  0.0],
    [0.0,  0.0, 1.0]
], dtype=np.float64)


# ============================================================
# CONVERSIONE MATRICE SMPL -> EULER
# ============================================================

def rotmats_to_euler_dict(theta_mats):
    """
    Converte le 24 matrici di rotazione HybrIK/SMPL
    in angoli Euler XYZ in gradi.

    theta_mats:
        shape = (24, 3, 3)

    ritorna:
        {
            "Pelvis": {"x": ..., "y": ..., "z": ...},
            ...
        }
    """

    rotations_dict = {}

    for i, name in enumerate(JOINT_NAMES):

        # Matrice di rotazione del joint
        R_smpl = np.asarray(
            theta_mats[i],
            dtype=np.float64
        )

        # Conversione del sistema di coordinate
        R_unity = (
            AXIS_CONVERSION
            @ R_smpl
            @ AXIS_CONVERSION.T
        )

        # Conversione matrice -> Euler
        euler = R.from_matrix(
            R_unity
        ).as_euler(
            "xyz",
            degrees=True
        )

        rotations_dict[name] = {
            "x": round(float(euler[0]), 2),
            "y": round(float(euler[1]), 2),
            "z": round(float(euler[2]), 2)
        }

    return rotations_dict


# ============================================================
# COSTRUZIONE MODELLO HYBRIK
# ============================================================

def build_hybrik(
    cfg_file,
    ckpt_file,
    gpu
):

    print("Configurazione HybrIK...")

    cfg = update_config(cfg_file)

    bbox_3d_shape = getattr(
        cfg.MODEL,
        "BBOX_3D_SHAPE",
        (2000, 2000, 2000)
    )

    bbox_3d_shape = [
        item * 1e-3
        for item in bbox_3d_shape
    ]

    dummy_set = edict({

        "joint_pairs_17": None,

        "joint_pairs_24": None,

        "joint_pairs_29": None,

        "bbox_3d_shape": bbox_3d_shape
    })

    transformation = SimpleTransform3DSMPLCam(

        dummy_set,

        scale_factor=cfg.DATASET.SCALE_FACTOR,

        color_factor=cfg.DATASET.COLOR_FACTOR,

        occlusion=cfg.DATASET.OCCLUSION,

        input_size=cfg.MODEL.IMAGE_SIZE,

        output_size=cfg.MODEL.HEATMAP_SIZE,

        depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,

        bbox_3d_shape=bbox_3d_shape,

        rot=cfg.DATASET.ROT_FACTOR,

        sigma=cfg.MODEL.EXTRA.SIGMA,

        train=False,

        add_dpg=False,

        loss_type=cfg.LOSS["TYPE"]
    )

    print(
        f"Loading HybrIK checkpoint from {ckpt_file}..."
    )

    save_dict = torch.load(
        ckpt_file,
        map_location="cpu"
    )

    if (
        isinstance(save_dict, dict)
        and "model" in save_dict
    ):
        state_dict = save_dict["model"]

    else:
        state_dict = save_dict

    hybrik_model = builder.build_sppe(
        cfg.MODEL
    )

    hybrik_model.load_state_dict(
        state_dict
    )

    hybrik_model.cuda(gpu)

    hybrik_model.eval()

    print("HybrIK caricato correttamente.")

    return (
        hybrik_model,
        transformation
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Webcam -> MMDet -> HybrIK -> Unity UDP"
    )

    parser.add_argument(
        "--webcam-id",
        default=1,
        type=int
    )

    parser.add_argument(
        "--det-config",
        default=DET_CONFIG
    )

    parser.add_argument(
        "--det-checkpoint",
        default=DET_CKPT
    )

    parser.add_argument(
        "--det-score-thr",
        default=0.5,
        type=float
    )

    parser.add_argument(
        "--hybrik-cfg",
        default=HYBRIK_CFG
    )

    parser.add_argument(
        "--hybrik-ckpt",
        default=HYBRIK_CKPT
    )

    parser.add_argument(
        "--gpu",
        default=0,
        type=int
    )

    parser.add_argument(
        "--unity-ip",
        default="127.0.0.1"
    )

    parser.add_argument(
        "--unity-port",
        default=5065,
        type=int
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Mostra la webcam con il bounding box"
    )

    opt = parser.parse_args()


    # ========================================================
    # GPU
    # ========================================================

    device = f"cuda:{opt.gpu}"

    if torch.cuda.is_available():

        torch.cuda.set_device(
            opt.gpu
        )

        print(
            "CUDA:",
            torch.cuda.get_device_name(opt.gpu)
        )

    else:

        raise RuntimeError(
            "CUDA non disponibile"
        )


    # ========================================================
    # MMDET
    # ========================================================

    print("Carico il detector...")

    with in_directory(MMDET_DIR):

        det_model = init_detector(

            opt.det_config,

            opt.det_checkpoint,

            device=device
        )


    print("Detector caricato.")


    # ========================================================
    # HYBRIK
    # ========================================================

    print("Carico HybrIK...")

    with in_directory(HYBRIK_DIR):

        hybrik_model, transformation = build_hybrik(

            opt.hybrik_cfg,

            opt.hybrik_ckpt,

            opt.gpu
        )


    # ========================================================
    # UDP
    # ========================================================

    import socket

    send_sock = socket.socket(
        socket.AF_INET,
        socket.SOCK_DGRAM
    )

    print(
        f"Streaming JSON avviato -> "
        f"{opt.unity_ip}:{opt.unity_port}"
    )


    # ========================================================
    # WEBCAM
    # ========================================================

    cap = cv2.VideoCapture(
        opt.webcam_id,
        cv2.CAP_V4L2
    )

    cap.set(
        cv2.CAP_PROP_FOURCC,
        cv2.VideoWriter_fourcc(*"MJPG")
    )

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    if not cap.isOpened():

        raise RuntimeError(
            "Impossibile aprire la webcam"
        )


    # ========================================================
    # DEBUG
    # ========================================================

    debug_printed = False

    frame_counter = 0

    start_time = time.time()


    # ========================================================
    # LOOP PRINCIPALE
    # ========================================================

    while cap.isOpened():

        ret, frame_bgr = cap.read()

        if not ret:

            print(
                "Errore nella lettura della webcam."
            )

            break


        frame_counter += 1


        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        input_image = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )


        # ====================================================
        # PERSON DETECTION
        # ====================================================

        with in_directory(MMDET_DIR):

            result = inference_detector(
                det_model,
                frame_bgr
            )


        pred = result.pred_instances


        # COCO:
        # label 0 = person

        mask = (
            (pred.labels == 0)
            &
            (pred.scores > opt.det_score_thr)
        )


        person_bboxes = (
            pred.bboxes[mask]
            .cpu()
            .numpy()
        )


        # ----------------------------------------------------
        # Nessuna persona
        # ----------------------------------------------------

        if len(person_bboxes) == 0:

            if opt.show:

                cv2.imshow(
                    "Live",
                    frame_bgr
                )

                if (
                    cv2.waitKey(1) & 0xFF
                    == ord("q")
                ):
                    break

            continue


        # ====================================================
        # PRENDIAMO LA PRIMA PERSONA
        # ====================================================

        tight_bbox = (
            person_bboxes[0][:4]
            .tolist()
        )


        # ====================================================
        # HYBRIK
        # ====================================================

        with in_directory(HYBRIK_DIR):

            with torch.no_grad():

                pose_input, bbox, img_center = (
                    transformation.test_transform(
                        input_image,
                        tight_bbox
                    )
                )


                pose_input = pose_input.to(
                    opt.gpu
                )[None, :, :, :]


                pose_output = hybrik_model(

                    pose_input,

                    flip_test=True,

                    bboxes=torch.from_numpy(
                        np.array(bbox)
                    )
                    .to(pose_input.device)
                    .unsqueeze(0)
                    .float(),

                    img_center=torch.from_numpy(
                        img_center
                    )
                    .to(pose_input.device)
                    .unsqueeze(0)
                    .float()
                )


                # =================================================
                # MATRICI DI ROTAZIONE SMPL
                # =================================================

                theta_mats = (
                    pose_output
                    .pred_theta_mats
                    .squeeze(dim=0)
                    .cpu()
                    .numpy()
                    .reshape(
                        NUM_JOINTS,
                        3,
                        3
                    )
                )


                # =================================================
                # TRASLAZIONE
                # =================================================

                transl = (
                    pose_output
                    .transl
                    .detach()
                    .squeeze(dim=0)
                    .cpu()
                    .numpy()
                )


                # =================================================
                # JOINT 3D
                # =================================================

                joints_3d = (

                    pose_output
                    .pred_xyz_jts_24
                    .detach()
                    .squeeze(dim=0)
                    .cpu()
                    .numpy()
                    .reshape(-1)
                    .tolist()
                )


        # ====================================================
        # ROTAZIONI
        # ====================================================

        unity_rotations = (
            rotmats_to_euler_dict(
                theta_mats
            )
        )


        # ====================================================
        # DEBUG PRIMA DEL PRIMO INVIO
        # ====================================================

        if not debug_printed:

            print("\n==============================")
            print("DEBUG HYBRIK")
            print("==============================")

            print(
                "Numero joint:",
                len(theta_mats)
            )

            print(
                "Numero rotazioni:",
                len(unity_rotations)
            )


            print("\nEsempi di rotazioni:")

            for name in [
                "Pelvis",
                "L_Shoulder",
                "R_Shoulder",
                "L_Elbow",
                "R_Elbow",
                "Head"
            ]:

                print(
                    name,
                    ":",
                    unity_rotations[name]
                )


            print(
                "\nRoot position:",
                transl
            )

            print("==============================\n")

            debug_printed = True


        # ====================================================
        # PACCHETTO JSON
        # ====================================================

        data_packet = {

            "person_id": 0,

            "timestamp": time.time(),

            "unity_rotations_deg":
                unity_rotations,

            "joint_xyz_3d":
                joints_3d,

            "root_position": {

                "x": float(transl[0]),

                "y": float(transl[1]),

                "z": float(transl[2])
            }
        }


        # ====================================================
        # JSON
        # ====================================================

        json_data = json.dumps(
            data_packet
        )


        # ====================================================
        # INVIO UDP
        # ====================================================

        try:

            send_sock.sendto(

                json_data.encode("utf-8"),

                (
                    opt.unity_ip,
                    opt.unity_port
                )
            )

        except Exception as e:

            print(
                "Errore invio UDP:",
                e
            )


        # ====================================================
        # VISUALIZZAZIONE
        # ====================================================

        if opt.show:

            x1, y1, x2, y2 = map(
                int,
                tight_bbox
            )


            cv2.rectangle(

                frame_bgr,

                (x1, y1),

                (x2, y2),

                (0, 255, 0),

                2
            )


            # FPS

            elapsed = (
                time.time()
                - start_time
            )

            if elapsed > 0:

                fps = (
                    frame_counter
                    / elapsed
                )

                cv2.putText(

                    frame_bgr,

                    f"FPS: {fps:.1f}",

                    (10, 30),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.8,

                    (0, 255, 0),

                    2
                )


            cv2.imshow(
                "Live",
                frame_bgr
            )


            if (
                cv2.waitKey(1) & 0xFF
                == ord("q")
            ):

                break


    # ========================================================
    # CHIUSURA
    # ========================================================

    cap.release()

    send_sock.close()

    if opt.show:

        cv2.destroyAllWindows()


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()