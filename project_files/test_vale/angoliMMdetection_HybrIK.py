#comnfigurazione per mac di Vale
"""
python angoliMMdetection_HybrIK.py --show
"""


#codice modificato per configurazione Mac Intel
import os

# Configurazione grafica e di sistema per macOS
os.environ["QT_QPA_PLATFORM"] = "cocoa"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "0"
os.environ["MAGMA_VERBOSE"] = "0"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import warnings
warnings.filterwarnings(
    "ignore",
    message=".*MAGMA.*"
)

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

@contextmanager
def in_directory(path):
    old_dir = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old_dir)

# --- GESTIONE PERCORSI E PREPARAZIONE AUTOMATICA SMPL-X ---
TESINA_DIR = '/Users/valentina/POLI/magistrale/1 anno/image processing/Tesina'
MMDET_DIR = os.path.join(TESINA_DIR, 'mmdetection')
HYBRIK_DIR = os.path.join(TESINA_DIR, 'HybrIK')

# Aggiungi i path dei repository al sistema
sys.path.insert(0, MMDET_DIR)
sys.path.insert(0, HYBRIK_DIR)

# Percorso in cui HybrIK si aspetta di trovare il file
target_smplx_dir = os.path.join(HYBRIK_DIR, 'model_files', 'smplx')
target_smplx_file = os.path.join(target_smplx_dir, 'SMPLX_NEUTRAL.npz')

# Il percorso in cui si trova attualmente il tuo file SMPLX_NEUTRAL.npz
source_smplx_file = os.path.join(HYBRIK_DIR, 'model_files', 'smplx', 'SMPLX_NEUTRAL.npz')

# Copia automatica PRIMA di importare hybrik
if not os.path.exists(target_smplx_file):
    if os.path.exists(source_smplx_file):
        os.makedirs(target_smplx_dir, exist_ok=True)
        shutil.copy(source_smplx_file, target_smplx_file)
        print(f"File SMPL-X copiato con successo in: {target_smplx_file}")
    else:
        raise FileNotFoundError(
            f"Non trovo il file SMPLX_NEUTRAL.npz in {source_smplx_file}! "
            "Assicurati che si trovi lì o aggiorna il percorso 'source_smplx_file'."
        )

# Import dei moduli di MMDetection e HybrIK
with in_directory(MMDET_DIR):
    from mmdet.apis import inference_detector, init_detector

from scipy.spatial.transform import Rotation as R

with in_directory(HYBRIK_DIR):
    from hybrik.models import builder
    from hybrik.utils.config import update_config
    from hybrik.utils.presets import SimpleTransform3DSMPLCam

# --- CONFIGURAZIONE PERCORSI ASSOLUTI ---
DET_CONFIG = os.path.join(MMDET_DIR, 'configs', 'rtmdet', 'rtmdet_tiny_8xb32-300e_coco.py')
DET_CKPT = os.path.join(MMDET_DIR, 'checkpoints', 'rtmdet_tiny.pth')

HYBRIK_CFG = os.path.join(HYBRIK_DIR, 'configs', '256x192_adam_lr1e-3-hrw48_cam_2x_w_pw3d_3dhp.yaml')
HYBRIK_CKPT = os.path.join(HYBRIK_DIR, 'pretrained_models', 'hybrik_hrnet48_wo3dpw.pth')

NUM_JOINTS = 24

JOINT_NAMES = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]

def rotmats_to_quat_dict(theta_mats):
    r = R.from_matrix(theta_mats)
    quats = r.as_quat()

    rotations_dict = {}
    for i, name in enumerate(JOINT_NAMES):
        x, y, z, w = quats[i]
        rotations_dict[name] = {
            "x": float(x),
            "y": float(y),
            "z": float(z),
            "w": float(w)
        }
    return rotations_dict


def build_hybrik(cfg_file, ckpt_file, device):
    cfg = update_config(cfg_file)

    bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
    bbox_3d_shape = [item * 1e-3 for item in bbox_3d_shape]

    dummy_set = edict({
        'joint_pairs_17': None,
        'joint_pairs_24': None,
        'joint_pairs_29': None,
        'bbox_3d_shape': bbox_3d_shape
    })

    transformation = SimpleTransform3DSMPLCam(
        dummy_set, scale_factor=cfg.DATASET.SCALE_FACTOR,
        color_factor=cfg.DATASET.COLOR_FACTOR,
        occlusion=cfg.DATASET.OCCLUSION,
        input_size=cfg.MODEL.IMAGE_SIZE,
        output_size=cfg.MODEL.HEATMAP_SIZE,
        depth_dim=cfg.MODEL.EXTRA.DEPTH_DIM,
        bbox_3d_shape=bbox_3d_shape,
        rot=cfg.DATASET.ROT_FACTOR, sigma=cfg.MODEL.EXTRA.SIGMA,
        train=False, add_dpg=False,
        loss_type=cfg.LOSS['TYPE'])

    hybrik_model = builder.build_sppe(cfg.MODEL)

    print(f'Loading HybrIK checkpoint from {ckpt_file}...')
    save_dict = torch.load(ckpt_file, map_location='cpu')
    state_dict = save_dict['model'] if isinstance(save_dict, dict) and 'model' in save_dict else save_dict
    hybrik_model.load_state_dict(state_dict)

    hybrik_model.to(device)
    hybrik_model.eval()

    return hybrik_model, transformation


def smooth_bbox(previous_bbox, new_bbox, alpha=0.25):
    if previous_bbox is None:
        return list(new_bbox)

    previous_bbox = np.array(previous_bbox, dtype=np.float32)
    new_bbox = np.array(new_bbox, dtype=np.float32)

    smoothed = (
        alpha * new_bbox +
        (1.0 - alpha) * previous_bbox
    )

    return smoothed.tolist()


def main():
    parser = argparse.ArgumentParser(description='Webcam -> mmdet -> HybrIK -> UDP live (Mac compatible)')
    parser.add_argument('--webcam-id', default=0, type=int) 
    parser.add_argument('--det-config', default=DET_CONFIG)
    parser.add_argument('--det-checkpoint', default=DET_CKPT)
    parser.add_argument('--det-score-thr', default=0.5, type=float)
    parser.add_argument('--hybrik-cfg', default=HYBRIK_CFG)
    parser.add_argument('--hybrik-ckpt', default=HYBRIK_CKPT)
    parser.add_argument('--unity-ip', default='127.0.0.1')
    parser.add_argument('--unity-port', default=5065, type=int)
    parser.add_argument('--show', action='store_true', help='mostra la finestra con il bbox rilevato')
    opt = parser.parse_args()

    # --- FORZIAMO LA CPU PER EVITARE I BUG DI MPS SU HYBRIK ---
    device = torch.device('cpu')
    print("Utilizzo dispositivo: CPU (Forzato per stabilità HybrIK)")

    print('Carico il detector...')
    with in_directory(MMDET_DIR):
        det_model = init_detector(
            opt.det_config,
            opt.det_checkpoint,
            device=torch.device('cpu') # <-- Forziamo il detector su CPU per evitare incompatibilità MPS/Torchvision
        )
        
    print('Carico HybrIK...')
    with in_directory(HYBRIK_DIR):
        hybrik_model, transformation = build_hybrik(
            opt.hybrik_cfg,
            opt.hybrik_ckpt,
            device
        )
        
    import socket
    send_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # --- INIZIO MODIFICA PER MAC WEB-CAM STABILITY ---
    cap = cv2.VideoCapture(opt.webcam_id)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    assert cap.isOpened(), 'Impossibile aprire la webcam'

    # Piccola pausa per lasciare che AVFoundation inizializzi l'hardware
    time.sleep(1.0)
    
    # Svuota i primi frame (fondamentale per evitare il ret=False iniziale su Mac)
    for _ in range(5):
        cap.read()
    # --- FINE MODIFICA ---

    DETECTION_INTERVAL = 10
    frame_counter = 0

    last_bbox = None
    smoothed_bbox = None
    BBOX_SMOOTHING = 0.25

    print(f'Streaming JSON avviato -> invio a {opt.unity_ip}:{opt.unity_port}  (q per uscire)')

    while cap.isOpened():
        ret, frame_bgr = cap.read()
        if not ret:
            break

        input_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_counter += 1

        run_detection = (
            last_bbox is None or
            frame_counter % DETECTION_INTERVAL == 0
        )

        if run_detection:
            with in_directory(MMDET_DIR):
                result = inference_detector(
                    det_model,
                    frame_bgr
                )

            pred_instances = result.pred_instances
            bboxes_all = pred_instances.bboxes.cpu().numpy()
            scores_all = pred_instances.scores.cpu().numpy()
            labels_all = pred_instances.labels.cpu().numpy()

            keep = (labels_all == 0) & (scores_all > opt.det_score_thr)
            person_bboxes = bboxes_all[keep]

            if len(person_bboxes) == 0:
                last_bbox = None
                smoothed_bbox = None
                if opt.show:
                    cv2.imshow('Live', frame_bgr)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                continue

            detected_bbox = person_bboxes[0][:4].tolist()
            last_bbox = detected_bbox
            smoothed_bbox = smooth_bbox(
                smoothed_bbox,
                detected_bbox,
                BBOX_SMOOTHING
            )
        else:
            if smoothed_bbox is None:
                continue

        tight_bbox = smoothed_bbox

        with in_directory(HYBRIK_DIR):
            with torch.no_grad():
                pose_input, bbox, img_center = transformation.test_transform(input_image, tight_bbox)
                pose_input = pose_input.to(device)[None, :, :, :]

                #Convertiamo in float32 PRIMA di .to(device)
                pose_output = hybrik_model(
                    pose_input,
                    flip_test=False,
                    bboxes=torch.from_numpy(np.array(bbox, dtype=np.float32)).unsqueeze(0).to(device),
                    img_center=torch.from_numpy(np.array(img_center, dtype=np.float32)).unsqueeze(0).to(device)
                )

                theta_mats = pose_output.pred_theta_mats.squeeze(dim=0).cpu().numpy().reshape(NUM_JOINTS, 3, 3)
                transl = pose_output.transl.detach().squeeze(dim=0).cpu().numpy()

                joints_3d = (
                    pose_output.pred_xyz_jts_24.detach()
                        .squeeze(dim=0)
                        .cpu()
                        .numpy()
                        .reshape(-1)
                        .tolist()
                )

        unity_rotations = rotmats_to_quat_dict(theta_mats)
        
        data_packet = {
            "person_id": 0,
            "timestamp": time.time(),
            "unity_rotations_deg": unity_rotations,
            "joint_xyz_3d": joints_3d,
            "root_position": {
                "x": float(transl[0]),
                "y": float(transl[1]),
                "z": -float(transl[2])
            }
        }

        json_data = json.dumps(data_packet)
        send_sock.sendto(json_data.encode('utf-8'), (opt.unity_ip, opt.unity_port))

        if opt.show:
            x1, y1, x2, y2 = map(int, tight_bbox)
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.imshow('Live', frame_bgr)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    if opt.show:
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()