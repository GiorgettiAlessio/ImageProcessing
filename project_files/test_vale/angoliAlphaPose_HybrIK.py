#comnfigurazione per mac di Vale
"""
cd "/Users/valentina/POLI/magistrale/1 anno/image processing/Tesina" \
source venv_tesina_310/bin/activate \
cd ImageProcessing/test_vale \
python angoliAlphaPose_HybrIK.py \
  --cfg "/Users/valentina/POLI/magistrale/1 anno/image processing/Tesina/AlphaPose/configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml" \
  --checkpoint "/Users/valentina/POLI/magistrale/1 anno/image processing/Tesina/HybrIK/pretrained_models/smpl/pretrained_w_cam.pth" \
  --webcam 0 \
  --detector yolo \
  --posebatch 16 \
  --udp-ip 127.0.0.1 \
  --udp-port 5065 \
  --print-fps \
  --vis
  """


# ---------------------------------------------------------------------------
# codice modificato per configurazione Mac Intel
#
# MODIFICHE PER LE PRESTAZIONI (rispetto alla versione precedente):
#   1. Cattura webcam e inferenza (detection + HybrIK) girano ora su due
#      thread separati. Il thread principale fa SOLO cv2.read()/imshow
#      (obbligatorio su macOS/AVFoundation - vedi vecchio commento sotto),
#      il thread di inferenza lavora sempre sull'ULTIMO frame disponibile,
#      saltando quelli che non fa in tempo a processare invece di
#      accodarli. Il framerate della webcam non è più il collo di
#      bottiglia: i pacchetti UDP escono al ritmo massimo che la CPU
#      riesce a sostenere.
#   2. La detection (YOLO) non gira più su OGNI frame ma ogni
#      --det-interval frame (default 8); negli altri si riusa l'ultimo
#      bounding box, filtrato con uno smoothing esponenziale, esattamente
#      come nello script MMDetection+HybrIK del progetto. Si traccia una
#      sola persona (quella con score più alto), dato che l'avatar in
#      Unity è comunque uno solo: elimina crop/transform inutili.
#   3. torch.set_num_threads() esplicito per sfruttare tutti i core del
#      processore Mac Intel invece di lasciare la scelta di default.
#   4. Risoluzione webcam forzata (--cam-width/--cam-height) per non far
#      lavorare il detector più del necessario.
#
# --vis resta disponibile: disegna l'ultimo overlay pronto senza bloccare
# la cattura, quindi la finestra resta fluida anche mentre l'inferenza è
# più lenta del framerate della camera (utile per registrare un video con
# Python e Unity affiancati).
# ---------------------------------------------------------------------------

import argparse
import json
import os
import socket
import sys
import threading
import time

import cv2
import numpy as np
import torch

os.environ["QT_QPA_PLATFORM"] = "cocoa"

# ---------------------------------------------------------------------------
# AlphaPose root
# ---------------------------------------------------------------------------
ALPHAPOSE_ROOT = "/Users/valentina/POLI/magistrale/1 anno/image processing/Tesina/AlphaPose"
os.chdir(ALPHAPOSE_ROOT)
if ALPHAPOSE_ROOT not in sys.path:
    sys.path.insert(0, ALPHAPOSE_ROOT)

from alphapose.models import builder
from alphapose.utils.config import update_config
from alphapose.utils.presets import SimpleTransform, SimpleTransform3DSMPL
from detector.apis import get_detector

from scipy.spatial.transform import Rotation as R


SMPL_JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]


# ---------------------------------------------------------------------------
# PoseProcessor: detection (a intervalli) + preprocessing HybrIK/AlphaPose
# su UN frame alla volta. Non fa I/O sulla webcam: viene chiamato dal
# thread di inferenza passandogli un frame BGR già catturato dal thread
# principale.
# ---------------------------------------------------------------------------
class PoseProcessor:
    """
    Sostituto di WebCamDetectionLoader pensato per essere chiamato da un
    thread separato da quello che possiede la webcam.

    Su macOS, cv2.VideoCapture().read() chiamato da un thread secondario
    puo' bloccarsi per sempre con AVFoundation, senza errori e senza uso
    di CPU: il frame non arriva mai perche' manca il run loop del thread
    principale. Per questo la CATTURA resta nel thread principale; questa
    classe invece non tocca affatto la webcam, quindi puo' tranquillamente
    girare in un thread di inferenza separato (i tensori PyTorch/numpy non
    hanno lo stesso vincolo).

    Traccia una sola persona (score piu' alto) con smoothing del bbox, per
    poter saltare la detection (costosa) alla maggior parte dei frame.
    """

    def __init__(self, detector, cfg, opt):
        self.detector = detector
        self.opt = opt

        self._input_size = cfg.DATA_PRESET.IMAGE_SIZE
        self._output_size = cfg.DATA_PRESET.HEATMAP_SIZE
        self._sigma = cfg.DATA_PRESET.SIGMA

        if cfg.DATA_PRESET.TYPE == "simple":
            self.transformation = SimpleTransform(
                self, scale_factor=0,
                input_size=self._input_size,
                output_size=self._output_size,
                rot=0, sigma=self._sigma,
                train=False, add_dpg=False)
        elif cfg.DATA_PRESET.TYPE == "simple_smpl":
            import inspect
            from easydict import EasyDict as edict
            dummy_set = edict({
                "joint_pairs_17": None,
                "joint_pairs_24": None,
                "joint_pairs_29": None,
                "bbox_3d_shape": (2.2, 2.2, 2.2),
            })

            candidate_kwargs = {
                "scale_factor": cfg.DATASET.SCALE_FACTOR,
                "color_factor": cfg.DATASET.COLOR_FACTOR,
                "occlusion": cfg.DATASET.OCCLUSION,
                "input_size": cfg.MODEL.IMAGE_SIZE,
                "output_size": cfg.MODEL.HEATMAP_SIZE,
                "depth_dim": cfg.MODEL.EXTRA.DEPTH_DIM,
                "bbox_3d_shape": (2.2, 2.2, 2.2),
                "rot": cfg.DATASET.ROT_FACTOR,
                "sigma": cfg.MODEL.EXTRA.SIGMA,
                "train": False,
                "add_dpg": False,
                "gpu_device": opt.device,
                "device": opt.device,
                "loss_type": cfg.LOSS["TYPE"],
            }

            sig_params = inspect.signature(SimpleTransform3DSMPL.__init__).parameters
            accepted_kwargs = {k: v for k, v in candidate_kwargs.items() if k in sig_params}

            dropped = set(candidate_kwargs) - set(accepted_kwargs)
            if dropped:
                print(f"DEBUG: SimpleTransform3DSMPL non accetta questi kwargs, li ignoro: {sorted(dropped)}", file=sys.stderr)

            self.transformation = SimpleTransform3DSMPL(dummy_set, **accepted_kwargs)
        else:
            raise RuntimeError(f"cfg.DATA_PRESET.TYPE non supportato: {cfg.DATA_PRESET.TYPE}")

        self.frame_idx = 0
        self.last_bbox = None       # ultimo bbox "grezzo" rilevato da YOLO
        self.smoothed_bbox = None   # bbox dopo lo smoothing, usato per il crop
        self.last_score = 0.0
        self.det_interval = max(1, opt.det_interval)
        self.bbox_alpha = opt.bbox_smoothing
        self.last_timings = {"detection_ms": 0.0, "transform_ms": 0.0, "ran_detection": False}

    @property
    def joint_pairs(self):
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]

    def _smooth_bbox(self, previous_bbox, new_bbox):
        """
        alpha = 1.0 -> nessun filtro (segue esattamente l'ultima detection)
        alpha = 0.1 -> molto stabile ma più lento a inseguire il movimento
        """
        if previous_bbox is None:
            return list(new_bbox)

        previous_bbox = np.array(previous_bbox, dtype=np.float32)
        new_bbox = np.array(new_bbox, dtype=np.float32)

        smoothed = self.bbox_alpha * new_bbox + (1.0 - self.bbox_alpha) * previous_bbox
        return smoothed.tolist()

    def process(self, frame):
        """
        frame: BGR np.ndarray già catturato dalla webcam (nessun I/O qui).
        Ritorna (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes)
        con al massimo UNA persona (boxes.shape[0] == 1).
        """
        im_name = f"{self.frame_idx}.jpg"

        run_detection = (
            self.last_bbox is None or
            self.frame_idx % self.det_interval == 0
        )
        self.frame_idx += 1

        orig_img = frame[:, :, ::-1]  # BGR -> RGB, coerente con la pipeline AlphaPose

        # Timing per-stage: azzerati ad ogni chiamata, letti dal thread di
        # inferenza subito dopo process() per capire dove va il tempo
        # (vedi --profile). ran_detection dice se in QUESTO frame la
        # detection è effettivamente girata (altrimenti detection_ms resta 0
        # perché è stata saltata, non perché è istantanea).
        self.last_timings = {"detection_ms": 0.0, "transform_ms": 0.0, "ran_detection": run_detection}

        if run_detection:
            t_det_start = time.perf_counter()

            img_k = self.detector.image_preprocess(frame)
            if isinstance(img_k, np.ndarray):
                img_k = torch.from_numpy(img_k)
            if img_k.dim() == 3:
                img_k = img_k.unsqueeze(0)

            with torch.no_grad():
                im_dim_list_k = torch.FloatTensor((frame.shape[1], frame.shape[0])).repeat(1, 2)
                dets = self.detector.images_detection(img_k, im_dim_list_k)

                if isinstance(dets, int) or dets.shape[0] == 0:
                    self.last_bbox = None
                    self.smoothed_bbox = None
                    self.last_timings["detection_ms"] = (time.perf_counter() - t_det_start) * 1000.0
                    return (None, orig_img, im_name, None, None, None, None)

                if isinstance(dets, np.ndarray):
                    dets = torch.from_numpy(dets)
                dets = dets.cpu()

                boxes = dets[:, 1:5]
                scores = dets[:, 5:6]

                boxes_k = boxes[dets[:, 0] == 0]
                scores_k = scores[dets[:, 0] == 0]

                if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                    self.last_bbox = None
                    self.smoothed_bbox = None
                    self.last_timings["detection_ms"] = (time.perf_counter() - t_det_start) * 1000.0
                    return (None, orig_img, im_name, None, None, None, None)

                # Teniamo solo la persona con score più alto: un solo
                # avatar in Unity, non serve tracciare più persone.
                best_idx = int(torch.argmax(scores_k.squeeze(-1)))
                detected_bbox = boxes_k[best_idx].tolist()
                self.last_score = float(scores_k[best_idx])

            self.last_bbox = detected_bbox
            self.smoothed_bbox = self._smooth_bbox(self.smoothed_bbox, detected_bbox)
            self.last_timings["detection_ms"] = (time.perf_counter() - t_det_start) * 1000.0

        else:
            if self.smoothed_bbox is None:
                return (None, orig_img, im_name, None, None, None, None)

        box_tensor = torch.tensor([self.smoothed_bbox], dtype=torch.float32)
        scores_tensor = torch.tensor([[self.last_score]], dtype=torch.float32)
        ids_tensor = torch.zeros_like(scores_tensor)

        inps = torch.zeros(1, 3, *self._input_size)
        cropped_boxes = torch.zeros(1, 4)

        t_transform_start = time.perf_counter()
        with torch.no_grad():
            inps[0], cropped_box = self.transformation.test_transform(orig_img, box_tensor[0])
            cropped_boxes[0] = torch.FloatTensor(cropped_box)
        self.last_timings["transform_ms"] = (time.perf_counter() - t_transform_start) * 1000.0

        return (inps, orig_img, im_name, box_tensor, scores_tensor, ids_tensor, cropped_boxes)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Realtime HybrIK Pose & Unity Euler Angles Output - Mac Compatible"
    )

    parser.add_argument("--cfg", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--webcam", type=int, default=0)
    parser.add_argument("--detector", type=str, default="yolo")

    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Ignorato su Mac (usa CPU se CUDA non c'è)"
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
        default=True,
        help="Mostra finestra con scheletro"
    )

    parser.add_argument("--print-fps", action="store_true", default=False)
    parser.add_argument(
        "--profile", action="store_true", default=False,
        help="Stampa il breakdown dei tempi per stage (detection/transform/hybrik/invio/vis) "
             "ogni 10 frame elaborati, per capire dove va il tempo"
    )

    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Usa AMP FP16 (disattivato automaticamente su CPU)"
    )

    parser.add_argument(
        "--tf32",
        action="store_true",
        default=True,
        help="Abilita TF32 su GPU compatibili"
    )

    parser.add_argument(
        "--no-stdout",
        action="store_true",
        default=False,
        help="Disattiva la stampa JSON/testuale su stdout"
    )

    parser.add_argument("--udp-ip", type=str, default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=5065)

    # --- NUOVI PARAMETRI PER LE PRESTAZIONI ---
    parser.add_argument(
        "--det-interval", type=int, default=8,
        help="Esegue la detection YOLO ogni N frame; negli altri riusa/filtra l'ultimo bbox (default: 8)"
    )
    parser.add_argument(
        "--bbox-smoothing", type=float, default=0.25,
        help="Alpha per lo smoothing esponenziale del bbox tra una detection e l'altra: "
             "1.0=nessun filtro, valori bassi=più stabile ma più lento a inseguire (default: 0.25)"
    )
    parser.add_argument("--cam-width", type=int, default=640)
    parser.add_argument("--cam-height", type=int, default=480)
    parser.add_argument(
        "--num-threads", type=int, default=0,
        help="Numero di thread CPU per PyTorch (0 = usa tutti i core disponibili)"
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
    Converte una batch di matrici di rotazione 3x3 (24, 3, 3) in un
    dizionario {nome_giunto: {x,y,z,w}} di QUATERNIONI veri e unitari.

    Stessa identica logica dello script MMDetection+HybrIK
    (rotmats_to_quat_dict), che AvatarController.cs si aspetta di
    ricevere per ogni giunto (vedi il commento "Python manda
    QUATERNION x,y,z,w" nella classe JointData). Prima questa
    funzione produceva angoli di Eulero (x,y,z, SENZA w): il campo
    w mancante veniva letto come 0 da Unity, e new Quaternion(x,y,z,0)
    con x,y,z in GRADI non è affatto la stessa rotazione — da qui gli
    arti storti e la spina dorsale "esplosa".
    """
    unity_rotations = {}

    if theta_mats is None:
        return unity_rotations

    if torch.is_tensor(theta_mats):
        theta_mats = theta_mats.detach().cpu().numpy()

    r = R.from_matrix(theta_mats)
    quats = r.as_quat()  # (N, 4) -> x, y, z, w

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
    if pred_rotations is None:
        return None

    if pred_rotations.dim() == 2:
        n_batch = pred_rotations.shape[0]
        if pred_rotations.shape[1] != 216:
            raise RuntimeError(f"pred_theta_mats inatteso: shape={tuple(pred_rotations.shape)}")
        pred_rotations = pred_rotations.reshape(n_batch, 24, 3, 3)

    elif pred_rotations.dim() == 3:
        if pred_rotations.shape == (24, 3, 3):
            pred_rotations = pred_rotations.unsqueeze(0)

    elif pred_rotations.dim() != 4:
        raise RuntimeError(f"Dimensione pred_theta_mats non supportata: {pred_rotations.dim()}")

    return pred_rotations


def root_to_unity(root_k):
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


# Coppie di giunti standard SMPL (parent -> child) per disegnare lo scheletro
SMPL_BONES = [
    ("Pelvis", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Spine3"),
    ("Spine3", "Neck"), ("Neck", "Head"),
    ("Spine3", "L_Collar"), ("L_Collar", "L_Shoulder"), ("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_Wrist"), ("L_Wrist", "L_Hand"),
    ("Spine3", "R_Collar"), ("R_Collar", "R_Shoulder"), ("R_Shoulder", "R_Elbow"), ("R_Elbow", "R_Wrist"), ("R_Wrist", "R_Hand"),
    ("Pelvis", "L_Hip"), ("L_Hip", "L_Knee"), ("L_Knee", "L_Ankle"), ("L_Ankle", "L_Foot"),
    ("Pelvis", "R_Hip"), ("R_Hip", "R_Knee"), ("R_Knee", "R_Ankle"), ("R_Ankle", "R_Foot")
]


def draw_skeleton_overlay(orig_img, pred_xyz, boxes):
    """Disegna bbox + scheletro proiettato 2D su una copia BGR del frame."""
    vis_img = orig_img.copy() if isinstance(orig_img, np.ndarray) else np.array(orig_img)
    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)  # AlphaPose restituisce RGB, imshow vuole BGR

    if pred_xyz is None or pred_xyz.shape[0] == 0:
        return vis_img

    for k in range(pred_xyz.shape[0]):
        jts = pred_xyz[k].detach().cpu().numpy().reshape(-1, 3)

        box_k = boxes[k]
        box_np = box_k.detach().cpu().numpy() if torch.is_tensor(box_k) else np.array(box_k)
        x1, y1, x2, y2 = box_np[:4]
        cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        box_height = y2 - y1

        cv2.rectangle(vis_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

        # Scala dinamica: non conosciamo a priori l'unita' di misura
        # esatta di pred_xyz_jts_29, quindi la deduciamo ad ogni frame
        # dall'escursione reale dei giunti rispetto al bacino, e la
        # mappiamo a una frazione sensata dell'altezza REALE del
        # bounding box in pixel. Usiamo il 90-esimo percentile (non il
        # max) per non farci rovinare la scala da un singolo giunto
        # rumoroso/outlier.
        core_jts = jts[:len(SMPL_JOINTS), :2]
        rel = core_jts - core_jts[0]  # relativo al bacino (Pelvis)
        dists = np.linalg.norm(rel, axis=1)
        ref_extent = np.percentile(dists, 90) + 1e-6
        scale = (box_height * 0.45) / ref_extent

        projected_jts = {}
        for idx, jt in enumerate(jts):
            if idx >= len(SMPL_JOINTS):
                break
            x_coord = int(cx + jt[0] * scale)
            y_coord = int(cy + jt[1] * scale)
            projected_jts[SMPL_JOINTS[idx]] = (x_coord, y_coord)
            cv2.circle(vis_img, (x_coord, y_coord), 3, (0, 0, 255), -1)

        for p_name, c_name in SMPL_BONES:
            if p_name in projected_jts and c_name in projected_jts:
                pt1 = projected_jts[p_name]
                pt2 = projected_jts[c_name]
                cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)

    return vis_img


def main():
    args = parse_args()

    # -----------------------------------------------------------------------
    # Thread CPU per PyTorch: su Mac Intel, se non lo si imposta
    # esplicitamente, PyTorch potrebbe non usare tutti i core disponibili.
    # -----------------------------------------------------------------------
    n_threads = args.num_threads if args.num_threads > 0 else (os.cpu_count() or 1)
    torch.set_num_threads(n_threads)
    print(f"PyTorch: {n_threads} thread CPU (di {os.cpu_count()} core disponibili)", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Gestione Device flessibile (Mac CPU / NVIDIA CUDA)
    # -----------------------------------------------------------------------
    if torch.cuda.is_available():
        gpu_ids = [int(i) for i in args.gpus.split(",")]
        torch.cuda.set_device(gpu_ids[0])
        device = torch.device(f"cuda:{gpu_ids[0]}")
        args.gpus = gpu_ids
        print("=" * 70, file=sys.stderr)
        print("GPU:", torch.cuda.get_device_name(gpu_ids[0]), file=sys.stderr)
        print("Device: CUDA", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
    else:
        device = torch.device("cpu")
        args.gpus = [-1]  # il detector di AlphaPose (yolo_api.py) si aspetta una lista di int: [-1] = CPU
        print("=" * 70, file=sys.stderr)
        print("⚠️ CUDA non disponibile. Esecuzione forzata su CPU (Mac Intel).", file=sys.stderr)
        print("=" * 70, file=sys.stderr)
        args.fp16 = False  # FP16 non va bene su CPU standard

    cfg = update_config(args.cfg)
    args.device = device
    args.tracking = args.pose_track or args.pose_flow or args.detector == "tracker"

    # -----------------------------------------------------------------------
    # Webcam: cattura pura, resta nel thread principale (macOS/AVFoundation)
    # -----------------------------------------------------------------------
    print("Inizializzazione webcam...", file=sys.stderr)
    stream = cv2.VideoCapture(args.webcam)
    stream.set(cv2.CAP_PROP_FRAME_WIDTH, args.cam_width)
    stream.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cam_height)
    assert stream.isOpened(), "Cannot capture source"

    # -----------------------------------------------------------------------
    # Detector
    # -----------------------------------------------------------------------
    print("Inizializzazione detector...", file=sys.stderr)
    processor = PoseProcessor(get_detector(args), cfg, args)

    # -----------------------------------------------------------------------
    # HybrIK
    # -----------------------------------------------------------------------
    print(f"Loading HybrIK pose model from {args.checkpoint}...", file=sys.stderr)
    pose_model = builder.build_sppe(
        cfg.MODEL,
        preset_cfg=cfg.DATA_PRESET,
    )

    if hasattr(pose_model, 'smpl') and hasattr(pose_model.smpl, 'shapedirs'):
        if pose_model.smpl.shapedirs.shape[-1] == 300:
            pose_model.smpl.shapedirs = pose_model.smpl.shapedirs[:, :, :10]

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
    )
    pose_model.load_state_dict(checkpoint)
    del checkpoint

    pose_model.to(device)
    pose_model.eval()

    # -----------------------------------------------------------------------
    # UDP
    # -----------------------------------------------------------------------
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp_dest = (args.udp_ip, args.udp_port)

    print(f"Invio dati via UDP verso {udp_dest[0]}:{udp_dest[1]}", file=sys.stderr)
    print(f"Detection ogni {args.det_interval} frame, smoothing bbox alpha={args.bbox_smoothing}", file=sys.stderr)
    print("Streaming avviato. Premi 'q' (con la finestra attiva) o Ctrl+C per uscire.", file=sys.stderr)

    # -----------------------------------------------------------------------
    # Stato condiviso tra thread di cattura (main) e thread di inferenza
    # -----------------------------------------------------------------------
    frame_lock = threading.Lock()
    latest_frame = {"frame": None, "id": 0}

    vis_lock = threading.Lock()
    latest_vis = {"frame": None}

    stop_event = threading.Event()
    stats = {"frame_count": 0, "t_start": time.perf_counter(), "n_people": 0}

    # Accumulatori per --profile: sommano i millisecondi di ogni stage sugli
    # ultimi N frame elaborati, per stampare una media ogni 10 frame e
    # azzerarsi. "n_det" conta separatamente quanti di quei frame hanno
    # davvero eseguito la detection (altrimenti la sua media risulterebbe
    # artificialmente bassa, diluita dai frame che l'hanno saltata).
    prof = {
        "detection_ms": 0.0, "n_det": 0,
        "transform_ms": 0.0,
        "hybrik_ms": 0.0,
        "send_ms": 0.0,
        "vis_ms": 0.0,
        "total_ms": 0.0,
        "n": 0,
    }

    def inference_loop():
        last_processed_id = -1

        while not stop_event.is_set():
            with frame_lock:
                frame = latest_frame["frame"]
                fid = latest_frame["id"]

            if frame is None or fid == last_processed_id:
                # Niente di nuovo da processare: piccola pausa per non
                # bruciare un core in busy-wait.
                time.sleep(0.001)
                continue
            last_processed_id = fid

            t_frame_start = time.perf_counter()

            with torch.inference_mode():
                inps, orig_img, im_name, boxes, scores, ids, cropped_boxes = processor.process(frame)

                if args.profile:
                    timings = getattr(processor, "last_timings", {})
                    if timings.get("ran_detection"):
                        prof["detection_ms"] += timings.get("detection_ms", 0.0)
                        prof["n_det"] += 1
                    prof["transform_ms"] += timings.get("transform_ms", 0.0)

                if boxes is None:
                    if args.vis:
                        vis_img = draw_skeleton_overlay(orig_img, None, None)
                        with vis_lock:
                            latest_vis["frame"] = vis_img
                    continue

                if not torch.is_tensor(inps):
                    inps = torch.tensor(inps, dtype=torch.float32)

                if inps.dim() != 4:
                    if inps.numel() == 3 * 256 * 256:
                        inps = inps.view(1, 3, 256, 256)
                    else:
                        continue

                inps = inps.to(device, non_blocking=True)

                t_hybrik_start = time.perf_counter()
                output = pose_model(inps)
                hybrik_ms = (time.perf_counter() - t_hybrik_start) * 1000.0

                (
                    pred_rotations,
                    pred_xyz,
                    pred_root,
                ) = extract_output(output)

                pred_rotations = prepare_rotations(pred_rotations)

                scores_cpu = (
                    scores.detach().cpu().tolist()
                    if torch.is_tensor(scores)
                    else list(scores)
                )

                n_people = len(scores_cpu)
                stats["n_people"] = n_people

                t_send_start = time.perf_counter()
                for k in range(n_people):
                    person_id = to_scalar(
                        ids[k] if ids is not None else None,
                        default=k,
                    )

                    unity_degrees = {}
                    if pred_rotations is not None and k < pred_rotations.shape[0]:
                        mats_k = pred_rotations[k].detach().float().cpu().numpy()
                        unity_degrees = rotmats_to_quat_dict(mats_k)

                    xyz_list = []
                    if pred_xyz is not None and k < pred_xyz.shape[0]:
                        xyz_list = pred_xyz[k].detach().float().cpu().tolist()

                    root_position_unity = None
                    if pred_root is not None and k < pred_root.shape[0]:
                        root_position_unity = root_to_unity(pred_root[k])

                    packet = {
                        "person_id": person_id,
                        "timestamp": time.time(),
                        "unity_rotations_deg": unity_degrees,
                        "joint_xyz_3d": xyz_list,
                        "root_position": root_position_unity,
                    }

                    data = json.dumps(packet, separators=(",", ":")).encode("utf-8")
                    sock.sendto(data, udp_dest)
                send_ms = (time.perf_counter() - t_send_start) * 1000.0

                vis_ms = 0.0
                if args.vis:
                    t_vis_start = time.perf_counter()
                    vis_img = draw_skeleton_overlay(orig_img, pred_xyz, boxes)
                    vis_ms = (time.perf_counter() - t_vis_start) * 1000.0
                    with vis_lock:
                        latest_vis["frame"] = vis_img

                if args.profile:
                    prof["hybrik_ms"] += hybrik_ms
                    prof["send_ms"] += send_ms
                    prof["vis_ms"] += vis_ms
                    prof["total_ms"] += (time.perf_counter() - t_frame_start) * 1000.0
                    prof["n"] += 1

                    if prof["n"] >= 10:
                        det_avg = prof["detection_ms"] / prof["n_det"] if prof["n_det"] > 0 else 0.0
                        print(
                            f"[PROFILE] su {prof['n']} frame elaborati "
                            f"({prof['n_det']} con detection) — medie in ms: "
                            f"detection={det_avg:.1f} (solo quando gira) | "
                            f"transform={prof['transform_ms'] / prof['n']:.1f} | "
                            f"hybrik={prof['hybrik_ms'] / prof['n']:.1f} | "
                            f"invio_udp={prof['send_ms'] / prof['n']:.2f} | "
                            f"vis={prof['vis_ms'] / prof['n']:.1f} | "
                            f"TOTALE={prof['total_ms'] / prof['n']:.1f}",
                            file=sys.stderr,
                        )
                        for key in prof:
                            prof[key] = 0.0 if key != "n" and key != "n_det" else 0

            stats["frame_count"] += 1
            if args.print_fps and stats["frame_count"] % 10 == 0:
                elapsed = time.perf_counter() - stats["t_start"]
                fps = stats["frame_count"] / elapsed if elapsed > 0 else 0.0
                print(f"--- FPS invio UDP: {fps:.2f} | Rilevati: {stats['n_people']} ---", file=sys.stderr)

    worker = threading.Thread(target=inference_loop, daemon=True)
    worker.start()

    try:
        while True:
            grabbed, frame = stream.read()
            if not grabbed:
                break

            with frame_lock:
                latest_frame["frame"] = frame
                latest_frame["id"] += 1

            if args.vis:
                with vis_lock:
                    vis_img = latest_vis["frame"]

                if vis_img is not None:
                    cv2.imshow("AlphaPose & HybrIK Realtime", vis_img)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\nInterrotto dall'utente.", file=sys.stderr)

    finally:
        stop_event.set()
        worker.join(timeout=2.0)
        stream.release()
        sock.close()
        if args.vis:
            cv2.destroyAllWindows()
        print("Chiuso pulito.", file=sys.stderr)


if __name__ == "__main__":
    main()