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



#codice modificato per configurazione Mac Intel

import argparse
import json
import os
import socket
import sys
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
 
 
class SyncWebcamLoader:
    """
    Sostituto sincrono di alphapose.utils.webcam_detector.WebCamDetectionLoader.
 
    Su macOS, cv2.VideoCapture().read() chiamato da un thread secondario
    (come fa WebCamDetectionLoader anche in modalita' --sp) puo' bloccarsi
    per sempre con AVFoundation, senza errori e senza uso di CPU: il frame
    non arriva mai perche' manca il run loop del thread principale.
    Questa classe fa tutto (cattura, detection, preprocessing pose) nel
    thread principale, in modo sincrono, un frame alla volta.
    """
 
    def __init__(self, input_source, detector, cfg, opt):
        self.cfg = cfg
        self.opt = opt
 
        self.stream = cv2.VideoCapture(int(input_source))
        assert self.stream.isOpened(), "Cannot capture source"
 
        self.detector = detector
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
 
    @property
    def joint_pairs(self):
        return [[1, 2], [3, 4], [5, 6], [7, 8],
                [9, 10], [11, 12], [13, 14], [15, 16]]
 
    def read(self):
        """
        Ritorna sempre una tupla a 7 elementi, stesso ordine e stessa
        semantica di WebCamDetectionLoader:
        (inps, orig_img, im_name, boxes, scores, ids, cropped_boxes)
        """
        grabbed, frame = self.stream.read()
        if not grabbed:
            return (None, None, None, None, None, None, None)
 
        im_name = f"{self.frame_idx}.jpg"
        self.frame_idx += 1
        orig_img = frame[:, :, ::-1]  # BGR -> RGB, coerente con la pipeline AlphaPose
 
        img_k = self.detector.image_preprocess(frame)
        if isinstance(img_k, np.ndarray):
            img_k = torch.from_numpy(img_k)
        if img_k.dim() == 3:
            img_k = img_k.unsqueeze(0)
 
        with torch.no_grad():
            im_dim_list_k = torch.FloatTensor((frame.shape[1], frame.shape[0])).repeat(1, 2)
            dets = self.detector.images_detection(img_k, im_dim_list_k)
 
            if isinstance(dets, int) or dets.shape[0] == 0:
                return (None, orig_img, im_name, None, None, None, None)
 
            if isinstance(dets, np.ndarray):
                dets = torch.from_numpy(dets)
            dets = dets.cpu()
 
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]
            ids = dets[:, 6:7] if self.opt.tracking else torch.zeros(scores.shape)
 
            boxes_k = boxes[dets[:, 0] == 0]
            if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                return (None, orig_img, im_name, None, None, None, None)
 
            scores_k = scores[dets[:, 0] == 0]
            ids_k = ids[dets[:, 0] == 0]
 
            inps = torch.zeros(boxes_k.size(0), 3, *self._input_size)
            cropped_boxes = torch.zeros(boxes_k.size(0), 4)
            for i, box in enumerate(boxes_k):
                inps[i], cropped_box = self.transformation.test_transform(orig_img, box)
                cropped_boxes[i] = torch.FloatTensor(cropped_box)
 
        return (inps, orig_img, im_name, boxes_k, scores_k, ids_k, cropped_boxes)
 
    def stop(self):
        self.stream.release()
 
 
SMPL_JOINTS = [
    "Pelvis", "L_Hip", "R_Hip", "Spine1", "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3", "L_Foot", "R_Foot", "Neck", "L_Collar",
    "R_Collar", "Head", "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand"
]
 
 
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
 
 
from scipy.spatial.transform import Rotation as R


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
 
 
def main():
    args = parse_args()
 
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
    # Detector / webcam
    # -----------------------------------------------------------------------
    print("Inizializzazione detector...", file=sys.stderr)
    det_loader = SyncWebcamLoader(
        args.webcam,
        get_detector(args),
        cfg,
        args,
    )
 
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
    print("Streaming avviato. Ctrl+C per uscire.", file=sys.stderr)
 
    frame_count = 0
    t_start = time.perf_counter()
 
    try:
        while True:
            with torch.inference_mode():
                read_data = det_loader.read()
                if len(read_data) == 7:
                    inps, orig_img, im_name, boxes, scores, ids, cropped_boxes = read_data
                elif len(read_data) == 6:
                    inps, orig_img, boxes, scores, ids, cropped_boxes = read_data
                    im_name = None
                else:
                    inps = read_data[0]
                    orig_img = read_data[1]
                    boxes = read_data[2] if len(read_data) > 2 else None
                    scores = read_data[3] if len(read_data) > 3 else None
                    ids = read_data[4] if len(read_data) > 4 else None
                    im_name, cropped_boxes = None, None
 
                if orig_img is None:
                    break
 
                if boxes is None or (hasattr(boxes, 'nelement') and boxes.nelement() == 0) or (isinstance(boxes, np.ndarray) and boxes.size == 0):
                    if args.vis:
                        print(f"DEBUG: imshow (no-box branch) frame {frame_count}", file=sys.stderr)
                        vis_img = orig_img.copy() if isinstance(orig_img, np.ndarray) else np.array(orig_img)
                        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)  # AlphaPose restituisce RGB, imshow vuole BGR
                        cv2.imshow("AlphaPose & HybrIK Realtime", vis_img)
                        key = cv2.waitKey(1) & 0xFF
                        print(f"DEBUG: waitKey returned {key}", file=sys.stderr)
                        if key == ord("q"):
                            break
                    continue
 
                if not torch.is_tensor(inps):
                    inps = torch.tensor(inps, dtype=torch.float32)
                
                if inps.dim() != 4:
                    if inps.numel() == 3 * 256 * 256:
                        inps = inps.view(1, 3, 256, 256)
                    else:
                        continue
                
                inps = inps.to(device, non_blocking=True)
 
                output = pose_model(inps)
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
 
                if args.vis:
                    vis_img = orig_img.copy() if isinstance(orig_img, np.ndarray) else np.array(orig_img)
                    vis_img = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
 
                    if pred_xyz is not None and pred_xyz.shape[0] > 0:
                        # Coppie di giunti standard SMPL (parent -> child) per disegnare lo scheletro
                        smpl_bones = [
                            ("Pelvis", "Spine1"), ("Spine1", "Spine2"), ("Spine2", "Spine3"),
                            ("Spine3", "Neck"), ("Neck", "Head"),
                            ("Spine3", "L_Collar"), ("L_Collar", "L_Shoulder"), ("L_Shoulder", "L_Elbow"), ("L_Elbow", "L_Wrist"), ("L_Wrist", "L_Hand"),
                            ("Spine3", "R_Collar"), ("R_Collar", "R_Shoulder"), ("R_Shoulder", "R_Elbow"), ("R_Elbow", "R_Wrist"), ("R_Wrist", "R_Hand"),
                            ("Pelvis", "L_Hip"), ("L_Hip", "L_Knee"), ("L_Knee", "L_Ankle"), ("L_Ankle", "L_Foot"),
                            ("Pelvis", "R_Hip"), ("R_Hip", "R_Knee"), ("R_Knee", "R_Ankle"), ("R_Ankle", "R_Foot")
                        ]
                        joint_name_to_idx = {name: idx for idx, name in enumerate(SMPL_JOINTS)}
 
                        for k in range(pred_xyz.shape[0]):
                            jts = pred_xyz[k].detach().cpu().numpy().reshape(-1, 3)
 
                            box_k = boxes[k]
                            box_np = box_k.detach().cpu().numpy() if torch.is_tensor(box_k) else np.array(box_k)
                            x1, y1, x2, y2 = box_np[:4]
                            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                            box_height = y2 - y1
 
                            # Scala dinamica: non conosciamo a priori l'unita' di
                            # misura esatta di pred_xyz_jts_29 (non e' detto sia
                            # in metri "puri"), quindi la deduciamo ad ogni frame
                            # dall'escursione reale dei giunti rispetto al bacino,
                            # e la mappiamo a una frazione sensata dell'altezza
                            # REALE del bounding box in pixel. Usiamo il 90-esimo
                            # percentile (non il max) per non farci rovinare la
                            # scala da un singolo giunto rumoroso/outlier.
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
                                # Disegniamo il pallino del giunto
                                cv2.circle(vis_img, (x_coord, y_coord), 3, (0, 0, 255), -1)
 
                            # Disegniamo le linee dello scheletro (ossa)
                            for p_name, c_name in smpl_bones:
                                if p_name in projected_jts and c_name in projected_jts:
                                    pt1 = projected_jts[p_name]
                                    pt2 = projected_jts[c_name]
                                    cv2.line(vis_img, pt1, pt2, (0, 255, 0), 2)
 
                    cv2.imshow("AlphaPose & HybrIK Realtime", vis_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        break
 
            frame_count += 1
            if args.print_fps and frame_count % 10 == 0:
                elapsed = time.perf_counter() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0.0
                print(f"--- FPS Medio: {fps:.2f} | Rilevati: {n_people} ---", file=sys.stderr)
 
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