import subprocess
import sys
import socket
import json
import time

def send_handshake(ip, port, pipeline_name):
    """Invia il tipo di pipeline scelta a Unity via UDP."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    handshake_data = {
        "pipeline_type": pipeline_name
    }
    message = json.dumps(handshake_data).encode('utf-8')
    
    print(f"[Launcher] Invio configurazione '{pipeline_name}' a Unity ({ip}:{port})...")
    for _ in range(3): # Invio multiplo per sicurezza di ricezione
        sock.sendto(message, (ip, port))
        time.sleep(0.1)
    sock.close()

def main():
    # Parametri di rete
    udp_ip = "127.0.0.1"
    udp_port = 5065
    webcam_id = 1

    # Percorsi assoluti basati sulla tua struttura locale
    base_dir = "/Users/valentina/POLI/magistrale/1 anno/image processing/Tesina"
    test_vale_dir = f"{base_dir}/ImageProcessing/test_vale"
    hybrik_dir = f"{base_dir}/HybrIK"

    while True:
        print("\n==============================================")
        print("    SELETTORE PIPELINE MOTION CAPTURE")
        print("==============================================")
        print("1. SAM3DBody-cpp (Scenario A)")
        print("2. MMDetection + HybrIK (Scenario C)")
        print("3. AlphaPose + HybrIK (Scenario B)")
        print("0. Esci")
        print("----------------------------------------------")
        
        scelta = input("Seleziona il numero della pipeline da avviare: ").strip()

        if scelta == "0":
            print("Uscita in corso...")
            break
        
        pipeline_key = ""
        cmd = []

        if scelta == "1":
            pipeline_key = "sam3d"
            print("\n[Launcher] Hai scelto: SAM3DBody-cpp")
            cmd = [
                f"{base_dir}/SAM3DBody-cpp/build/fast_sam_3dbody_run", 
                "--from", str(webcam_id), 
                "--udp-ip", udp_ip, 
                "--udp-port", str(udp_port)
            ]

        elif scelta == "2":
            pipeline_key = "mmdet-hybrik"
            print("\n[Launcher] Hai scelto: MMDetection + HybrIK")
            script_path = f"{test_vale_dir}/angoliMMdetection_HybrIK.py"
            cmd = [
                sys.executable, script_path, 
                "--webcam-id", str(webcam_id), 
                "--unity-ip", udp_ip, 
                "--unity-port", str(udp_port)
            ]

        elif scelta == "3":
            pipeline_key = "alphapose-hybrik"
            print("\n[Launcher] Hai scelto: AlphaPose + HybrIK")
            script_path = f"{test_vale_dir}/angoliAlphaPose_HybrIK.py"
            
            # Passiamo i path corretti di config e checkpoint dentro HybrIK
            cfg_path = f"{hybrik_dir}/configs/smpl/256x192_adam_lr1e-3-res34_smpl_24_3d_base_2x_mix.yaml"
            ckpt_path = f"{hybrik_dir}/pretrained_models/smpl/pretrained_w_cam.pth"
            
            cmd = [
                sys.executable, script_path,
                "--cfg", cfg_path,
                "--checkpoint", ckpt_path,
                "--webcam", str(webcam_id),
                "--detector", "yolo",
                "--udp-ip", udp_ip,
                "--udp-port", str(udp_port)
            ]
        else:
            print("❌ Scelta non valida. Riprova.")
            continue

        # 1. Invia l'handshake a Unity prima di far partire il modello
        send_handshake(udp_ip, udp_port, pipeline_key)

        # 2. Esegue lo script della pipeline scelta
        print(f"[Launcher] Esecuzione in corso... (premi Ctrl+C per fermare e tornare al menu)\n")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n[Launcher] Pipeline interrotta. Ritorno al menu principale...")
            time.sleep(1)

if __name__ == "__main__":
    main()