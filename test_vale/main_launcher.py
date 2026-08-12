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
    # Parametri di rete (modificali se necessario)
    udp_ip = "127.0.0.1"
    udp_port = 5065
    webcam_id = 1

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
            # Comando per avviare il binario C++ (adatta i path se necessario)
            cmd = [
                "./build/fast_sam_3dbody_run", 
                "--from", str(webcam_id), 
                "--udp-ip", udp_ip, 
                "--udp-port", str(udp_port)
            ]

        elif scelta == "2":
            pipeline_key = "mmdet-hybrik"
            print("\n[Launcher] Hai scelto: MMDetection + HybrIK")
            script_path = "/home/alessio/Desktop/progettoImage/ImageProcessing/angoliMMdetection_HybrIK.py"
            cmd = [
                sys.executable, script_path, 
                "--webcam-id", str(webcam_id), 
                "--unity-ip", udp_ip, 
                "--unity-port", str(udp_port)
            ]

        elif scelta == "3":
            pipeline_key = "alphapose-hybrik"
            print("\n[Launcher] Hai scelto: AlphaPose + HybrIK")
            script_path = "/home/alessio/Desktop/progettoImage/ImageProcessing/angoliAlphaPose_HybrIK.py"
            cmd = [
                sys.executable, script_path, 
                "--webcam", str(webcam_id), 
                "--udp-ip", udp_ip, 
                "--udp-port", str(udp_port)
            ]
        else:
            print("❌ Scelta non valida. Riprova.")
            continue

        # 1. Invia subito la scelta via rete a Unity prima di far partire il modello pesante
        send_handshake(udp_ip, udp_port, pipeline_key)

        # 2. Esegue lo script della pipeline scelta
        print(f"[Launcher] Esecuzione in corso... (premi Ctrl+C per fermare la pipeline e tornare al menu)\n")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n[Launcher] Pipeline interrotta dall'utente. Ritorno al menu principale...")
            time.sleep(1)

if __name__ == "__main__":
    main()