"""
udp_test_receiver.py

Riceve e salva su file i pacchetti UDP inviati da fast_sam_3dbody_run (con --udp).
Usa il formato JSON Lines (un pacchetto JSON per riga).

Uso:
    python3 udp_test_receiver.py --port 5065 --output udp_log.jsonl
"""

import argparse
import json
import socket

parser = argparse.ArgumentParser()
parser.add_argument("--ip", default="127.0.0.1")
parser.add_argument("--port", type=int, default=5065)
parser.add_argument("--output", default="udp_packets.jsonl", help="File di output per i pacchetti")
args = parser.parse_args()

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((args.ip, args.port))

print(f"In ascolto su {args.ip}:{args.port} — Scrittura su '{args.output}' — Ctrl+C per interrompere.")

count = 0
try:
    with open(args.output, "a", encoding="utf-8") as f:
        while True:
            data, addr = sock.recvfrom(65536)
            try:
                packet = json.loads(data.decode("utf-8"))
            except json.JSONDecodeError as e:
                print(f"[{count}] JSON non valido ricevuto da {addr[0]}:{addr[1]}: {e}")
                continue

            count += 1
            # Scrive il pacchetto JSON su una singola riga nel file
            f.write(json.dumps(packet) + "\n")
            f.flush()  # Forza la scrittura immediata su disco
            
            print(f"Salurato pacchetto #{count} su file (da {addr[0]}:{addr[1]})")
            
except KeyboardInterrupt:
    print(f"\nInterrotto. Pacchetti totali salvati in '{args.output}': {count}")
finally:
    sock.close()