using UnityEngine;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

public class UDPReceiver : MonoBehaviour
{
    private Thread receiveThread;
    private UdpClient client;
    public int port = 5065;
    
    // Riferimento al ConnectionManager che elabora i dati
    public ConnectionManager connectionManager; 
    private bool running = true;

    void Start()
    {
        InitializeServer();
    }

    private void InitializeServer()
    {
        receiveThread = new Thread(new ThreadStart(ReceiveData));
        receiveThread.IsBackground = true;
        receiveThread.Start();
    }

    private void ReceiveData()
    {
        client = new UdpClient(port);
        IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);

        while (running)
        {
            try {
                byte[] data = client.Receive(ref remoteEndPoint);
                string text = Encoding.UTF8.GetString(data);
                
                // Invia i dati al manager sul main thread tramite un'azione
                // (fondamentale in Unity per evitare crash con i trasform)
                UnityMainThreadDispatcher.Instance().Enqueue(() => {
                    connectionManager.RouteData(text);
                });
            }
            catch (Exception e) { Debug.LogError("UDP Error: " + e.Message); }
        }
    }

    void OnDisable() {
        running = false;
        if (client != null) client.Close();
    }
}