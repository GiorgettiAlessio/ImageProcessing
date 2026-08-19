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

    [HideInInspector] public string latestJSON = "";
    private readonly object lockObject = new object();
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
        Debug.Log("Thread UDP avviato in ascolto su 127.0.0.1:" + port);
    }

    private void ReceiveData()
    {
        try
        {
            // Colleghiamo esplicitamente il client UDP all'indirizzo di loopback locale (127.0.0.1)
            IPEndPoint localEndPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), port);
            client = new UdpClient(localEndPoint);

            IPEndPoint remoteEndPoint = new IPEndPoint(IPAddress.Any, 0);

            while (running)
            {
                byte[] data = client.Receive(ref remoteEndPoint);
                string text = Encoding.UTF8.GetString(data);

                lock (lockObject)
                {
                    latestJSON = text;
                }
            }
        }
        catch (Exception e)
        {
            Debug.LogError("Errore critico nel thread UDP: " + e.Message);
        }
    }

    void OnDisable()
    {
        running = false;
        if (client != null) client.Close();
        if (receiveThread != null && receiveThread.IsAlive) receiveThread.Abort();
    }
}
