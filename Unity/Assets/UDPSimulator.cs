using UnityEngine;
using System.Net.Sockets;
using System.Text;
using System.Collections.Generic;
using Newtonsoft.Json;

public class UDPSimulator : MonoBehaviour
{
    private UdpClient client;
    public int port = 5065;
    public string ipAddress = "127.0.0.1";

    void Start()
    {
        client = new UdpClient();
        Debug.Log("Simulatore UDP Full-Body avviato verso " + ipAddress + ":" + port);
    }

    void Update()
    {
        // Genera oscillazioni periodiche di test
        float braccio = Mathf.Sin(Time.time * 3f) * 45f;
        float testa = Mathf.Cos(Time.time * 2f) * 30f;
        float dito = Mathf.Sin(Time.time * 6f) * 20f; // Movimento falange

        var payload = new
        {
            unity_rotations_deg = new Dictionary<string, object>
            {
                { "L_Shoulder", new { x = 0f, y = 0f, z = braccio } },
                { "R_Shoulder", new { x = 0f, y = 0f, z = -braccio } },
                { "Head",       new { x = 0f, y = testa,  z = 0f } },
                { "L_Index1",   new { x = dito, y = 0f,   z = 0f } } // Test falange indice sinistro
            },
            root_position = new { x = 0f, y = 0f, z = 0f }
        };

        string json = JsonConvert.SerializeObject(payload);
        byte[] data = Encoding.UTF8.GetBytes(json);

        try
        {
            client.Send(data, data.Length, ipAddress, port);
        }
        catch (System.Exception e)
        {
            Debug.LogError("Errore invio UDP simulato: " + e.Message);
        }
    }

    void OnDisable()
    {
        if (client != null) client.Close();
    }
}
