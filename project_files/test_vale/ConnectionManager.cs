using UnityEngine;
using Newtonsoft.Json;

public class ConnectionManager : MonoBehaviour
{
    public MonoBehaviour[] providers; // Trascina qui i componenti HybrIK/SAM
    private IPoseProvider activeProvider;
    public Animator targetAnimator;

    public void RouteData(string jsonPayload)
    {
        // 1. Handshake per cambiare pipeline
        try {
            var hs = JsonConvert.DeserializeObject<Handshake>(jsonPayload);
            if (hs != null && !string.IsNullOrEmpty(hs.pipeline_type)) {
                SwitchPipeline(hs.pipeline_type);
                return;
            }
        } catch { }

        // 2. Dati Posa
        activeProvider?.ApplyPose(jsonPayload, targetAnimator);
    }

    void SwitchPipeline(string type)
    {
        foreach (var p in providers) {
            bool isActive = p.GetType().Name.ToLower().Contains(type.ToLower());
            p.enabled = isActive;
            if (isActive) activeProvider = (IPoseProvider)p;
        }
    }

    class Handshake { public string pipeline_type; }
}
