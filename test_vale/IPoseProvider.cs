using UnityEngine;

public interface IPoseProvider 
{
    void ApplyPose(string jsonPayload, Animator animator);
}
