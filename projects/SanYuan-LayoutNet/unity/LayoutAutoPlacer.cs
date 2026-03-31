using System;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

// 把这个脚本挂到一个空物体上，配置 json 路径和 prefab 映射后点击 Play。
// Unity 原生脚本语言是 C#；如果你一定要 C++，需要走原生插件（不建议做这个场景）。

[Serializable]
public class LayoutObjectItem
{
    public int id;
    public string @class;
    public float[] scene_position_xyz;
    public float[] scene_scale_xyz;
    public float depth_order_score;
    public int depth_rank_far_to_near;
}

[Serializable]
public class LayoutRoot
{
    public string image_id;
    public string image_path;
    public List<LayoutObjectItem> objects;
}

[Serializable]
public class ClassPrefabPair
{
    public string className;
    public GameObject prefab;
}

public class LayoutAutoPlacer : MonoBehaviour
{
    [Header("Input")]
    public string layoutJsonPath; // 例如: C:/xxx/layout.json

    [Header("Prefab Mapping")]
    public List<ClassPrefabPair> classPrefabMap = new List<ClassPrefabPair>();
    public GameObject defaultPrefab;

    [Header("Placement")]
    public bool clearChildrenBeforePlace = true;
    public float globalScale = 1.0f;
    public Vector3 globalOffset = Vector3.zero;

    private Dictionary<string, GameObject> _map;

    void Start()
    {
        try
        {
            BuildMap();
            var root = LoadLayout(layoutJsonPath);
            if (root == null || root.objects == null)
            {
                Debug.LogError("Layout json is empty or invalid.");
                return;
            }

            if (clearChildrenBeforePlace)
            {
                ClearChildren();
            }

            PlaceObjects(root.objects);
            Debug.Log($"LayoutAutoPlacer done. image_id={root.image_id}, count={root.objects.Count}");
        }
        catch (Exception e)
        {
            Debug.LogError("LayoutAutoPlacer failed: " + e);
        }
    }

    void BuildMap()
    {
        _map = new Dictionary<string, GameObject>(StringComparer.OrdinalIgnoreCase);
        foreach (var p in classPrefabMap)
        {
            if (string.IsNullOrEmpty(p.className) || p.prefab == null) continue;
            _map[p.className] = p.prefab;
        }
    }

    LayoutRoot LoadLayout(string path)
    {
        if (string.IsNullOrEmpty(path) || !File.Exists(path))
        {
            Debug.LogError("Layout json not found: " + path);
            return null;
        }

        string txt = File.ReadAllText(path);
        return JsonUtility.FromJson<LayoutRoot>(txt);
    }

    void ClearChildren()
    {
        for (int i = transform.childCount - 1; i >= 0; i--)
        {
            var c = transform.GetChild(i).gameObject;
#if UNITY_EDITOR
            DestroyImmediate(c);
#else
            Destroy(c);
#endif
        }
    }

    void PlaceObjects(List<LayoutObjectItem> objs)
    {
        // 按 far->near 放置，便于调试观察
        objs.Sort((a, b) => a.depth_rank_far_to_near.CompareTo(b.depth_rank_far_to_near));

        foreach (var o in objs)
        {
            var prefab = ResolvePrefab(o.@class);
            if (prefab == null)
            {
                Debug.LogWarning($"No prefab for class={o.@class}, id={o.id}");
                continue;
            }

            Vector3 pos = ReadVec3(o.scene_position_xyz) * globalScale + globalOffset;
            Vector3 scl = ReadVec3(o.scene_scale_xyz) * globalScale;

            GameObject go = Instantiate(prefab, pos, Quaternion.identity, transform);
            go.name = $"obj_{o.id}_{o.@class}_rank{o.depth_rank_far_to_near}";
            go.transform.localScale = scl;
        }
    }

    GameObject ResolvePrefab(string cls)
    {
        if (!string.IsNullOrEmpty(cls) && _map != null && _map.TryGetValue(cls, out var p))
            return p;
        return defaultPrefab;
    }

    Vector3 ReadVec3(float[] arr)
    {
        if (arr == null || arr.Length < 3) return Vector3.zero;
        return new Vector3(arr[0], arr[1], arr[2]);
    }
}
