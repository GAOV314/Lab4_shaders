import numpy as np

def phong_fragment_shader(fragment, uniforms):
    """
    Phong Lighting Model Fragment Shader
    Args:
        fragment: dict with keys 'position', 'normal', 'uv'
        uniforms: dict with keys 'light_pos', 'view_pos', 'light_color', 'object_color', 'ambient', 'diffuse', 'specular', 'shininess'
    Returns:
        tuple: (r, g, b) in range [0,1]
    """
    # Inputs
    pos = np.array(fragment['position'])
    normal = np.array(fragment['normal'])
    uv = fragment.get('uv', (0, 0))
    light_pos = np.array(uniforms.get('light_pos', (10,10,10)))
    view_pos = np.array(uniforms.get('view_pos', (0,0,10)))
    light_color = np.array(uniforms.get('light_color', (1,1,1)))
    object_color = np.array(uniforms.get('object_color', (1,0.8,0.4)))
    ambient_strength = uniforms.get('ambient', 0.1)
    diffuse_strength = uniforms.get('diffuse', 0.8)
    specular_strength = uniforms.get('specular', 0.5)
    shininess = uniforms.get('shininess', 32)

    # Normalized vectors
    norm = normal / (np.linalg.norm(normal) + 1e-8)
    light_dir = (light_pos - pos)
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-8)
    view_dir = (view_pos - pos)
    view_dir = view_dir / (np.linalg.norm(view_dir) + 1e-8)

    # Ambient
    ambient = ambient_strength * light_color

    # Diffuse
    diff = max(np.dot(norm, light_dir), 0.0)
    diffuse = diffuse_strength * diff * light_color

    # Specular
    reflect_dir = 2 * np.dot(norm, light_dir) * norm - light_dir
    spec = np.power(max(np.dot(view_dir, reflect_dir), 0.0), shininess)
    specular = specular_strength * spec * light_color

    # Result
    result = (ambient + diffuse + specular) * object_color
    result = np.clip(result, 0, 1)
    return tuple(result)

def pbr_fragment_shader(fragment, uniforms):
    """
    Simplified PBR (Physically Based Rendering) Fragment Shader
    Args:
        fragment: dict with keys 'position', 'normal', 'uv'
        uniforms: dict with keys 'light_pos', 'view_pos', 'light_color', 'albedo', 'metallic', 'roughness', 'ao'
    Returns:
        tuple: (r, g, b) in range [0,1]
    """
    # Get inputs
    pos = np.array(fragment['position'])
    normal = np.array(fragment['normal'])
    uv = fragment.get('uv', (0,0))
    light_pos = np.array(uniforms.get('light_pos', (10,10,10)))
    view_pos = np.array(uniforms.get('view_pos', (0,0,10)))
    light_color = np.array(uniforms.get('light_color', (1,1,1)))
    albedo = np.array(uniforms.get('albedo', (1,0.8,0.4)))
    metallic = uniforms.get('metallic', 0.0)
    roughness = uniforms.get('roughness', 0.5)
    ao = uniforms.get('ao', 1.0)

    # Normalized vectors
    N = normal / (np.linalg.norm(normal) + 1e-8)
    V = (view_pos - pos)
    V = V / (np.linalg.norm(V) + 1e-8)
    L = (light_pos - pos)
    L = L / (np.linalg.norm(L) + 1e-8)
    H = (V + L)
    H = H / (np.linalg.norm(H) + 1e-8)

    # Constants
    F0 = np.array([0.04, 0.04, 0.04])
    F0 = F0 * (1.0 - metallic) + albedo * metallic

    # Cook-Torrance BRDF
    # Distribution GGX
    def DistributionGGX(N, H, roughness):
        a = roughness ** 2
        a2 = a * a
        NdotH = max(np.dot(N, H), 0.0)
        NdotH2 = NdotH * NdotH
        denom = (NdotH2 * (a2 - 1.0) + 1.0)
        denom = np.pi * denom * denom
        return a2 / (denom + 1e-8)

    # Geometry Schlick-GGX
    def GeometrySchlickGGX(NdotV, roughness):
        r = roughness + 1.0
        k = (r * r) / 8.0
        return NdotV / (NdotV * (1.0 - k) + k + 1e-8)

    def GeometrySmith(N, V, L, roughness):
        return GeometrySchlickGGX(max(np.dot(N, V), 0.0), roughness) * \
               GeometrySchlickGGX(max(np.dot(N, L), 0.0), roughness)

    # Fresnel Schlick
    def fresnelSchlick(cosTheta, F0):
        return F0 + (1.0 - F0) * np.power(1.0 - cosTheta, 5.0)

    NDF = DistributionGGX(N, H, roughness)
    G = GeometrySmith(N, V, L, roughness)
    F = fresnelSchlick(max(np.dot(H, V), 0.0), F0)

    NdotL = max(np.dot(N, L), 0.0)
    NdotV = max(np.dot(N, V), 0.0)

    numerator = NDF * G * F
    denominator = 4.0 * NdotV * NdotL + 1e-8
    specular = numerator / denominator

    kS = F
    kD = 1.0 - kS
    kD *= 1.0 - metallic

    # Diffuse
    diffuse = kD * albedo / np.pi

    # Light radiance
    radiance = light_color

    # Final color
    color = (diffuse + specular) * radiance * NdotL
    # Add ambient
    ambient = 0.03 * albedo * ao
    color += ambient
    color = np.clip(color, 0, 1)
    return tuple(color)