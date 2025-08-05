# Rasterizador 3D por Software (OBJ + Texturas + Transformaciones + Shaders)

Proyecto en Python que implementa un pipeline gráfico básico completamente en software (sin OpenGL / DirectX), capaz de:
- Cargar un modelo `.obj` con su textura.
- Aplicar transformaciones de Model, View, Projection y Viewport.
- Renderizar triángulos texturizados.
- Cambiar entre 4 tomas de cámara “fotográficas”.
- Aplicar 4 shaders en tiempo real (implementados manualmente):
  1. Hologram
  2. X-Ray
  3. Water
  4. Noise / Static

Todo usando únicamente Python y `pygame` para la ventana y carga de imagen.

---

## 1. Estructura de Archivos

| Archivo        | Descripción |
|----------------|-------------|
| `main.py`      | Punto de entrada. Configura ventana, carga modelo, maneja entrada del usuario y selecciona cámara / shader. |
| `gl.py`        | Núcleo del rasterizador: matrices, cámara, proyección, rasterización de triángulos, cálculo baricéntrico, shaders y guardado BMP. |
| `model.py`     | Representación del modelo: vértices, índices, UVs, textura, normalización, escalado y entrega de triángulos. |
| `obj_loader.py`| Parser simple de archivos `.obj` (soporta `v`, `vt`, y caras trianguladas o polígonos triangulados por “fan”). |
| `texture.png`  | Textura usada por el modelo (debes colocarla junto al `.obj`). |
| `model.obj`    | Modelo 3D a cargar (debes proveerlo). |
| `output.bmp`   | Imagen exportada al presionar ENTER (se genera). |

---

## 2. Requisitos

- Python 3.8+ (recomendado)
- Pygame

Instalación de pygame:

```bash
pip install pygame
```

---

## 3. Ejecución

Coloca en la misma carpeta:
```
main.py
gl.py
model.py
obj_loader.py
model.obj
Body.png
```

Luego ejecuta:

```bash
python main.py
```

Si `model.obj` o `Body.png` no existen, el programa no podrá mostrar el modelo texturizado (no crea un modelo por defecto en esta versión).

---

## 4. Pipeline de Render

Orden de transformaciones (por vértice):

1. **Model**: Vértices se centran en el origen y se escalan a un diámetro ~2 (para estandarizar el tamaño) en `model.center_and_scale()`. (La matriz de modelo se mantiene como identidad salvo que quisieras añadir animaciones locales).
2. **View**: Se genera con `look_at(eye, target, up)` para las diferentes tomas (cambia la posición y orientación de la cámara).
3. **Projection**: Matriz de proyección en perspectiva (FOV ~60°, near/far configurados).
4. **Division por W**: Conversión a coordenadas NDC (Normalized Device Coordinates).
5. **Viewport**: Mapeo NDC [-1..1] a coordenadas de pantalla `[0..width] x [0..height]`.
6. **Rasterización**: 
   - Triangulación por índices.
   - Cálculo baricéntrico por píxel.
   - Interpolación lineal de UV (no perspectiva-correct).
   - Muestreo de textura y aplicación de shader.
   - Escritura en framebuffer (sin z-buffer: el último triángulo pintado sobreescribe).

---

## 5. Cámaras / Tomas (“Photoshoot”)

Teclas numéricas (en tiempo real):

| Tecla | Toma         | Descripción |
|-------|--------------|-------------|
| 1     | Medium Shot  | Cámara frontal centrada. |
| 2     | Low Angle    | Cámara baja apuntando hacia arriba (sensación de grandeza). |
| 3     | High Angle   | Cámara elevada apuntando hacia abajo. |
| 4     | Dutch Angle  | Cámara girada (roll) para un efecto dramático. |

El modelo siempre se mantiene completamente visible (distancia calculada a partir del radio de bounding sphere).

---

## 6. Shaders

Seleccionables con F1–F4:

| Shader | Tecla | Concepto | Detalles |
|--------|-------|----------|----------|
| Hologram | F1 | Efecto tintado cian con líneas de escaneo y flicker | Modulación seno sobre Y y tiempo; realce de G/B. |
| X-Ray    | F2 | Apariencia “rayos X” con bordes brillantes | Usa min(a,b,c) de baricéntricas para detectar borde y atenúa interior por z. |
| Water    | F3 | Ondas y refracción | Distorsión UV senoidal animada; normal procedimental y luz direccional. |
| Noise    | F4 | Interferencia / estática | Ruido pseudo aleatorio por píxel con mezcla temporal. |

### 6.1. Notas Técnicas del Water Shader
- Distorsiona UV con combinación de dos sinusoides en diferente frecuencia y fase.
- Calcula una “altura” h(u,v,t) y deriva aproximaciones ∂h/∂u, ∂h/∂v para levantar una normal.
- Iluminación: `diffuse = base + N·L`, simple especular con half-vector.
- Tinte azul se mezcla con la textura distorsionada.
- No hay refracción física real ni Fresnel; solo estilización.

---

## 7. Controles Completos

| Tecla       | Acción |
|-------------|--------|
| 1 / 2 / 3 / 4 | Cambiar vista (Medium / Low / High / Dutch). |
| F1 / F2 / F3 / F4 | Cambiar shader (Hologram / X-Ray / Water / Noise). |
| M           | Cicla modo de primitivas (Puntos → Líneas → Triángulos). |
| ENTER       | Guardar frame actual como `output.bmp`. |
| R           | Reset (recalcula centro, escala y vuelve a Medium Shot). |
| ESC / Cerrar ventana | Salir. |

---

## 8. Limitaciones / Consideraciones

- **Sin Z-Buffer**: Triángulos se pintan en orden de carga. Si tu modelo está bien triangulado, suele bastar; de lo contrario puede haber sobreescrituras incorrectas.
- **Sin corrección de perspectiva para UV**: Las texturas pueden mostrar ligera deformación en polígonos muy inclinados.
- **Sin normales del modelo**: Shaders que simulan iluminación (Water) generan normales procedurales. Para iluminación real, habría que parsear `vn` y hacer transformación.
- **Sin clipping explícito**: Se confía en la división por W; triángulos totalmente detrás pueden causar artefactos en casos extremos (poco frecuente si el modelo cabe en el frustum).
- **Rendimiento**: Es CPU-bound; tamaños de ventana grandes o modelos con muchos triángulos bajarán FPS.

---

## 9. Extender el Proyecto

### 9.1. Agregar un nuevo Shader
1. Define una función `shader_miShader(self, model, base_color, u,v, x,y, z, ...)` en `gl.py`.
2. Ajusta `apply_shader` agregando un nuevo índice.
3. Agrega el nombre en `shader_names` en `main.py` y asigna una tecla (ej: F5).
4. (Opcional) Pasa datos adicionales (normales, etc.) si los implementas.

### 9.2. Añadir Z-Buffer (idea)
1. Crear una matriz `zbuffer = [[inf]*height for _ in range(width)]`.
2. Durante rasterización, antes de escribir el color, compara z actual con zbuffer[x][y].
3. Si z < zbuffer[x][y], actualiza zbuffer y pinta.

### 9.3. UV Perspective Correct (idea)
Guardar 1/w y (u/w, v/w); interpolar baricéntricamente y al final dividir:  
```
u = (u_over_w) / (one_over_w)
v = (v_over_w) / (one_over_w)
```

---

## 10. Solución de Problemas

| Problema | Posibles Causas | Solución |
|----------|-----------------|----------|
| Textura distorsionada / invertida | Eje V invertido | Ya se invierte al samplear (1 - v); si tu textura luce al revés, elimina esa inversión. |
| Colores extraños en Water | UV fuera de [0,1] tras distorsión | El sampler ya clampéa; reduce `amp_uv` si ves estiramientos. |
| Modelo no aparece | `model.obj` mal referenciado o sin triángulos válidos | Verifica ruta y formato de líneas `f`. |
| Muy lento | Modelo muy denso o ventana grande | Reduce resolución (ej. 400x400) o simplifica el modelo. |

---

