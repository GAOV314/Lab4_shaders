import pygame
import os
import math
from gl import *
from model import Model
from obj_loader import OBJLoader

width = 512
height = 512
pygame.init()
screen = pygame.display.set_mode((width, height), pygame.SCALED)
clock = pygame.time.Clock()
rend = Renderer(screen)

# Cargar modelo/texture
obj_loader = OBJLoader()
model_path = os.path.join(os.path.dirname(__file__), "model.obj")
texture_path = os.path.join(os.path.dirname(__file__), "Body.png")

model = None
if os.path.exists(model_path):
    if os.path.exists(texture_path):
        model = obj_loader.load_obj(model_path, texture_path)
    else:
        model = obj_loader.load_obj(model_path)
else:
    print("No se encontró model.obj")

if not model:
    print("No se pudo cargar el modelo.")
    pygame.quit()
    raise SystemExit

model.center_and_scale()
rend.models.append(model)

fov = rend.fov_deg
r = model.bounding_radius
distance = (r / math.tan(math.radians(fov)/2)) * 1.3

def set_medium_shot():
    rend.set_camera((0,0,distance), (0,0,0), (0,1,0))
    print("Vista: Medium Shot")

def set_low_angle():
    rend.set_camera((0,-r*0.5,distance*0.9), (0,0,0), (0,1,0))
    print("Vista: Low Angle")

def set_high_angle():
    rend.set_camera((0,r*0.5,distance*0.9), (0,0,0), (0,1,0))
    print("Vista: High Angle")

def set_dutch_angle():
    roll_deg = 25
    roll_rad = math.radians(roll_deg)
    up = (math.sin(roll_rad), math.cos(roll_rad), 0)
    rend.set_camera((0,0,distance), (0,0,0), up)
    print("Vista: Dutch Angle")

set_medium_shot()

shader_names = ["Hologram", "X-Ray", "Water", "Noise"]
def print_shader():
    print(f"Shader activo: {shader_names[rend.shader_mode]}")

print("Controles:")
print("1 = Medium, 2 = Low angle, 3 = High angle, 4 = Dutch angle")
print("F1 = Hologram shader")
print("F2 = X-Ray shader")
print("F3 = Water shader")
print("F4 = Noise shader")
print("M = Ciclar modo de primitivas (Puntos/Líneas/Triángulos)")
print("ENTER = Guardar BMP")
print("R = Reset modelo (recalcular centro/escala)")
print_shader()

mode_cycle = [POINTS, LINES, TRIANGLES]
mode_index = 2
rend.primitiveType = TRIANGLES

running = True
while running:
    for e in pygame.event.get():
        if e.type == pygame.QUIT:
            running = False
        elif e.type == pygame.KEYDOWN:
            if e.key == pygame.K_1: set_medium_shot()
            elif e.key == pygame.K_2: set_low_angle()
            elif e.key == pygame.K_3: set_high_angle()
            elif e.key == pygame.K_4: set_dutch_angle()
            elif e.key == pygame.K_F1:
                rend.shader_mode = 0; print_shader()
            elif e.key == pygame.K_F2:
                rend.shader_mode = 1; print_shader()
            elif e.key == pygame.K_F3:
                rend.shader_mode = 2; print_shader()
            elif e.key == pygame.K_F4:
                rend.shader_mode = 3; print_shader()
            elif e.key == pygame.K_m:
                mode_index = (mode_index + 1) % len(mode_cycle)
                rend.primitiveType = mode_cycle[mode_index]
                print(f"Modo render: {['Puntos','Líneas','Triángulos'][mode_index]}")
            elif e.key == pygame.K_RETURN:
                rend.save_bmp("output.bmp")
            elif e.key == pygame.K_r:
                model.reset_transform()
                model.center_and_scale()
                r = model.bounding_radius
                distance = (r / math.tan(math.radians(fov)/2)) * 1.3
                set_medium_shot()
                print("Reset modelo")

    rend.glClear()
    rend.glRender()

    pygame.display.set_caption(
        f"Shots 1-4 | Shader: {shader_names[rend.shader_mode]} | Modo: {['Pts','Line','Tri'][mode_index]}"
    )
    pygame.display.flip()
    clock.tick(60)

pygame.quit()