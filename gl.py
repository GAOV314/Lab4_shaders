import pygame
import math
import random

POINTS = 0
LINES = 1
TRIANGLES = 2

class Renderer(object):
    def __init__(self, screen):
        self.screen = screen
        _, _, self.width, self.height = self.screen.get_rect()
        self.glColor(1,1,1)
        self.glClearColor(0.1,0.1,0.15)
        self.glClear()
        self.primitiveType = TRIANGLES
        self.models = []

        # Matrices / cámara
        self.view_matrix = self.identity()
        self.proj_matrix = self.identity()
        self.fov_deg = 60
        self.near = 0.1
        self.far = 1000.0

        
        self.shader_mode = 0
        self.frame_count = 0

    
    def identity(self):
        return [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ]

    def multiply(self, A,B):
        r = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r[i][j] += A[i][k]*B[k][j]
        return r

    def vec4_mul_mat(self, M, v):
        return [
            M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2] + M[0][3]*v[3],
            M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2] + M[1][3]*v[3],
            M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2] + M[2][3]*v[3],
            M[3][0]*v[0] + M[3][1]*v[1] + M[3][2]*v[2] + M[3][3]*v[3],
        ]

    def normalize_vec3(self, v):
        l = math.sqrt(v[0]*v[0] + v[1]*v[1] + v[2]*v[2])
        if l == 0:
            return [0,0,0]
        return [v[0]/l, v[1]/l, v[2]/l]

    def cross(self, a,b):
        return [
            a[1]*b[2] - a[2]*b[1],
            a[2]*b[0] - a[0]*b[2],
            a[0]*b[1] - a[1]*b[0]
        ]

    def look_at(self, eye, target, up):
        zaxis = self.normalize_vec3([eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]])
        xaxis = self.normalize_vec3(self.cross(up, zaxis))
        yaxis = self.cross(zaxis, xaxis)
        return [
            [xaxis[0], xaxis[1], xaxis[2], - (xaxis[0]*eye[0] + xaxis[1]*eye[1] + xaxis[2]*eye[2])],
            [yaxis[0], yaxis[1], yaxis[2], - (yaxis[0]*eye[0] + yaxis[1]*eye[1] + yaxis[2]*eye[2])],
            [zaxis[0], zaxis[1], zaxis[2], - (zaxis[0]*eye[0] + zaxis[1]*eye[1] + zaxis[2]*eye[2])],
            [0,0,0,1]
        ]

    def perspective(self, fov_deg, aspect, near, far):
        f = 1 / math.tan(math.radians(fov_deg)/2)
        nf = 1 / (near - far)
        return [
            [f/aspect, 0, 0, 0],
            [0, f, 0, 0],
            [0, 0, (far+near)*nf, (2*far*near)*nf],
            [0, 0, -1, 0]
        ]

    
    def glClearColor(self, r,g,b):
        self.clearColor = [max(0,min(1,r)),max(0,min(1,g)),max(0,min(1,b))]

    def glColor(self, r,g,b):
        self.currColor = [max(0,min(1,r)),max(0,min(1,g)),max(0,min(1,b))]

    def glClear(self):
        c = [int(i*255) for i in self.clearColor]
        self.screen.fill(c)
        self.frameBuffer = [[self.clearColor for _ in range(self.height)] for _ in range(self.width)]

    def glPoint(self, x,y, color=None):
        x = int(round(x))
        y = int(round(y))
        if 0 <= x < self.width and 0 <= y < self.height:
            src = color or self.currColor
            col = [int(max(0,min(1,c))*255) for c in src]
            self.screen.set_at((x, self.height - 1 - y), col)
            self.frameBuffer[x][y] = [c/255 for c in col]

    def glLine(self, p0, p1, color=None):
        x0,y0 = p0; x1,y1 = p1
        dx = abs(x1-x0); dy = abs(y1-y0)
        steep = dy > dx
        if steep:
            x0,y0 = y0,x0
            x1,y1 = y1,x1
        if x0 > x1:
            x0,x1 = x1,x0
            y0,y1 = y1,y0
        dx = x1 - x0
        dy = abs(y1 - y0)
        error = 0
        y = y0
        ystep = 1 if y0 < y1 else -1
        for x in range(x0, x1+1):
            if steep:
                self.glPoint(y,x,color)
            else:
                self.glPoint(x,y,color)
            error += dy
            if (error * 2) >= dx:
                y += ystep
                error -= dx

    
    def project_vertex(self, v4):
        v_view = self.vec4_mul_mat(self.view_matrix, v4)
        clip = self.vec4_mul_mat(self.proj_matrix, v_view)
        w = clip[3]
        if abs(w) < 1e-8:
            return None
        ndc_x = clip[0] / w
        ndc_y = clip[1] / w
        ndc_z = clip[2] / w
        sx = (ndc_x + 1) * 0.5 * self.width
        sy = (ndc_y + 1) * 0.5 * self.height
        return (sx, sy, ndc_z)

    
    def shader_hologram(self, base_color, u,v, x,y, z, time):
        scan = 0.5 + 0.5 * math.sin((y * 0.15) + time * 6.0)
        flicker = 0.85 + 0.15 * math.sin(time * 20 + x * 0.05)
        r = base_color[0] * 0.4
        g = min(1.0, base_color[1] * 0.9 + 0.3)
        b = min(1.0, base_color[2] * 1.0 + 0.4)
        factor = scan * flicker
        return [min(1,r*factor), min(1,g*factor), min(1,b*factor)]

    def shader_xray(self, base_color, u,v, x,y, z, bary):
        edge = min(bary[0], bary[1], bary[2])
        edge_threshold = 0.06
        depth_intensity = 1.0 - max(0.0, min(1.0, (z + 1) / 2))
        if edge < edge_threshold:
            return [0.2 + 0.8*depth_intensity, 0.8*depth_intensity, 1.0]
        gray = (base_color[0] + base_color[1] + base_color[2]) / 3
        return [gray*0.4, gray*0.8, gray*depth_intensity]

    def shader_water(self, model, base_color, u,v, x,y, z, time):
        
        if model.texture is None:
            
            base_color = [0.2, 0.4, 0.7]

        
        wave_speed1 = 0.9
        wave_speed2 = 1.3
        freq1 = 6.0
        freq2 = 9.5
        amp_uv = 0.015  
        
        du = amp_uv * math.sin(freq1 * (u + time * wave_speed1)) \
           + amp_uv * 0.6 * math.sin(freq2 * (v + time * wave_speed2) + 1.2)
        dv = amp_uv * math.sin(freq1 * (v + time * wave_speed1) + 0.7) \
           + amp_uv * 0.6 * math.sin(freq2 * (u + time * wave_speed2) + 2.3)

        u2 = u + du
        v2 = v + dv

        tex_color = model.get_texture_color(u2, v2) if model.texture else base_color[:]

        
        amp_h = 0.05
        h = (
            amp_h * math.sin(freq1 * (u + time * wave_speed1)) +
            amp_h * 0.6 * math.sin(freq2 * (v + time * wave_speed2) + 1.2)
        )

        
        eps = 0.002
        h_du = (
            amp_h * math.cos(freq1 * (u + eps + time * wave_speed1)) * freq1 -
            amp_h * math.cos(freq1 * (u + time * wave_speed1)) * freq1
        ) / eps
        h_dv = (
            amp_h * 0.6 * math.cos(freq2 * (v + eps + time * wave_speed2) + 1.2) * freq2 -
            amp_h * 0.6 * math.cos(freq2 * (v + time * wave_speed2) + 1.2) * freq2
        ) / eps

        
        nx = -h_du
        ny = -h_dv
        nz = 1.0
        nl = math.sqrt(nx*nx + ny*ny + nz*nz)
        if nl > 0:
            nx /= nl; ny /= nl; nz /= nl

        
        L = self.normalize_vec3([0.3, 0.7, 0.6])
        NdotL = max(0.0, nx*L[0] + ny*L[1] + nz*L[2])

        
        diffuse = 0.4 + 0.6 * NdotL  

        
        V = [0,0,1]
        H = self.normalize_vec3([L[0]+V[0], L[1]+V[1], L[2]+V[2]])
        NdotH = max(0.0, nx*H[0] + ny*H[1] + nz*H[2])
        spec = pow(NdotH, 32) * 0.6  

        
        water_tint = [0.2, 0.45, 0.75]

        
        mixed = [
            tex_color[0] * water_tint[0],
            tex_color[1] * water_tint[1],
            tex_color[2] * water_tint[2]
        ]

        
        shaded = [
            mixed[0] * diffuse + spec,
            mixed[1] * diffuse + spec * 0.9,
            mixed[2] * (diffuse + spec * 0.8)
        ]

        
        for i in range(3):
            if shaded[i] > 1:
                shaded[i] = 1 - (shaded[i]-1)*0.5

        
        return [max(0,min(1,shaded[0])),
                max(0,min(1,shaded[1])),
                max(0,min(1,shaded[2]))]

    def shader_noise(self, base_color, u,v, x,y, z, frame):
        seed = int(x)*374761393 + int(y)*668265263 + frame*69069
        seed = (seed ^ (seed >> 13)) & 0xFFFFFFFF
        n = ((seed * 1274126177) & 0xFFFFFFFF) / 0xFFFFFFFF
        noise_strength = 0.45
        return [
            base_color[0]*(1-noise_strength) + n*noise_strength,
            base_color[1]*(1-noise_strength) + n*noise_strength,
            base_color[2]*(1-noise_strength) + n*noise_strength
        ]

    def apply_shader(self, mode, model, base_color, u,v, x,y, z, bary=None):
        if mode == 0:
            col = self.shader_hologram(base_color, u,v, x,y, z, self.frame_count/60.0)
        elif mode == 1:
            col = self.shader_xray(base_color, u,v, x,y, z, bary)
        elif mode == 2:
            col = self.shader_water(model, base_color, u,v, x,y, z, self.frame_count/60.0)
        elif mode == 3:
            col = self.shader_noise(base_color, u,v, x,y, z, self.frame_count)
        else:
            col = base_color
        return [max(0.0, min(1.0, col[0])),
                max(0.0, min(1.0, col[1])),
                max(0.0, min(1.0, col[2]))]

    
    def barycentric(self, x,y, A,B,C):
        denom = (B[1]-C[1])*(A[0]-C[0]) + (C[0]-B[0])*(A[1]-C[1])
        if abs(denom) < 1e-10:
            return None
        a = ((B[1]-C[1])*(x-C[0]) + (C[0]-B[0])*(y-C[1])) / denom
        b = ((C[1]-A[1])*(x-C[0]) + (A[0]-C[0])*(y-C[1])) / denom
        c = 1 - a - b
        return a,b,c

    def draw_textured_triangle(self, A,B,C, uvA,uvB,uvC, zA,zB,zC, model):
        min_x = max(0, int(min(A[0],B[0],C[0])))
        max_x = min(self.width-1, int(max(A[0],B[0],C[0])))
        min_y = max(0, int(min(A[1],B[1],C[1])))
        max_y = min(self.height-1, int(max(A[1],B[1],C[1])))

        for y in range(min_y, max_y+1):
            for x in range(min_x, max_x+1):
                bc = self.barycentric(x,y, A,B,C)
                if bc is None:
                    continue
                a,b,c = bc
                if a >= 0 and b >= 0 and c >= 0:
                    u = a*uvA[0] + b*uvB[0] + c*uvC[0]
                    v = a*uvA[1] + b*uvB[1] + c*uvC[1]
                    z = a*zA + b*zB + c*zC
                    base_color = model.get_texture_color(u,v) if model.texture else [
                        random.uniform(0.3,1.0),
                        random.uniform(0.3,1.0),
                        random.uniform(0.3,1.0)
                    ]
                    shaded = self.apply_shader(self.shader_mode, model, base_color, u,v, x,y, z, (a,b,c))
                    self.glPoint(x,y, shaded)

    def glRender(self):
        for model in self.models:
            tris = model.get_triangles()

            if self.primitiveType == POINTS:
                for tri in tris:
                    for (x,y,z,u,v) in tri:
                        pv = self.project_vertex((x,y,z,1))
                        if pv:
                            self.glPoint(pv[0], pv[1], [1,1,1])
                self.frame_count += 1
                continue

            if self.primitiveType == LINES:
                for tri in tris:
                    pts2d = []
                    for (x,y,z,u,v) in tri:
                        pv = self.project_vertex((x,y,z,1))
                        if pv:
                            pts2d.append(pv)
                    if len(pts2d) == 3:
                        self.glLine((pts2d[0][0], pts2d[0][1]), (pts2d[1][0], pts2d[1][1]), [1,1,1])
                        self.glLine((pts2d[1][0], pts2d[1][1]), (pts2d[2][0], pts2d[2][1]), [1,1,1])
                        self.glLine((pts2d[2][0], pts2d[2][1]), (pts2d[0][0], pts2d[0][1]), [1,1,1])
                self.frame_count += 1
                continue

            
            for tri in tris:
                proj_pts = []
                uvs = []
                for (x,y,z,u,v) in tri:
                    pv = self.project_vertex((x,y,z,1))
                    if not pv:
                        break
                    proj_pts.append(pv)
                    uvs.append((u,v))
                if len(proj_pts) == 3:
                    A = (proj_pts[0][0], proj_pts[0][1])
                    B = (proj_pts[1][0], proj_pts[1][1])
                    C = (proj_pts[2][0], proj_pts[2][1])
                    self.draw_textured_triangle(
                        A,B,C,
                        uvs[0],uvs[1],uvs[2],
                        proj_pts[0][2], proj_pts[1][2], proj_pts[2][2],
                        model
                    )

        self.frame_count += 1

    
    def set_camera(self, eye, target, up):
        aspect = self.width / self.height
        self.view_matrix = self.look_at(eye, target, up)
        self.proj_matrix = self.perspective(self.fov_deg, aspect, self.near, self.far)

    
    def save_bmp(self, filename):
        file_size = 54 + 3*self.width*self.height
        header = bytearray(54)
        header[0:2] = b'BM'
        header[2:6] = file_size.to_bytes(4,'little')
        header[10:14] = (54).to_bytes(4,'little')
        header[14:18] = (40).to_bytes(4,'little')
        header[18:22] = self.width.to_bytes(4,'little')
        header[22:26] = self.height.to_bytes(4,'little')
        header[26:28] = (1).to_bytes(2,'little')
        header[28:30] = (24).to_bytes(2,'little')

        pixel_data = bytearray()
        for y in range(self.height):
            for x in range(self.width):
                c = self.frameBuffer[x][y]
                b = int(c[2]*255); g = int(c[1]*255); r = int(c[0]*255)
                pixel_data.extend([b,g,r])
            pad = (4 - (self.width*3)%4)%4
            pixel_data.extend([0]*pad)

        with open(filename,'wb') as f:
            f.write(header)
            f.write(pixel_data)
        print(f"Archivo BMP guardado: {filename}")