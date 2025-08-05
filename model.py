import math

class Model(object):
    def __init__(self):
        self.vertices = []        
        self.faces = []           
        self.texture_coords = []  
        self.texture_faces = []   
        self.texture = None

        
        self.transform_matrix = [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ]

        self.initial_vertices = []
        self.initial_center = [0,0,0]
        self.initial_scale = 1.0
        self.bounding_radius = 1.0 

    def add_vertex(self, x,y,z):
        self.vertices.extend([x,y,z])

    def add_texture_coord(self, u,v):
        self.texture_coords.extend([u,v])

    def add_face(self, v1,v2,v3, vt1=0,vt2=0,vt3=0):
        self.faces.extend([v1,v2,v3])
        self.texture_faces.extend([vt1,vt2,vt3])

    def load_texture(self, texture_path):
        try:
            import pygame
            self.texture = pygame.image.load(texture_path)
            print(f"Textura cargada: {texture_path} ({self.texture.get_width()}x{self.texture.get_height()})")
            return True
        except Exception as e:
            print(f"Error cargando textura: {e}")
            return False

    def get_texture_color(self, u, v):
        u = max(0.0, min(1.0, u))
        v = max(0.0, min(1.0, v))
        if self.texture is None:
            return [1,1,1]
        v_img = 1.0 - v
        w = self.texture.get_width()
        h = self.texture.get_height()
        x = int(u * (w - 1))
        y = int(v_img * (h - 1))
        try:
            c = self.texture.get_at((x,y))
            return [c.r/255.0, c.g/255.0, c.b/255.0]
        except:
            return [1,1,1]

    def save_initial_state(self):
        self.initial_vertices = self.vertices.copy()

    def reset_transform(self):
        self.transform_matrix = [
            [1,0,0,0],
            [0,1,0,0],
            [0,0,1,0],
            [0,0,0,1]
        ]
        if self.initial_vertices:
            self.vertices = self.initial_vertices.copy()

    def multiply_matrices(self, a,b):
        r = [[0]*4 for _ in range(4)]
        for i in range(4):
            for j in range(4):
                for k in range(4):
                    r[i][j] += a[i][k]*b[k][j]
        return r

    def apply_transform(self, x,y,z):
        p = [x,y,z,1]
        r = [0,0,0,0]
        for i in range(4):
            for j in range(4):
                r[i] += self.transform_matrix[i][j]*p[j]
        return r[0], r[1], r[2]

    def get_bounds(self):
        if not self.vertices:
            return 0,0,0,0,0,0
        xs = self.vertices[0::3]
        ys = self.vertices[1::3]
        zs = self.vertices[2::3]
        return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)

    def normalize_vertices(self):
        min_x, max_x, min_y, max_y, min_z, max_z = self.get_bounds()
        cx = (min_x + max_x)/2
        cy = (min_y + max_y)/2
        cz = (min_z + max_z)/2
        self.initial_center = [cx,cy,cz]
        for i in range(0, len(self.vertices), 3):
            self.vertices[i]   -= cx
            self.vertices[i+1] -= cy
            self.vertices[i+2] -= cz

    def compute_bounding_radius(self):
        max_r2 = 0
        for i in range(0, len(self.vertices), 3):
            x = self.vertices[i]
            y = self.vertices[i+1]
            z = self.vertices[i+2]
            r2 = x*x + y*y + z*z
            if r2 > max_r2:
                max_r2 = r2
        self.bounding_radius = math.sqrt(max_r2) if max_r2 > 0 else 1.0

    def auto_scale(self, target_size):
        
        min_x, max_x, min_y, max_y, min_z, max_z = self.get_bounds()
        w = max_x - min_x
        h = max_y - min_y
        d = max_z - min_z
        m = max(w,h,d)
        if m > 0:
            s = target_size / m
            self.initial_scale = s
            for i in range(0, len(self.vertices), 3):
                self.vertices[i]   *= s
                self.vertices[i+1] *= s
                self.vertices[i+2] *= s

    def center_and_scale(self):
        if not self.initial_vertices:
            self.save_initial_state()
        self.normalize_vertices()
        
        self.auto_scale(2.0)
        self.compute_bounding_radius()
        print(f"Modelo centrado y escalado. Radio: {self.bounding_radius:.3f}")

    
    def get_triangles(self):
        
        tris = []
        for i in range(0, len(self.faces), 3):
            tri = []
            for j in range(3):
                v_idx = self.faces[i+j] - 1
                x = self.vertices[v_idx*3]
                y = self.vertices[v_idx*3+1]
                z = self.vertices[v_idx*3+2]
                
                tx,ty,tz = self.apply_transform(x,y,z)

                vt_idx = self.texture_faces[i+j] - 1 if i+j < len(self.texture_faces) and self.texture_faces[i+j] > 0 else -1
                if vt_idx >= 0 and (vt_idx*2 + 1) < len(self.texture_coords):
                    u = self.texture_coords[vt_idx*2]
                    v = self.texture_coords[vt_idx*2+1]
                else:
                    u,v = 0.0,0.0
                tri.append((tx,ty,tz,u,v))
            if len(tri) == 3:
                tris.append(tri)
        return tris