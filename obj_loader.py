from model import Model

class OBJLoader:
    def __init__(self):
        pass

    def load_obj(self, filename, texture_filename=None):
        
        try:
            model = Model()

            vertices_temp = []       
            texcoords_temp = []

            with open(filename, 'r', encoding='utf-8', errors='ignore') as file:
                for raw_line in file:
                    line = raw_line.strip()
                    if not line or line.startswith('#'):
                        continue

                    if line.startswith('v '):
                        parts = line.split()
                        # v x y z
                        x = float(parts[1]); y = float(parts[2])
                        z = float(parts[3]) if len(parts) > 3 else 0.0
                        model.add_vertex(x, y, z)

                    elif line.startswith('vt '):
                        parts = line.split()
                        # vt u v (w opcional ignorada)
                        u = float(parts[1])
                        v = float(parts[2]) if len(parts) > 2 else 0.0
                        model.add_texture_coord(u, v)

                    elif line.startswith('f '):
                        parts = line.split()[1:]

                        
                        v_idx_list = []
                        vt_idx_list = []

                        v_count = len(model.vertices) // 3
                        vt_count = len(model.texture_coords) // 2

                        for token in parts:
                            
                            comps = token.split('/')
                            v_str = comps[0]
                            vt_str = comps[1] if len(comps) > 1 and comps[1] != '' else None

                            
                            vi = int(v_str)
                            if vi < 0:
                                vi = v_count + 1 + vi  
                            v_idx_list.append(vi)

                            
                            if vt_str is not None:
                                vti = int(vt_str)
                                if vti < 0:
                                    vti = vt_count + 1 + vti
                                vt_idx_list.append(vti)
                            else:
                                vt_idx_list.append(0)  

                        
                        for i in range(1, len(v_idx_list) - 1):
                            v1, v2, v3 = v_idx_list[0], v_idx_list[i], v_idx_list[i + 1]
                            vt1, vt2, vt3 = vt_idx_list[0], vt_idx_list[i], vt_idx_list[i + 1]
                            model.add_face(v1, v2, v3, vt1, vt2, vt3)

            if texture_filename:
                model.load_texture(texture_filename)

            print(f"Modelo cargado: {len(model.vertices)//3} vértices, "
                  f"{len(model.faces)//3} triángulos, {len(model.texture_coords)//2} UV")

            return model

        except FileNotFoundError:
            print(f"Error: No se pudo encontrar el archivo {filename}")
            return None
        except Exception as e:
            print(f"Error cargando archivo OBJ: {e}")
            return None