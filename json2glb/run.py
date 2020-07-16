import json
import numpy as np
import math
import os,argparse
import math
import igl
from shutil import copyfile
import struct

class NPFakeFloat(float):
    def __init__(self, formatted):
        self.formatted = formatted

    def __repr__(self):
        return self.formatted

    def __float__(self):
        return self

def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])

def float_format(a):
    return [NPFakeFloat(np.format_float_positional(e, trim='0')) for e in a]

parser = argparse.ArgumentParser()
parser.add_argument(
        '--future_path',
        default = './3D-FUTURE-model',
        help = 'path to 3D FUTURE'
        )
parser.add_argument(
        '--json_path',
        default = './3D-FRONT',
        help = 'path to 3D FRONT'
        )

parser.add_argument(
        '--save_path',
        default = './outputs',
        help = 'path to save result dir'
        )

args = parser.parse_args()

files = os.listdir(args.json_path)

uv_mul = np.array([1, -1], dtype=np.float32)
uv_off = np.array([0, 1], dtype=np.float32)

for m in files:
    with open(args.json_path+'/'+m, 'r', encoding='utf-8') as f:
        gltf_buffer = bytearray()
        gltf_views = []
        gltf_accessors = []
        gltf_images = []
        gltf_textures = []
        gltf_materials = []
        gltf_meshes = []
        gltf_nodes = []

        def load_mesh(p, n, u, pidx, nidx, uidx, texture_path=None):
            min_pos = np.array([np.inf] * 3).astype(np.float32)
            max_pos = np.array([-np.inf] * 3).astype(np.float32)
            min_uv = np.array([np.inf] * 2).astype(np.float32)
            max_uv = np.array([-np.inf] * 2).astype(np.float32)
            min_normal = np.array([1] * 3).astype(np.float32)
            max_normal = np.max([-1] * 3).astype(np.float32)

            vertex_data = bytearray()
            index_data = bytearray()

            assert(pidx.shape[1] == uidx.shape[1] == nidx.shape[1] == 3)
            pidx = pidx.reshape(-1)
            uidx = uidx.reshape(-1)
            nidx = nidx.reshape(-1)

            visited = {}
            cur_vertex = 0
            total_indices = 0
            for index_tuple in zip(pidx, uidx, nidx):
                if index_tuple in visited:
                    index_data.extend(struct.pack("I", visited[index_tuple]))
                else:
                    position_idx, uv_idx, normal_idx = index_tuple
                    visited[index_tuple] = cur_vertex
                    index_data.extend(struct.pack("I", cur_vertex))

                    position = p[position_idx]
                    normal = n[normal_idx]
                    normal_l2norm = np.linalg.norm(normal, ord=2)
                    if normal_l2norm > 0:
                        normal /= normal_l2norm

                    uv = u[uv_idx]
                    uv = uv_mul * uv + uv_off

                    min_pos = np.minimum(min_pos, position)
                    max_pos = np.maximum(max_pos, position)

                    min_normal = np.minimum(min_normal, normal)
                    max_normal = np.maximum(max_normal, normal)

                    min_uv = np.minimum(min_uv, uv)
                    max_uv = np.maximum(max_uv, uv)

                    vertex_data.extend(struct.pack("ffffffff", *position, *normal, *uv))
                    cur_vertex += 1

                total_indices += 1

            total_vertices = cur_vertex
            assert(total_vertices > 0)
            assert(total_indices > 0)

            start_pad = 4 - (len(gltf_buffer) % 4)
            if start_pad < 4:
                gltf_buffer.extend(b' ' * start_pad)

            mesh_start = len(gltf_buffer)

            gltf_buffer.extend(vertex_data + index_data)

            vertex_size = 4*3 + 4*3 + 4*2

            # Position view
            gltf_views.append({
                "buffer": 0,
                "byteOffset": mesh_start,
                "byteLength": len(vertex_data) - 4*3 - 4*2,
                "byteStride": vertex_size
            })

            # Position Accessor
            gltf_accessors.append({
                "bufferView": len(gltf_views) - 1,
                "byteOffset": 0,
                "type": "VEC3",
                "componentType": 5126,
                "count": total_vertices,
                "min": float_format(min_pos),
                "max": float_format(max_pos)
            })

            # Normal View
            gltf_views.append({
                "buffer": 0,
                "byteOffset": mesh_start + 4*3,
                "byteLength": len(vertex_data) - 4*3 - 4*2,
                "byteStride": vertex_size
            })
            
            # Normal Accessor
            gltf_accessors.append({
                "bufferView": len(gltf_views) - 1,
                "byteOffset": 0,
                "type": "VEC3",
                "componentType": 5126,
                "count": total_vertices,
                "min": float_format(min_normal),
                "max": float_format(max_normal)
            })

            # UV View
            gltf_views.append({
                "buffer": 0,
                "byteOffset": mesh_start + 4*3 + 4*3,
                "byteLength": len(vertex_data) - 4*3 - 4*3,
                "byteStride": vertex_size
            })

            # UV Accessor
            gltf_accessors.append({
                "bufferView": len(gltf_views) - 1,
                "byteOffset": 0,
                "type": "VEC2",
                "componentType": 5126,
                "count": total_vertices,
                "min": float_format(min_uv),
                "max": float_format(max_uv)
            })

            # Index View
            gltf_views.append({
                "buffer": 0,
                "byteOffset": mesh_start + len(vertex_data),
                "byteLength": len(index_data)
            })

            # Index accessor
            gltf_accessors.append({
                "bufferView": len(gltf_views) - 1,
                "byteOffset": 0,
                "type": "SCALAR",
                "componentType": 5125,
                "count": total_indices,
                "min": [0],
                "max": [total_vertices - 1]
            })

            if texture_path != None:
                with open(texture_path, "rb") as texture_file:
                    texture_data = texture_file.read()

                texture_start = len(gltf_buffer)
                texture_size = len(texture_data)

                gltf_buffer.extend(texture_data)

                # Texture View
                gltf_views.append({
                    "buffer": 0,
                    "byteOffset": texture_start,
                    "byteLength": texture_size
                })

                gltf_images.append({
                    "bufferView": len(gltf_views) - 1,
                    "mimeType": "image/png"
                })

                gltf_textures.append({
                    "source": len(gltf_images) - 1,
                    "sampler": 0
                })

                gltf_materials.append({
                    "pbrMetallicRoughness": {
                        "baseColorTexture": {
                            "index": len(gltf_textures) - 1,
                            "texCoord": 0
                        },
                        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 1.0
                    }
                })
            else:
                gltf_materials.append({
                    "pbrMetallicRoughness": {
                        "baseColorFactor": [1.0, 1.0, 1.0, 1.0],
                        "metallicFactor": 0.0,
                        "roughnessFactor": 1.0
                    }
                })

            gltf_meshes.append({"primitives": [{
                "mode": 4,
                "indices": len(gltf_accessors) - 1,
                "attributes": {
                    "POSITION": len(gltf_accessors) - 4,
                    "NORMAL": len(gltf_accessors) - 3,
                    "TEXCOORD_0": len(gltf_accessors) - 2
                },
                "material": len(gltf_materials) - 1
            }]})

            return len(gltf_meshes) - 1

        class ExternalObject:
            def __init__(self, jid, bbox):
                self.jid = jid
                self.bbox = bbox
                self.mesh_idx = None

        data = json.load(f)

        external_objects = {}

        print(m[:-5])
        for ff in data['furniture']:
            if 'valid' in ff and ff['valid']:
                external_objects[ff['uid']] = ExternalObject(
                        ff['jid'], ff['bbox'])

        inline_meshes = {}

        class InlineMesh:
            def __init__(self, vertices, normals, uvs, faces):
                self.v = vertices
                self.n = normals
                self.uv = uvs
                self.f = faces
                self.mesh_idx = None 

        for mm in data['mesh']:
            inline_meshes[mm['uid']] = InlineMesh(
                    np.reshape(np.array(mm['xyz']).astype(np.float32), [-1, 3]),
                    np.reshape(np.array(mm['normal']).astype(np.float32), [-1, 3]),
                    np.reshape(np.array(mm['uv']).astype(np.float32), [-1, 2]),
                    np.reshape(np.array(mm['faces']).astype(np.uint), [-1, 3]))

        scene = data['scene']
        room = scene['room']
        for r in room:
            room_id = r['instanceid']
            meshes=[]
            children = r['children']
            for c in children:
                
                ref = c['ref']
                mesh_type = 'f'
                try:
                    external_object = external_objects[ref]
                    if external_object.mesh_idx == None:
                        if os.path.exists(args.future_path+'/' + external_object.jid):
                            try:
                                p, u, n, pidx, uidx, nidx = igl.read_obj(args.future_path+'/' + external_object.jid + '/raw_model.obj', dtype='float32')
                            except ValueError:
                                raise KeyError

                            u = u[:, 0:2]

                            texture_path = args.future_path + '/' + external_object.jid + '/texture.png'

                            new_mesh_idx = load_mesh(p, n, u, pidx, nidx, uidx, texture_path)

                            external_object.mesh_idx = new_mesh_idx
                        else:
                            raise KeyError

                    mesh_idx = external_object.mesh_idx

                except KeyError:
                    try:
                        inline_mesh = inline_meshes[ref]
                    except KeyError:
                        continue

                    if inline_mesh.mesh_idx == None:
                        new_mesh_idx = load_mesh(inline_mesh.v, inline_mesh.n, inline_mesh.uv, inline_mesh.f, inline_mesh.f, inline_mesh.f)

                        inline_mesh.mesh_idx = new_mesh_idx

                    mesh_idx = inline_mesh.mesh_idx

                ref = np.array([0,0,1], dtype=np.float32)
                rot = np.array(c['rot'], dtype=np.float32)

                axis = np.cross(ref, rot[1:])
                axis_l2norm = np.linalg.norm(axis, ord=2)
                theta = np.arccos(np.dot(ref, rot[1:]))*2

                if axis_l2norm > 0 and not math.isnan(theta):
                    axis /= axis_l2norm
                    halfsin = np.sin(theta / 2)
                    halfcos = np.cos(theta / 2)
                    quaternion = np.array([axis[0] * halfsin,
                                           axis[1] * halfsin, 
                                           axis[2] * halfsin,
                                           halfcos], dtype=np.float32)

                else:
                    quaternion = np.array([0.0, 0.0, 0.0, 1.0], np.float32)

                gltf_nodes.append({
                    "mesh": mesh_idx,
                    "translation": float_format(np.array(c['pos'], dtype=np.float32)),
                    "rotation": float_format(quaternion),
                    "scale": float_format(np.array(c['scale'], dtype=np.float32))
                })

        master = {
            "asset": {
                "version": "2.0"
            },
            "samplers": [{
                "magFilter": 9729,
                "minFilter": 9987,
                "wrapS": 10497,
                "wrapT": 10497
            }],
            "buffers": [{
                "byteLength": len(gltf_buffer)
            }],
            "bufferViews": gltf_views,
            "accessors": gltf_accessors,
            "images": gltf_images,
            "textures": gltf_textures,
            "materials": gltf_materials,
            "meshes": gltf_meshes,
            "nodes": gltf_nodes,
            "scene": 0,
            "scenes": [{
                "nodes": list(range(len(gltf_nodes)))
            }]
        }

        master_serialized = json.dumps(master, ensure_ascii=True).encode("ascii")
        json_pad = len(master_serialized) % 4
        json_pad = 4 - json_pad if json_pad > 0 else 0
        master_serialized += b' ' * json_pad
        buffer_pad = len(gltf_buffer) % 4
        buffer_pad = 4 - buffer_pad if buffer_pad > 0 else 0
        gltf_buffer += b' ' * buffer_pad

        total_len = 12 + 8 + len(master_serialized) + 8 + len(gltf_buffer)

        with open(f'{args.save_path}/{m[:-5]}.glb', 'wb') as glb_file:
            glb_file.write(struct.pack("III", 0x46546C67, 2, total_len))
            glb_file.write(struct.pack("II", len(master_serialized), 0x4e4f534a))
            glb_file.write(master_serialized)
            glb_file.write(struct.pack("II", len(gltf_buffer), 0x004e4942))
            glb_file.write(gltf_buffer)
