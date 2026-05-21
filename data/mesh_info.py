import trimesh

# mesh = trimesh.load("mesh_denmark_normalized.obj")
mesh = trimesh.load("mesh_superior_normalized.obj")
print(mesh.bounds)
print("size:", mesh.bounds[1] - mesh.bounds[0])
print("mesh bounds in meters", mesh.bounds * 10)