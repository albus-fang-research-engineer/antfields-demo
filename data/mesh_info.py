import trimesh

mesh = trimesh.load("mesh.obj")

print(mesh.bounds)
print("size:", mesh.bounds[1] - mesh.bounds[0])
print("mesh bounds in meters", mesh.bounds * 10)