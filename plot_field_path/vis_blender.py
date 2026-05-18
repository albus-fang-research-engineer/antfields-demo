import math, bpy

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

bpy.ops.wm.obj_import(filepath="gibson/mesh.obj")

# Plane parallel to XZ (vertical)
bpy.ops.mesh.primitive_plane_add(
    location=(0.0, 0.16, -0.0),
    rotation=(math.pi / 2, 0, 0)
)
plane = bpy.context.object
plane.scale = (0.20, 0.10, 1)
bpy.ops.object.transform_apply(scale=True)

# UVs are already correct on a primitive plane — no manual unwrap needed

mat = bpy.data.materials.new("SpeedMap")
mat.use_nodes = True
nodes = mat.node_tree.nodes
links = mat.node_tree.links
nodes.clear()

tex  = nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load("/plot_field_path/epoch_sweep_50_250_with_noise.png")

emit = nodes.new("ShaderNodeEmission")
out  = nodes.new("ShaderNodeOutputMaterial")

links.new(tex.outputs["Color"],     emit.inputs["Color"])
links.new(emit.outputs["Emission"], out.inputs["Surface"])

plane.data.materials.append(mat)