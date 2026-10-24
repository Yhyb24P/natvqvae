"""Build the Blender structure demo from existing CIFs and summary metadata.

Run with Blender's bundled Python, e.g.:
    blender -b --python src/natvqvae/blender_demo.py -- \
        --config <demo>/render_input.json --output <demo>
This module intentionally does not import the natvqvae package.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import bpy
import numpy as np
from mathutils import Matrix, Vector

OUTPUTS = {
    'wt_overview': 'wt_overview.png',
    'a_class_panels': 'a_class_panels.png',
    'nudix_site': 'nudix_site.png',
    'nmnat_inset': 'nmnat_inset.png',
    'animation': 'demo.mp4',
    'blend': 'demo.blend',
    'blend_session': 'demo.blend.MNSession',
    'identity': 'render_identity.json',
}

TEXT_DARK = (0.02, 0.022, 0.026, 1.0)
BACKGROUND = (3.00, 2.92, 2.78, 1.0)
BOLD_FONT = next((path for path in (
    Path('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'),
    Path('/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf'),
) if path.is_file()), None)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path):
    path = Path(path)
    if path.is_file():
        return sha256(path)
    digest = hashlib.sha256()
    for item in sorted(p for p in path.rglob('*') if p.is_file()):
        digest.update(str(item.relative_to(path)).encode())
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def parse_args():
    argv = sys.argv[sys.argv.index('--') + 1:] if '--' in sys.argv else sys.argv[1:]
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--stills-only', action='store_true')
    return parser.parse_args(argv)


def reset_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.preferences.addon_enable(module='bl_ext.user_default.molecularnodes')


def setup_render(render, device_preference='GPU'):
    scene = bpy.context.scene
    scene.render.engine = render.get('engine', 'CYCLES')
    scene.render.resolution_x = int(render['width'])
    scene.render.resolution_y = int(render['height'])
    scene.render.resolution_percentage = 100
    scene.render.film_transparent = False
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.cycles.samples = int(render.get('samples', 24))
    scene.cycles.use_denoising = True
    scene.cycles.device = 'CPU'
    device_info = {'type': 'CPU', 'name': 'CPU'}
    if device_preference == 'GPU':
        prefs = bpy.context.preferences.addons['cycles'].preferences
        prefs.compute_device_type = 'CUDA'
        prefs.refresh_devices()
        for device in prefs.devices:
            device.use = device.type == 'CUDA'
        cuda = [device for device in prefs.devices if device.type == 'CUDA']
        if cuda:
            scene.cycles.device = 'GPU'
            device_info = {'type': 'CUDA', 'name': cuda[0].name}
    return device_info


def set_world():
    world = bpy.data.worlds.new('World')
    world.use_nodes = True
    nodes = world.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputWorld')
    camera_background = nodes.new('ShaderNodeBackground')
    camera_background.inputs['Color'].default_value = BACKGROUND
    camera_background.inputs['Strength'].default_value = 1.0
    lighting_background = nodes.new('ShaderNodeBackground')
    lighting_background.inputs['Color'].default_value = (0.55, 0.55, 0.55, 1.0)
    lighting_background.inputs['Strength'].default_value = 0.25
    light_path = nodes.new('ShaderNodeLightPath')
    mix = nodes.new('ShaderNodeMixShader')
    world.node_tree.links.new(light_path.outputs['Is Camera Ray'], mix.inputs['Fac'])
    world.node_tree.links.new(lighting_background.outputs['Background'], mix.inputs[1])
    world.node_tree.links.new(camera_background.outputs['Background'], mix.inputs[2])
    world.node_tree.links.new(mix.outputs['Shader'], output.inputs['Surface'])
    bpy.context.scene.world = world


def look_at(obj, target, track_axis='TRACK_NEGATIVE_Z'):
    constraint = obj.constraints.new('TRACK_TO')
    constraint.target = target
    constraint.track_axis = track_axis
    constraint.up_axis = 'UP_Y'


def make_empty(name, location):
    empty = bpy.data.objects.new(name, None)
    bpy.context.scene.collection.objects.link(empty)
    empty.location = location
    return empty


def make_camera(center, size, lens=55, ortho=False, factor=1.85, height=0.55,
                ortho_factor=1.8):
    data = bpy.data.cameras.new('Camera')
    data.lens = lens
    data.clip_start = 0.01
    data.clip_end = 200.0
    camera = bpy.data.objects.new('Camera', data)
    bpy.context.scene.collection.objects.link(camera)
    if ortho:
        data.type = 'ORTHO'
        data.ortho_scale = size * ortho_factor * 16.0 / 9.0
        camera.location = Vector(center) + Vector((0.0, -size * 2.0, 0.0))
    else:
        camera.location = Vector(center) + Vector((0.0, -size * factor, size * height))
    target = make_empty('CameraTarget', center)
    look_at(camera, target)
    bpy.context.scene.camera = camera
    return camera, target


def add_lights(center, size):
    key = bpy.data.lights.new('Key', 'AREA')
    key.energy = 420.0
    key.shape = 'DISK'
    key.size = max(size * 1.5, 2.0)
    key_obj = bpy.data.objects.new('Key', key)
    bpy.context.scene.collection.objects.link(key_obj)
    key_obj.location = Vector(center) + Vector((size * 1.4, -size * 1.5, size * 1.6))
    look_at(key_obj, make_empty('KeyTarget', center))
    fill = bpy.data.lights.new('Fill', 'AREA')
    fill.energy = 160.0
    fill.size = max(size * 2.0, 3.0)
    fill_obj = bpy.data.objects.new('Fill', fill)
    bpy.context.scene.collection.objects.link(fill_obj)
    fill_obj.location = Vector(center) + Vector((-size * 1.8, -size * 1.2, size * 0.8))
    look_at(fill_obj, make_empty('FillTarget', center))
    rim = bpy.data.lights.new('Rim', 'AREA')
    rim.energy = 240.0
    rim.size = max(size * 1.2, 2.0)
    rim_obj = bpy.data.objects.new('Rim', rim)
    bpy.context.scene.collection.objects.link(rim_obj)
    rim_obj.location = Vector(center) + Vector((-size * 1.2, size * 1.0, size * 1.4))
    look_at(rim_obj, make_empty('RimTarget', center))
    set_world()


def import_molecule(path, label):
    before = set(bpy.data.objects)
    result = bpy.ops.mn.import_molecule(method='local', filepath=str(path),
                                       file_format='cif', style='cartoon')
    if result != {'FINISHED'}:
        raise RuntimeError(f'Import failed: {path}')
    new = [obj for obj in bpy.data.objects if obj not in before and obj.type == 'MESH']
    if not new:
        raise RuntimeError(f'No mesh imported: {path}')
    obj = new[-1]
    obj.name = label
    return obj


def local_vertices(obj):
    return obj.data.attributes['res_id'].data, obj.data.attributes['is_alpha_carbon'].data


def ca_map(obj, lo=None, hi=None):
    res, ca = local_vertices(obj)
    positions = {}
    for index, vertex in enumerate(obj.data.vertices):
        residue = res[index].value
        if ca[index].value and (lo is None or lo <= residue <= hi):
            positions[residue] = obj.matrix_world @ vertex.co
    return positions


def centre_of(obj):
    positions = list(ca_map(obj).values())
    if not positions:
        positions = [obj.matrix_world @ vertex.co for vertex in obj.data.vertices]
    array = np.array([[v.x, v.y, v.z] for v in positions])
    return Vector(array.mean(axis=0)), float(np.max(array.max(axis=0) - array.min(axis=0)))


def domain_attributes(obj, domains):
    res, _ = local_vertices(obj)
    names = {}
    for name, (lo, hi) in domains.items():
        key = f'domain_{name}'
        attribute = obj.data.attributes.new(key, 'BOOLEAN', 'POINT')
        for index in range(len(obj.data.vertices)):
            attribute.data[index].value = lo <= res[index].value <= hi
        names[name] = key
    other = obj.data.attributes.new('domain_other', 'BOOLEAN', 'POINT')
    covered = set()
    for lo, hi in domains.values():
        covered.update(range(lo, hi + 1))
    for index in range(len(obj.data.vertices)):
        other.data[index].value = res[index].value not in covered
    names['other'] = 'domain_other'
    return names


def remove_default_style(obj):
    modifier = obj.modifiers.get('Molecular Nodes')
    if modifier is None or modifier.node_group is None:
        raise RuntimeError(f'Missing Molecular Nodes modifier: {obj.name}')
    tree = modifier.node_group
    for node in [node for node in tree.nodes if node.name.startswith('Style ')]:
        bpy.ops.mn.remove_style(name_tree=tree.name, name_node=node.name)


def add_style(obj, style, selection, color=None, name=None):
    kwargs = {
        'name_object': obj.name,
        'style': style,
        'selection': selection,
        'name': name or style,
    }
    if color is None:
        kwargs['use_uniform_color'] = False
    else:
        kwargs['use_uniform_color'] = True
        kwargs['uniform_color'] = color
    result = bpy.ops.mn.add_style(**kwargs)
    if result != {'FINISHED'}:
        raise RuntimeError(f'Style failed: {obj.name} {style} {selection}')


def material(name, color, emission=0.0):
    mat = bpy.data.materials.get(name)
    if mat is not None:
        return mat
    mat = bpy.data.materials.new(name)
    mat.use_nodes = True
    nodes = mat.node_tree.nodes
    nodes.clear()
    output = nodes.new('ShaderNodeOutputMaterial')
    if emission > 0:
        shader = nodes.new('ShaderNodeEmission')
        shader.inputs['Color'].default_value = color
        shader.inputs['Strength'].default_value = emission
    else:
        shader = nodes.new('ShaderNodeBsdfPrincipled')
        shader.inputs['Base Color'].default_value = color
        shader.inputs['Roughness'].default_value = 0.45
    mat.node_tree.links.new(shader.outputs[0], output.inputs['Surface'])
    return mat


def add_marker(location, color, radius=0.11, label=None):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=20, ring_count=12,
                                         radius=radius, location=location)
    marker = bpy.context.object
    marker.name = f'marker_{label or "site"}'
    marker.data.materials.append(material(f'marker_{label or "site"}', (*color[:3], 1.0), 0.9))


def add_leader(start, end, color, label=None, radius=0.02):
    direction = Vector(end) - Vector(start)
    length = direction.length
    if length <= 1e-6:
        return
    bpy.ops.mesh.primitive_cylinder_add(radius=radius, depth=length,
                                         location=(Vector(start) + Vector(end)) / 2)
    leader = bpy.context.object
    leader.name = f'leader_{label or "site"}'
    leader.rotation_mode = 'QUATERNION'
    leader.rotation_quaternion = direction.to_track_quat('Z', 'Y')
    leader.data.materials.append(material(f'leader_{label or "site"}', (*color[:3], 1.0), 0.9))


def add_text(text, location, color, size, camera, name=None, align_x='CENTER', bold=False):
    curve = bpy.data.curves.new(name or text, type='FONT')
    curve.body = text
    curve.size = size
    curve.align_x = align_x
    curve.align_y = 'CENTER'
    if bold and BOLD_FONT is not None:
        curve.font = bpy.data.fonts.load(str(BOLD_FONT), check_existing=True)
    obj = bpy.data.objects.new(name or text, curve)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(material(f'text_{name or text}', (*color[:3], 1.0), 1.0))
    look_at(obj, camera, track_axis='TRACK_Z')
    return obj


def add_site(obj, position, color, label, camera, centre, size=0.30, label_shift=(0.0, 0.0, 0.0),
             label_depth=6.0):
    ca = ca_map(obj, position, position)
    if position not in ca:
        raise ValueError(f'Missing C-alpha for residue {position} in {obj.name}')
    point = ca[position]
    outward = (point - Vector(centre))
    if outward.length == 0:
        outward = Vector((0.0, 0.0, 1.0))
    outward.normalize()
    label_location = point + outward * 0.55 + Vector((0.0, -0.55, 0.18)) + Vector(label_shift)
    label_location.y = centre.y - label_depth
    add_marker(point, color, radius=0.1, label=f'{label}_{obj.name}')
    add_leader(point, label_location, color, label=f'{label}_{obj.name}')
    add_text(label, label_location, color, size, camera, name=f'label_{label}_{obj.name}')


def add_ligand_label(center, text, camera, color, offset, size=0.2, label_y=None):
    center = Vector(center)
    label_location = center + Vector(offset)
    if label_y is not None:
        label_location.y = label_y
    add_leader(center, label_location, color, label=text)
    add_text(text, label_location, color, size, camera, name=f'label_ligand_{text}')


def add_color_legend(camera, center, size, colors, corner='lower_left', z_position=None):
    names = ('NMNAT', 'linker', 'Nudix')
    text_size = max(0.32, size * 0.018)
    radius = max(0.07, size * 0.0045)
    if corner == 'upper_left':
        x = center.x - size * 0.42
        z = center.z + 5.0
        y = center.y - size * 0.8
    else:
        x = center.x - size * 0.45
        z = center.z - size * 0.45
        y = center.y - size * 0.8
    if z_position is not None:
        z = z_position
    step = max(0.40, size * 0.026)
    for index, name in enumerate(names):
        location = Vector((x, y, z - index * step))
        add_marker(location, colors[name], radius=radius, label=f'legend_{name}')
        add_text(name, location + Vector((0.36, 0.0, 0.0)),
                 TEXT_DARK, text_size, camera, align_x='LEFT', bold=True,
                 name=f'legend_text_{name}')


def add_candidate_legend(camera, center, size, config):
    x = center.x - size * 0.85
    z = center.z + size * 0.25
    y = center.y - size * 0.8
    step = max(0.40, size * 0.026)
    text_size = max(0.28, size * 0.016)
    for index, name in enumerate(('A1', 'A2', 'A3', 'B1', 'B2')):
        color = config['marker_colors'][name]
        location = Vector((x, y, z - index * step))
        add_marker(location, color, radius=max(0.07, size * 0.0045), label=f'legend_group_{name}')
        add_text(f'{name} {config["candidate_edits"][name]}',
                 location + Vector((0.46, 0.0, 0.0)), TEXT_DARK, text_size, camera,
                 align_x='LEFT', bold=True, name=f'legend_group_text_{name}')


def combined_bounds(objects):
    points = []
    for obj in objects:
        points.extend(ca_map(obj).values())
    if not points:
        raise ValueError('No C-alpha coordinates for bounds')
    array = np.array([[v.x, v.y, v.z] for v in points])
    minimum, maximum = array.min(0), array.max(0)
    centre = Vector(((minimum + maximum) / 2).tolist())
    return centre, float(np.max(maximum - minimum))


def configure_animation(render, output, frames, fps):
    scene = bpy.context.scene
    frames_dir = Path(output).parent / 'demo_frames'
    if frames_dir.exists():
        for path in frames_dir.glob('frame_*.png'):
            path.unlink()
    frames_dir.mkdir(parents=True, exist_ok=True)
    scene.render.resolution_x = int(render.get('animation_width', 960))
    scene.render.resolution_y = int(render.get('animation_height', 540))
    scene.cycles.samples = int(render.get('animation_samples', 16))
    scene.render.image_settings.file_format = 'PNG'
    scene.render.image_settings.color_mode = 'RGBA'
    scene.render.fps = int(fps)
    scene.frame_start = 1
    scene.frame_end = int(frames)
    scene.render.filepath = str(frames_dir / 'frame_')
    return frames_dir


def encode_animation(frames_dir, output, fps, ffmpeg):
    import subprocess

    subprocess.run(
        [ffmpeg, '-y', '-framerate', str(fps), '-i', str(frames_dir / 'frame_%04d.png'),
         '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',
         '-movflags', '+faststart', str(output)],
        check=True,
    )


def set_linear_interpolation(obj):
    # Blender 5.x stores object animation in layered action channelbags.
    action = obj.animation_data.action
    for layer in action.layers:
        for strip in layer.strips:
            for channelbag in strip.channelbags:
                for fcurve in channelbag.fcurves:
                    for point in fcurve.keyframe_points:
                        point.interpolation = 'LINEAR'


def animate_turntable(objects, centre, frames):
    pivot = make_empty('WT_Rotation', centre)
    for obj in objects:
        world = obj.matrix_world.copy()
        obj.parent = pivot
        obj.matrix_parent_inverse = pivot.matrix_world.inverted()
        obj.matrix_world = world
    pivot.rotation_mode = 'XYZ'
    pivot.rotation_euler = (0.0, 0.0, 0.0)
    pivot.keyframe_insert('rotation_euler', frame=1)
    pivot.rotation_euler = (0.0, 0.0, 2.0 * math.pi)
    pivot.keyframe_insert('rotation_euler', frame=frames + 1)
    set_linear_interpolation(pivot)
    bpy.context.scene.frame_set(1)


def build_wt(config, render):
    wt = import_molecule(config['wt_cif'], 'WT')
    domains = config['domains']
    names = domain_attributes(wt, domains)
    remove_default_style(wt)
    for name, key in names.items():
        add_style(wt, 'cartoon', key, config['colors'][name], name=name)
    centre, size = centre_of(wt)
    camera, _ = make_camera(centre, size, ortho=True, ortho_factor=1.5)
    add_lights(centre, size)
    add_text('WT design sites | candidate-group colors', centre + Vector((0.0, -size * 0.8, size * 0.66)),
             TEXT_DARK, 0.32, camera, name='wt_title')
    add_color_legend(camera, centre, size, config['colors'], corner='lower_left')
    add_candidate_legend(camera, centre, size, config)
    for mutation in config['mutations']:
        group = mutation['candidates'][0][:2]
        color = config['marker_colors'].get(group, (0.9, 0.9, 0.9, 1.0))
        ca = ca_map(wt, int(mutation['position']), int(mutation['position']))
        add_marker(ca[int(mutation['position'])], color, radius=0.12,
                   label=f"{mutation['wt']}{mutation['position']}{mutation['alt']}_{wt.name}")
    bpy.context.scene.render.filepath = config['outputs']['wt_overview']
    bpy.ops.render.render(write_still=True)
    if not render.get('skip_animation', False):
        frames_dir = configure_animation(render, config['outputs']['animation'],
                                         int(render.get('frames', 120)), int(render.get('fps', 24)))
        group = [wt] + [item for item in bpy.data.objects
                        if (item.name.startswith('marker_')
                            and not item.name.startswith('marker_legend_'))
                        or item.name.startswith(('label_', 'leader_'))]
        animate_turntable(group, centre, int(render.get('frames', 120)))
        for item in group:
            if item.type in {'FONT', 'MESH'} and item.name.startswith(('label_', 'leader_')):
                item.hide_render = True
    bpy.ops.wm.save_as_mainfile(filepath=config['outputs']['blend'])
    if not render.get('skip_animation', False):
        bpy.ops.render.render(animation=True)
        encode_animation(frames_dir, config['outputs']['animation'],
                         int(render.get('fps', 24)),
                         render.get('ffmpeg', '/usr/bin/ffmpeg'))
        for path in frames_dir.glob('frame_*.png'):
            path.unlink()
        frames_dir.rmdir()


def build_a_class(config):
    objects = []
    for name in ('A1', 'A2', 'A3'):
        obj = import_molecule(config['candidate_cifs'][name], name)
        transform = config.get('align_transforms', {}).get(name)
        if transform is None:
            raise ValueError(f'Missing WT alignment transform: {name}')
        obj.matrix_world = Matrix(transform) @ obj.matrix_world
        objects.append(obj)
    spacing = 10.0
    offsets = {'A1': -spacing, 'A2': 0.0, 'A3': spacing}
    for obj in objects:
        obj.matrix_world = Matrix.Translation((offsets[obj.name], 0.0, 0.0)) @ obj.matrix_world
    centre, size = combined_bounds(objects)
    camera, _ = make_camera(centre, size, ortho=True, ortho_factor=0.68)
    add_lights(centre, size)
    add_color_legend(camera, centre, size, config['colors'], corner='lower_left',
                     z_position=centre.z - 6.0)
    add_text('A1/A2/A3 aligned to the same WT NMNAT C-alpha frame',
             Vector((centre.x, centre.y, centre.z + 6.5)),
             TEXT_DARK, 0.5, camera, name='panel_title')
    for obj in objects:
        names = domain_attributes(obj, config['domains'])
        remove_default_style(obj)
        for name, key in names.items():
            add_style(obj, 'cartoon', key, config['colors'][name], name=name)
        obj_centre, _ = centre_of(obj)
        for mutation in config['mutations']:
            if obj.name not in mutation['candidates']:
                continue
            color = config['marker_colors'].get(obj.name, (0.9, 0.9, 0.9, 1.0))
            ca = ca_map(obj, int(mutation['position']), int(mutation['position']))
            add_marker(ca[int(mutation['position'])], color, radius=0.1,
                       label=f"{mutation['wt']}{mutation['position']}{mutation['alt']}_{obj.name}")
        edits = config['candidate_edits'][obj.name]
        add_text(f'{obj.name}  {edits}',
                 Vector((obj.matrix_world.translation.x, centre.y, obj_centre.z + 4.0)),
                 TEXT_DARK, 0.50, camera, name=f'panel_{obj.name}')
    bpy.context.scene.render.filepath = config['outputs']['a_class_panels']
    bpy.ops.render.render(write_still=True)


def build_nudix(config):
    wt = import_molecule(config['wt_cif'], 'WT_nudix')
    names = domain_attributes(wt, config['domains'])
    remove_default_style(wt)
    for name, key in names.items():
        add_style(wt, 'cartoon', key, config['colors'][name], name=name)
    points = list(ca_map(wt, 204, 338).values())
    array = np.array([[v.x, v.y, v.z] for v in points])
    minimum, maximum = array.min(0), array.max(0)
    centre = Vector(((minimum + maximum) / 2).tolist())
    size = float(np.max(maximum - minimum))
    camera, _ = make_camera(centre, size * 1.1, ortho=True, ortho_factor=1.5)
    add_lights(centre, size)
    for mutation in config['mutations']:
        if mutation['domain'] != 'Nudix':
            continue
        group = mutation['candidates'][0]
        color = config['marker_colors'].get(group, (0.9, 0.9, 0.9, 1.0))
        shift = (-0.7, -0.8, 0.5) if mutation['position'] == 248 else (0.0, -0.8, -0.6)
        add_site(wt, int(mutation['position']), color,
                 f"{mutation['wt']}{mutation['position']}{mutation['alt']}",
                 camera, centre, size=0.30, label_shift=shift,
                 label_depth=max(3.0, size * 1.2))
    for reference in config.get('references', []):
        add_site(wt, int(reference['position']), TEXT_DARK,
                 reference['wt'] + str(reference['position']) + ' ref',
                 camera, centre, size=0.20, label_shift=(-0.7, -1.0, 0.8))
    add_text('Nudix apo WT', centre + Vector((0.0, 0.0, size * 0.75)),
             TEXT_DARK, 0.34, camera, name='nudix_title')
    add_text('E248Q (B1), E252 reference, D308N (B2)\nmetal/hydrolysis mechanism unresolved',
             centre + Vector((0.0, 0.0, size * 0.62)),
             TEXT_DARK, 0.24, camera, name='nudix_note')
    bpy.context.scene.render.filepath = config['outputs']['nudix_site']
    bpy.ops.render.render(write_still=True)


def build_nmnat_inset(config):
    joint = import_molecule(config['joint']['cif'], 'WT_joint')
    remove_default_style(joint)
    domains = domain_attributes(joint, config['domains'])
    add_style(joint, 'cartoon', domains['NMNAT'], (0.30, 0.32, 0.38, 1.0), name='NMNAT')
    ligands = config['joint'].get('ligands', [])
    mg_centres = [Vector(item['center_blender_units']) for item in ligands if item['name'] == 'MG']
    mg_center = mg_centres[0] if mg_centres else None
    hetero = joint.data.attributes['is_hetero'].data
    keep = joint.data.attributes.new('ligand_keep', 'BOOLEAN', 'POINT')
    for index, vertex in enumerate(joint.data.vertices):
        world = joint.matrix_world @ vertex.co
        keep.data[index].value = bool(hetero[index].value) and (
            mg_center is None or (world - mg_center).length > 0.35)
    add_style(joint, 'ball_and_stick', 'ligand_keep', None, name='ligands')
    atp = next((Vector(item['center_blender_units']) for item in ligands if item['name'] == 'ATP'), None)
    nmn = next((Vector(item['center_blender_units']) for item in ligands if item['name'] == 'NMN'), None)
    centre_points = [v for v in (atp, nmn) if v is not None]
    nmnat_points = list(ca_map(joint, 4, 181).values())
    all_points = nmnat_points + centre_points
    array = np.array([[v.x, v.y, v.z] for v in all_points])
    minimum, maximum = array.min(0), array.max(0)
    centre = Vector(((minimum + maximum) / 2).tolist())
    size = float(np.max(maximum - minimum))
    camera, _ = make_camera(centre, size, ortho=True, ortho_factor=1.9)
    add_lights(centre, size)
    if atp is not None:
        add_ligand_label(atp, 'ATP', camera, (0.62, 0.42, 0.0, 1.0),
                         (0.4, 0.0, -0.6), size=0.30, label_y=centre.y - size * 1.2)
    if nmn is not None:
        add_ligand_label(nmn, 'NMN', camera, (0.0, 0.42, 0.18, 1.0),
                         (-0.9, 0.0, 0.6), size=0.30, label_y=centre.y - size * 1.2)
    add_text('NMNAT ATP/NMN predicted co-occupancy',
             centre + Vector((0.0, 0.0, size * 0.72)),
             TEXT_DARK, 0.34, camera, name='nmnat_title')
    add_text('one joint model; input includes 1 Mg\nthis model places Mg in Nudix, outside this view\nproductive catalytic geometry not established',
             centre + Vector((0.0, 0.0, size * 0.58)),
             TEXT_DARK, 0.24, camera, name='nmnat_note', bold=True)
    bpy.context.scene.render.filepath = config['outputs']['nmnat_inset']
    bpy.ops.render.render(write_still=True)


def mn_version():
    path = Path(bpy.utils.user_resource('EXTENSIONS')) / 'user_default/molecularnodes/blender_manifest.toml'
    if path.exists():
        match = re.search(r'^version\s*=\s*"([^"]+)"', path.read_text(), re.MULTILINE)
        if match:
            return match.group(1)
    return None


def remove_path(path):
    path = Path(path)
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def main():
    args = parse_args()
    config = json.load(open(args.config))
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config['outputs'] = {key: str(output / name) for key, name in OUTPUTS.items()}
    identity_path = Path(config['outputs']['identity'])
    identity_path.unlink(missing_ok=True)
    for key in ('wt_overview', 'a_class_panels', 'nudix_site', 'nmnat_inset',
                'animation', 'blend', 'blend_session'):
        remove_path(config['outputs'][key])
    render = config['render']
    render['skip_animation'] = args.stills_only
    reset_scene()
    device_info = setup_render(render, render.get('device', 'GPU'))
    build_wt(config, render)
    reset_scene()
    setup_render(render, render.get('device', 'GPU'))
    build_a_class(config)
    reset_scene()
    setup_render(render, render.get('device', 'GPU'))
    build_nudix(config)
    reset_scene()
    setup_render(render, render.get('device', 'GPU'))
    build_nmnat_inset(config)
    mode = 'stills_only' if args.stills_only else 'full'
    produced = ['wt_overview', 'a_class_panels', 'nudix_site', 'nmnat_inset',
                'blend', 'blend_session']
    if mode == 'full':
        produced.append('animation')
    outputs = {}
    for key in produced:
        path = Path(config['outputs'][key])
        if not path.exists():
            raise RuntimeError(f'Missing render output: {path}')
        outputs[key] = {'path': str(path), 'sha256': sha256_path(path)}
    identity = {
        'status': 'COMPLETED',
        'mode': mode,
        'blend_state': 'animation_labels_hidden' if mode == 'full' else 'still_labels_visible',
        'blender_version': bpy.app.version_string,
        'molecular_nodes_version': mn_version(),
        'render': {
            'engine': 'CYCLES',
            'device': device_info,
            'still_samples': int(render.get('samples', 24)),
            'still_resolution': [int(render['width']), int(render['height'])],
            'animation': None if mode == 'stills_only' else {
                'samples': int(render.get('animation_samples', 16)),
                'resolution': [int(render.get('animation_width', 960)),
                               int(render.get('animation_height', 540))],
                'frames': int(render.get('frames', 120)),
                'fps': int(render.get('fps', 24)),
                'motion': 'static camera; 360 deg turntable',
            },
        },
        'render_input_sha256': sha256(args.config),
        'script_sha256': sha256(__file__),
        'outputs': outputs,
        'generated_utc': datetime.now(timezone.utc).isoformat(),
        'limits': 'Headless predicted structures; turntable motion is display only; no dynamics, binding, or activity claim.',
    }
    identity_path.write_text(json.dumps(identity, indent=2, ensure_ascii=False) + '\n')
    print(json.dumps(identity, ensure_ascii=False))


if __name__ == '__main__':
    main()
