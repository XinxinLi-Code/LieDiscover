"""Affine coordinate interface for the unchanged official cropped grid.

The official resampler uses t=i/nt and x=j/nx. The spatial scale below is
nx*dx of the retained patch, NOT the full-domain physical period. This keeps
the upstream periodic-patch assumption; it does not repair that assumption.
"""

import math

import torch


PROTOCOL = 'lienlsd_official_7d85a987_coords_v1'


def frame_from_axes(t, x):
    frame = {'version': 'affine_cropped_grid_v1',
             'boundary_convention': 'official_periodic_crop'}
    for name, axis in (('t', t), ('x', x)):
        axis = axis.detach().double()
        if axis.ndim != 1 or len(axis) < 2 or not torch.isfinite(axis).all():
            raise ValueError(f'Expected a finite one-dimensional {name} grid.')
        origin = float(axis[0])
        step = float((axis[-1] - axis[0]) / (len(axis) - 1))
        expected = origin + torch.arange(len(axis), dtype=axis.dtype, device=axis.device) * step
        if step <= 0 or not torch.allclose(axis, expected, rtol=1e-6, atol=1e-6):
            raise ValueError(f'The official coordinate interface requires a uniform increasing {name} grid.')
        frame.update({f'{name}_origin': origin, f'd{name}': step,
                      f'n{name}': len(axis), f'{name}_scale': len(axis) * step})
    return frame


def frames_match(left, right):
    if set(left) != set(right):
        return False
    return all(math.isclose(a, right[key], rel_tol=1e-6, abs_tol=1e-7)
               if isinstance(a, (float, int)) else a == right[key]
               for key, a in left.items())


def bind_coordinates(args, *datasets):
    frame = datasets[0].coordinate_map
    if any(not frames_match(frame, data.coordinate_map) for data in datasets[1:]):
        raise ValueError('Train/validation/test coordinate frames disagree; use matching saved grids.')
    args.coordinate_map = dict(frame)


def physical_points(points, frame):
    """Return physical t,x,u without mutating normalized ODE states."""
    scale = points.new_tensor([frame['t_scale'], frame['x_scale'], 1.])
    origin = points.new_tensor([frame['t_origin'], frame['x_origin'], 0.])
    return points * scale + origin


def pushforward(delta, frame):
    """Convert all physical vector-field channels to normalized coordinates."""
    scale = delta.new_tensor([frame['t_scale'], frame['x_scale'], 1.])
    return delta / scale[None, :, None]
