"""Shared rounded connector geometry for the animated thesis diagrams."""
import math


def rounded_route(points, radius=18, steps=16):
    """Return an SVG path and sampled points following the same quadratic bends."""
    points = [tuple(p) for p in points]
    path = [f'M {points[0][0]},{points[0][1]}']
    sampled = [points[0]]
    for i in range(1, len(points) - 1):
        a, b, c = points[i - 1:i + 2]
        incoming, outgoing = math.dist(a, b), math.dist(b, c)
        if not incoming or not outgoing:
            continue
        trim = min(radius, incoming / 2, outgoing / 2)
        if i == len(points) - 2:
            trim = min(trim, max(0, outgoing - 20))
        before = tuple(b[j] + (a[j] - b[j]) * trim / incoming for j in (0, 1))
        after = tuple(b[j] + (c[j] - b[j]) * trim / outgoing for j in (0, 1))
        path.append(f'L {before[0]},{before[1]} Q {b[0]},{b[1]} {after[0]},{after[1]}')
        sampled.append(before)
        for k in range(1, steps + 1):
            t = k / steps
            sampled.append(tuple((1-t)**2 * before[j] + 2*(1-t)*t*b[j] + t*t*after[j] for j in (0, 1)))
    path.append(f'L {points[-1][0]},{points[-1][1]}')
    sampled.append(points[-1])
    # Avoid zero-length segments in packet interpolation.
    sampled = [p for i, p in enumerate(sampled) if i == 0 or p != sampled[i-1]]
    return ' '.join(path), sampled
