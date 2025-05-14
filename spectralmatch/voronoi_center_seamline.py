from typing import List, Tuple
import os
import rasterio
import numpy as np
from rasterio.features import shapes
from affine import Affine
from shapely.geometry import shape, Polygon, LineString, mapping, GeometryCollection, Point
from shapely.ops import split, voronoi_diagram
import networkx as nx
import fiona
from itertools import combinations


def voronoi_center_seamline(
    paths: List[str],
    mask_out: str,
    *,
    dist_min: float = 10,
    min_cut_length: float = 0,
    debug: bool = False,
    image_field_name: str = 'image',
    )-> None:
    """
    Generates a Voronoi-based seamline from the centers of edge-matching polygons (EMPs) and saves the result as a vector mask.

    Args:
        paths (List[str]): List of input EMP vector file paths.
        mask_out (str): Output path for the generated seamline vector mask.
        dist_min (float, optional): Minimum spacing between Voronoi points. Defaults to 10.
        min_cut_length (float, optional): Minimum seamline segment length to retain. Defaults to 0.
        debug (bool, optional): If True, enables debug output. Defaults to False.
        image_field_name (str, optional): Field name for the output image. Defaults to 'image'.

    Outputs:
        Writes the seamline vector file to the specified output file.
    """

    emps = []
    crs = None
    for p in paths:
        mask, transform = _read_mask(p, debug)
        if debug: print(f"{p}: valid pixels = {mask.sum()}/{mask.size}")
        emp = _seamline_mask(mask, transform, debug)
        if debug: print(f"Extracted EMP from {p}: area={emp.area:.2f}, bounds={emp.bounds}")
        emps.append(emp)
        if crs is None:
            with rasterio.open(p) as src:
                crs = src.crs

    for i, emp in enumerate(emps):
        if debug: print(f"EMP[{i}]: area={emp.area:.2f}, bounds={emp.bounds}")

    cuts: List[LineString] = []
    for i, (a, b) in enumerate(combinations(emps, 2)):
        ov = a.intersection(b)
        if debug: print(f"Overlap {i} area: {ov.area:.2f}")
        if not ov.is_empty:
            cut = _compute_centerline(a, b, dist_min, min_cut_length, debug)
            if debug: print(f"Cutline {i}: length={cut.length:.2f}")
            cuts.append(cut)

    segmented: List[Polygon] = []
    for idx, emp in enumerate(emps):
        relevant = [c for c in cuts if emp.intersects(c)]
        if debug: print(f"EMP[{idx}] has {len(relevant)} intersecting cuts")
        seg = _segment_emp(emp, relevant, debug)
        if debug: print(f"EMP[{idx}] segmented result area: {seg.area:.2f}")
        segmented.append(seg)

    schema = {'geometry': 'Polygon', 'properties': {image_field_name: 'str'}}
    with fiona.open(mask_out, 'w', driver='GPKG', crs=crs, schema=schema, layer='seamlines') as dst:
        for img, poly in zip(paths, segmented):
            if debug: print(f"Writing polygon for {os.path.basename(img)}: area={poly.area:.2f}")
            dst.write({'geometry': mapping(poly), 'properties': {image_field_name: os.path.splitext(os.path.basename(img))[0]}})


def _read_mask(
    path: str,
    debug: bool = False
    ) -> Tuple[np.ndarray, Affine]:

    with rasterio.open(path) as src:
        arr = src.read(1)
        nod = src.nodatavals[0]
        if debug: print(f"Nodata value for {path}: {nod}")
        return arr != nod, src.transform


def _seamline_mask(
    mask: np.ndarray,
    transform: Affine,
    debug: bool = False
    ) -> Polygon:

    polys = [shape(geom) for geom, val in shapes(mask.astype(np.uint8), mask=mask, transform=transform) if val == 1]
    if debug: print(f"Extracted {len(polys)} polygons from mask")
    if not polys:
        raise ValueError("No valid EMP polygons found")
    largest = max(polys, key=lambda p: p.area)
    if debug: print(f"Largest EMP area: {largest.area:.2f}")
    return largest


def _densify_polygon(
    poly: Polygon,
    dist: float,
    debug: bool = False
    ) -> List[Tuple[float, float]]:

    coords = list(poly.exterior.coords)
    dense = []
    for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
        d = ((x1 - x0)**2 + (y1 - y0)**2)**0.5
        n = max(int(d // dist), 1)
        for j in range(n):
            dense.append((x0 + (x1 - x0) * j / n, y0 + (y1 - y0) * j / n))
    if debug: print(f"Densified {len(dense)} points from polygon with {len(coords)} original points")
    return dense


def _compute_centerline(
    a: Polygon,
    b: Polygon,
    dist_min: float,
    min_cut_length: float,
    debug: bool = False
    ) -> LineString:

    voa = a.intersection(b)
    pts = _densify_polygon(voa, dist_min, debug)
    if debug: print(f"Densified {len(pts)} points on overlap boundary")

    multi = voronoi_diagram(GeometryCollection([Point(p) for p in pts]))
    G = nx.Graph()
    for seg in multi.geoms:
        if isinstance(seg, LineString) and seg.length >= min_cut_length:
            G.add_edge(seg.coords[0], seg.coords[-1], weight=seg.length)
    if debug: print(f"Voronoi graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    boundary_pts = a.boundary.intersection(b.boundary)
    if isinstance(boundary_pts, Point):
        coords = [(boundary_pts.x, boundary_pts.y)]
    elif hasattr(boundary_pts, "geoms"):
        coords = [(pt.x, pt.y) for pt in boundary_pts.geoms if isinstance(pt, Point)]
    else:
        coords = []

    if len(coords) >= 2:
        u, v = coords[0], coords[1]
        if debug: print(f"Using boundary intersections: {u}, {v}")
    else:
        u, v = max(combinations(pts, 2), key=lambda p: (p[0]-p[1])**2 + (p[1]-p[1])**2)
        if debug: print(f"Fallback to farthest densified points: {u}, {v}")

    nodes = list(G.nodes())
    if not nodes:
        if debug: print("Empty Voronoi graph, using fallback straight line")
        return LineString([u, v])

    start = min(nodes, key=lambda n: (n[0]-u[0])**2 + (n[1]-u[1])**2)
    end = min(nodes, key=lambda n: (n[0]-v[0])**2 + (n[1]-v[1])**2)
    if debug: print(f"Snapped start={start}, end={end}")

    try:
        path = nx.shortest_path(G, source=start, target=end, weight='weight')
        if debug: print(f"Computed shortest path with {len(path)} nodes")
        return LineString(path)
    except nx.NetworkXNoPath:
        if debug: print("No path found; fallback straight line used")
        return LineString([u, v])


def _segment_emp(
    emp: Polygon,
    cuts: List[LineString],
    debug: bool = False
    ) -> Polygon:

    for i, ln in enumerate(cuts):
        if emp.intersects(ln):
            if debug: print(f"EMP intersects cut {i}, performing split")
            result = split(emp, ln)
            pieces = list(result.geoms)
            if debug: print(f"Split created {len(pieces)} pieces with areas: {[p.area for p in pieces]}")
            emp = max(pieces, key=lambda p: p.area)
            if debug: print(f"Retained piece with area={emp.area:.2f}")
        elif debug:
            print(f"EMP does not intersect cut {i}, skipping")
    return emp