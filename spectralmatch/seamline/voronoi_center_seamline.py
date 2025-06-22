import os
import rasterio
import numpy as np
import networkx as nx
import fiona

from fiona.errors import DriverError
from rasterio.features import shapes
from affine import Affine
from shapely.geometry import MultiPoint
from shapely.geometry import (
    shape,
    Polygon,
    LineString,
    mapping,
    GeometryCollection,
    Point,
)
from shapely.ops import split, voronoi_diagram
from itertools import combinations
from typing import List, Tuple

from ..handlers import search_paths, _resolve_paths
from ..types_and_validation import Universal


def voronoi_center_seamline(
    input_images: Universal.CreateInFolderOrListFiles,
    output_mask: str,
    *,
    image_field_name: str = "image",
    min_point_spacing: float = 10,
    min_cut_length: float = 0,
    debug_logs: Universal.DebugLogs = False,
    debug_vectors_path: str | None = None,
) -> None:
    """
    Generates a Voronoi-based seamline mask from edge-matching polygons (EMPs) and writes the result to a vector file.

    Args:
        input_images (str | List[str], required): Defines input files from a glob path, folder, or list of paths. Specify like: "/input/files/*.tif", "/input/folder" (assumes *.tif), ["/input/one.tif", "/input/two.tif"].
        output_mask (str): Output path for the final seamline polygon vector file.
        min_point_spacing (float, optional): Minimum spacing between Voronoi seed points; default is 10.
        min_cut_length (float, optional): Minimum cutline segment length to retain; default is 0.
        debug_logs (Universal.DebugLogs, optional): Enables debug print statements if True; default is False.
        image_field_name (str, optional): Name of the attribute field for image ID in output; default is 'image'.
        debug_vectors_path (str | None, optional): Optional path to save debug layers (cutlines, intersections).

    Outputs:
        Saves a polygon seamline layer to `output_mask`, and optionally saves intermediate cutlines to `debug_vectors_path`.
    """

    print("Start voronoi center seamline")

    Universal.validate(
        input_images=input_images,
    )
    input_image_paths = _resolve_paths(
        "search", input_images, kwargs={"default_file_pattern": "*.tif"}
    )

    emps = []
    crs = None
    for p in input_image_paths:
        mask, transform = _read_mask(p, debug_logs)
        emp = _seamline_mask(mask, transform, debug_logs)
        emps.append(emp)
        if crs is None:
            with rasterio.open(p) as src:
                crs = src.crs

    for i, emp in enumerate(emps):
        if debug_logs:
            print(f"EMP{i}: area={emp.area:.2f}, bounds={emp.bounds}")

    cuts: List[LineString] = []
    for i, (a, b) in enumerate(combinations(emps, 2)):
        ov = a.intersection(b)
        if debug_logs:
            print(f"Overlap {i} area: {ov.area:.2f}")
        if not ov.is_empty:
            if debug_vectors_path:
                _save_intersection_points(a, b, debug_vectors_path, crs, f"{i}")
            cut = _compute_centerline(
                a,
                b,
                min_point_spacing,
                min_cut_length,
                debug_logs,
                crs,
                debug_vectors_path,
            )
            cuts.append(cut)

    # Optionally save cutlines
    if debug_vectors_path:
        schema = {"geometry": "LineString", "properties": {"pair_id": "str"}}
        with fiona.open(
            debug_vectors_path,
            "w",
            driver="GPKG",
            crs=crs,
            schema=schema,
            layer="cutlines",
        ) as dst:
            for idx, line in enumerate(cuts):
                dst.write(
                    {
                        "geometry": mapping(line),
                        "properties": {"pair_id": f"{idx}"},
                    }
                )

    segmented: List[Polygon] = []
    for idx, emp in enumerate(emps):
        relevant = [c for c in cuts if emp.intersects(c)]
        seg = _segment_emp(emp, relevant, debug_logs)
        if debug_logs:
            print(
                f"EMP{idx} has {len(relevant)} intersecting cuts and {seg.area:.2f} segmented area"
            )
        segmented.append(seg)

    schema = {"geometry": "Polygon", "properties": {image_field_name: "str"}}
    with fiona.open(
        output_mask, "w", driver="GPKG", crs=crs, schema=schema, layer="seamlines"
    ) as dst:
        for img, poly in zip(input_image_paths, segmented):
            dst.write(
                {
                    "geometry": mapping(poly),
                    "properties": {
                        image_field_name: os.path.splitext(os.path.basename(img))[0]
                    },
                }
            )


def _read_mask(path: str, debug_logs: bool = False) -> Tuple[np.ndarray, Affine]:
    """
    Reads a raster mask and returns a binary array where valid data is True.

    Args:
        path (str): Path to the input raster file.
        debug_logs (bool, optional): If True, enables debug output; default is False.

    Returns:
        Tuple[np.ndarray, Affine]: A binary mask array and the associated affine transform.
    """

    with rasterio.open(path) as src:
        arr = src.read(1)
        nod = src.nodatavals[0]
        return arr != nod, src.transform


def _seamline_mask(
    mask: np.ndarray, transform: Affine, debug_logs: bool = False
) -> Polygon:
    """
    Extracts polygons from a binary mask and returns the largest as the EMP.

    Args:
        mask (np.ndarray): Binary mask where True indicates valid area.
        transform (Affine): Affine transform associated with the mask.
        debug_logs (bool, optional): If True, prints debug info; default is False.

    Returns:
        Polygon: The largest extracted polygon from the mask.
    """

    polys = [
        shape(geom)
        for geom, val in shapes(mask.astype(np.uint8), mask=mask, transform=transform)
        if val == 1
    ]
    if debug_logs:
        print(f"Extracted {len(polys)} polygons from mask")
    if not polys:
        raise ValueError("No valid EMP polygons found")
    largest = max(polys, key=lambda p: p.area)
    return largest


def _densify_polygon(
    poly: Polygon | GeometryCollection, dist: float, debug_logs: bool = False
) -> List[Tuple[float, float]]:
    """
    Densifies the exterior of the largest polygon by inserting points at regular intervals.

    Args:
        poly (Polygon | GeometryCollection): Input geometry to densify.
        dist (float): Maximum distance between inserted points.
        debug_logs (bool, optional): If True, prints debug info; default is False.

    Returns:
        List[Tuple[float, float]]: List of (x, y) coordinates with added intermediate points.
    """

    # Extract coordinates from valid polygon geometries
    if isinstance(poly, Polygon):
        geometries = [poly]
    elif isinstance(poly, GeometryCollection):
        geometries = [g for g in poly.geoms if isinstance(g, Polygon)]
    else:
        raise TypeError(f"Unsupported geometry type: {type(poly)}")

    if not geometries:
        raise ValueError("No polygon geometry found to densify")

    # Use the largest polygon
    largest = max(geometries, key=lambda p: p.area)
    coords = list(largest.exterior.coords)

    dense = []
    for (x0, y0), (x1, y1) in zip(coords, coords[1:]):
        d = ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5
        n = max(int(d // dist), 1)
        for j in range(n):
            dense.append((x0 + (x1 - x0) * j / n, y0 + (y1 - y0) * j / n))
    return dense


def _compute_centerline(
    a: Polygon,
    b: Polygon,
    min_point_spacing: float,
    min_cut_length: float,
    debug_logs: bool = False,
    crs=None,
    debug_vectors_path=None,
) -> LineString:
    """
    Computes a Voronoi-based centerline between two overlapping polygons.

    Args:
        a (Polygon): First polygon.
        b (Polygon): Second polygon.
        min_point_spacing (float): Minimum spacing between seed points for Voronoi generation.
        min_cut_length (float): Minimum segment length to include in the centerline graph.
        debug_logs (bool, optional): If True, prints debug information; default is False.
        crs (optional): Coordinate reference system used for optional debug output.
        debug_vectors_path (optional): Path to save debug Voronoi cells; if None, skips saving.

    Returns:
        LineString: Shortest centerline path computed through the Voronoi diagram of the overlap.
    """

    voa = a.intersection(b)
    pts = _densify_polygon(voa, min_point_spacing, debug_logs)

    # Compute intersection and extract both Voronoi and anchor points
    boundary_pts = a.boundary.intersection(b.boundary)
    coords = []

    if isinstance(boundary_pts, Point):
        pt = (boundary_pts.x, boundary_pts.y)
        pts.append(pt)
        coords.append(pt)
    elif isinstance(boundary_pts, LineString):
        mid = boundary_pts.interpolate(0.5, normalized=True)
        pt = (mid.x, mid.y)
        pts.append(pt)
        coords.append(pt)
    elif hasattr(boundary_pts, "geoms"):
        for geom in boundary_pts.geoms:
            if isinstance(geom, Point):
                pt = (geom.x, geom.y)
            elif isinstance(geom, LineString):
                mid = geom.interpolate(0.5, normalized=True)
                pt = (mid.x, mid.y)
            else:
                continue
            pts.append(pt)
            coords.append(pt)

    if debug_logs:
        print(f"Densified {len(pts)} points")
        print(f"Convex hull area: {MultiPoint(pts).convex_hull.area}")

    multi = voronoi_diagram(GeometryCollection([Point(p) for p in pts]), edges=False)

    if debug_vectors_path:
        _save_voronoi_cells(
            multi, debug_vectors_path, crs, layer_name=f"voronoi_{int(voa.area)}"
        )

    G = nx.Graph()
    for poly in multi.geoms:
        if isinstance(poly, Polygon):
            coords_poly = list(poly.exterior.coords)
            for i in range(len(coords_poly) - 1):
                p1, p2 = coords_poly[i], coords_poly[i + 1]
                seg = LineString([p1, p2])
                if seg.length >= min_cut_length:
                    G.add_edge(p1, p2, weight=seg.length)

    if debug_logs:
        print(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    if len(coords) >= 2:
        u, v = coords[0], coords[1]
    else:
        u, v = max(
            combinations(pts, 2),
            key=lambda p: (p[0][0] - p[1][0]) ** 2 + (p[0][1] - p[1][1]) ** 2,
        )

    nodes = list(G.nodes())
    if not nodes:
        raise ValueError(
            "Empty Voronoi graph: no centerline could be computed for the overlap"
        )

    start = min(nodes, key=lambda n: (n[0] - u[0]) ** 2 + (n[1] - u[1]) ** 2)
    end = min(nodes, key=lambda n: (n[0] - v[0]) ** 2 + (n[1] - v[1]) ** 2)
    if debug_logs:
        print(f"Snapped start={start}, end={end}")

    path = nx.shortest_path(G, source=start, target=end, weight="weight")
    return LineString([u] + path + [v])


def _segment_emp(
    emp: Polygon, cuts: List[LineString], debug_logs: bool = False
) -> Polygon:
    """
    Segments an EMP polygon by sequentially applying centerline cuts, retaining the piece containing the centroid.

    Args:
        emp (Polygon): The original EMP polygon to segment.
        cuts (List[LineString]): List of cutlines to apply.
        debug_logs (bool, optional): If True, prints debug info; default is False.

    Returns:
        Polygon: The segmented portion of the EMP containing the original centroid.
    """

    # sequentially cut EMP by each centerline, choosing the segment containing the EMP centroid
    for i, ln in enumerate(cuts):
        if not emp.intersects(ln):
            # Force cut if it's close enough (e.g., < 1 unit)
            dist = emp.distance(ln)
            if dist > 1e-3:
                if debug_logs:
                    print(f"Cut {i} too far (distance={dist:.4f}), skipping")
                continue
            if debug_logs:
                print(f"Cut {i} near EMP (distance={dist:.4f}), forcing split")

        pieces = list(split(emp, ln).geoms)
        if not pieces:
            continue

        # choose the piece that contains the original centroid
        centroid = emp.centroid
        chosen = None
        for p in pieces:
            if p.contains(centroid):
                chosen = p
                break
        if chosen is None:
            # fallback to largest area if centroid-based selection fails
            chosen = max(pieces, key=lambda p: p.area)

        emp = chosen
        if debug_logs:
            print(
                f"After cut {i}: {len(pieces)} pieces, selected piece area={emp.area:.2f}"
            )

    return emp


def _save_intersection_points(
    a: Polygon,
    b: Polygon,
    path: str,
    crs,
    pair_id: str,
) -> None:
    """
    Saves intersection points between the boundaries of two polygons to a GeoPackage layer.

    Args:
        a (Polygon): First polygon.
        b (Polygon): Second polygon.
        path (str): Path to the output GeoPackage file.
        crs: Coordinate reference system for the output.
        pair_id (str): Identifier for the polygon pair, saved as an attribute.

    Returns:
        None
    """

    inter = a.boundary.intersection(b.boundary)
    if isinstance(inter, Point):
        points = [inter]
    elif hasattr(inter, "geoms"):
        points = [g for g in inter.geoms if isinstance(g, Point)]
    else:
        points = []

    if not points:
        return

    schema = {"geometry": "Point", "properties": {"pair_id": "str"}}
    layer_name = "intersections"

    mode = "a"
    if not os.path.exists(path) or layer_name not in fiona.listlayers(path):
        mode = "w"

    with fiona.open(
        path, mode=mode, driver="GPKG", crs=crs, schema=schema, layer=layer_name
    ) as dst:
        for pt in points:
            dst.write(
                {
                    "geometry": mapping(pt),
                    "properties": {"pair_id": pair_id},
                }
            )


def _save_voronoi_cells(
    voronoi_cells: GeometryCollection, path: str, crs, layer_name: str = "voronoi_cells"
) -> None:
    """
    Saves Voronoi polygon geometries to a specified GeoPackage layer.

    Args:
        voronoi_cells (GeometryCollection): Collection of Voronoi polygon geometries.
        path (str): Path to the output GeoPackage file.
        crs: Coordinate reference system for the output layer.
        layer_name (str, optional): Name of the layer to write; default is "voronoi_cells".

    Returns:
        None
    """

    schema = {"geometry": "Polygon", "properties": {}}

    # Determine if file and layer exist
    layer_exists = False
    if os.path.exists(path):
        try:
            with fiona.open(path, mode="r", driver="GPKG") as src:
                layer_exists = (
                    layer_name in src.listlayers()
                    if hasattr(src, "listlayers")
                    else False
                )
        except DriverError:
            pass

    mode = "a" if layer_exists else "w"

    with fiona.open(
        path, mode=mode, driver="GPKG", crs=crs, schema=schema, layer=layer_name
    ) as dst:
        for geom in voronoi_cells.geoms:
            if isinstance(geom, Polygon):
                dst.write(
                    {
                        "geometry": mapping(geom),
                        "properties": {},
                    }
                )
