import numpy as np
np.float = float
np.float128 = np.float64

import rasterio
from rasterio import features
from shapely.geometry import Polygon, LineString, Point, MultiLineString
from shapely.ops import unary_union
from scipy.spatial import Delaunay
import gco
import networkx as nx
import fiona
import cv2
import geopandas as gpd
from itertools import product
from typing import List, Optional, Tuple, Any


def _load_image_footprints(image_paths: List[str]) -> List[Polygon]:
    """Load image geospatial footprints as shapely Polygons.

    Args:
        image_paths (List[str]): List of file paths to georeferenced images.

    Returns:
        List[Polygon]: Shapely Polygons representing each image footprint.
    """
    footprints = []
    for path in image_paths:
        with rasterio.open(path) as src:
            bounds = src.bounds
        footprints.append(Polygon([(bounds.left, bounds.bottom),
                                   (bounds.left, bounds.top),
                                   (bounds.right, bounds.top),
                                   (bounds.right, bounds.bottom)]))
    return footprints


def _load_building_masks(mask_paths: List[str]) -> List[Polygon]:
    """Load vector building masks from shapefiles.

    Args:
        mask_paths (List[str]): List of file paths to building mask shapefiles.

    Returns:
        List[Polygon]: Shapely Polygons for building footprints.
    """
    masks = []
    for shp in mask_paths:
        with fiona.open(shp, 'r') as src:
            for feat in src:
                geom = Polygon(feat['geometry']['coordinates'][0])
                masks.append(geom)
    return masks


def _generate_triangulation(bounds: Polygon, triangle_size: float) -> List[Polygon]:
    minx, miny, maxx, maxy = bounds.bounds

    # Compute exact number of steps including last point exactly at max
    nx = int(np.round((maxx - minx) / triangle_size)) + 1
    ny = int(np.round((maxy - miny) / triangle_size)) + 1

    xs = np.linspace(minx, maxx, nx)
    ys = np.linspace(miny, maxy, ny)
    pts = np.array(list(product(xs, ys)))

    delaunay = Delaunay(pts)
    triangles = []
    for simplex in delaunay.simplices:
        tri_pts = pts[simplex]
        poly = Polygon(tri_pts)
        if poly.within(bounds):
            triangles.append(poly)
    return triangles


def _build_adjacency_graph(triangles: List[Polygon]) -> nx.Graph:
    """Build adjacency graph of triangles: edges where triangles share a segment.

    Args:
        triangles (List[Polygon]): List of shapely triangle Polygons.

    Returns:
        nx.Graph: Graph with nodes for triangles and 'geom' attr on edges.
    """
    G = nx.Graph()
    for i, tri in enumerate(triangles):
        G.add_node(i)
    for i in range(len(triangles)):
        for j in range(i + 1, len(triangles)):
            inter = triangles[i].boundary.intersection(triangles[j].boundary)
            if isinstance(inter, LineString) and inter.length > 0:
                G.add_edge(i, j, geom=inter)
    return G


def _compute_data_cost(triangles: List[Polygon],
                       footprints: List[Polygon],
                       infinite_cost: float = 1e6) -> np.ndarray:
    """Compute data cost: 0 if triangle visible in image, else infinite.

    Args:
        triangles (List[Polygon]): List of triangle polygons.
        footprints (List[Polygon]): List of image footprint polygons.
        infinite_cost (float): Cost for invisibility.

    Returns:
        np.ndarray: Array of shape (n_triangles, n_images).
    """
    nT = len(triangles)
    nI = len(footprints)
    cost = np.full((nT, nI), infinite_cost, dtype=np.int32)
    for t, tri in enumerate(triangles):
        cen = tri.centroid
        for i, fp in enumerate(footprints):
            if cen.within(fp):
                cost[t, i] = 0
    return cost


def _compute_image_gradients(image_paths: List[str]) -> Tuple[List[np.ndarray], List[Any]]:
    """Precompute gradient magnitude arrays and affine transforms for all images.

    Args:
        image_paths (List[str]): List of file paths to images.

    Returns:
        Tuple[List[np.ndarray], List[Any]]: Gradient magnitude arrays and transforms.
    """
    grads, transforms = [], []
    for path in image_paths:
        with rasterio.open(path) as src:
            arr = src.read([1, 2, 3]).astype(np.float32)
            inten = np.mean(arr, axis=0)
            gx = cv2.Sobel(inten, cv2.CV_32F, 1, 0, ksize=3)
            gy = cv2.Sobel(inten, cv2.CV_32F, 0, 1, ksize=3)
            grad = np.sqrt(gx**2 + gy**2)
            grads.append(grad)
            transforms.append(src.transform)
    return grads, transforms


def _compute_edge_weights(G: nx.Graph,
                          labels: List[int],
                          grads: List[np.ndarray],
                          transforms: List[Any],
                          lambda_param: float,
                          building_masks: Optional[List[Polygon]] = None,
                          samples: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute edges, weights, and pairs for MRF optimization.

    Args:
        G (nx.Graph): Adjacency graph of triangles.
        labels (List[int]): List of label indices.
        grads (List[np.ndarray]): Gradient arrays.
        transforms (List[Any]): Affine transforms for images.
        lambda_param (float): Smoothing parameter.
        building_masks (Optional[List[Polygon]]): Building footprints.
        samples (int): Number of sample points per edge.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: edges_array, weights_array, smooth_matrix.
    """
    L = len(grads)
    smooth_mat = np.ones((L, L), dtype=np.int32)
    np.fill_diagonal(smooth_mat, 0)

    edges = []
    weights = []
    inf = 1000000
    for i, j, data in G.edges(data=True):
        edge_geom: LineString = data['geom']
        # get a handful of points along the edge
        samples = 10
        pts = [edge_geom.interpolate(α, normalized=True)
               for α in np.linspace(0, 1, samples)]
        # for each image, pull gradient at those pts
        grad_vals = []
        for img_idx, (grad, trans) in enumerate(zip(grads, transforms)):
            for p in pts:
                # world→pixel
                row, col = ~trans * (p.x, p.y)
                grad_vals.append(grad[int(row), int(col)])
        mean_grad = np.mean(grad_vals)
        # make weight proportional to boundary strength
        w = int(lambda_param * mean_grad)
        edges.append((i, j))
        weights.append(w)
    edges_arr = np.array(edges, dtype=np.int32)
    weights_arr = np.array(weights, dtype=np.int32)
    return edges_arr, weights_arr, smooth_mat


def _solve_mrf(data_cost: np.ndarray,
               smooth_mat: np.ndarray,
               edges: np.ndarray,
               weights: np.ndarray) -> np.ndarray:
    """Solve labeling via graph cuts (alpha-expansion) using gco.

    Args:
        data_cost (np.ndarray): Array shape (n_nodes, n_labels).
        smooth_mat (np.ndarray): Label transition cost matrix.
        edges (np.ndarray): Edge indices array.
        weights (np.ndarray): Edge weights.

    Returns:
        np.ndarray: 1D array of labels.
    """
    labels = gco.cut_general_graph(edges,
                                     weights,
                                     data_cost,
                                     smooth_mat,
                                     n_iter=10)
    return labels


def _extract_seamlines(G: nx.Graph,
                       triangles: List[Polygon],
                       labels: np.ndarray) -> MultiLineString:
    """Extract seamline segments as edges where adjacent triangles have different labels.

    Args:
        G (nx.Graph): Adjacency graph.
        triangles (List[Polygon]): List of triangle polygons.
        labels (np.ndarray): Array of labels per triangle.

    Returns:
        MultiLineString: Seamline network.
    """
    segments = []
    for i, j, data in G.edges(data=True):
        if labels[i] != labels[j]:
            segments.append(data['geom'])
    return MultiLineString(segments)


def _save_seamlines(
    seams: MultiLineString,
    output_vector_path: str,
    crs,
) -> None:
    """Save the seamlines to a vector file."""
    gdf = gpd.GeoDataFrame(geometry=[seams], crs=crs)
    gdf.to_file(output_vector_path, driver="GPKG")


def _compute_overlap_polygon(footprints: List[Polygon]) -> Polygon:
    """Compute the union of all pairwise intersections (areas where ≥2 images overlap)."""
    overlaps = []
    for i in range(len(footprints)):
        for j in range(i+1, len(footprints)):
            inter = footprints[i].intersection(footprints[j])
            if not inter.is_empty:
                overlaps.append(inter)
    return unary_union(overlaps)


def _save_triangles(
    triangles: List[Polygon],
    output_path: str,
    crs,
) -> None:
    """Save triangle polygons to a GeoPackage.

    Args:
        triangles (List[Polygon]): List of triangle geometries.
        output_path (str): Path to the output GeoPackage file.
        crs: Coordinate reference system of the geometries.
    """
    gdf = gpd.GeoDataFrame(geometry=triangles, crs=crs)
    gdf.to_file(output_path, driver="GPKG", layer="triangles")


def markov_random_field_seamline(
    image_paths: List[str],
    output_vector_path: str,
    building_mask_paths: Optional[List[str]] = None,
    triangle_size: float = 500.0,
    lambda_param: float = 2.0,
) -> MultiLineString:
    """Generate a seamline network from a set of overlapping georeferenced images."""

    # Get CRS from first image
    with rasterio.open(image_paths[0]) as src:
        out_crs = src.crs
        print(f"[DEBUG] Loaded CRS: {out_crs}")

    # Load image footprints and optional building masks
    footprints = _load_image_footprints(image_paths)
    print(f"[DEBUG] Loaded {len(footprints)} image footprints.")

    masks = _load_building_masks(building_mask_paths) if building_mask_paths else None
    if masks:
        print(f"[DEBUG] Loaded {len(masks)} building masks.")

    # Compute overlapping region of all images
    overlap_poly = _compute_overlap_polygon(footprints)
    print(f"[DEBUG] Overlap polygon area: {overlap_poly.area}")
    if overlap_poly.is_empty:
        raise ValueError("No overlapping area between images!")

    # Generate triangle mesh within overlap
    triangles = _generate_triangulation(overlap_poly, triangle_size)
    print(f"[DEBUG] Generated {len(triangles)} triangles.")
    _save_triangles(triangles, '/Users/kanoalindiwe/Downloads/Projects/spectralmatch/docs/examples/data_worldview3/triangles.gpkg', out_crs)  # Reuse same path, different layer

    # Build adjacency graph of triangles
    G = _build_adjacency_graph(triangles)
    print(f"[DEBUG] Graph has {G.number_of_nodes()} nodes and {G.number_of_edges()} edges.")

    # Compute image visibility cost per triangle
    data_cost = _compute_data_cost(triangles, footprints)
    print(f"[DEBUG] Data cost shape: {data_cost.shape}")

    # Compute gradient magnitude maps and affine transforms
    grads, transforms = _compute_image_gradients(image_paths)
    print(f"[DEBUG] Loaded gradients for {len(grads)} images.")
    for i, grad in enumerate(grads):
        print(f"[DEBUG] Gradient {i} shape: {grad.shape}")

    # Compute edge weights for graph cut optimization
    edges, weights, smooth_mat = _compute_edge_weights(
        G, list(range(len(image_paths))), grads, transforms, lambda_param, masks)
    print(f"[DEBUG] Computed {len(edges)} edge weights.")
    print(f"[DEBUG] Smooth matrix shape: {smooth_mat.shape}")

    # Solve MRF to assign each triangle to an image
    labels = _solve_mrf(data_cost, smooth_mat, edges, weights)
    print(f"[DEBUG] MRF labeling done. Label count: {len(labels)}")

    # Extract seamline edges between differently labeled triangles
    seams = _extract_seamlines(G, triangles, labels)
    print(f"[DEBUG] Extracted {len(seams.geoms)} seamline segments.")

    # Save seamlines to vector file
    _save_seamlines(seams, output_vector_path, out_crs)
    print(f"[DEBUG] Saved seamlines to: {output_vector_path}")

    return seams