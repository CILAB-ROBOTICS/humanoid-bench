import os


IMAGE_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'hand.png'))

VERTICES = {
    'tactile_lh_palm': [(128, 539), (329, 540), (330, 475), (343, 459), (342, 340), (297, 335), (297, 372), (285, 361), (286, 311), (240, 306), (228, 309), (183, 306), (175, 312), (125, 312), (130, 377), (172, 412), (185, 441), (128, 473)],
    'tactile_lh_ffproximal': [(127, 279), (176, 277), (175, 173), (127, 173)],
    'tactile_lh_ffmiddle': [(128, 155), (173, 155), (172, 96), (127, 100)],
    'tactile_lh_ffdistal': [(128, 74), (175, 75), (167, 19), (138, 14)],
    'tactile_lh_mfproximal': [(182, 270), (236, 270), (230, 171), (185, 165)],
    'tactile_lh_mfmiddle': [(185, 139), (233, 138), (227, 68), (183, 73)],
    'tactile_lh_mfdistal': [(188, 61), (225, 62), (218, 7), (194, 7)],
    'tactile_lh_rfproximal': [(239, 282), (285, 283), (282, 167), (243, 168)],
    'tactile_lh_rfmiddle': [(244, 145), (284, 148), (285, 77), (244, 78)],
    'tactile_lh_rfdistal': [(244, 65), (282, 70), (275, 13), (250, 14)],
    'tactile_lh_lfmetacarpal': [(291, 380), (343, 446), (343, 316), (299, 319)],
    'tactile_lh_lfproximal': [(297, 303), (344, 305), (342, 199), (299, 194)],
    'tactile_lh_lfmiddle': [(299, 171), (340, 170), (340, 118), (297, 115)],
    'tactile_lh_lfdistal': [(299, 96), (340, 97), (334, 38), (305, 38)],
    'tactile_lh_thproximal': [(127, 460), (173, 415), (104, 347), (60, 383)],
    'tactile_lh_thmiddle': [(60, 379), (104, 348), (64, 284), (27, 308)],
    'tactile_lh_thdistal': [(24, 292), (67, 263), (44, 215), (11, 225)],

    'tactile_rh_ffproximal': [(729, 274), (780, 273), (776, 173), (734, 177)],
    'tactile_rh_ffmiddle': [(732, 147), (773, 147), (773, 102), (734, 103)],
    'tactile_rh_ffdistal': [(731, 80), (774, 75), (768, 17), (744, 13)],
    'tactile_rh_mfproximal': [(671, 269), (720, 269), (716, 167), (677, 163)],
    'tactile_rh_mfmiddle': [(675, 131), (719, 131), (716, 78), (675, 78)],
    'tactile_rh_mfdistal': [(675, 65), (719, 65), (707, 0), (680, 3)],
    'tactile_rh_rfproximal': [(619, 273), (667, 273), (659, 174), (619, 173)],
    'tactile_rh_rfmiddle': [(619, 148), (664, 154), (661, 94), (619, 94)],
    'tactile_rh_rfdistal': [(620, 75), (662, 77), (655, 13), (626, 14)],
    'tactile_rh_lfmetacarpal': [(564, 430), (612, 377), (606, 290), (559, 289)],
    'tactile_rh_lfproximal': [(559, 289), (609, 290), (606, 197), (564, 196)],
    'tactile_rh_lfmiddle': [(562, 171), (609, 168), (603, 119), (564, 116)],
    'tactile_rh_lfdistal': [(564, 96), (603, 97), (597, 41), (571, 36)],
    'tactile_rh_thproximal': [(722, 438), (764, 469), (838, 392), (796, 357)],
    'tactile_rh_thmiddle': [(800, 354), (838, 386), (876, 312), (838, 290)],
    'tactile_rh_thdistal': [(841, 277), (882, 299), (892, 216), (864, 210)],
}


import numpy as np
def subdivide_quad(corners, H, W):
    """
    corners: list of four (x,y) points in order TL, TR, BR, BL
    H, W: number of rows and columns
    return: list of H*W cells, each cell is [tl, tr, br, bl]
    """
    TL, TR, BR, BL = [np.array(p) for p in corners]
    # interpolate left and right edges into H+1 points
    left_pts  = [TL + (BL - TL) * (i / H) for i in range(H + 1)]
    right_pts = [TR + (BR - TR) * (i / H) for i in range(H + 1)]
    cells = []
    for i in range(H):
        topL, botL = left_pts[i],  left_pts[i+1]
        topR, botR = right_pts[i], right_pts[i+1]
        for j in range(W):
            u0, u1 = j / W, (j + 1) / W
            tl = topL + (topR - topL) * u0
            tr = topL + (topR - topL) * u1
            bl = botL + (botR - botL) * u0
            br = botL + (botR - botL) * u1
            cells.append([tl, tr, br, bl])
    return cells


if __name__ == '__main__':
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from matplotlib.collections import PatchCollection

    # Load background image
    img = plt.imread('hand.png')
    h, w = img.shape[:2]

    # Prepare figure and axes
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.imshow(img)
    ax.axis('off')

    # Create PatchCollections for each tactile region
    collections = {}
    for name, verts in VERTICES.items():
        # Determine grid size: torso uses 4×8, others use 2×4
        if 'torso' in name:
            H, W = 4, 8
        else:
            H, W = 2, 4

        # Approximate region as quad: use bounding box if polygon has more than 4 points
        if len(verts) == 4:
            quad = verts
        else:
            xs, ys = zip(*verts)
            quad = [
                (min(xs), min(ys)),  # TL
                (max(xs), min(ys)),  # TR
                (max(xs), max(ys)),  # BR
                (min(xs), max(ys)),  # BL
            ]

        # Subdivide quad into H×W cells and create polygons
        cell_quads = subdivide_quad(quad, H, W)
        patches = [Polygon(c, closed=True) for c in cell_quads]
        coll = PatchCollection(patches, match_original=True)
        coll.set_facecolor((0, 0, 0, 0))  # initial fully transparent
        ax.add_collection(coll)
        collections[name] = (coll, H, W)

    plt.ion()
    plt.show()

    # Visualization loop (example with random data)
    MAX_PRESSURE = 5.0  # scale factor for normalization

    # for _ in range(100):  # replace with actual simulation steps
    while True:
        # Simulate touch sensor data as random
        touch_obs = {
            name: np.random.rand(3, H, W) * MAX_PRESSURE
            for name, (_, H, W) in collections.items()
        }

        # Update PatchCollection for each region
        for name, (coll, H, W) in collections.items():
            grid3 = touch_obs[name]
            # Use the absolute z-component as intensity
            vals = np.abs(grid3[2]).flatten()
            norm = plt.Normalize(0, MAX_PRESSURE)
            colors = plt.cm.jet(norm(vals))  # RGBA array length H*W
            coll.set_facecolor(colors)

        # Refresh display
        fig.canvas.draw_idle()
        fig.canvas.flush_events()
        plt.pause(0.05)

