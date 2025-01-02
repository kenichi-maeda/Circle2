import numpy as np
from itertools import combinations
import sys
import json
import csv

def is_inside_circle(circle, point):
    """Checks if a point is inside a circle."""
    center, radius = circle
    dist = np.linalg.norm(np.array(point) - np.array(center))
    return dist < radius

def circle_from_points(a, b, c, tolerance=1e-9):
    """Creates a circle from three points."""
    A, B, C = np.array(a), np.array(b), np.array(c)
    AB_mid, BC_mid = (A + B) / 2, (B + C) / 2
    AB_slope = -(A[0] - B[0]) / (A[1] - B[1] + tolerance) if abs(A[1] - B[1]) > tolerance else None
    BC_slope = -(B[0] - C[0]) / (B[1] - C[1] + tolerance) if abs(B[1] - C[1]) > tolerance else None

    if AB_slope is not None and BC_slope is not None:
        A_intercept = AB_mid[1] - AB_slope * AB_mid[0]
        B_intercept = BC_mid[1] - BC_slope * BC_mid[0]

        if abs(AB_slope - BC_slope) < tolerance:
            return None  # We cannot form a circle

        x_center = (B_intercept - A_intercept) / (AB_slope - BC_slope)
        y_center = AB_slope * x_center + A_intercept
    else:
        return None

    center = (x_center, y_center)
    radius = np.linalg.norm(A - np.array(center))
    return center, radius

def generate_circles(points):
    """Generates exactly 4 circles from the combinations of the points."""
    circles = []

    for comb in combinations(points, 3):
        circle = circle_from_points(comb[0], comb[1], comb[2])
        if circle is None:
            continue

        # Remaining points
        remaining_points = [p for p in points if p not in comb]

        # Count how many points are inside and outside
        inside_points = [p for p in remaining_points if is_inside_circle(circle, p)]
        outside_points = [p for p in remaining_points if not is_inside_circle(circle, p)] 

        # Check if there is exactly one point inside and one point outside
        if len(inside_points) == 1 and len(outside_points) == 1:
            center, radius = circle
            x = center[0]
            y = center[1]
            circles.append((x, y, radius))

    return circles

# Randomly generate 5 points within a 100x100 canvas
n = 100000
csv_filename = 'circles_data.csv'

with open(csv_filename, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)

    # Write header
    header = [
        'x1', 'y1', 'x2', 'y2', 'x3', 'y3', 'x4', 'y4', 'x5', 'y5',
        'X1', 'Y1', 'R1', 'X2', 'Y2', 'R2', 'X3', 'Y3', 'R3', 'X4', 'Y4', 'R4'
    ]
    csv_writer.writerow(header)


    for _ in range(n):
        points = np.random.uniform(0, 100, size=(5, 2)).tolist()

        # Generate 4 circles
        circles = generate_circles(points)

        # Ensure we have exactly 4 circles
        if len(circles) < 4:
            sys.exit("4 circles not found!!!!!")

        # Sort circles by radius in ascending order
        circles.sort(key=lambda c: c[2])

        # Write the points and circle data to the CSV
        row = []
        for p in points:
            row.extend(p)
        for c in circles:
            row.extend([c[0], c[1], c[2]])
        csv_writer.writerow(row)

    print("Finished!")
