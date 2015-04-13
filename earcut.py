# coding: utf-8

__author__ = 'Alexander Soulimov (alexander.soulimov@gmail.com)'


def earcut(points, return_indices=False):
    """

    :param points:
    :param return_indices:
    :return:

    >>> earcut([[[10,0],[0,50],[60,60],[70,10]]])
    [[0, 50], [10, 0], [70, 10], [70, 10], [60, 60], [0, 50]]
    >>> earcut([[[10,0],[0,50],[60,60],[70,10]]], True)
    {'indices': [0, 1, 2, 2, 3, 0], 'vertices': [0, 50, 10, 0, 70, 10, 60, 60]}
    """
    outer_node = filter_points(create_linked_list(points[0], True))
    triangles = {'vertices': [], 'indices': []} if return_indices else []

    if outer_node is None:
        return triangles

    threshold = 80
    size = min_x = min_y = None

    for p in points:
        if threshold < 0:
            break
        threshold -= len(p)

    # if the shape is not too simple, we'll use z-order curve hash later; calculate polygon bbox
    if threshold < 0:
        node = outer_node.next
        min_x = max_x = node.p[0]
        min_y = max_y = node.p[1]
        while True:
            x = node.p[0]
            y = node.p[1]
            if x < min_x:
                min_x = x
            if y < min_y:
                min_y = y
            if x > max_x:
                max_x = x
            if y > max_y:
                max_y = y
            node = node.next
            if node == outer_node:
                break

        #  min_x, min_y and size are later used to transform coords into integers for z-order calculation
        size = max(max_x - min_x, max_y - min_y)

    if len(points) > 1:
        outer_node = eliminate_holes(points, outer_node)

    earcut_linked(outer_node, triangles, min_x, min_y, size)

    return triangles


def add_indexed_vertex(triangles, node):
    """

    :param triangles:
    :param node:
    :return:
    """
    if node.source:
        node = node.source

    i = node.index
    if i is None:
        dim = len(node.p)
        vertices = triangles.get("vertices")
        node.index = i = len(vertices) / dim
        for d in range(0, dim):
            vertices.append(node.p[d])

    triangles["indices"].append(i)


def index_curve(start, min_x, min_y, size):
    """
    Interlink polygon nodes in z-order

    :param start:
    :param min_x:
    :param min_y:
    :param size:
    :return:
    """
    node = start
    while True:
        if node.z is None:
            node.z = z_order(node.p[0], node.p[1], min_x, min_y, size)
        node.prevZ = node.prev
        node.nextZ = node.next
        node = node.next
        if node == start:
            break

    node.prevZ.nextZ = None
    node.prevZ = None

    sort_linked(node)


def sort_linked(linked_list):
    """
    Simon Tatham's linked list merge sort algorithm
    http://www.chiark.greenend.org.uk/~sgtatham/algorithms/listsort.html
    :param linked_list:
    :return:
    """
    in_size = 1

    while True:
        p = linked_list
        linked_list = None
        tail = None
        num_merges = 0

        while p:
            num_merges += 1
            q = p
            p_size = 0

            for i in range(in_size):
                p_size += 1
                q = q.nextZ
                if not q:
                    break

            q_size = in_size

            while p_size > 0 or (q_size > 0 and q):
                if p_size == 0:
                    e = q
                    q = q.nextZ
                    q_size -= 1
                elif q_size == 0 or not q:
                    e = p
                    p = p.nextZ
                    p_size -= 1
                elif p.z <= q.z:
                    e = p
                    p = p.nextZ
                    p_size -= 1
                else:
                    e = q
                    q = q.nextZ
                    q_size -= 1

                if tail:
                    tail.nextZ = e
                else:
                    linked_list = e

                e.prevZ = tail
                tail = e

            p = q

        tail.nextZ = None
        in_size *= 2

        if num_merges == 1:
            break
    return linked_list


def z_order(x, y, min_x, min_y, size):
    """
    Z-order of a point given coords and size of the data bounding box
    :param x:
    :param y:
    :param min_x:
    :param min_y:
    :param size:
    :return:
    """
    # coords are transformed into (0..1000) integer range
    x = 1000 * (x - min_x) / size
    x = (x | (x << 8)) & 0x00FF00FF
    x = (x | (x << 4)) & 0x0F0F0F0F
    x = (x | (x << 2)) & 0x33333333
    x = (x | (x << 1)) & 0x55555555

    y = 1000 * (y - min_y) / size
    y = (y | (y << 8)) & 0x00FF00FF
    y = (y | (y << 4)) & 0x0F0F0F0F
    y = (y | (y << 2)) & 0x33333333
    y = (y | (y << 1)) & 0x55555555

    return x | (y << 1)


def earcut_linked(ear, triangles, min_x, min_y, size, p=0):
    """

    :param ear:
    :param triangles:
    :param min_x:
    :param min_y:
    :param size:
    :param p:
    :return:
    """
    if ear is None:
        return

    indexed = type(triangles) == dict

    # interlink polygon nodes in z-order
    if not p and min_x is not None:
        index_curve(ear, min_x, min_y, size)

    stop = ear
    # iterate through ears, slicing them one by one
    while ear.prev != ear.next:
        prev = ear.prev
        next = ear.next

        if is_ear(ear, min_x, min_y, size):
            # cut off the triangle
            if indexed:
                add_indexed_vertex(triangles, prev)
                add_indexed_vertex(triangles, ear)
                add_indexed_vertex(triangles, next)
            else:
                triangles.append(prev.p)
                triangles.append(ear.p)
                triangles.append(next.p)

            # remove ear node
            next.prev = prev
            prev.next = next

            if ear.prevZ:
                ear.prevZ.nextZ = ear.nextZ
            if ear.nextZ:
                ear.nextZ.prevZ = ear.prevZ

            # skipping the next vertice leads to less sliver triangles
            ear = next.next
            stop = next.next
            continue

        ear = next
        # if we looped through the whole remaining polygon and can't find any more ears
        if ear == stop:
            # try filtering points and slicing again
            if not p:
                earcut_linked(filter_points(ear), triangles, min_x, min_y, size, 1)
            # if this didn't work, try curing all small self-intersections locally
            elif p == 1:
                ear = cure_local_intersections(ear, triangles)
                earcut_linked(ear, triangles, min_x, min_y, size, 2)

            # as a last resort, try splitting the remaining polygon into two
            elif p == 2:
                split_earcut(ear, triangles, min_x, min_y, size)
            break


def cure_local_intersections(start, triangles):
    """
    Go through all polygon nodes and cure small local self-intersections
    :param start:
    :param triangles:
    :return:
    """
    indexed = type(triangles) == dict
    node = start

    while True:
        a = node.prev
        b = node.next.next

        # a self-intersection where edge (v[i-1],v[i]) intersects (v[i+1],v[i+2])
        if a.p != b.p and intersects(a.p, node.p, node.next.p, b.p) and locally_inside(a, b) and locally_inside(b, a):
            if indexed:
                add_indexed_vertex(triangles, a)
                add_indexed_vertex(triangles, node)
                add_indexed_vertex(triangles, b)
            else:
                triangles.append(a.p)
                triangles.append(node.p)
                triangles.append(b.p)

            # remove two nodes involved
            a.next = b
            b.prev = a

            az = node.prevZ
            bz = node.nextZ and node.nextZ.nextZ

            if az:
                az.nextZ = bz

            if bz:
                bz.prevZ = az

            node = start = b

        node = node.next
        if node == start:
            break
    return node


def is_ear(ear, min_x, min_y, size):
    """
    Check whether a polygon node forms a valid ear with adjacent nodes
    :param ear:
    :param min_x:
    :param min_y:
    :param size:
    :return:
    """
    a = ear.prev.p
    b = ear.p
    c = ear.next.p

    ax = a[0]
    bx = b[0]
    cx = c[0]
    ay = a[1]
    by = b[1]
    cy = c[1]

    abd = ax * by - ay * bx
    acd = ax * cy - ay * cx
    cbd = cx * by - cy * bx
    A = abd - acd - cbd

    if A <= 0:
        return False  # reflex, can't be an ear

    # now make sure we don't have other points inside the potential ear;
    # the code below is a bit verbose and repetitive but this is done for performance

    cay = cy - ay
    acx = ax - cx
    aby = ay - by
    bax = bx - ax

    # if we use z-order curve hashing, iterate through the curve
    if min_x is not None:
        min_tx = (ax if ax < cx else cx) if ax < bx else (bx if bx < cx else cx)
        min_ty = (ay if ay < cy else cy) if ay < by else (by if by < cy else cy)
        max_tx = (ax if ax > cx else cx) if ax > bx else (bx if bx > cx else cx)
        max_ty = (ay if ay > cy else cy) if ay > by else (by if by > cy else cy)

        #  z-order range for the current triangle bbox
        min_z = z_order(min_tx, min_ty, min_x, min_y, size)
        max_z = z_order(max_tx, max_ty, min_x, min_y, size)

        # first look for points inside the triangle in increasing z-order
        node = ear.nextZ

        while node and node.z <= max_z:
            p = node.p
            node = node.nextZ
            if p == a or p == c:
                continue

            px = p[0]
            py = p[1]

            s = cay * px + acx * py - acd
            if s >= 0:
                t = aby * px + bax * py + abd
                if t >= 0:
                    k = A - s - t
                    if (k >= 0) and ((s and t) or (s and k) or (t and k)):
                        return False

        # then look for points in decreasing z-order
        node = ear.prevZ
        while node and node.z >= min_z:
            p = node.p
            node = node.prevZ
            if p == a or p == c:
                continue

            px = p[0]
            py = p[1]

            s = cay * px + acx * py - acd
            if s >= 0:
                t = aby * px + bax * py + abd
                if t >= 0:
                    k = A - s - t
                    if (k >= 0) and ((s and t) or (s and k) or (t and k)):
                        return False

    # if we don't use z-order curve hash, simply iterate through all other points
    else:
        node = ear.next.next
        while node != ear.prev:
            p = node.p
            node = node.next

            px = p[0]
            py = p[1]

            s = cay * px + acx * py - acd
            if s >= 0:
                t = aby * px + bax * py + abd
                if t >= 0:
                    k = A - s - t
                    if (k >= 0) and ((s and t) or (s and k) or (t and k)):
                        return False

    return True


def split_earcut(start, triangles, min_x, min_y, size):
    """
    Try splitting polygon into two and triangulate them independently
    :param start:
    :param triangles:
    :param min_x:
    :param min_y:
    :param size:
    :return:
    """
    # look for a valid diagonal that divides the polygon into two
    a = start
    while True:
        b = a.next.next
        while b != a.prev:
            if a.p != b.p and is_valid_diagonal(a, b):
                # split the polygon in two by the diagonal
                c = split_polygon(a, b)
                # filter colinear points around the cuts
                a = filter_points(a, a.next)
                c = filter_points(c, c.next)

                # run earcut on each half
                earcut_linked(a, triangles, min_x, min_y, size)
                earcut_linked(c, triangles, min_x, min_y, size)
                return
            b = b.next

        a = a.next
        if a == start:
            break


def is_valid_diagonal(a, b):
    """
    Check if a diagonal between two polygon nodes is valid (lies in polygon interior)

    :param a:
    :param b:
    :return:
    """
    return not intersects_polygon(a, a.p, b.p) and locally_inside(a, b) and locally_inside(b, a) and middle_inside(a, a.p, b.p)


def compare_x(a, b):
    return a.p[0] - b.p[0]


def get_leftmost(start):
    """
    Find the leftmost node of a polygon ring
    :param start:
    :return:
    """
    node = start
    leftmost = start

    while True:
        if node.p[0] < leftmost.p[0]:
            leftmost = node
        node = node.next
        if node == start:
            break
    return leftmost


def eliminate_holes(points, outer_node):
    """
    Link every hole into the outer loop, producing a single-ring polygon without holes

    :param points:
    :param outer_node:
    :return:
    """
    length = len(points)
    queue = list()

    for p in points:
        filtered_points_list = filter_points(create_linked_list(p, False))
        if filtered_points_list:
            queue.append(get_leftmost(filtered_points_list))

    queue.sort(compare_x)

    # process holes from left to right
    for q in queue:
        eliminate_hole(q, outer_node)
        outer_node = filter_points(outer_node, outer_node.next)

    return outer_node


def eliminate_hole(hole_node, outer_node):
    """
    Find a bridge between vertices that connects hole with an outer ring and and link it

    :param hole_node:
    :param outer_node:
    :return:
    """
    outer_node = find_hole_bridge(hole_node, outer_node)
    if outer_node:
        b = split_polygon(outer_node, hole_node)
        filter_points(b, b.next)


def find_hole_bridge(hole_node, outer_node):
    """
    David Eberly's algorithm for finding a bridge between hole and outer polygon
    :param hole_node:
    :param outer_node:
    :return:
    """
    node = outer_node
    p = hole_node.p
    px = p[0]
    py = p[1]
    q_max = -float('inf')
    m_node = None

    # find a segment intersected by a ray from the hole's leftmost point to the left;
    # segment's endpoint with lesser x will be potential connection point

    while True:
        a = node.p
        b = node.next.p

        if a[1] >= py >= b[1]:
            qx = a[0] + (py - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
            if px >= qx > q_max:
                q_max = qx
                m_node = node if a[0] < b[0] else node.next
        node = node.next
        if node == outer_node:
            break

    if m_node is None:
        return None

    # look for points strictly inside the triangle of hole point, segment intersection and endpoint;
    # if there are no points found, we have a valid connection;
    # otherwise choose the point of the minimum angle with the ray as connection point
    bx = m_node.p[0]
    by = m_node.p[1]
    pbd = px * by - py * bx
    pcd = px * py - py * q_max
    cpy = py - py
    pcx = px - q_max
    pby = py - by
    bpx = bx - px
    a_cap = pbd - pcd - (q_max * by - py * bx)
    sign = -1 if a_cap <= 0 else 1,
    stop = m_node,
    tan_min = float('inf')

    node = m_node.next

    while node != stop:
        mx = node.p[0]
        my = node.p[1]
        amx = px - mx

        if amx >= 0 and mx >= bx:
            s = (cpy * mx + pcx * my - pcd) * sign
            if s >= 0:
                t = (pby * mx + bpx * my + pbd) * sign

                if t >= 0 and (a_cap * sign - s - t >= 0):
                    tan = abs(py - my) / amx  # tangential
                    if tan < tan_min and locally_inside(node, hole_node):
                        m_node = node
                        tan_min = tan

        node = node.next

    return m_node


def intersects(p1, q1, p2, q2):
    """
    Check if two segments intersect
    :param p1:
    :param q1:
    :param p2:
    :param q2:
    :return:
    """
    return orient(p1, q1, p2) != orient(p1, q1, q2) and orient(p2, q2, p1) != orient(p2, q2, q1)


def intersects_polygon(start, a, b):
    """
    Check if a polygon diagonal intersects any polygon segments
    :param start:
    :param a:
    :param b:
    :return:
    """
    node = start
    while True:
        p1 = node.p
        p2 = node.next.p

        if p1 != a and p2 != a and p1 != b and p2 != b and intersects(p1, p2, a, b):
            return True

        node = node.next
        if node == start:
            break

    return False


def locally_inside(a, b):
    """
    Сheck if a polygon diagonal is locally inside the polygon
    :param a:
    :param b:
    :return:
    """
    return orient(a.p, b.p, a.next.p) != -1 and orient(a.p, a.prev.p, b.p) != -1 if orient(a.prev.p, a.p, a.next.p) == -1 else orient(a.p, b.p, a.prev.p) == -1 or orient(a.p, a.next.p, b.p) == -1


def middle_inside(start, a, b):
    """
    Check if the middle point of a polygon diagonal is inside the polygon
    :param start:
    :param a:
    :param b:
    :return:
    """
    node = start
    inside = False
    px = (a[0] + b[0]) / 2
    py = (a[1] + b[1]) / 2

    while True:
        p1 = node.p
        p2 = node.next.p

        if ((p1[1] > py) != (p2[1] > py)) and (px < (p2[0] - p1[0]) * (py - p1[1]) / (p2[1] - p1[1]) + p1[0]):
            inside = not inside

        node = node.next
        if node == start:
            break

    return inside


def split_polygon(a, b):
    """
    Link two polygon vertices with a bridge; if the vertices belong to the same ring, it splits polygon into two;
    if one belongs to the outer ring and another to a hole, it merges it into a single ring

    :param a:
    :param b:
    :return:
    """
    a2 = Node(a.p)
    b2 = Node(b.p)
    an = a.next
    bp = b.prev

    a2.source = a
    b2.source = b

    a.next = b
    b.prev = a

    a2.next = an
    an.prev = a2

    b2.next = a2
    a2.prev = b2

    bp.next = b2
    b2.prev = bp

    return b2


class Node(object):
    def __init__(self, point):
        # vertex coordinates
        self.p = point

        # previous and next vertice nodes in a polygon ring
        self.prev = None
        self.next = None

        # z-order curve value
        self.z = None

        # previous and next nodes in z-order
        self.prevZ = None
        self.nextZ = None

        # used for indexed output
        self.source = None
        self.index = None


def insert_node(point, last):
    """
    Create a node and optionally link it with previous one (in a circular doubly linked list)

    :param point:
    :param last:
    :return:
    """
    node = Node(point)

    if last is None:
        node.prev = node
        node.next = node
    else:
        node.next = last.next
        node.prev = last
        last.next.prev = node
        last.next = node

    return node


def create_linked_list(points, clockwise):
    """
    Create a circular doubly linked list from polygon points in the specified winding order

    :param points:
    :param clockwise:
    :return:
    """
    coord_sum = 0
    length = len(points)
    last = None

    # calculate original winding order of a polygon ring
    for i, j in zip(range(length), [length-1] + range(length-1)):
        p1 = points[i]
        p2 = points[j]
        coord_sum += (p2[0] - p1[0]) * (p1[1] + p2[1])

    # link points into circular doubly-linked list in the specified winding order
    points_range = range(length) if clockwise == (coord_sum > 0) else range(length-1, -1, -1)
    for i in points_range:
        last = insert_node(points[i], last)

    return last


def orient(p, q, r):
    """
    Winding order of triangle formed by 3 given points

    :param p:
    :param q:
    :param r:
    :return:
    """
    o = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    return 1 if o > 0 else -1 if o < 0 else 0


def equals(point1, point2):
    """
    Check if two points are equal

    :param point1:
    :param point2:
    :return:
    """
    return point1[0] == point2[0] and point1[1] == point2[1]


def filter_points(start, end=None):
    """
    Eliminate colinear or duplicate points

    :param start:
    :param end:
    :return:
    """
    if end is None:
        end = start

    node = start
    again = False
    while True:
        again = False

        if equals(node.p, node.next.p) or orient(node.prev.p, node.p, node.next.p) == 0:
            # remove node
            node.prev.next = node.next
            node.next.prev = node.prev

            if node.prevZ:
                node.prevZ.nextZ = node.nextZ
            if node.nextZ:
                node.nextZ.prevZ = node.prevZ

            node = end = node.prev
            if node == node.next:
                return None
            again = True
        else:
            node = node.next

        if not again and node == end:
            break

    return end
