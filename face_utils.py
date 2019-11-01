def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h)
    x = rect.left()
    y = rect.top()
    width = rect.right() - x
    height = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, width, height)

# dlib return shape object 68 points in the face
#  convert this object to a NumPy array


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)

    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)

    # return the list of (x, y)-coordinates
    return coords
