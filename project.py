def rect_to_bb(rect):
    # take a bounding predicted by dlib and convert it
    # to the format (x, y, w, h)
    x = rect.left()
    y = rect.top()
    width = rect.right() - x
    height = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, width, height)
