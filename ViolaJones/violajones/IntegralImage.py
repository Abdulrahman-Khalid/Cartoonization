import numpy as np


def get_integral_image(imgArr):
    rowAccumelate = np.zeros(imgArr.shape)
    integralImg = np.zeros((imgArr.shape[0] + 1, imgArr.shape[1] + 1))
    for y in range(imgArr.shape[1]):
        for x in range(imgArr.shape[0]):
            rowAccumelate[x, y] = rowAccumelate[x-1, y] + imgArr[x, y]
            integralImg[x + 1, y + 1] = integralImg[x + 1, y] + \
                rowAccumelate[x, y]
    return integralImg


def sum_region(integralImg, topLeftCorner, bottomRightCorner):
    topLeftCorner = (topLeftCorner[1], topLeftCorner[0])
    bottomRightCorner = (bottomRightCorner[1], bottomRightCorner[0])
    topRightCorner = (bottomRightCorner[0], topLeftCorner[1])
    BottomLeftCorner = (topLeftCorner[0], bottomRightCorner[1])
    return integralImg[bottomRightCorner] + integralImg[topLeftCorner] - integralImg[topRightCorner] - integralImg[BottomLeftCorner]


# Test get_integral_image
# R = int(input("Enter the number of rows:"))
# C = int(input("Enter the number of columns:"))
# matrix = []
# print("Enter the entries rowwise:")
# for i in range(R):
#     a = []
#     for j in range(C):
#         a.append(int(input()))
#     matrix.append(a)
# matrix = np.array(matrix)
# integralMatrix = get_integral_image(matrix)
# for i in range(R+1):
#     for j in range(C+1):
#         print(integralMatrix[i, j], end=" ")
#     print()

# print("sum region:", sum_region(integralMatrix, (0, 0), (3, 3)))
