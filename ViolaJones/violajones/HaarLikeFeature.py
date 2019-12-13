import violajones.IntegralImage as IImg


def enum(**enums):
    return type('Enum', (), enums)


FeatureType = enum(TWO_HORIZONTAL=(
    2, 1), THREE_HORIZONTAL=(3, 1), TWO_VERTICAL=(1, 2), THREE_VERTICAL=(1, 3), FOUR=(2, 2))
FeatureTypes = [FeatureType.TWO_VERTICAL, FeatureType.TWO_HORIZONTAL,
                FeatureType.THREE_VERTICAL, FeatureType.THREE_HORIZONTAL, FeatureType.FOUR]


class HaarLikeFeature(object):
    def __init__(self, featureType, topLeft, width, height, threshold, polarity):
        self.featureType = featureType
        self.topLeft = topLeft
        self.width = width
        self.height = height
        self.threshold = threshold
        self.polarity = polarity
        self.bottomRight = (topLeft[0] + width, topLeft[1] + height)
        self.weight = 1

    def __repr__(self):
        return "HaarLikeFeature()"

    def __str__(self):
        return "{} {} {} {} {} {}".format(self.featureType, self.topLeft, self.width, self.height, self.threshold, self.polarity)

    def get_score(self, integralImg):
        score = 0
        if self.featureType == FeatureType.TWO_VERTICAL:
            # left part of the feature
            leftPart = IImg.sum_region(integralImg, self.topLeft, (
                self.topLeft[0] + self.width, int(self.topLeft[1] + self.height / 2)))
            # right part of the feature
            rightPart = IImg.sum_region(integralImg, (self.topLeft[0], int(
                self.topLeft[1] + self.height / 2)), self.bottomRight)
            score = leftPart - rightPart
        elif self.featureType == FeatureType.TWO_HORIZONTAL:
            # top part of the feature
            topPart = IImg.sum_region(integralImg, self.topLeft, (int(
                self.topLeft[0] + self.width / 2), self.topLeft[1] + self.height))
            # bottom part of the feature
            bottomPart = IImg.sum_region(integralImg, (int(
                self.topLeft[0] + self.width / 2), self.topLeft[1]), self.bottomRight)
            score = topPart - bottomPart
        elif self.featureType == FeatureType.THREE_HORIZONTAL:
            # top part of the feature
            topPart = IImg.sum_region(integralImg, self.topLeft, (int(
                self.topLeft[0] + self.width / 3), self.topLeft[1] + self.height))
            # middle part of the feature
            middlePart = IImg.sum_region(integralImg, (int(self.topLeft[0] + self.width / 3), self.topLeft[1]), (int(
                self.topLeft[0] + 2 * self.width / 3), self.topLeft[1] + self.height))
            # bottom part of the feature
            bottomPart = IImg.sum_region(integralImg, (int(
                self.topLeft[0] + 2 * self.width / 3), self.topLeft[1]), self.bottomRight)
            score = topPart - middlePart + bottomPart
        elif self.featureType == FeatureType.THREE_VERTICAL:
            # left part of the feature
            leftPart = IImg.sum_region(integralImg, self.topLeft, (self.bottomRight[0], int(
                self.topLeft[1] + self.height / 3)))
            # middle part of the feature
            middlePart = IImg.sum_region(integralImg, (self.topLeft[0], int(
                self.topLeft[1] + self.height / 3)), (self.bottomRight[0], int(self.topLeft[1] + 2 * self.height / 3)))
            # right part of the feature
            rightPart = IImg.sum_region(integralImg, (self.topLeft[0], int(
                self.topLeft[1] + 2 * self.height / 3)), self.bottomRight)
            score = leftPart - middlePart + rightPart
        elif self.featureType == FeatureType.FOUR:
            # top left part of the feature
            topLeftPart = IImg.sum_region(integralImg, self.topLeft, (int(
                self.topLeft[0] + self.width / 2), int(self.topLeft[1] + self.height / 2)))
            # top right part of the feature
            topRightPart = IImg.sum_region(integralImg, (int(
                self.topLeft[0] + self.width / 2), self.topLeft[1]), (self.bottomRight[0], int(self.topLeft[1] + self.height / 2)))
            # bottom left part of the feature
            bottomLeftPart = IImg.sum_region(integralImg, (self.topLeft[0], int(
                self.topLeft[1] + self.height / 2)), (int(self.topLeft[0] + self.width / 2), self.bottomRight[1]))
            # bottom right part of the feature
            bottomRightPart = IImg.sum_region(integralImg, (int(self.topLeft[0] + self.width / 2), int(
                self.topLeft[1] + self.height / 2)), self.bottomRight)
            score = topLeftPart - topRightPart - bottomLeftPart + bottomRightPart
        return score

    def get_vote(self, integralImg):
        if self.get_score(integralImg) < self.polarity * self.threshold:
            return self.weight
        else:
            return -1 * self.weight


# index 3 4 5 6
# 1*2 height width width width
# 1*3 height width width width
# 2*1 height heigth heigth width
# 3*1 height heigth heigth width
# 2*2 height heigth width width
