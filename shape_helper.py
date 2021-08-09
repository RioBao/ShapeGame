import math
import cv2
import numpy as np
import random

# We only need x and z coordinates, y is irrelevant in a top down scene vizaulisation
class CornerPoint:
    def __init__(self, x, z):
        self.coord = (x,z)
        self.hit_list = [False] #initialize with False to make sure 5 hits are needed for a corner detection

    def hit(self):
        self.hit_list.append(True)
        self.hit_list = self.hit_list[-5:] # only last 5 items  
    def miss(self):
        self.hit_list.append(False)
        self.hit_list = self.hit_list[-5:] # only last 5 items

    def correct_position(self):
        return all(self.hit_list)


class Shape:
    def __init__(self, name):
        self.numCorners = 0
        self.name = name
        self.cornerPoints = []
        self.imagePath = ""

    def add_corner(self,x,z):
        self.cornerPoints.append(CornerPoint(x,z))
        self.numCorners = self.numCorners+1
 
    def all_positions_correct(self):
        for corner in self.cornerPoints:
            if not corner.correct_position():
                return False
        return True

    def cornerPoints_to_array(self):
        a = np.zeros((self.numCorners,2))
        for itt in range(0,self.numCorners):
            #print(self.cornerPoints[itt].coord)
            a[itt] = self.cornerPoints[itt].coord
        return a

    def get_polyLines(self):
        a = self.cornerPoints_to_array()
        return a.reshape((1,-1,2)).astype(int) # (-1,1,2)


class Square(Shape):
    def __init__(self, videoWidth):
        super(Square, self).__init__('Square')  #The init in this class overrides the one in the parent class, so use super to call it
        self.add_corner(videoWidth + 50,280)  # Add corners in clockwise direction, starting top left
        self.add_corner(videoWidth + 230,280)
        self.add_corner(videoWidth + 230,120)
        self.add_corner(videoWidth + 50,120)
        self.imagePath = "square.jpg"

class Triangle1(Shape):
    def __init__(self, videoWidth):
        super(Triangle1, self).__init__('Triangle1')
        self.add_corner(videoWidth + 320,480)
        self.add_corner(videoWidth + 25, 480)
        self.add_corner(videoWidth + 170,175)
        self.imagePath = "triangle1.jpg"


class Triangle2(Shape):
    def __init__(self, videoWidth):
        super(Triangle2, self).__init__('Triangle1')
        self.add_corner(videoWidth + 280,260)
        self.add_corner(videoWidth + 70, 260)
        self.add_corner(videoWidth + 170,530)
        self.imagePath = "triangle2.jpg"


def get_random_shape(name, videoWidth):
    #allClasses = {'Square','Triangle1','Triangle2'}
    allClasses = ['Triangle1','Triangle1'] # Todo: Should use number of players to select appropriate shapes
    rand = random.choice(allClasses)
    if rand == 'Square':
        return Square(videoWidth)
    elif rand == 'Triangle1':
        return Triangle1(videoWidth)
    elif rand == 'Triangle2':
        return Triangle2(videoWidth)
    else:
        return []
