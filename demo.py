# -*- coding: utf-8 -*-
class Vector:
	def __init__(self,x,y):
		self.x = x
		self.y = y
		
	def __add__(self,other):
		return Vector(self.x + other.x, self.y + other.y)
			
	def __str__(self):
		return "Vector : " + str(self.x)+ " "+ str(self.y)

p1 = Vector(10,20)
p2 = Vector(2,3)
p3 = p1 + p2		
print(p3)

