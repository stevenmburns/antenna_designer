import numpy as np

class Transform:
  def __init__(self, A=np.eye(4)):
    self.A = A

  @staticmethod
  def inverse(tr):
      return Transform(np.linalg.inv(tr.A))

  @staticmethod
  def translate(x, y, z):
      return Transform(
          np.array([[1, 0, 0, x],
                    [0, 1, 0, y],
                    [0, 0, 1, z],
                    [0, 0, 0, 1]])
      )


  @staticmethod
  def rotX(theta_degrees):
      c = np.cos(np.pi/180*theta_degrees)
      s = np.sin(np.pi/180*theta_degrees)
      return Transform(
          np.array([[1, 0,  0, 0],
                    [0, c, -s, 0],
                    [0, s,  c, 0],
                    [0, 0,  0, 1]])
      )

  @staticmethod
  def rotY(theta_degrees):
      c = np.cos(np.pi/180*theta_degrees)
      s = np.sin(np.pi/180*theta_degrees)
      return Transform(
          np.array([[ c, 0,  s, 0],
                    [ 0, 1,  0, 0],
                    [-s, 0,  c, 0],
                    [ 0, 0,  0, 1]])
      )

  @staticmethod
  def rotZ(theta_degrees):
      c = np.cos(np.pi/180*theta_degrees)
      s = np.sin(np.pi/180*theta_degrees)
      return Transform(
          np.array([[c, -s, 0, 0],
                    [s,  c, 0, 0],
                    [0,  0, 1, 0],
                    [0,  0, 0, 1]])
      )

  def hit(self, coords):
    v = np.array( coords + (1,))
    V = self.A.dot(v)
    return V[0], V[1], V[2]

  def premult(self, other):
      return Transform(other.A @ self.A)

  def postmult(self, other):
      return Transform(self.A @ other.A)

class TransformStack:
  def __init__(self):
    self.stack = [Transform()]

  def push(self, tr):
    self.stack.append(self.stack[-1].postmult(tr))

  def pop(self):
    self.stack.pop()

  def hit(self, v):
    return self.stack[-1].hit(v)
