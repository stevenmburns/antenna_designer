import numpy as np
from antenna_designer import Transform, TransformStack

def test_hit():
  tr = Transform()
  coords = (1, 2, 3)
  assert coords == tr.hit(coords)

def test_translate():
    tr0 = Transform.translate(1,1,1)
    tr1 = Transform.translate(-1,-1,-1)
    print(tr0.A)
    print(tr1.A)

    assert np.allclose(tr0.premult(tr1).A, Transform().A)
    assert np.allclose(tr0.postmult(tr1).A, Transform().A)

def test_rotate():
    for rot in (Transform.rotX, Transform.rotY, Transform.rotZ):

        tr0 = rot(30)
        tr1 = rot(-30)
        tr2 = Transform.inverse(tr0)
        print(tr0.A)
        print(tr1.A)
        print(tr2.A)

        assert np.allclose(tr0.premult(tr1).A, Transform().A)
        assert np.allclose(tr0.postmult(tr1).A, Transform().A)
    
        assert np.allclose(tr1.A, tr2.A)

def test_stack():
  st = TransformStack()

  st.push(Transform.translate(1,1,1))
  st.push(Transform.translate(-1,-1,-1))

  coords = (1, 2, 3)
  assert coords == st.hit(coords)

  assert np.allclose(st.stack[-1].A, Transform().A)
