import numpy as np
from icecream import ic

from .abstract_pysim import AbstractPySim

from .spline import NaturalSpline # , PiecewiseLinear

class SplinePySim(AbstractPySim):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_impedance(self, *, ntrap=0, engine='python', N=20):

        y0, y1 = np.float64(0), np.float64(2*self.halfdriver)

        p0, p1 = np.array((0, y0, 0),dtype=np.float64), np.array((0, y1, 0),dtype=np.float64)

        spl = NaturalSpline(N=N)
        #spl = PiecewiseLinear(N=N)
        spl.gen_constraint(midderivs_free=True)
        spl.gen_Vandermonde(nsegs=self.nsegs*ntrap)
        spl.gen_deriv_ops()

        n_start = p0
        n_end   = p1
        total_length = np.sqrt((n_end-n_start)**2).sum(axis=0)
        ic(total_length)

        xs = np.linspace(n_start, n_end, self.nsegs*ntrap+1)
        #ic(xs)

        m_endpoints = np.linspace(n_start, n_end, self.nsegs)
        m_centers = 0.5*(m_endpoints[1:, :] + m_endpoints[:-1, :])
        m_delta = m_endpoints[1:, :] - m_endpoints[:-1, :]

        # m are the test points (can be -1, 0, or +1) because we need to take numerical derivatives of the voltage
        # n are the integration intervals (centers on 1/2 points) tied to the spline

        def Spline_Integral_Standalone(m, *, ntrap, wire_radius, k, use_deriv):
            assert ntrap != 0

            m_vec_delta = m[1:, :] - m[:-1, :]
            delta = np.sqrt((m_vec_delta*m_vec_delta).sum(axis=1))
            assert m.shape[0] - 1 == delta.shape[0]


            if use_deriv:
                eval_mat = spl.Vandermonde @ (N/self.nsegs*spl.deriv_op) @ spl.S
            else:
                eval_mat = spl.Vandermonde @ spl.S
            ic(eval_mat.shape)

            def Aux(i):
                local_n = xs[i:len(xs)-ntrap+i:ntrap,:]

                ic(i, local_n.shape, m.shape)

                diffs = local_n[np.newaxis, :, :] - m[:, np.newaxis, :]
                R = np.sqrt((diffs*diffs).sum(axis=2))

                ic(R.shape)

                # not always diagonal indices
                diag_indices = np.where(R < 0.00001)
                ic(diag_indices[0].shape, diag_indices[1].shape, diag_indices[0], delta.shape)

                #new_delta = delta[diag_indices[0]]
                #hack assuming all deltas are identical
                new_delta = delta[0]

                RR = R
                RR[diag_indices] = 1

                local_res = np.exp(-(0+1j)*k*R)/(4*np.pi*RR)
                diag = 1/(2*np.pi*new_delta) * np.log(new_delta/wire_radius) - (0+1j)*k/(4*np.pi) 
                local_res[diag_indices] = diag

                ic(local_res.shape)

                restricted_eval_mat = eval_mat[i:len(xs)-ntrap+i:ntrap,:] 
                ic(restricted_eval_mat.shape)

                tmp = local_res @ restricted_eval_mat
                ic(tmp.shape)
                return tmp

            res = np.zeros(shape=(m.shape[0], spl.S.shape[1]),dtype=np.complex128)
            ic(res.shape)
            for i in range(0, ntrap+1):
                coeff = (2 if i > 0 and i < ntrap else 1)/(2*ntrap)
                res += coeff * Aux(i)


            return res


        def Integral(m, ntrap, use_deriv):
            return Spline_Integral_Standalone(m, ntrap=ntrap, wire_radius=self.wire_radius, k=self.k, use_deriv=use_deriv)

        #z = self.jomega * self.mu * (vec_delta_l[np.newaxis, :, :] * vec_delta_l[:, np.newaxis, :]).sum(axis=2)
        z = self.jomega * self.mu * (m_delta[0, :] * m_delta[0, :]).sum(axis=0)

        z *= Integral(m_centers, ntrap=ntrap, use_deriv=False)

        s = 1/(self.jomega*self.eps) * Integral(m_endpoints, ntrap=ntrap, use_deriv=True)

        z += s[1:, :] - s[:-1, :]

        self.z = z

        ic(z.shape, self.nsegs)

        v = np.zeros(shape=(self.z.shape[0],), dtype=np.complex128)
        v[self.driver_seg_idx] = 1

        i = spl.Vandermonde @ spl.S @ spl.pseudo_solve(self.z, v)

        i_driver = i[i.shape[0]//2]
        #i_driver = i[self.driver_seg_idx]
        ic(i.shape)

        driver_impedance = v[self.driver_seg_idx]/i_driver
        ic(np.abs(driver_impedance), np.angle(driver_impedance)*180/np.pi)

        return driver_impedance, i
