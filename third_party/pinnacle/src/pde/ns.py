import numpy as np
import torch

import deepxde as dde
from . import baseclass


class NS2D_LidDriven(baseclass.BasePDE):

    def __init__(self, datapath="ref/lid_driven_a4.dat", a=4, nu=1 / 100, bbox=[0, 1, 0, 1]):
        super().__init__()
        self.output_config = [{'name': s} for s in ['u', 'v', 'p']]
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(
            xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]]
        )

        def ns_pde(x, u):
            u_vel, v_vel, _ = u[:, 0:1], u[:, 1:2], u[:, 2:]
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            momentum_x = (
                u_vel * u_vel_x
                + v_vel * u_vel_y
                + p_x
                - nu * (u_vel_xx + u_vel_yy)
            )
            momentum_y = (
                u_vel * v_vel_x
                + v_vel * v_vel_y
                + p_y
                - nu * (v_vel_xx + v_vel_yy)
            )
            continuity = u_vel_x + v_vel_y
            return [momentum_x, momentum_y, continuity]

        self.pde = ns_pde
        self.set_pdeloss(names=["momentum_x", "momentum_y", "continuity"])
        self.load_ref_data(datapath)

        def boundary_top(x, on_boundary):
            return on_boundary and np.isclose(x[1], bbox[3])

        def boundary_not_top(x, on_boundary):
            return on_boundary and not np.isclose(x[1], bbox[3])

        self.add_bcs([{
            'component': 0,
            'function': (lambda x: a * x[:, 0:1] * (1 - x[:, 0:1])),
            'bc': boundary_top,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_top,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_not_top,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_not_top,
            'type': 'dirichlet'
        }, {
            'component': 2,
            'points': np.array([[0, 0]]),
            'values': np.array([[0]]),
            'type': 'pointset'
        }])
        self.training_points()


class NS2D_BackStep(baseclass.BasePDE):

    def __init__(
        self,
        datapath="ref/ns_0_obstacle.dat",
        nu=1 / 100,
        bbox=[0, 4, 0, 2],
        # obstacle={
        #     'circ': [(1, 0.5, 0.2), (2, 0.5, 0.3), (2.7, 1.6, 0.2)],
        #     'rec': [(2.8, 3.6, 0.8, 1.1)]
        # },
        obstacle={},
    ):
        super().__init__()
        # output dim
        self.output_config = [{'name': s} for s in ['u', 'v', 'p']]
        # geom
        eps = 1e-5
        self.bbox = bbox
        self.geom = dde.geometry.Rectangle(xmin=[bbox[0], bbox[2]], xmax=[bbox[1], bbox[3]])
        rec = dde.geometry.Rectangle(xmin=[bbox[0] - eps, bbox[3] / 2], xmax=[bbox[1] / 2, bbox[3] + eps])
        self.geom = dde.geometry.csg.CSGDifference(self.geom, rec)

        # TODO: use CSGMultiDifference
        for name, configs in obstacle.items():
            if name == 'circ':
                for c in configs:
                    diff_geom = dde.geometry.Disk(c[:2], c[2])
                    self.geom = dde.geometry.csg.CSGDifference(self.geom, diff_geom)
            elif name == 'rec':
                for c in configs:
                    diff_geom = dde.geometry.Rectangle(xmin=[c[0], c[2]], xmax=[c[1], c[3]])
                    self.geom = dde.geometry.csg.CSGDifference(self.geom, diff_geom)

        # PDE
        def ns_pde(x, u):
            u_vel, v_vel, _ = u[:, 0:1], u[:, 1:2], u[:, 2:]
            u_vel_x = dde.grad.jacobian(u, x, i=0, j=0)
            u_vel_y = dde.grad.jacobian(u, x, i=0, j=1)
            u_vel_xx = dde.grad.hessian(u, x, component=0, i=0, j=0)
            u_vel_yy = dde.grad.hessian(u, x, component=0, i=1, j=1)

            v_vel_x = dde.grad.jacobian(u, x, i=1, j=0)
            v_vel_y = dde.grad.jacobian(u, x, i=1, j=1)
            v_vel_xx = dde.grad.hessian(u, x, component=1, i=0, j=0)
            v_vel_yy = dde.grad.hessian(u, x, component=1, i=1, j=1)

            p_x = dde.grad.jacobian(u, x, i=2, j=0)
            p_y = dde.grad.jacobian(u, x, i=2, j=1)

            momentum_x = (u_vel * u_vel_x + v_vel * u_vel_y + p_x - nu * (u_vel_xx + u_vel_yy))
            momentum_y = (u_vel * v_vel_x + v_vel * v_vel_y + p_y - nu * (v_vel_xx + v_vel_yy))
            continuity = u_vel_x + v_vel_y

            return [momentum_x, momentum_y, continuity]

        self.pde = ns_pde
        self.set_pdeloss(names=["momentum_x", "momentum_y", "continuity"])

        self.load_ref_data(datapath)

        # bcs
        def boundary_in(x, on_boundary):
            return on_boundary and np.isclose(x[0], bbox[0])

        def boundary_out(x, on_boundary):
            return on_boundary and np.isclose(x[0], bbox[1])

        def boundary_other(x, on_boundary):
            return on_boundary and not (boundary_in(x, on_boundary) or boundary_out(x, on_boundary))

        def u_func(x):
            return x[:, 1:2] * (1 - x[:, 1:2]) * 4

        self.add_bcs([{
            'component': 0,
            'function': u_func,
            'bc': boundary_in,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_in,
            'type': 'dirichlet'
        }, {
            'component': 2,
            'function': (lambda _: 0),
            'bc': boundary_out,
            'type': 'dirichlet'
        }, {
            'component': 0,
            'function': (lambda _: 0),
            'bc': boundary_other,
            'type': 'dirichlet'
        }, {
            'component': 1,
            'function': (lambda _: 0),
            'bc': boundary_other,
            'type': 'dirichlet'
        }])

        # traing config
        self.training_points()


