import numpy as np
import torch
import scipy.io as sio
import math
import matplotlib.pyplot as plt
import pickle

import utils.defects as defects

class KineticData(object):
    '''
        Storge class for kinetic solver, note the vars are in frequency domain.
    '''
    def __init__(self, psi_h, psim1_h, u_h, v_h, Bm1_h, simu_args):
        self.psi_h = psi_h
        self.psim1_h = psim1_h
        self.u_h = u_h
        self.v_h = v_h
        self.Bm1_h = Bm1_h
        self.simu_args = simu_args

    def getter(self):
        return (self.psi_h, self.psim1_h, self.u_h, self.v_h, self.Bm1_h)
    
    def setter(self, psi_h, psim1_h, u_h, v_h, Bm1_h):
        self.psi_h = psi_h
        self.psim1_h = psim1_h
        self.u_h = u_h
        self.v_h = v_h
        self.Bm1_h = Bm1_h

    def get_D(self):
        dth_ = self.simu_args[5]
        psi_h = self.psi_h
        psi_h0 = torch.fft.ifft2(psi_h[:, :, 0]) * dth_
        psi_h2 = torch.fft.ifft2(psi_h[:, :, -2]) * dth_
        psi_h4 = torch.fft.ifft2(psi_h[:, :, -4]) * dth_
        # Concentration
        c = torch.real(psi_h0)
        d11 = 0.5 * (torch.real(psi_h0) + torch.real(psi_h2))
        d12 = 0.5 * torch.imag(psi_h2)
        d22 = 0.5 * (torch.real(psi_h0) - torch.real(psi_h2))
        # transfer to float32
        d11 = d11.float()
        d12 = d12.float()
        return d11, d12

    def dumper(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def loader(path, device='cuda:0'):
        with open(path, 'rb') as f:
            return pickle.load(f).to_device(device)
        
    def to_device(self, device):
        self.psi_h = self.psi_h.to(device)
        self.psim1_h = self.psim1_h.to(device)
        self.u_h = self.u_h.to(device)
        self.v_h = self.v_h.to(device)
        self.Bm1_h = self.Bm1_h.to(device)
        return self


class KineticSolver(object):
    '''Kinetic solver for the 2D nematics flow, using pusedo-spectral method.'''
    def __init__(self, geo_params, flow_params, simu_params, device='cuda:0'):
        self.geo_params = geo_params
        self.flow_params = flow_params
        self.simu_params = simu_params
        self.device = device

        dT = self.flow_params['dT']
        dR = self.flow_params['dR']
        alpha = self.flow_params['alpha']
        beta = self.flow_params['beta']
        zeta = self.flow_params['zeta']
        V0 = self.flow_params['V0']
        self.flow_args = (dT, dR, alpha, beta, zeta, V0)

        seed_ = self.simu_params['seed']
        self.inner_steps = self.simu_params['inner_steps']
        self.outer_steps = self.simu_params['outer_steps']
        # self.duration = self.simu_params['T']
        self.step = 0
        self.total_steps = self.inner_steps * self.outer_steps
        self.seed = seed_

        N = self.geo_params['N']
        Nth = self.geo_params['Nth']
        L = self.geo_params['L']
        Lth = 2*np.pi
        dx_ = L / N
        dt = self.simu_params['dt']
        dth_ = Lth / Nth
        print(" dx_ ", dx_, " dth_ ", dth_, " dt ", dt)
        self.simu_args = (N, Nth, L, Lth, dx_, dth_, dt, seed_)

        x_ = np.linspace(0, L-dx_, N)
        y_ = np.linspace(0, L - dx_, N)
        th_ = np.linspace(0, Lth - dth_, Nth)
        xx_, yy_, th_ = np.meshgrid(x_, y_, th_)
        xx_ = np.transpose(xx_, [1, 0, 2])
        yy_ = np.transpose(yy_, [1, 0, 2])

        ik_x = (1. / L) * np.hstack((np.arange(0, int(N / 2) + 1), np.arange(-int(N / 2) + 1, 0)))
        ik_y = (1. / L) * np.hstack((np.arange(0, int(N / 2) + 1), np.arange(-int(N / 2) + 1, 0)))
        ik_th = (1. / Lth) * np.hstack((np.arange(0, int(Nth / 2) + 1), np.arange(-int(Nth / 2) + 1, 0)))
        kx_, ky_, kth_ = np.meshgrid(ik_x, ik_y, ik_th)

        kx2d_, ky2d_ = np.meshgrid(ik_x, ik_y)
        kx2d_ = kx2d_.T
        ky2d_ = ky2d_.T
        kx_op = 2.0 * np.pi * 1j * kx2d_
        ky_op = 2.0 * np.pi * 1j * ky2d_
        kth_op = 2.0 * np.pi * 1j * kth_

        Lx_h = kx_op ** 2 + ky_op ** 2
        Lth_h = kth_op ** 2

        ksq = np.abs(Lx_h)
        ksq[0, 0] = 1  # ksq.at[0,0].set(1) # magnitude of Fourier modes
        kx_n = np.imag(kx_op) / np.sqrt(ksq)
        ky_n = np.imag(ky_op) / np.sqrt(ksq)
        linear_ = self.to_NNZ(dT * Lx_h + dR * self.to_ZNN(Lth_h))
        K_h = 1. / (3. - 2. * dt * linear_)

        L11_h = (1. / ksq) * (1.0 - kx_n * kx_n)
        L12_h = (1. / ksq) * (0.0 - kx_n * ky_n)
        L21_h = (1. / ksq) * (0.0 - ky_n * kx_n)
        L22_h = (1. / ksq) * (1.0 - ky_n * ky_n)
        max_k = np.max(np.abs(ky_[:, :, :]))
        max_th = np.max(np.abs(kth_[:, :, :]))
        filter_ = (np.abs(kx_) < (2 / 3.) * max_k) & (np.abs(ky_) < (2 / 3.) * max_k) & (np.abs(kth_) < (2 / 3.) * max_th)
    
        ik_x = (1. / L) * np.hstack((np.arange(0, int(N / 2) + 1), np.arange(-int(N / 2) + 1, 0)))
        ik_y = (1. / L) * np.hstack((np.arange(0, int(N / 2) + 1), np.arange(-int(N / 2) + 1, 0)))
        ik_x[int(N / 2)] = 0
        ik_y[int(N / 2)] = 0
        kx2d_, ky2d_ = np.meshgrid(ik_x, ik_y)
        kx2d_ = kx2d_.T
        ky2d_ = ky2d_.T

        kx_op = 2.0 * np.pi * 1j * kx2d_
        ky_op = 2.0 * np.pi * 1j * ky2d_

        # to torch and self
        p1 = torch.tensor(np.cos(th_), dtype=torch.float64, device=device)
        p2 = torch.tensor(np.sin(th_), dtype=torch.float64, device=device)

        xx_ = torch.tensor(xx_, dtype=torch.float64, device=device)
        yy_ = torch.tensor(yy_, dtype=torch.float64, device=device)
        th_ = torch.tensor(th_, dtype=torch.float64, device=device)
        K_h = torch.tensor(K_h, dtype=torch.complex128, device=device)
        kx_op = torch.tensor(kx_op, dtype=torch.complex128, device=device)
        ky_op = torch.tensor(ky_op, dtype=torch.complex128, device=device)
        kth_op = torch.tensor(kth_op, dtype=torch.complex128, device=device)
        L11_h = torch.tensor(L11_h, dtype=torch.float64, device=device)
        L12_h = torch.tensor(L12_h, dtype=torch.float64, device=device)
        L21_h = torch.tensor(L21_h, dtype=torch.float64, device=device)
        L22_h = torch.tensor(L22_h, dtype=torch.float64, device=device)
        filter_ = torch.tensor(filter_, dtype=torch.bool, device=device)

        self.ops = (p1, p2, xx_, yy_, th_, K_h, kx_op, ky_op, kth_op, L11_h, L12_h,
                    L21_h, L22_h, filter_)
        
  
    def con(self, arr):
        return arr.cpu().data.numpy()


    def to_ZNN(self, func_):
        return np.transpose(func_, [2, 0, 1])


    def to_NNZ(self, func_):
        return np.transpose(func_, [1, 2, 0])


    def torch_NNZ(self, func_):
        return torch.transpose(func_, 1, 2).transpose(0, 2)


    def torch_ZNN(self, func_):
        return torch.transpose(func_, 2, 0).transpose(1, 2)

    def initialize2_pytorch(self, seed=1234):
        N, Nth, L, Lth, dx_, dth_, dt, seed_ = self.simu_args
        p1, p2, xx_, yy_, th_, K_h, kx_op, ky_op, kth_op, L11_h, L12_h, L21_h, L22_h, filter_ = self.ops
        psi = torch.ones((N, N, Nth), dtype=torch.float64, device=self.device)
        Nk = 8
        pert = 0.1
        seed_ = seed
        torch.manual_seed(seed_)
        np.random.seed(seed_)
        # print(L)
        for i in range(0, Nk):
            k1 = np.random.randint(1, Nk + 1)
            k2 = np.random.randint(1, Nk + 1)
            rand = np.random.uniform(0, 1, 4)
            psi = psi + pert * (rand[0] - 0.5) \
                * torch.cos(2. * torch.pi * k1 * xx_ / L + rand[1]) \
                * torch.cos(2. * torch.pi * k2 * yy_ / L + rand[2]) \
                * torch.cos(2. * th_ + rand[3]) ** (i + 1)

        psi = (L ** 2) * psi / (torch.sum(psi * dx_ * dx_ * dth_))
        psim1_h = torch.fft.fftn(psi)

        psi_h = psim1_h + 0
        ################
        psi_h0 = torch.fft.ifft2(psi_h[:, :, 0]) * dth_
        psi_h2 = torch.fft.ifft2(psi_h[:, :, -2]) * dth_
        psi_h4 = torch.fft.ifft2(psi_h[:, :, -4]) * dth_
        # Concentration
        c = torch.real(psi_h0)
        # print(" c ", np.mean(self.con(c).flatten()), np.std(self.con(c).flatten()))
        d11 = 0.5 * (torch.real(psi_h0) + torch.real(psi_h2))
        d12 = 0.5 * torch.imag(psi_h2)
        d22 = 0.5 * (torch.real(psi_h0) - torch.real(psi_h2))


        # print(" done with initialization ")
        # print(" max  c ", torch.max((c)), " min c ", torch.min(c))
        # print(" max  d11 ", torch.max(d11), " min d11 ", torch.min(d11))
        # print(" max  d12 ", torch.max(d12), " min d12 ", torch.min(d12), " sum ", torch.max(d12) + torch.min(d12))
        # print(" max  d22 ", torch.max(d22), " min d22 ", torch.min(d22))

        # # initialize u_h, v_h
        u = torch.zeros((N, N), dtype=torch.float64, device=self.device)
        v = torch.zeros((N, N), dtype=torch.float64, device=self.device)
        u_h = torch.fft.fft2(u)
        v_h = torch.fft.fft2(v)
        s11_h, s12_h, s21_h, s22_h = self.sigma_h(psim1_h, u_h, v_h)
        u_h, v_h = self.Stokes(s11_h, s12_h, s21_h, s22_h)
        Bm1_h = self.flux(psim1_h, u_h, v_h)
        # self.arr_ = (psi_h, psim1_h, u_h, v_h, Bm1_h)
        # self.var = KineticData(psi_h, psim1_h, u_h, v_h, Bm1_h)


        return psi_h, psim1_h, u_h, v_h, Bm1_h
        
    def sigma_h(self, psi_h, u_h, v_h):
        # get params
        p1, p2, xx_, yy_, th_, K_h, kx_op, ky_op, kth_op, L11_h, L12_h, L21_h, L22_h, filter_ = self.ops
        dT, dR, alpha, beta, zeta, V0 = self.flow_args
        N, Nth, L, Lth, dx_, dth_, dt, seed_ = self.simu_args

        e11 = torch.real(torch.fft.ifft2(kx_op * u_h))  ## ux
        e12 = 0.5 * torch.real(torch.fft.ifft2(ky_op * u_h + kx_op * v_h))  ## 0.5(uy + vx)
        e22 = -e11  ## vy

        # Moments of the distribution functions
        psi_h0 = torch.fft.ifft2(psi_h[:, :, 0]) * dth_
        psi_h2 = torch.fft.ifft2(psi_h[:, :, -2]) * dth_
        psi_h4 = torch.fft.ifft2(psi_h[:, :, -4]) * dth_

        # Concentration
        c = torch.real(psi_h0)
        # print(" c mean ", np.mean(con(c).flatten()), np.std(con(c).flatten()))
        # print(" max c ", jnp.max(c), " min c ", jnp.min(c))

        # D tensor
        d11 = 0.5 * (torch.real(psi_h0) + torch.real(psi_h2))
        d12 = 0.5 * torch.imag(psi_h2)
        d22 = 0.5 * (torch.real(psi_h0) - torch.real(psi_h2))

        # S tensor
        s1111 = 0.125 * (
                    3.0 * torch.real(psi_h0) + 4.0 * torch.real(psi_h2) + torch.real(psi_h4))  # verified in mathematica
        s1112 = 0.125 * (2.0 * torch.imag(psi_h2) + torch.imag(psi_h4))  # verified in mathematica
        s1122 = d11 - s1111
        s1222 = d12 - s1112
        s2222 = d22 - s1122

        # D*D
        dd11 = d11 * d11 + d12 * d12
        dd12 = d11 * d12 + d12 * d22
        dd22 = d12 * d12 + d22 * d22

        # S:E
        se11 = s1111 * e11 + 2.0 * s1112 * e12 + s1122 * e22
        se12 = s1112 * e11 + 2.0 * s1122 * e12 + s1222 * e22
        se22 = s1122 * e11 + 2.0 * s1222 * e12 + s2222 * e22

        # S:D
        sd11 = s1111 * d11 + 2.0 * s1112 * d12 + s1122 * d22
        sd12 = s1112 * d11 + 2.0 * s1122 * d12 + s1222 * d22
        sd22 = s1122 * d11 + 2.0 * s1222 * d12 + s2222 * d22

        # print(" alpha ", alpha, " beta ", beta,  " zeta ", zeta)

        # Tensor components
        s11_h = torch.fft.fft2(alpha * (d11 - c / 2.) + beta * se11 - 2 * zeta * beta * (dd11 - sd11))
        s12_h = torch.fft.fft2(alpha * d12 + beta * se12 - 2 * zeta * beta * (dd12 - sd12))
        s22_h = torch.fft.fft2(alpha * (d22 - c / 2.) + beta * se22 - 2 * zeta * beta * (dd22 - sd22))
        s21_h = s12_h

        

        return s11_h, s12_h, s21_h, s22_h

    def flux(self, psi_h, u_h, v_h):
        # get params
        p1, p2, xx_, yy_, th_, K_h, kx_op, ky_op, kth_op, L11_h, L12_h, L21_h, L22_h, filter_ = self.ops
        dT, dR, alpha, beta, zeta, V0 = self.flow_args
        N, Nth, L, Lth, dx_, dth_, dt, seed_ = self.simu_args
        # velocity and distribution function in real space
        u = torch.real(torch.fft.ifft2(u_h))
        v = torch.real(torch.fft.ifft2(v_h))
        psi = torch.real(torch.fft.ifftn(psi_h))  # ifft2 was done instead of ifftn

        # compute derivatives
        ux = torch.real(torch.fft.ifft2(kx_op * u_h))
        uy = torch.real(torch.fft.ifft2(ky_op * u_h))
        vx = torch.real(torch.fft.ifft2(kx_op * v_h))
        vy = -ux

        # get integral values from Fourier transform
        psi_h0 = torch.fft.ifft2(psi_h[:, :, 0]) * dth_  # can we store this value and call maybe ?
        psi_h2 = torch.fft.ifft2(psi_h[:, :, -2]) * dth_  # can we store this value and call maybe ?

        # D-tensor
        d11 = 0.5 * (torch.real(psi_h0) + torch.real(psi_h2))
        d12 = 0.5 * torch.imag(psi_h2)
        d21 = d12
        d22 = 0.5 * (torch.real(psi_h0) - torch.real(psi_h2))

        # T = grad(u) + 2*zeta*D
        t11 = ux + 2.0 * zeta * d11
        t12 = uy + 2.0 * zeta * d12
        t21 = vx + 2.0 * zeta * d21
        t22 = vy + 2.0 * zeta * d22

        # Conformational fluxes
        xdot_psi_h = torch.fft.fftn(self.torch_NNZ(u + V0 * self.torch_ZNN(p1)) * psi)
        ydot_psi_h = torch.fft.fftn(self.torch_NNZ(v + V0 * self.torch_ZNN(p2)) * psi)

        pdot = -p2 * self.torch_NNZ(t11 * self.torch_ZNN(p1) + t12 * self.torch_ZNN(p2)) + p1 * self.torch_NNZ(
            t21 * self.torch_ZNN(p1) + t22 * self.torch_ZNN(p2))
        # print(" type ", pdot.dtype, psi.dtype, pdot.shape, psi.shape)
        divp_psi = kth_op * (torch.fft.fftn(pdot * psi))
        divx_psi = self.torch_NNZ(kx_op * self.torch_ZNN(xdot_psi_h) + ky_op * self.torch_ZNN(ydot_psi_h))

        B_h = -(divx_psi + divp_psi) * filter_  # design this filter
        # print(" B_h device ", B_h.get_device())

        return B_h

    def Stokes(self, s11_h, s12_h, s21_h, s22_h):
        p1, p2, xx_, yy_, th_, K_h, kx_op, ky_op, kth_op, L11_h, L12_h, L21_h, L22_h, filter_ = self.ops
        # Stokes solver
        u_hat = (L11_h * (kx_op * s11_h + ky_op * s21_h)
                + L12_h * (kx_op * s12_h + ky_op * s22_h))
        v_hat = (L21_h * (kx_op * s11_h + ky_op * s21_h)
                + L22_h * (kx_op * s12_h + ky_op * s22_h))
        return u_hat, v_hat
    
    def sbdf2(self, psi_h, psim1_h, B_h, Bm1_h):
        p1, p2, x_, yy_, th_, K_h, kx_op, ky_op, kth_op, L11_h, L12_h, L21_h, L22_h, filter_ = self.ops
        N, Nth, L, Lth, dx_, dth_, dt, seed_ = self.simu_args
        return K_h * (4.0 * psi_h - psim1_h + 2.0 * dt * (2.0 * B_h - Bm1_h))

    def agent_act(self, s11_h, s12_h, s21_h, s22_h, action):
        # action is a 2d bool field, indicating which area on the stress field is under control
        # the region under control multiplies by matrix action
        # put action to device
        action = torch.tensor(action, dtype=torch.float64, device=self.device)
        # for s_h in [s11_h, s12_h, s21_h, s22_h]:
        #     s = torch.real(torch.fft.ifft2(s_h))
        #     s = s * action
        #     s_h = torch.fft.fft2(s)
        s11 = torch.real(torch.fft.ifft2(s11_h)) * action
        s12 = torch.real(torch.fft.ifft2(s12_h)) * action
        s21 = torch.real(torch.fft.ifft2(s21_h)) * action
        s22 = torch.real(torch.fft.ifft2(s22_h)) * action
        s11_h = torch.fft.fft2(s11)
        s12_h = torch.fft.fft2(s12)
        s21_h = torch.fft.fft2(s21)
        s22_h = torch.fft.fft2(s22)

        return s11_h, s12_h, s21_h, s22_h

    # the actual loop
    def kinetic(self, psi_h, psim1_h, u_h, v_h, Bm1_h, action=None):
        # TODO: add control for stress Sigma
        # psi_h, psim1_h, u_h, v_h, Bm1_h = vars.getter()

        for i in range(0, self.inner_steps):
            s11_h, s12_h, s21_h, s22_h = self.sigma_h(psi_h, u_h, v_h)

            # if action
            if action is not None:
                s11_h, s12_h, s21_h, s22_h = self.agent_act(s11_h, s12_h, s21_h, s22_h, action)

            u_hat, v_hat = self.Stokes(s11_h, s12_h, s21_h, s22_h)
            B_h = self.flux(psi_h, u_hat, v_hat)
            psip1_h = self.sbdf2(psi_h, psim1_h, B_h, Bm1_h)

            psim1_h = psi_h + 0
            psi_h = psip1_h + 0
            Bm1_h = B_h + 0

        self.step += 1
        # print self.step and self.outer_steps
        # print(f'Step {self.step} of {self.outer_steps} done.')

        # vars.setter(psi_h, psim1_h, u_hat, v_hat, Bm1_h)
        return (psi_h, psim1_h, u_hat, v_hat, Bm1_h), (self.step==self.outer_steps)
    
    def single_kinetic(self, array_):
        psi_h, psim1_h, u_h, v_h, Bm1_h = array_

        s11_h, s12_h, s21_h, s22_h = self.sigma_h(psi_h, u_h, v_h)
        u_hat, v_hat = self.Stokes(s11_h, s12_h, s21_h, s22_h)
        B_h = self.flux(psi_h, u_hat, v_hat)
        psip1_h = self.sbdf2(psi_h, psim1_h, B_h, Bm1_h)

        psim1_h = psi_h + 0
        psi_h = psip1_h + 0
        Bm1_h = B_h + 0
        return (psi_h, psim1_h, u_hat, v_hat, Bm1_h)

    def loop_kinetic(self, arr_):
        for _ in range(0, self.inner_steps):
            arr_ = self.single_kinetic(arr_)
        return arr_

    def preloop_kinetic(self, vars, num_itr=5000):
        psi_h, psim1_h, u_h, v_h, Bm1_h = vars.getter()
        arr_ = (psi_h, psim1_h, u_h, v_h, Bm1_h)

        for i in range(0, num_itr):
            arr_, _ = self.kinetic(*arr_)
            arr_ = (arr_[0], arr_[1], arr_[2], arr_[3], arr_[4])
                # prof.step()
            if i % 1000 == 0:
                print(f'Pre-loop iteration {i}.')
        vars.setter(*arr_)
        self.step = 0
        return vars

    def cleanup(self):
        pass

    def save_loop(self, vars):
        psi_h, psim1_h, u_h, v_h, Bm1_h = vars.getter()
        arr_ = (psi_h, psim1_h, u_h, v_h, Bm1_h)
        outer_steps = self.outer_steps
        N = self.geo_params['N']
        tensor_psi_h = np.zeros((N, N, N, outer_steps + 1), dtype=np.complex128)
        tensor_psim1_h = np.zeros((N, N, N, outer_steps + 1), dtype=np.complex128)
        tensor_u_h = np.zeros((N, N, N, outer_steps + 1), dtype=np.complex128)
        tensor_v_h = np.zeros((N, N, N, outer_steps + 1), dtype=np.complex128)
        tensor_Bm1_h = np.zeros((N, N, N, outer_steps + 1), dtype=np.complex128)
        for _write in range(0, outer_steps + 1):
            arr_ = self.loop_kinetic(arr_)
            psi_h, psim1_h, u_h, v_h, Bm1_h = arr_
            tensor_psi_h[:, :, :, _write] = psi_h.cpu().data.numpy()
            tensor_psim1_h[:, :, :, _write] = psim1_h.cpu().data.numpy()
            tensor_u_h[:, :, :, _write] = u_h.cpu().data.numpy()
            tensor_v_h[:, :, :, _write] = v_h.cpu().data.numpy()
            tensor_Bm1_h[:, :, :, _write] = Bm1_h.cpu().data.numpy()
            print(f'Write iteration {_write}.')
        return tensor_psi_h, tensor_psim1_h, tensor_u_h, tensor_v_h, tensor_Bm1_h

    def save_D_loop(self, vars):
        psi_h, psim1_h, u_h, v_h, Bm1_h = vars.getter()
        arr_ = (psi_h, psim1_h, u_h, v_h, Bm1_h)
        outer_steps = self.outer_steps
        N = self.geo_params['N']
        tensor_D = np.zeros((N, N, 3, outer_steps + 1))
        for _write in range(0, outer_steps + 1):
            arr_ = self.loop_kinetic(arr_)
            psi_h, psim1_h, u_h, v_h, Bm1_h = arr_
            d11, d12, d22 = self.get_D(psi_h)
            tensor_D[:, :, 0, _write] = d11.cpu().data.numpy()
            tensor_D[:, :, 1, _write] = d12.cpu().data.numpy()
            tensor_D[:, :, 2, _write] = d22.cpu().data.numpy()
            print(f'Write iteration {_write}.')
        return tensor_D

    def get_D(self, psi_h):
        dth_ = self.simu_args[5]
        psi_h0 = torch.fft.ifft2(psi_h[:, :, 0]) * dth_
        psi_h2 = torch.fft.ifft2(psi_h[:, :, -2]) * dth_
        psi_h4 = torch.fft.ifft2(psi_h[:, :, -4]) * dth_
        # Concentration
        c = torch.real(psi_h0)
        d11 = 0.5 * (torch.real(psi_h0) + torch.real(psi_h2))
        d12 = 0.5 * torch.imag(psi_h2)
        d22 = 0.5 * (torch.real(psi_h0) - torch.real(psi_h2))
        return d11, d12, d22

    @staticmethod
    def visualize_flow_field(data, ax):
        d11, d12 = data.get_D()
        d11 = d11.cpu().data.numpy()
        d12 = d12.cpu().data.numpy()
        theta = defects.theta_cal(d11, d12)
        defects_arr = defects.calculate_defects(theta)

        defects.plot_defects(defects_arr, theta, ax)

    # def loop_kinetic(self, psi_h, psim1_h, u_h, v_h, Bm1_h):
    #     arr_ = (psi_h, psim1_h, u_h, v_h, Bm1_h)
    #     for _ in range(0, self.inner_steps):
    #         arr_ = self.kinetic(arr_)
    #     return arr_

    # def get_arr(self):
    #     return self.var.getter()
    
    def __str__(self):
        return f'KineticSolver params: \n {self.geo_params} \n {self.flow_params} \n {self.simu_params} \n {self.device}'
    
if __name__ == '__main__':
    geo_params = {
        'N': 256,
        'Nth': 256,
        'L': 20
    }
    flow_params = {
        'dT': 0.3,
        'dR': 0.3,
        'alpha': -10,
        'beta': 1.0,
        'zeta': 2,
        'V0': 0.0
    }
    simu_params = {
        'dt': 0.0004,
        'seed': 1234,
        'inner_steps': 640,
        'outer_steps': 200
    }
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    solver_paras = (geo_params, flow_params, simu_params)
    solver = KineticSolver(*solver_paras, device=device)
    simulation_data = KineticData(*solver.initialize2_pytorch(seed=918), solver.simu_args)
    d11, d12 = simulation_data.get_D()
    print(d11.shape)
        