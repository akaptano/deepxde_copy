import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

class GS_Linear:
    def __init__(self, eps, kappa, delta,divertor = 0, beta_limit=False):

        self.divertor = divertor
        self.beta_limit = beta_limit
        self.eps = eps
        self.kappa = kappa
        self.delta = delta

        self.N1 = - (1 + np.arcsin(self.delta)) ** 2 / (self.eps * self.kappa ** 2)
        self.N2 = (1 - np.arcsin(self.delta)) ** 2 / (self.eps * self.kappa ** 2)
        self.N3 = - self.kappa / (self.eps * np.cos(np.arcsin(self.delta)) ** 2)
        self.get_UX = [self.get_U1,self.get_U2,self.get_U3,self.get_U4,self.get_U5,self.get_U6,self.get_U7,self.get_UP]

        if divertor == 1:
            self.get_UX = [self.get_U1,self.get_U2,self.get_U3,self.get_U4,self.get_U5,self.get_U6,self.get_U7,
                            self.get_U8,self.get_U9,self.get_U10,self.get_U11,self.get_U12,
                            self.get_UP]


        # Define Boundary Constraints
        #self.get_BCs()
        #self.solve_coefficients()

    def visualize(self, X, nx, ny):
        '''
        Input:
            X: is a vstack from a meshgrid -> points where psi is evaluated
            nx: number of points in x
            ny: number of points in y
        Output:
            psi_true: analytical solution for the given points
        '''
        # Calculate corresponding psi
        psi = []
        for point in X:
            psi.append(self.psi_func(point[0],point[1]))
        psi_true = np.reshape(psi, [nx, ny])
        return psi_true


    def psi_func(self, x, y, A=None):
        '''
        Input:
        cj: a list of coefficients for each polynomials
        x: x point to be calculated
        y: y point to be calculated
        Output:
        psi: flux function at a given point for GS w/ Solovev
        '''
        if A is not None:
            self.A = A

        ## Added this line to calculate both sides
        if x < 0:
            x *= -1
        # Get polynomial values for a given point

        UH = np.array([i(x,y) for i in self.get_UX[:-1]])
        APart, nonAPart = self.get_UP(x,y)
        if self.beta_limit == False:
            cj_UH = UH*self.cj[0]
            UP = self.A*APart+nonAPart

        elif self.beta_limit == True:
            cj_UH = UH*self.cj[0][:-1]
            UP = self.cj[0][-1]*APart+nonAPart

        '''
        UH = np.array([i(x,y) for i in self.get_UX[:-1]])
        cj_UH = UH*self.cj[0]
        return self.get_UP(x,y)+ sum(cj_UH)'''

        return UP + sum(cj_UH)

    def solve_coefficients(self):
        # UH
        if self.divertor != 1:
            self.UH = np.concatenate(([self.BC1[0]], [self.BC2[0]],[self.BC3[0]],
                                        [self.BC4[0]], [self.BC5[0]], [self.BC6[0]], [self.BC7[0]]), axis=0)
            self.UP = np.concatenate(([-1*self.BC1[1]], [-1*self.BC2[1]],[-1*self.BC3[1]],
                                        [-1*self.BC4[1]], [-1*self.BC5[1]], [-1*self.BC6[1]], [-1*self.BC7[1]]), axis=0)
        elif self.divertor == 1:
            self.UH = np.concatenate(([self.BC1[0]], [self.BC2[0]],[self.BC3[0]],
                                        [self.BC4[0]], [self.BC5[0]], [self.BC6[0]], [self.BC7[0]],
                                        [self.BC8[0]], [self.BC9[0]], [self.BC10[0]], [self.BC11[0]], [self.BC12[0]]), axis=0)
            self.UP = np.concatenate(([-1*self.BC1[1]], [-1*self.BC2[1]],[-1*self.BC3[1]],
                                        [-1*self.BC4[1]], [-1*self.BC5[1]], [-1*self.BC6[1]], [-1*self.BC7[1]],
                                        [-1*self.BC8[1]], [-1*self.BC9[1]], [-1*self.BC10[1]], [-1*self.BC11[1]], [-1*self.BC12[1]]), axis=0)

        if self.beta_limit == True:
            self.UH = np.concatenate((self.UH,[self.BC0[0]]), axis=0)
            self.UP = np.concatenate((self.UP,[-1*self.BC0[1]]), axis=0)
        # print(np.shape(self.UH))
        # print(np.shape(self.UP))
        self.cj = np.linalg.solve(self.UH, self.UP).T

        if self.beta_limit == True and self.A is None:
            self.A = self.cj[0][-1]

    def get_BCs(self, A):
        # outer point
        self.A = A
        x,y = self.outer_point()
        self.BC1 = self.get_point_flux(x,y)
        dUH_dyy, dUP_dyy = self.get_point_curvature_dyy(x,y)
        dUH_dx,  dUP_dx  = self.get_point_slope_dx(x,y)
        self.BC2 = (dUH_dyy + self.N1*dUH_dx, \
                     dUP_dyy + self.N1*dUP_dx)

        # inner point
        x,y = self.inner_point()
        self.BC3 = self.get_point_flux(x,y)
        dUH_dyy, dUP_dyy = self.get_point_curvature_dyy(x,y)
        dUH_dx,  dUP_dx  = self.get_point_slope_dx(x,y)
        self.BC4 = (dUH_dyy + self.N2*dUH_dx, \
                    dUP_dyy + self.N2*dUP_dx)

        if self.divertor == 0:
            # high point
            x,y = self.high_point()
            self.BC5 = self.get_point_flux(x,y)
            self.BC6  = self.get_point_slope_dx(x,y)
            dUH_dxx, dUP_dxx = self.get_point_curvature_dxx(x,y)
            dUH_dy,  dUP_dy  = self.get_point_slope_dy(x,y)
            self.BC7 = (dUH_dxx + self.N3*dUH_dy, \
                        dUP_dxx + self.N3*dUP_dy)

        if self.divertor == 1:
            # high point
            x,y = self.high_point()
            self.BC5 = self.get_point_flux(x,y)
            self.BC6  = self.get_point_slope_dx(x,y)
            dUH_dxx, dUP_dxx = self.get_point_curvature_dxx(x,y)
            dUH_dy,  dUP_dy  = self.get_point_slope_dy(x,y)
            self.BC7 = (dUH_dxx + self.N3*dUH_dy, \
                        dUP_dxx + self.N3*dUP_dy)
            # lower separatrix point
            x,y = self.sep_point()
            self.BC8 = self.get_point_flux(x,y*-1)
            self.BC9  = self.get_point_slope_dx(x,y*-1)
            self.BC10 = self.get_point_slope_dy(x,y*-1)

            # inner point up-down symmetry
            x,y = self.inner_point()
            self.BC11 = self.get_point_slope_dy(x,y)
            # outer point up-down symmetry
            x,y = self.outer_point()
            self.BC12 = self.get_point_slope_dy(x,y)


        if self.divertor == 2:
            # separatrix point
            x,y = self.sep_point()
            self.BC5 = self.get_point_flux(x,y)
            self.BC6  = self.get_point_slope_dx(x,y)
            self.BC7 = self.get_point_slope_dy(x,y)

        if self.beta_limit == True:
            x,y = self.inner_point()
            self.BC0 = self.get_point_slope_dx(x,y)

    def get_UP(self,x,y):
        APart = (1/2.0*x**2*np.log(x)- x**4 / 8.0)
        nonAPart = x**4/8.0 + y**2*0
        return (APart,nonAPart)
    def get_UP_dx(self,x,y):
        APart = (x/2.0- x**3/2.0 + x*np.log(x))
        nonAPart = x**3/2.0 + y**2*0
        return (APart,nonAPart)
    def get_UP_dxx(self,x,y):
        APart = ( -3.0/2.0*x**2 + 3.0/2.0 + np.log(x))
        nonAPart = 3.0/2.0*x**2 + y**2*0
        return (APart,nonAPart)
    def get_UP_dy(self,x,y):
        APart = 0.0
        nonAPart = 0.0
        return (APart,nonAPart)
    def get_UP_dyy(self,x,y):
        APart = 0.0
        nonAPart = 0.0
        return (APart,nonAPart)

    def get_U1(self,x,y):
        return 1.0 + x*0 +y*0
    def get_U1_dx(self,x,y):
        return 0.0
    def get_U1_dxx(self,x,y):
        return 0.0
    def get_U1_dy(self,x,y):
        return 0.0
    def get_U1_dyy(self,x,y):
        return 0.0

    def get_U2(self,x,y):
        return x ** 2 + y*0
    def get_U2_dx(self,x,y):
        return 2.0*x
    def get_U2_dxx(self,x,y):
        return 2.0
    def get_U2_dy(self,x,y):
        return 0.0
    def get_U2_dyy(self,x,y):
        return 0.0

    def get_U3(self,x,y):
        return y ** 2 - x ** 2 * np.log(x) + y*0
    def get_U3_dx(self,x,y):
        return -x*(2.0*np.log(x)+1.0)
    def get_U3_dxx(self,x,y):
        return -2.0 * np.log(x) - 3.0
    def get_U3_dy(self,x,y):
        return x*0 + 2.0*y
    def get_U3_dyy(self,x,y):
        return 2.0 + x*0 +y*0

    def get_U4(self,x,y):
        return x ** 4 - 4.0 * x ** 2 * y ** 2
    def get_U4_dx(self,x,y):
        return 4*x**3-8*y**2*x
    def get_U4_dxx(self,x,y):
        return 12*x**2-8*y**2
    def get_U4_dy(self,x,y):
        return -8.0*x**2*y
    def get_U4_dyy(self,x,y):
        return -8*x**2

    def get_U5(self,x,y):
        return 2.0 * y ** 4 - 9.0 * x ** 2 * y ** 2 - (12.0 * x ** 2 * y ** 2 - 3 * x ** 4) * np.log(x)
    def get_U5_dx(self,x,y):
        return 3.0*x*((4*x**2-8*y**2)*np.log(x)+x**2-10.0*y**2)
    def get_U5_dxx(self,x,y):
        return (36.0*x**2-24.0*y**2)*np.log(x)+21*x**2-54.0*y**2
    def get_U5_dy(self,x,y):
        return 8.0*y**3+(-24.0*x**2*np.log(x)-18.0*x**2)*y
    def get_U5_dyy(self,x,y):
        return 24.0*y**2-24*x**2*np.log(x)-18*x**2

    def get_U6(self,x,y):
        return x ** 6 - 12.0 * x ** 4 * y ** 2 + 8.0 * x ** 2 * y ** 4
    def get_U6_dx(self,x,y):
        return 6.0*x**5-48.0*y**2*x**3+16*y**4*x
    def get_U6_dxx(self,x,y):
        return 30*x**4-144*y**2*x**2+16*y**4
    def get_U6_dy(self,x,y):
        return 32.0*x**2*y**3-24.0*x**4*y
    def get_U6_dyy(self,x,y):
        return 96.0*x**2*y**2-24.0*x**4

    def get_U7(self,x,y):
        return 8.0 * y ** 6 - 140 * x ** 2 * y ** 4 + 75.0 * x ** 4 * y ** 2 - (120.0 * x ** 2 * y ** 4 - 180 * x ** 4 * y ** 2 + 15.0 * x ** 6) * np.log(x)
    def get_U7_dx(self,x,y):
        return -5*x*((18.0*x**4 - 144 * y**2*x**2 + 48.0*y**4)*np.log(x)+3.0*x**4-96.0*y**2*x**2+80.0*y**4)
    def get_U7_dxx(self,x,y):
        return (-450.0*x**4+2160*y**2*x**2-240*y**4)*np.log(x)-165.0*x**4+2160*y**2*x**2-640*y**4
    def get_U7_dy(self,x,y):
        return 48.0*y**5+(-480.0*x**2*np.log(x)-560*x**2)*y**3+(360*x**4*np.log(x)+150*x**4)*y
    def get_U7_dyy(self,x,y):
        return 240.0*y**4+(-1440.0*x**2*np.log(x)-1680*x**2)*y**2+(360*x**4*np.log(x)+150*x**4)

    # For general up-down asymmetric case
    def get_U8(self,x,y):
        return x*0 + y*1
    def get_U8_dx(self,x,y):
        return 0.0
    def get_U8_dxx(self,x,y):
        return 0.0
    def get_U8_dy(self,x,y):
        return 1.0
    def get_U8_dyy(self,x,y):
        return 0.0

    def get_U9(self,x,y):
        return x ** 2 * y
    def get_U9_dx(self,x,y):
        return 2.0*x*y
    def get_U9_dxx(self,x,y):
        return 2.0*y
    def get_U9_dy(self,x,y):
        return x**2
    def get_U9_dyy(self,x,y):
        return 0.0

    def get_U10(self,x,y):
        return y ** 3 - 3 * y * x ** 2 * np.log(x)
    def get_U10_dx(self,x,y):
        return -3.0*y*(2*x*np.log(x) + x)
    def get_U10_dxx(self,x,y):
        return -3.0*y*(2*np.log(x) + 3)
    def get_U10_dy(self,x,y):
        return 3*y ** 2 - 3*x ** 2 * np.log(x)
    def get_U10_dyy(self,x,y):
        return 6*y

    def get_U11(self,x,y):
        return 3*y*x ** 4 - 4.0 * x ** 2 * y ** 3
    def get_U11_dx(self,x,y):
        return 12 * y * x ** 3 - 8 * y ** 3*x
    def get_U11_dxx(self,x,y):
        return 36 * y * x ** 2 - 8 * y ** 3
    def get_U11_dy(self,x,y):
        return 3*x ** 4 - 12 * x ** 2 * y ** 2
    def get_U11_dyy(self,x,y):
        return -24 * y * x ** 2

    def get_U12(self,x,y):
        return 8.0*y ** 5 - 45*y*x ** 4 - 80 * y ** 3 * x ** 2 *np.log(x) + 60 * y * x ** 4 * np.log(x)
    def get_U12_dx(self,x,y):
        return -160*y ** 3 * x * np.log(x) - 80*y ** 3 * x + 240 * y * x ** 3 * np.log(x) - 120*y*x ** 3
    def get_U12_dxx(self,x,y):
        return -160 * y ** 3 * np.log(x) - 240* y ** 3 + 720*y*x ** 2 * np.log(x) - 120*y*x ** 2
    def get_U12_dy(self,x,y):
        return 40*y ** 4 - 45*x ** 4 - 240 * x ** 2*y**2* np.log(x) + 60 * x ** 4 * np.log(x)
    def get_U12_dyy(self,x,y):
        return 160*y ** 3 - 480*y*x ** 2 * np.log(x)

    def outer_point(self):
        return 1.0+self.eps, 0.0

    def inner_point(self):
        return 1.0-self.eps, 0.0

    def high_point(self):
        return 1.0-self.delta*self.eps, self.kappa*self.eps

    def sep_point(self):
        shift = 0.1
        return 1.0-(1.0+shift)*self.delta*self.eps, (1.0+shift)*self.kappa*self.eps

    def get_point_flux(self,x,y):
        U1 = self.get_U1(x,y)
        U2 = self.get_U2(x,y)
        U3 = self.get_U3(x,y)
        U4 = self.get_U4(x,y)
        U5 = self.get_U5(x,y)
        U6 = self.get_U6(x,y)
        U7 = self.get_U7(x,y)
        APart, nonAPart = self.get_UP(x,y)
        UH = [U1,U2,U3,U4,U5,U6,U7]

        # If calculating single null, add 5 more polynomials for up-down assymmetry
        if self.divertor == 1:
            U8 = self.get_U8(x,y)
            U9 = self.get_U9(x,y)
            U10 = self.get_U10(x,y)
            U11 = self.get_U11(x,y)
            U12 = self.get_U12(x,y)
            UH = UH + [U8,U9,U10,U11,U12]

        if self.beta_limit == True:
            U0 = APart
            UH = UH + [U0]
            UP = [nonAPart]
        elif self.beta_limit == False:
            UP = [self.A*APart+nonAPart]

        return np.array(UH), np.array(UP)

    def get_point_slope_dx(self, x,y):
        U1 = self.get_U1_dx(x,y)
        U2 = self.get_U2_dx(x,y)
        U3 = self.get_U3_dx(x,y)
        U4 = self.get_U4_dx(x,y)
        U5 = self.get_U5_dx(x,y)
        U6 = self.get_U6_dx(x,y)
        U7 = self.get_U7_dx(x,y)
        APart, nonAPart = self.get_UP_dx(x,y)
        UH = [U1,U2,U3,U4,U5,U6,U7]


        # If calculating single null, add 5 more polynomials for up-down assymmetry
        if self.divertor == 1:
            U8 = self.get_U8_dx(x,y)
            U9 = self.get_U9_dx(x,y)
            U10 = self.get_U10_dx(x,y)
            U11 = self.get_U11_dx(x,y)
            U12 = self.get_U12_dx(x,y)
            UH = UH + [U8,U9,U10,U11,U12]

        if self.beta_limit == True:
            U0 = APart
            UH = UH + [U0]
            UP = [nonAPart]
        elif self.beta_limit == False:
            UP = [self.A*APart+nonAPart]

        return np.array(UH), np.array(UP)

    def get_point_slope_dy(self, x,y):
        U1 = self.get_U1_dy(x,y)
        U2 = self.get_U2_dy(x,y)
        U3 = self.get_U3_dy(x,y)
        U4 = self.get_U4_dy(x,y)
        U5 = self.get_U5_dy(x,y)
        U6 = self.get_U6_dy(x,y)
        U7 = self.get_U7_dy(x,y)
        APart, nonAPart = self.get_UP_dy(x,y)
        UH = [U1,U2,U3,U4,U5,U6,U7]

        # If calculating single null, add 5 more polynomials for up-down assymmetry
        if self.divertor == 1:
            U8 = self.get_U8_dy(x,y)
            U9 = self.get_U9_dy(x,y)
            U10 = self.get_U10_dy(x,y)
            U11 = self.get_U11_dy(x,y)
            U12 = self.get_U12_dy(x,y)
            UH = UH + [U8,U9,U10,U11,U12]

        if self.beta_limit == True:
            U0 = APart
            UH = UH + [U0]
            UP = [nonAPart]
        elif self.beta_limit == False:
            UP = [self.A*APart+nonAPart]

        return np.array(UH), np.array(UP)

    def get_point_curvature_dyy(self, x,y):
        U1 = self.get_U1_dyy(x,y)
        U2 = self.get_U2_dyy(x,y)
        U3 = self.get_U3_dyy(x,y)
        U4 = self.get_U4_dyy(x,y)
        U5 = self.get_U5_dyy(x,y)
        U6 = self.get_U6_dyy(x,y)
        U7 = self.get_U7_dyy(x,y)
        APart, nonAPart = self.get_UP_dyy(x,y)
        UH = [U1,U2,U3,U4,U5,U6,U7]

        # If calculating single null, add 5 more polynomials for up-down assymmetry
        if self.divertor == 1:
            U8 = self.get_U8_dyy(x,y)
            U9 = self.get_U9_dyy(x,y)
            U10 = self.get_U10_dyy(x,y)
            U11 = self.get_U11_dyy(x,y)
            U12 = self.get_U12_dyy(x,y)
            UH = UH + [U8,U9,U10,U11,U12]

        if self.beta_limit == True:
            U0 = APart
            UH = UH + [U0]
            UP = [nonAPart]
        elif self.beta_limit == False:
            UP = [self.A*APart+nonAPart]

        return np.array(UH), np.array(UP)

    def get_point_curvature_dxx(self, x,y):
        U1 = self.get_U1_dxx(x,y)
        U2 = self.get_U2_dxx(x,y)
        U3 = self.get_U3_dxx(x,y)
        U4 = self.get_U4_dxx(x,y)
        U5 = self.get_U5_dxx(x,y)
        U6 = self.get_U6_dxx(x,y)
        U7 = self.get_U7_dxx(x,y)
        APart, nonAPart = self.get_UP_dxx(x,y)
        UH = [U1,U2,U3,U4,U5,U6,U7]


        # If calculating single null, add 5 more polynomials for up-down assymmetry
        if self.divertor == 1:
            U8 = self.get_U8_dxx(x,y)
            U9 = self.get_U9_dxx(x,y)
            U10 = self.get_U10_dxx(x,y)
            U11 = self.get_U11_dxx(x,y)
            U12 = self.get_U12_dxx(x,y)
            UH = UH + [U8,U9,U10,U11,U12]

        if self.beta_limit == True:
            U0 = APart
            UH = UH + [U0]
            UP = [nonAPart]
        elif self.beta_limit == False:
            UP = [self.A*APart+nonAPart]

        return np.array(UH), np.array(UP)
