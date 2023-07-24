import numpy as np
from params_emt import *
from numba import jit_module
import matplotlib.pyplot as plt

def pos_hill(x, x0, n):
	'''
	positive hill function
	'''
	return ((x/x0)**n)/(1+(x/x0)**n)

def neg_hill(x, x0, n):
	'''
	negative hill function
	'''
	return 1./(1+(x/x0)**n)

def joint_hill(x1, x2, x01, x02, n1, n2):
	'''
	negative hill fucntion with two arguments - added
	'''
	return 1./(1 + (x1/x01)**n1 + (x2/x02)**n2 )

def pos_hill_der(x, x0, n):
    '''
    first derivative of positive Hill function
    '''
    return (n*((x/x0)**n))/( x*(1 + (x/x0)**n)**2 )

def neg_hill_der(x, x0, n):
    '''
    first derivative of negative Hill function
    '''
    return -(n*((x/x0)**n))/( x*(1 + (x/x0)**n)**2 )

def joint_der(x1, x2, x01, x02, n1, n2):
	'''
	first derivative of joint hill function (derivative taken on variable x1)
	'''
	return -(n1/x1)*( (x1/x01)**n1 )*joint_hill(x1, x2, x01, x02, n1, n2)**2


# all chemical terms below
def FT(R2, T):
	return k0_T + kt*neg_hill(R2, JT, nr2) #- kd_T*T

def Fs(T, s, tgf):
	return k0_s + ks*pos_hill(T+tgf, Js, nt) #- kd_s*s

def FS(s, R3, S):
	return kS*s*neg_hill(R3, JS, nr3) #- kd_S*S

def FR3(S, Z, R3):
	return k0_3 + k3*joint_hill(S, Z, J13, J23, ns, nz) #- kd_3*R3

def Fz(S, z):
	return k0_z + kz*pos_hill(S, Jz, ns) #- kd_z*z

def FZ(z, R2, Z):
	return kZ*z*neg_hill(R2, JZ, nr2) #- kd_Z*Z

def FR2(S, Z, R2):
	return k0_2 + k2*joint_hill(S, Z, J12, J22, ns, nz) #- kd_2*R2

def FE(S, Z, E):
	return ke_1*neg_hill(S, J1e, ns) + ke_2*neg_hill(Z, J2e, nz) #- kd_e*E

def FN(S, Z, N):
	return kn_1*pos_hill(S, J1n, ns) + kn_2*pos_hill(Z, J2n, nz) #- kd_n*N


def time_course_emt(b, tgf, tm, ic_vec, sigma=0., check_neg=False,dt = 0.001):
	'''
	time course for the full model with unspliced/spliced split
	option to set up stochastic ODEs with sigma>0
	b: unspliced-spliced conversion rate
	tgf: external TGFB input
	tm: length of simulation
	ic_vec: numpy array of initial conditions (must be of length 18)
	sigma: amplitude of white noise (default=0.)
	check_neg: if True, set negative values to zero after each iteration
	'''
	dt = 0.001
	t = int(tm/dt)+1

	T, Tu = np.zeros(t), np.zeros(t)
	s, su = np.zeros(t), np.zeros(t)
	S, Su = np.zeros(t), np.zeros(t)
	R3, R3u = np.zeros(t), np.zeros(t)
	z, zu = np.zeros(t), np.zeros(t)
	Z, Zu = np.zeros(t), np.zeros(t)
	R2, R2u = np.zeros(t), np.zeros(t)
	E, Eu = np.zeros(t), np.zeros(t)
	N, Nu = np.zeros(t), np.zeros(t)

	T[0], s[0], S[0], R3[0], z[0], Z[0], R2[0], E[0], N[0] = ic_vec[9],ic_vec[10],ic_vec[11],ic_vec[12],ic_vec[13],\
																	  ic_vec[14],ic_vec[15],ic_vec[16],ic_vec[17]
	Tu[0], su[0], Su[0], R3u[0], zu[0], Zu[0], R2u[0], Eu[0], Nu[0] = ic_vec[0],ic_vec[1],ic_vec[2],ic_vec[3],ic_vec[4],\
																	  ic_vec[5],ic_vec[6],ic_vec[7],ic_vec[8]

	for i in range(t-1):
		Tu[i+1] = Tu[i] + dt*FT(R2[i], T[i]) - dt*b*Tu[i] + np.sqrt(Tu[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)
		T[i + 1] = T[i] + dt*b*Tu[i] - dt*kd_T*T[i] + np.sqrt(T[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)

		su[i+1] = su[i] + dt*Fs(T[i], s[i], tgf) - dt*b*su[i] + np.sqrt(su[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)
		s[i + 1] = s[i] + dt*b*su[i] - dt*kd_s*s[i] + np.sqrt(s[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)

		Su[i + 1] = Su[i] + dt * FS(s[i], R3[i], S[i]) - dt*b*Su[i] + np.sqrt(Su[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)
		S[i + 1] = S[i] + dt*b*Su[i] - dt*kd_S*S[i] + np.sqrt(S[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)

		R3u[i+1] = R3u[i] + dt*FR3(S[i], Z[i], R3[i]) - dt*b*R3u[i] + np.sqrt(R3u[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)
		R3[i + 1] = R3[i] + dt*b*R3u[i] - dt*kd_3*R3[i] + np.sqrt(R3[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)

		zu[i+1] = zu[i] + dt*Fz(S[i], z[i]) - dt*b*zu[i] + np.sqrt(zu[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)
		z[i + 1] = z[i] + dt*b*zu[i] - dt*kd_z*z[i] + np.sqrt(z[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)

		Zu[i+1] = Zu[i] + dt*FZ(z[i], R2[i], Z[i]) - dt*b*Zu[i] + np.sqrt(dt*Zu[i])*sigma*np.random.normal(loc=0., scale=1.)
		Z[i + 1] = Z[i] + dt*b*Zu[i] - dt*kd_Z*Z[i] + np.sqrt(Z[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)

		R2u[i+1] = R2u[i] + dt*FR2(S[i], Z[i], R2[i]) - dt*b*R2u[i] + np.sqrt(R2u[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)
		R2[i + 1] = R2[i] + dt*b*R2u[i] - dt*kd_2*R2[i] + np.sqrt(R2[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)

		Eu[i+1] = Eu[i] + dt*FE(S[i], Z[i], E[i]) - dt*b*Eu[i] + np.sqrt(Eu[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)
		E[i + 1] = E[i] + dt*b*Eu[i] - dt*kd_e*E[i] + np.sqrt(E[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)

		Nu[i+1] = Nu[i] + dt*FN(S[i], Z[i], N[i]) - dt*b*Nu[i] + np.sqrt(Nu[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)
		N[i + 1] = N[i] + dt*b*Nu[i] - dt*kd_n*N[i] + np.sqrt(N[i]*dt)*sigma*np.random.normal(loc=0., scale=1.)

		### enforce non-negative variables by hand ###
		if check_neg:
			### unspliced species
			if Tu[i + 1] < 0.:
				Tu[i + 1] = 0.
			if su[i + 1] < 0.:
				su[i + 1] = 0.
			if Su[i + 1] < 0.:
				Su[i + 1] = 0.
			if R3u[i + 1] < 0.:
				R3u[i + 1] = 0.
			if zu[i + 1] < 0.:
				zu[i + 1] = 0.
			if Zu[i + 1] < 0.:
				Zu[i + 1] = 0.
			if R2u[i + 1] < 0.:
				R2u[i + 1] = 0.
			if Eu[i + 1] < 0.:
				Eu[i + 1] = 0.
			if Nu[i + 1] < 0.:
				Nu[i + 1] = 0.
			### spliced species
			if T[i + 1] < 0.:
				T[i + 1] = 0.
			if s[i + 1] < 0.:
				s[i + 1] = 0.
			if S[i + 1] < 0.:
				S[i + 1] = 0.
			if R3[i + 1] < 0.:
				R3[i + 1] = 0.
			if z[i + 1] < 0.:
				z[i + 1] = 0.
			if Z[i + 1] < 0.:
				Z[i + 1] = 0.
			if R2[i + 1] < 0.:
				R2[i + 1] = 0.
			if E[i + 1] < 0.:
				E[i + 1] = 0.
			if N[i + 1] < 0.:
				N[i + 1] = 0.
	X_u = np.vstack((Tu, su, Su, R3u, zu, Zu, R2u, Eu, Nu))
	X_s = np.vstack((T, s, S, R3, z, Z, R2, E, N))
	return X_u, X_s
    #return Tu, su, Su, R3u, zu, Zu, R2u, Eu, Nu, T, s, S, R3, z, Z, R2, E, N

    #return {'Tu':Tu, 'su':su, 'Su':Su, 'R3u':R3u, 'zu':zu, 'Zu':Zu, 'R2u':R2u, 'Eu':Eu, 'Nu':Nu, 'T':T, 's':s, 'S':S, 'R3':R3, 'z':z, 'Z':Z, 'R2':R2, 'E':E, 'N':N}


jit_module()



def plot_traj(ax, tm, E, E_epi, E_hyb, E_mes):
    ax.plot(np.linspace(0, tm, E.size), E)
    plt.plot([0, tm], [E_epi, E_epi], 'g--', lw=1, label='E FP')
    plt.plot([0, tm], [E_hyb, E_hyb], '--', color='orange', lw=1, label='E/M FP')
    plt.plot([0, tm], [E_mes, E_mes], 'r--', lw=1, label='M FP')
    plt.xlim([0, tm])
    plt.ylim([-0.1, 1.2 * np.amax(E)])
    plt.xlabel('t')
    plt.ylabel('E-cad')
    plt.legend(loc='upper center', ncol=3)