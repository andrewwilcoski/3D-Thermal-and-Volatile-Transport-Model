"""
Adapted from heat1d.py

Thermal model for a 3D surface (triangular mesh) on the Moon. Created to model the interior temperatures of lunar pits.

Created 07/06/2021
"""

# Physical constants:
sigma = 5.67051196e-8 # Stefan-Boltzmann Constant
#S0 = 1361.0 # Solar constant at 1 AU [W.m-2]
chi = 2.7 # Radiative conductivity parameter [Mitchell and de Pater, 1994]
R350 = chi/350**3 # Useful form of the radiative conductivity

# Numerical parameters:
F = 0.5 # Fourier Mesh Number, must be <= 0.5 for stability
m = 10 # Number of layers in upper skin depth [default: 10]
n = 5 # Layer increase with depth: dz[i] = dz[i-1]*(1+1/n) [default: 5]
b = 20 # Number of skin depths to bottom layer [default: 20]

# Accuracy of temperature calculations
# The model will run until the change in temperature of the bottom layer
# is less than DTBOT over one diurnal cycle
DTSURF = 0.1 # surface temperature accuracy [K]
DTBOT = DTSURF # bottom layer temperature accuracy [K]
#NYEARSEQ = 1 # equilibration time [orbits]
NPERDAY = 24 # minimum number of time steps per diurnal cycle

# NumPy is needed for various math operations
import numpy as np

# MatPlotLib and Pyplot are used for plotting
import matplotlib.pyplot as plt
import matplotlib as mpl

# Methods for calculating solar angles from orbits
import orbits_slopes

# Planets database
import planets

# Additional libraries
import perftest as pt
import trimesh
#import open3d as o3d
import time
import multiprocessing as mp
from numba import jit, njit
from numba import boolean
import os
import numpy.ma as ma
from tqdm import tqdm

# Models contain the profiles and model results
class model(object):
    
    # Initialization
    def __init__(self, mesh, A=0.12, emis=0.95, num_ref=10, planet=planets.Moon, lat=0, ndays=1, nu=0, skip_VF=False, F_los=0, \
                idx_los=0, shadow=True, daz=1*np.pi/180, dzen=1*np.pi/180, shadowfax1=False, skip_horizon=False, horizon_az=0, \
                horizon_zen=0, intersect_sky=0, open3d=False, pro_num_VF=4, num_split_VF=4, \
                pro_num_Hor=4, num_split_Hor=4, path='pit_model', save_VF=False, save_Hor=False, save_Geom=False, \
                skip_Geom=False, tri_vert=0, tri_cent=0, tri_area=0, tri_norm=0, tri_rays=0, save_Scat=False, skip_Scat=False, \
                f_vis=0, f_IR=0, update_dir=False, shadowfax3=False, NYEARSEQ=1, num_save='N_steps', save_surf=True):
        
        # Initialize
        self.planet = planet
        self.lat = lat
        self.A = A #Surface albedo
        self.emis = emis #Surface emissivity
        self.Sabs = self.planet.S * (1.0 - self.A)
        self.r = self.planet.rAU # solar distance [AU]
        self.nu = nu # orbital true anomaly [rad]
        self.nuout = nu # orbital true anomaly [rad] at which simulation starts
        self.nudot = np.float() # rate of change of true anomaly [rad/s]
        self.dec = np.float() # solar declination [rad]
        self.shadow = shadow #Whether or not to include shadowing
        self.shadowfax1 = shadowfax1
        self.shadowfax3 = shadowfax3
        self.num_save = num_save
        self.save_surf = save_surf
        
        #Make directory to save files in
        if (update_dir == False): #Are you trying to update an existing directory or not?
            if ( (save_VF==True) | (save_Hor==True) | (save_Geom==True) | (save_Scat==True) ):
                path_num = 0
                path_org = path
                while (os.path.exists(path) == True):
                    path = path_org+'_'+str(path_num)
                    path_num += 1
                os.mkdir(path)
        
        # Initialize mesh triangle geometry and view factors for multi-scattering
        
        ########### Triangle Geometry ###########
        self.mesh = mesh
        t0 = time.time()
        if (skip_Geom==True):
            self.tri_vert, self.tri_cent, self.tri_area, self.tri_norm, self.tri_rays = tri_vert, tri_cent, tri_area, tri_norm,\
                                                                                        tri_rays
        else:
            if (open3d==True):
                self.tri_vert, self.tri_cent, self.tri_area, self.tri_norm, self.tri_rays = Triangle_Geometry(self.mesh)
            else:
                self.tri_vert, self.tri_cent, self.tri_area, self.tri_norm, self.tri_rays = Triangle_Geometry_trimesh(self.mesh)
            
        if (save_Geom==True):
            np.save(path+'/tri_vert', self.tri_vert)
            np.save(path+'/tri_cent', self.tri_cent)
            np.save(path+'/tri_area', self.tri_area)
            np.save(path+'/tri_norm', self.tri_norm)
            np.save(path+'/tri_rays', self.tri_rays)
        
        t1 = time.time()
        print ('Tri Geom:', t1-t0)
        self.num_facet = self.tri_cent.shape[0] #Number of facets (triangles) in triangle mesh
        #self.F_los, self.idx_los = viewFactors_3D(self.tri_vert, self.tri_cent, self.tri_area, self.tri_norm)
        
        ########### View Factors ###########
        if (skip_VF==True):
            self.F_los, self.idx_los = F_los, idx_los
        elif (skip_VF=='vf3'):
            self.F_los = viewFactors_3D_3(self.tri_vert, self.tri_cent, self.tri_area, self.tri_norm, \
                                                        self.tri_rays)
        elif (skip_VF=='multi'):
            self.F_los = np.zeros([self.tri_cent.shape[0], self.tri_vert.shape[0]])
            #pro_num: Number of processes allowed at once
            #num_split: Number of arrays to split pro_idx into
            pro_idx_VF = np.arange(0, self.tri_cent.shape[0]) #Make array of indices to match number of facets
            #Split index array into multiple index arrays of ~equal length
            pro_idx_VF = np.array_split(pro_idx_VF, num_split_VF)
            #print ('pro_idx:', pro_idx)
            
            pool = mp.Pool(processes=pro_num_VF)
            results = [pool.apply_async(viewFactors_3D_5, args=(self.tri_vert[:,:,:], self.tri_cent[i,:],\
                                         self.tri_area[:], self.tri_norm[:,:], self.tri_rays[i,:,:], i)) for i in pro_idx_VF]
            
            output = np.array([m.get() for m in results])
            output = np.vstack(output)
            F_los, idx_VF = output[:,:-1], output[:,-1].astype(int)
            self.F_los[idx_VF,:] = F_los
            
            pool.close()
            pool.join()
            
        else:
            #self.F_los, self.idx_los = viewFactors_3D_2(self.tri_vert, self.tri_cent, self.tri_area, self.tri_norm, \
            #                                            self.tri_rays)
            self.F_los = viewFactors_3D_4(self.tri_vert, self.tri_cent, self.tri_area, self.tri_norm, \
                                                        self.tri_rays)
        
        if (save_VF==True):
            np.save(path+'/F_los', self.F_los)
        
        t2 = time.time()
        print ('View Factors:', t2-t1)
        
        ########### Multiple-Scattering ###########
        self.num_ref = num_ref #Maximum number of reflections accounted for
        #Scattering coefficients for visible light (f_vis) and IR (f_IR) reflection
        if (skip_Scat==True):
            self.f_vis, self.f_IR = f_vis, f_IR
        else:
            self.f_vis, self.f_IR = Multi_Scattering(self.F_los, self.A, self.emis, self.num_ref)
        
        if (save_Scat==True):
            np.save(path+'/f_vis', self.f_vis)
            np.save(path+'/f_IR', self.f_IR)
        
        t3 = time.time()
        print ('Multi Scat Coeffs:', t3-t2)
        
        ########### Find Horizons ###########
        self.daz = daz
        self.dzen = dzen
        self.skip_horizon = skip_horizon
        if (skip_horizon==True):
            self.horizon_az, self.horizon_zen, self.intersect_sky = horizon_az, horizon_zen, intersect_sky
        elif(skip_horizon=='horizon_3'):
            self.horizon_az, self.horizon_zen, self.intersect_sky = Find_Horizon_3(self.tri_cent, self.tri_vert, self.daz,\
                                                                                   self.dzen)
        elif (skip_horizon=='jit'):
            horizon_az = np.arange(0, 2*np.pi+self.daz, self.daz) #azimuth angles [rad]
            #horizon_zen = np.arange(0, np.pi, self.dzen) #zenith angles [rad]
            horizon_zen = np.arange(0, np.pi/2+self.dzen, self.dzen) #zenith angles [rad]
            intersect = np.ones([self.tri_cent.shape[0], self.tri_vert.shape[0]], dtype=bool)
            intersect_sky = np.zeros([self.tri_cent.shape[0], horizon_az.size, horizon_zen.size], dtype=bool)
            
            self.horizon_az, self.horizon_zen, self.intersect_sky = Find_Horizon_4(self.tri_cent, self.tri_vert, self.tri_norm,\
                                                                                   horizon_az, horizon_zen, intersect,\
                                                                                   intersect_sky, self.daz, self.dzen)
        elif (skip_horizon=='multi'):
            self.horizon_az = np.arange(0, 2*np.pi, self.daz) #azimuth angles [rad]
            #self.horizon_zen = np.arange(0, np.pi, self.dzen) #zenith angles [rad]
            self.horizon_zen = np.arange(0, np.pi/2+self.dzen, self.dzen) #zenith angles [rad]
            
            self.intersect_sky = np.zeros([self.tri_cent.shape[0], self.horizon_az.size, self.horizon_zen.size], dtype=bool)
            #pro_num: Number of processes allowed at once
            #num_split: Number of arrays to split pro_idx into
            pro_idx_Hor = np.arange(0, self.horizon_az.size) #Make array of indices to match size of horizon_az
            #Split index array into multiple index arrays of ~equal length
            pro_idx_Hor = np.array_split(pro_idx_Hor, num_split_Hor)
            #print ('pro_idx_Hor:', pro_idx_Hor)
            
            pool = mp.Pool(processes=pro_num_Hor)
            results = [pool.apply_async(Find_Horizon_5, args=(self.tri_cent, self.tri_vert, self.tri_norm, self.horizon_az[i], \
                                                              self.horizon_zen, i, self.daz, self.dzen)) for i in pro_idx_Hor]
            
            output = np.array([m.get() for m in results])
            output = np.hstack(output)
            intersect_sky, idx_Hor = output[:,:,:-1], output[0,:,-1].astype(int)
            self.intersect_sky[:,idx_Hor,:] = intersect_sky
            
            pool.close()
            pool.join()
            
        elif (skip_horizon=='multi1'):
            self.horizon_az = np.arange(0, 2*np.pi, self.daz) #azimuth angles [rad]
            self.horizon_zen = np.arange(0, np.pi/2+self.dzen, self.dzen) #zenith angles [rad]
            
            self.r_dir, self.theta_sky = Solar_Rays(self.horizon_az, self.horizon_zen, self.tri_norm)
            
            self.intersect_sky = np.zeros([self.tri_cent.shape[0], self.horizon_az.size, self.horizon_zen.size], dtype=bool)
            #pro_num: Number of processes allowed at once
            #num_split: Number of arrays to split pro_idx into
            pro_idx_Hor = np.arange(0, self.tri_cent.shape[0]) #Make array of indices to match number of facets
            #Split index array into multiple index arrays of ~equal length
            pro_idx_Hor = np.array_split(pro_idx_Hor, num_split_Hor)
            #print ('pro_idx_Hor:', pro_idx_Hor)
            
            pool = mp.Pool(processes=pro_num_Hor)
            results = [pool.apply_async(Find_Horizon_6, args=(self.tri_cent[i,:], self.tri_vert, self.horizon_az, \
                                                  self.horizon_zen, self.r_dir, self.theta_sky[i,:], i)) for i in pro_idx_Hor]
            
            output = np.array([m.get() for m in results])
            output = np.hstack(output)
            intersect_sky, idx_Hor = output[:,:,:-1], output[0,:,-1].astype(int)
            self.intersect_sky[idx_Hor,:,:] = intersect_sky
            
            pool.close()
            pool.join()
            
        if (save_Hor==True):
            np.save(path+'/horizon_az', self.horizon_az)
            np.save(path+'/horizon_zen', self.horizon_zen)
            np.save(path+'/intersect_sky', self.intersect_sky)
            
        t4 = time.time()
        print ('Find Horizon:', t4-t3)
        
        # Calculate slope and azimuth of each facet
        r = np.sqrt(np.sum(self.tri_norm**2, axis=1))
        #beta: azimuth of facet [rad]
        self.tri_norm[self.tri_norm[:,0]==0,0] = np.abs(self.tri_norm[self.tri_norm[:,0]==0,0]) #Some of the x values were -0
        self.beta = np.arctan(self.tri_norm[:,1]/self.tri_norm[:,0])
        self.beta[np.isnan(self.beta)] = 0
        self.beta[(self.tri_norm[:,0]<0) & (self.tri_norm[:,1]>0)] += np.pi
        self.beta[(self.tri_norm[:,0]<0) & (self.tri_norm[:,1]<0)] -= np.pi
        #alpha: slope of facet [rad]
        self.alpha = np.arccos(self.tri_norm[:,2]/r)
        self.alpha_max = np.amax(self.alpha)
        
        # Initialize arrays
        self.Qs = np.zeros(self.num_facet) # total surface flux
        self.Q_solar = np.zeros(self.num_facet) # solar flux
        self.Q_IR = np.zeros(self.num_facet) # scattered IR emission
        self.Q_vis = np.zeros(self.num_facet) # scattered visible light
        self.cosz = np.float() #cosSolarZenith
        
        # Initialize model profile
        self.profile = profile(planet, lat, self.num_facet, self.emis, self.alpha_max)
        
        # Model run times
        # Equilibration time -- TODO: change to convergence check
        self.equiltime = NYEARSEQ*planet.year - \
                        (NYEARSEQ*planet.year)%planet.day
        # Run time for output
        self.endtime = self.equiltime + ndays*planet.day
        self.t = 0.
        self.dt = getTimeStep(self.profile, self.planet.day)
        # Check for maximum time step
        self.dtout = self.dt
        dtmax = self.planet.day/NPERDAY
        if self.dt > dtmax:
            self.dtout = dtmax
            
        ####
        self.dtout = self.dtout/2 #self.dtout/2
        self.dt = self.dtout
        ####
        
        print ('self.dtout:', self.dtout)
        
        # Array for output temperatures and local times
        if (self.num_save == 'N_steps'):
            self.N_steps = np.int( (ndays*planet.day)/self.dtout )
            print ('N_steps:', self.N_steps)
            self.N_z = self.profile.z.shape[0]
            if (self.save_surf==True):
                self.T = np.zeros([self.N_steps, self.num_facet])
            else:
                self.T = np.zeros([self.N_steps, self.N_z, self.num_facet])
            self.Qst = np.zeros([self.N_steps, self.num_facet]) #Qs as a funciton of t
            self.Q_solart = np.zeros([self.N_steps, self.num_facet])
            self.Q_IRt = np.zeros([self.N_steps, self.num_facet])
            self.Q_vist = np.zeros([self.N_steps, self.num_facet])
            self.coszt = np.zeros([self.N_steps]) #cosSolarZenith as a funciton of t
            self.lt = np.zeros([self.N_steps])
            
        else:
            self.N_steps = np.int( (ndays*planet.day)/self.dtout )
            print ('N_steps:', self.N_steps)
            self.N_z = self.profile.z.shape[0]
            if (self.save_surf==True):
                self.T = np.zeros([self.num_save, self.num_facet])
            else:
                self.T = np.zeros([self.num_save, self.N_z, self.num_facet])
            self.Qst = np.zeros([self.num_save, self.num_facet]) #Qs as a funciton of t
            self.Q_solart = np.zeros([self.num_save, self.num_facet])
            self.Q_IRt = np.zeros([self.num_save, self.num_facet])
            self.Q_vist = np.zeros([self.num_save, self.num_facet])
            self.coszt = np.zeros([self.num_save]) #cosSolarZenith as a funciton of t
            self.lt = np.zeros([self.num_save])
            
    
    def run(self):
        
        # Equilibrate the model
        while (self.t < self.equiltime):
            self.advance()
        
        # Run through end of model and store output
        self.dt = self.dtout
        self.t = 0. # reset simulation time
        self.nu = self.nuout
        save_count = 0
        print ('START SIM: ', self.nu)
        for i in tqdm(range(0,self.N_steps)):
            self.advance()
            if (self.num_save == 'N_steps'):
                if (self.save_surf==True):
                    self.T[i,:] = self.profile.T[0,:] # temperature [K]
                else:
                    self.T[i,:,:] = self.profile.T # temperature [K]
                self.Qst[i,:] = self.Qs # Surface flux
                self.Q_solart[i,:] = self.Q_solar
                self.Q_IRt[i,:] = self.Q_IR
                self.Q_vist[i,:] = self.Q_vis
                self.coszt[i] = self.cosz # Surface flux
                self.lt[i] = self.t/self.planet.day*24.0 # local time [hr]
            
            elif ( (i == save_count*int(self.N_steps/self.num_save)) & (save_count<self.num_save) ):
                if (self.save_surf==True):
                    self.T[save_count,:] = self.profile.T[0,:] # temperature [K]
                else:
                    self.T[save_count,:,:] = self.profile.T # temperature [K]
                self.Qst[save_count,:] = self.Qs # Surface flux
                self.Q_solart[save_count,:] = self.Q_solar
                self.Q_IRt[save_count,:] = self.Q_IR
                self.Q_vist[save_count,:] = self.Q_vis
                self.coszt[save_count] = self.cosz # Surface flux
                self.lt[save_count] = self.t/self.planet.day*24.0 # local time [hr]
                save_count += 1
            
    def advance(self):
        self.updateOrbit()
        t4 = time.time()
        self.surfFlux()
        #print ('self.Qs[1551]:', self.Qs[1551])
        #print ('self.Q_solar[1551]:', self.Q_solar[1551])
        #print ('self.Q_vis[1551]:', self.Q_vis[1551])
        #print ('self.Q_IR[1551]:', self.Q_IR[1551])
        ##print ('np.count_nonzero(np.isnan(self.Q_IR):', np.count_nonzero(np.isnan(self.Q_IR)))
        ##print ('np.where(np.isnan(self.Q_IR)):', np.where(np.isnan(self.Q_IR)))
        t5 = time.time()
        #print ('surfFlux:', t5-t4)
        #print ('\n nu:', self.nu)
        self.profile.update_T(self.dt, self.Qs, self.planet.Qb)
        ##print ('np.count_nonzero(np.isnan(self.profile.T[0,:]):', np.count_nonzero(np.isnan(self.profile.T[0,:])))
        ##print ('np.count_nonzero(np.isnan(self.profile.T):', np.count_nonzero(np.isnan(self.profile.T)))
        ##print ('np.where(np.isnan(self.profile.T)):', np.where(np.isnan(self.profile.T)))
        #print ('self.profile.T[:,1551]:', self.profile.T[:,1551])
        self.profile.update_cp()
        ##print ('np.count_nonzero(np.isnan(self.profile.cp):', np.count_nonzero(np.isnan(self.profile.cp)))
        ##print ('self.profile.cp:', self.profile.cp, '\n')
        self.profile.update_k()
        #print ('self.profile.k:', self.profile.k, '\n')
        #print ('self.profile.T[0,:]:', self.profile.T[0,:])
        #print ('self.profile.T[0,1944]:', self.profile.T[0,1944])
        #print ('np.any(np.isnan(self.profile.T[0,:])):', np.any(np.isnan(self.profile.T[0,:])))
        #print ('np.where(np.isnan(self.profile.T[0,:])==True):', np.where(np.isnan(self.profile.T[0,:])==True), '\n')
        #print ('self.profile.k[0,:]:', self.profile.k[0,:])
        #print ('np.any(np.isnan(self.profile.k[0,:])):', np.any(np.isnan(self.profile.k[0,:])))
        #print ('np.where(np.isnan(self.profile.k[0,:])==True):', np.where(np.isnan(self.profile.k[0,:])==True))
        #print ('self.profile.cp[0,:]:', self.profile.cp[0,:])
        #print ('np.any(np.isnan(self.profile.cp[0,:])):', np.any(np.isnan(self.profile.cp[0,:])))
        #print ('np.where(np.isnan(self.profile.cp[0,:])==True):', np.where(np.isnan(self.profile.cp[0,:])==True), '\n')
        self.t += self.dt # Increment time
    
    def updateOrbit(self):
        orbits_slopes.orbitParams(self)
        self.nu += self.nudot * self.dt
    
    # Surface heating rate
    # May include solar and infrared contributions, reflectance phase function
    def surfFlux(self):
        h = orbits_slopes.hourAngle(self.t, self.planet.day) # hour angle
        
        #cosine of incidence angle and solar azimuth angle
        c, sa = orbits_slopes.cosSlopeSolarZenith(self.lat, self.dec, h, self.alpha, self.beta)
        cosz = orbits_slopes.cosSolarZenith(self.lat, self.dec, h) #cosine of incidence angle for horizontal surface
        self.cosz = cosz
        
        ### Ignore variable albedo for now
        #i = np.arccos(c) # solar incidence angle [rad]
        #a = self.planet.albedoCoef[0]
        #b = self.planet.albedoCoef[1]
        #f = (1.0 - albedoVar(self.planet.albedo, a, b, i))/(1.0 - self.planet.albedo)
        ###
        
        #Insolation (Albedo included in Sabs)
        f = 1
        self.Q_solar = f * self.Sabs * (self.r/self.planet.rAU)**-2 * c
        t6 = time.time()
        if (self.shadowfax3==True):
            frac_shadow = Shadowfax_3D_3(sa, cosz, self.horizon_az, self.horizon_zen, self.intersect_sky)
            self.Q_solar = self.Q_solar*frac_shadow
        else:
            idx_shadow = Shadowfax_3D_2(sa, cosz, self.horizon_az, self.horizon_zen, self.intersect_sky)
            #print ('idx_shadow (surfFlux):', idx_shadow)
            self.Q_solar[idx_shadow] = 0 # insolation is zero where facets are shadowed by other facets.
        t7 = time.time()
        #print ('     Shadowfax:', t7-t6)
        
        #Scattering
        self.Q_vis = np.dot(self.Q_solar, self.f_vis.T)
        self.Q_IR = np.dot(self.profile.T[0,:]**4, self.f_IR.T) #np.zeros_like(self.Q_IR) #
        
        self.Qs = self.Q_solar + self.Q_vis + self.Q_IR
        
        #print ('   self.Qs[1944]:', self.Qs[1944])
        #print ('   self.Q_solar[1944]:', self.Q_solar[1944])
        #print ('   self.Q_vis[1944]:', self.Q_vis[1944])
        #print ('   self.profile.T[:,1944]:', self.profile.T[:,1944])
        #print ('np.any(np.isnan(self.profile.T[0,:])):', np.any(np.isnan(self.profile.T[0,:])))
        #print ('   self.Q_IR:', self.Q_IR[1944])
        
    def surfFlux_test(self, h):
        #h = orbits_slopes.hourAngle(self.t, self.planet.day) # hour angle
        
        #cosine of incidence angle and solar azimuth angle
        c, sa = orbits_slopes.cosSlopeSolarZenith(self.lat, self.dec, h, self.alpha, self.beta)
        cosz = orbits_slopes.cosSolarZenith(self.lat, self.dec, h) #cosine of incidence angle for horizontal surface
        self.cosz = cosz
        
        ### Ignore variable albedo for now
        #i = np.arccos(c) # solar incidence angle [rad]
        #a = self.planet.albedoCoef[0]
        #b = self.planet.albedoCoef[1]
        #f = (1.0 - albedoVar(self.planet.albedo, a, b, i))/(1.0 - self.planet.albedo)
        ###
        
        #Insolation
        f = 1
        self.Q_solar = f * self.Sabs * (self.r/self.planet.rAU)**-2 * c
        t6 = time.time()
        idx_shadow = np.zeros_like(self.alpha)
        closest_ray_idx = 0
        intersect = 0
        
        ########## TO ADD: if (np.all(self.Q_solar == 0)): don't even bother to check for shadowing!
        
        if (self.shadow==True):
            if (self.shadowfax1==False):
                idx_shadow, closest_ray_idx, intersect = Shadowfax_3D(sa, cosz, self.tri_rays, self.tri_cent, self.tri_vert)
                print ('idx_shadow:', idx_shadow)
                print ('self.Q_solar:', self.Q_solar)
                self.Q_solar[idx_shadow] = 0 # insolation is zero where facets are shadowed by other facets.
            else:
                idx_shadow = Shadowfax_3D_1(sa, cosz, self.horizon_az, self.horizon_zen1, self.horizon_zen2)
                print ('idx_shadow:', idx_shadow)
                print ('self.Q_solar:', self.Q_solar)
                self.Q_solar[idx_shadow] = 0 # insolation is zero where facets are shadowed by other facets.
            
        t7 = time.time()
        print ('Shadowfax:', t7-t6)
        
        print ('np.arccos(cosz):', np.arccos(cosz)*180/np.pi)
        print ('sa:', sa*180/np.pi)
        
        #if (np.all(self.Q_solar==0)):
        #    print ('h:', h*180./np.pi)
        #    print ('idx_shadow:', idx_shadow)
        
        #Scattering
        self.Q_vis = np.dot(self.Q_solar, self.f_vis.T)
        self.Q_IR = np.dot(self.profile.T[0,:]**4, self.f_IR.T)
        
        self.Qs = self.Q_solar + self.Q_vis + self.Q_IR
        
        return self.Qs, self.Q_solar, self.Q_vis, self.Q_IR, idx_shadow, closest_ray_idx, intersect

    def surfFlux_test1(self, h):
        #h = orbits_slopes.hourAngle(self.t, self.planet.day) # hour angle
        
        #cosine of incidence angle and solar azimuth angle
        c, sa = orbits_slopes.cosSlopeSolarZenith(self.lat, self.dec, h, self.alpha, self.beta)
        cosz = orbits_slopes.cosSolarZenith(self.lat, self.dec, h) #cosine of incidence angle for horizontal surface
        self.cosz = cosz
        
        ### Ignore variable albedo for now
        #i = np.arccos(c) # solar incidence angle [rad]
        #a = self.planet.albedoCoef[0]
        #b = self.planet.albedoCoef[1]
        #f = (1.0 - albedoVar(self.planet.albedo, a, b, i))/(1.0 - self.planet.albedo)
        ###
        
        #Insolation
        f = 1
        self.Q_solar = f * self.Sabs * (self.r/self.planet.rAU)**-2 * c
        t6 = time.time()
        if (self.skip_horizon==False):
            idx_shadow = Shadowfax_3D_1(sa, cosz, self.horizon_az, self.horizon_zen1, self.horizon_zen2)
        else:
            idx_shadow = Shadowfax_3D_2(sa, cosz, self.horizon_az, self.horizon_zen, self.intersect_sky)
        #print ('idx_shadow (surfFlux):', idx_shadow)
        self.Q_solar[idx_shadow] = 0 # insolation is zero where facets are shadowed by other facets.
        t7 = time.time()
        #print ('     Shadowfax:', t7-t6)
        
        #Scattering
        self.Q_vis = np.dot(self.Q_solar, self.f_vis.T)
        self.Q_IR = np.dot(self.profile.T[0,:]**4, self.f_IR.T)
        
        self.Qs = self.Q_solar + self.Q_vis + self.Q_IR
        
        return self.Qs, self.Q_solar, self.Q_vis, self.Q_IR, idx_shadow
    
class profile(object):
    """
    Profiles are objects that contain the model layers
    
    The profile class defines methods for initializing and updating fields
    contained in the model layers, such as temperature and conductivity.
    
    """
    
    def __init__(self, planet=planets.Moon, lat=0, num_facet=1, emis=0.95, alpha_max=0):
        
        self.planet = planet
        self.num_facet = num_facet
        
        # The spatial grid
        self.emissivity = emis
        self.alpha_max = alpha_max
        ks = planet.ks
        kd = planet.kd
        rhos = planet.rhos
        rhod = planet.rhod
        H = planet.H
        cp0 = planet.cp0
        kappa = ks/(rhos*cp0)
        
        self.z = spatialGrid(skinDepth(planet.day, kappa), m, n, b, self.num_facet)
        self.nlayers = np.shape(self.z)[0] # number of model layers
        self.dz = np.diff(self.z[:,0])
        self.d3z = self.dz[1:]*self.dz[0:-1]*(self.dz[1:] + self.dz[0:-1])
        self.g1 = 2*self.dz[1:]/self.d3z[0:] # A.K.A. "p" in the Appendix
        self.g2 = 2*self.dz[0:-1]/self.d3z[0:] # A.K.A. "q" in the Appendix
        
        # Thermophysical properties
        self.kc = kd - (kd-ks)*np.exp(-self.z/H) #0.1*np.ones_like(self.z)
        self.rho = rhod - (rhod-rhos)*np.exp(-self.z/H) #2900*np.ones_like(self.z)
        #For Rock:
        #self.kc = 2*np.ones_like(self.z) #2 from Robertson (1988) Thermal Properties of Rocks, USGS Report
        #self.rho = 3000*np.ones_like(self.z) #[kg.m-3]
        
        # Initialize temperature profile
        self.init_T(planet, lat)
        
        # Initialize conductivity profile
        self.update_k()
        
        # Initialize heat capacity profile
        self.update_cp()
        
        #Thermal Inertia
        self.Gamma = np.sqrt(self.kc*self.rho*self.cp)
        print ('Thermal Inertia:', self.Gamma)
    
    # Temperature initialization
    def init_T(self, planet=planets.Moon, lat=0):
        self.T = np.zeros([self.nlayers, self.num_facet]) \
                 + T_eq(planet, 0) #120 #T_eq(planet, 0)#np.abs(lat) - self.alpha_max)
        #print ('T_eq(planet, 0):', T_eq(planet, 0))
        print ('self.T:', self.T)
    
    # Heat capacity initialization
    def update_cp(self):
        self.cp = heatCapacity(self.planet, self.T)
        ##self.cp = heatCapacity_ice(self.T)
        #self.cp = 840*np.ones_like(self.T)
    
    # Thermal conductivity initialization (temperature-dependent)
    def update_k(self):
        self.k = thermCond(self.kc, self.T)
    
    ##########################################################################
    # Core thermal computation                                               #
    # dt -- time step [s]                                                    #
    # Qs -- surface heating rate [W.m-2]                                     #
    # Qb -- bottom heating rate (interior heat flow) [W.m-2]                 #
    ##########################################################################
    def update_T(self, dt, Qs=0, Qb=0):
        #print ('self.T:', self.T)
        # Coefficients for temperature-derivative terms
        #alpha = self.g1*self.k[0:-2]
        #beta = self.g2*self.k[1:-1]
        alpha = np.transpose(self.g1*self.k[0:-2].T)
        beta = np.transpose(self.g2*self.k[1:-1].T)
        
        # Temperature of first layer is determined by energy balance
        # at the surface
        surfTemp(self, Qs)
        
        # Temperature of the last layer is determined by the interior
        # heat flux
        botTemp(self, Qb)
        
        # This is an efficient vectorized form of the temperature
        # formula, which is much faster than a for-loop over the layers
        self.T[1:-1,:] = self.T[1:-1,:] + dt/(self.rho[1:-1,:]*self.cp[1:-1,:]) * \
                     ( alpha*self.T[0:-2,:] - \
                       (alpha+beta)*self.T[1:-1,:] + \
                       beta*self.T[2:,:] )
        
        #print ('self.T:', self.T, '\n')
        #print ('self.T[(self.T>600) | (self.T<0)]:', self.T[(self.T>600) | (self.T<0)], '\n')
        #print ('np.where(np.isnan(self.T)==True):', np.where(np.isnan(self.T)==True), '\n')
        #print ('self.T[6,2470]:', self.T[6,2470])
        #print ('self.T[7,2470]:', self.T[7,2470], '\n')
        #print ('self.T[:,2470]:', self.T[:,2470], '\n')
     ##########################################################################   
    
    # Simple plot of temperature profile
    def plot(self):
        ax = plt.axes(xlim=(0,400),ylim=(np.min(self.z),np.max(self.z)))
        plt.plot(self.T, self.z)
        ax.set_ylim(1.0,0)
        plt.xlabel('Temperature, $T$ (K)')
        plt.ylabel('Depth, $z$ (m)')
        mpl.rcParams['font.size'] = 14

#---------------------------------------------------------------------------
"""

The functions defined below are used by the thermal code.

"""
#---------------------------------------------------------------------------

# Thermal skin depth [m]
# P = period (e.g., diurnal, seasonal)
# kappa = thermal diffusivity = k/(rho*cp) [m2.s-1]
def skinDepth(P, kappa):
    return np.sqrt(kappa*P/np.pi)

# The spatial grid is non-uniform, with layer thickness increasing downward
def spatialGrid(zs, m, n, b, num_facet):
    dz = np.zeros([1, num_facet]) + zs/m # thickness of uppermost model layer
    z = np.zeros([1, num_facet]) # initialize depth array at zero
    zmax = zs*b # depth of deepest model layer

    i = 0
    while (np.any(z[i, :] < zmax)):
        i += 1
        h = dz[i-1, :]*(1+1/n) # geometrically increasing thickness
        dz = np.append(dz, [h], axis=0) # thickness of layer i
        z = np.append(z, [z[i-1, :] + dz[i, :]], axis=0) # depth of layer i
    
    return z

# Solar incidence angle-dependent albedo model
# A0 = albedo at zero solar incidence angle
# a, b = coefficients
# i = solar incidence angle
#def albedoVar(A0, a, b, i):
#    return A0 + a*(i/(np.pi/4))**3 + b*(i/(np.pi/2))**8
def albedoVar(A0, a, b, i):
    x = A0 + a*(i/(np.pi/4))**3 + b*(i/(np.pi/2))**8
    x[x>1] = 1 #EDIT Make sure albedo never exceeds 1
    return x

# Radiative equilibrium temperature at local noontime
def T_radeq(planet, lat):
    return ((1-planet.albedo)/(sigma*planet.emissivity) * planet.S * np.cos(lat))**0.25

# Equilibrium mean temperature for rapidly rotating bodies
def T_eq(planet, lat):
    return T_radeq(planet, lat)/np.sqrt(2)

# Heat capacity of regolith (temperature-dependent)
# This polynomial fit is based on data from Ledlow et al. (1992) and
# Hemingway et al. (1981), and is valid for T > ~10 K
# The formula yields *negative* (i.e. non-physical) values for T < 1.3 K
def heatCapacity(planet, T):
    c = planet.cpCoeff
    return np.polyval(c, T)

# Temperature-dependent thermal conductivity
# Based on Mitchell and de Pater (1994) and Vasavada et al. (2012)
def thermCond(kc, T):
    return kc*(1 + R350*T**3)

# Surface temperature calculation using Newton's root-finding method
# p -- profile object
# Qs -- heating rate [W.m-2] (e.g., insolation and infared heating)
def surfTemp(p, Qs):
    Ts = p.T[0, :]
    deltaT = Ts
    
    while (np.any(np.abs(deltaT) > DTSURF)):
        x = p.emissivity*sigma*Ts**3
        y = 0.5*thermCond(p.kc[0,:], Ts)/p.dz[0]
    
        # f is the function whose zeros we seek
        f = x*Ts - Qs - y*(-3*Ts+4*p.T[1, :]-p.T[2, :])
        # fp is the first derivative w.r.t. temperature        
        fp = 4*x - \
             3*p.kc[0,:]*R350*Ts**2 * \
                0.5*(4*p.T[1, :]-3*Ts-p.T[2, :])/p.dz[0] + 3*y
        
        # Estimate of the temperature increment
        deltaT = -f/fp
        Ts += deltaT
        
    # Update surface temperature
    p.T[0, :] = Ts

# Bottom layer temperature is calculated from the interior heat
# flux and the temperature of the layer above
def botTemp(p, Qb):
    p.T[-1,:] = p.T[-2,:] + (Qb/p.k[-2,:])*p.dz[-1]

def getTimeStep(p, day):
    dt_min = np.min( F * p.rho[:-1, 0] * p.cp[:-1, 0] * p.dz**2 / p.k[:-1, 0] )
    return dt_min

def Triangle_Geometry(mesh):
    #Returns vertices, centroids, areas, and normals of each triangle in mesh
    mesh_tri = np.asarray(mesh.triangles)
    mesh_vert = np.asarray(mesh.vertices)
    
    tri_vert = np.zeros([mesh_tri.shape[0], mesh_tri.shape[1], 3])
    tri_cent = np.zeros([mesh_tri.shape[0], mesh_tri.shape[1]])
    tri_area = np.zeros([mesh_tri.shape[0]])
    
    tri_norm = np.asarray(mesh.triangle_normals)
    
    tri_rays = np.zeros([mesh_tri.shape[0], mesh_tri.shape[0], 3]) #Rays from each facet to every other facet
    
    tri_idx = np.arange(0, tri_cent.shape[0])
    for i in tri_idx: #for i in range(0, tri.shape[0]):
        tri_vert[i, :, :] = mesh_vert[mesh_tri[i,:], :] #Find vertices of each triangle
        tri_cent[i, :] = np.mean(tri_vert[i, :, :], axis=0) #Calculate centroid of each triangle
        
        v0 = pt.Vec3(tri_vert[i,0,0], tri_vert[i,0,1], tri_vert[i,0,2])
        v1 = pt.Vec3(tri_vert[i,1,0], tri_vert[i,1,1], tri_vert[i,1,2])
        v2 = pt.Vec3(tri_vert[i,2,0], tri_vert[i,2,1], tri_vert[i,2,2])
        
        v0v1 = v1.sub(v0)
        v0v2 = v2.sub(v0)
        
        tri_area[i] = 0.5 * v0v1.cross(v0v2).length() #Calculate area of each triangle
        
    #If this takes a super long time I could save myself a loop by including this calculation in viewFactors_3D
    for i in tri_idx: 
        tri_rays[i,i,:] = 0 #The ray from facet i to itself is 0
        for j in np.where(tri_idx!=i)[0]:
            sub = tri_cent[j,:] - tri_cent[i,:]
            tri_rays[i,j,:] = sub/np.sqrt(np.sum(sub**2)) #normalized ray from i centroid to j centroid
            
            #Make tri_rays nan where ray is on the opposite side of the face from the normal
            cos = np.dot(tri_rays[i,j,:], tri_norm[i,:])
            if (cos<0):
                tri_rays[i,j,:] = np.nan
            
    
    return tri_vert, tri_cent, tri_area, tri_norm, tri_rays

def Triangle_Geometry_trimesh(mesh):
    #Returns vertices, centroids, areas, and normals of each triangle in mesh
    #This version uses the trimesh library instead of the open3d library
    #mesh_tri = np.asarray(mesh.triangles)
    #mesh_vert = np.asarray(mesh.vertices)
    
    tri_vert = mesh.triangles
    tri_cent = np.zeros([tri_vert.shape[0], 3])
    tri_area = np.zeros([tri_vert.shape[0]])
    
    tri_norm = trimesh.triangles.normals(mesh.triangles)[0]
    
    tri_rays = np.zeros([tri_vert.shape[0], tri_vert.shape[0], 3]) #Rays from each facet to every other facet
    
    tri_idx = np.arange(0, tri_cent.shape[0])
    for i in tri_idx: #for i in range(0, tri.shape[0]):
        #tri_vert[i, :, :] = mesh_vert[mesh_tri[i,:], :] #Find vertices of each triangle
        tri_cent[i, :] = np.mean(tri_vert[i, :, :], axis=0) #Calculate centroid of each triangle
        
        v0 = pt.Vec3(tri_vert[i,0,0], tri_vert[i,0,1], tri_vert[i,0,2])
        v1 = pt.Vec3(tri_vert[i,1,0], tri_vert[i,1,1], tri_vert[i,1,2])
        v2 = pt.Vec3(tri_vert[i,2,0], tri_vert[i,2,1], tri_vert[i,2,2])
        
        v0v1 = v1.sub(v0)
        v0v2 = v2.sub(v0)
        
        tri_area[i] = 0.5 * v0v1.cross(v0v2).length() #Calculate area of each triangle
        
    #If this takes a super long time I could save myself a loop by including this calculation in viewFactors_3D
    for i in tri_idx: 
        tri_rays[i,i,:] = 0 #The ray from facet i to itself is 0
        for j in np.where(tri_idx!=i)[0]:
            sub = tri_cent[j,:] - tri_cent[i,:]
            tri_rays[i,j,:] = sub/np.sqrt(np.sum(sub**2)) #normalized ray from i centroid to j centroid
            
            #Make tri_rays nan where ray is on the opposite side of the face from the normal
            cos = np.dot(tri_rays[i,j,:], tri_norm[i,:])
            if (cos<0):
                tri_rays[i,j,:] = np.nan
            
    
    return tri_vert, tri_cent, tri_area, tri_norm, tri_rays

def viewFactors_3D(tri_vert, tri_cent, tri_area, tri_norm):
    #Calculate view factors of every facet seen by each other facet
    tri_idx = np.arange(0, tri_cent.shape[0])
    
    idx_los = [] #Empty Line-Of-Sight index. Will store indicies of facets within LOS of each facet
    F_los = np.zeros([tri_cent.shape[0], tri_cent.shape[0]]) #[] #View factors of facets within LOS of each facet
    
    #Calculate rays from centroid of facet of interest to all other facet centroids
    #Determine if any of these rays intersect other facets.
    #If any does, then the further facet is not seen by the facet of interest.
    
    for i in tri_idx:
        idx_i = np.array([]) #Will hold indicies of facets within LOS of i
        #F_i = np.array([]) #Will hold view factors of facets within LOS of i
        
        norm_i = pt.Vec3(tri_norm[i,0], tri_norm[i,1], tri_norm[i,2]) #Normal vector of facet i
        r = pt.Ray(orig=pt.Vec3(tri_cent[i,0], tri_cent[i,1], tri_cent[i,2])) #Initialize ij ray with origin at i
        
        for j in np.where(tri_idx!=i)[0]:
            idx_j = np.zeros([3,1])
            idx_j[0,0] = j
            
            r.direction = pt.Vec3(tri_cent[j,0], tri_cent[j,1], \
                                  tri_cent[j,2]).sub(r.orig).normalize()
            
            v0 = pt.Vec3(tri_vert[j,0,0], tri_vert[j,0,1], tri_vert[j,0,2])
            v1 = pt.Vec3(tri_vert[j,1,0], tri_vert[j,1,1], tri_vert[j,1,2])
            v2 = pt.Vec3(tri_vert[j,2,0], tri_vert[j,2,1], tri_vert[j,2,2])
            
            idx_j[1:,0] = pt.ray_triangle_intersect(r, v0, v1, v2) #Returns [intersect T/F, dist]
            
            #Check if there is any other facet blocking facet j from LOS of facet i
            if (idx_j[1,0] == True): #First check if facet j is oriented towards facet i
                for k in np.where((tri_idx!=i) & (tri_idx!=j))[0]:
                    v0 = pt.Vec3(tri_vert[k,0,0], tri_vert[k,0,1], tri_vert[k,0,2])
                    v1 = pt.Vec3(tri_vert[k,1,0], tri_vert[k,1,1], tri_vert[k,1,2])
                    v2 = pt.Vec3(tri_vert[k,2,0], tri_vert[k,2,1], tri_vert[k,2,2])

                    intersect = pt.ray_triangle_intersect(r, v0, v1, v2)
                    
                    if (intersect[0]==True): #If ray intersects a facet, add it to the list!
                        idx_j = np.append(idx_j, np.array([[k, intersect[0], intersect[1]]]).T, axis=1)
                
                #Have to check which facets are intersected AND which is the closest to the facet of interest.
                if (idx_j[0, np.argmin(idx_j[2,:])] == j): #If there is no other facet blocking LOS from i to j,
                    idx_i = np.append(idx_i, j)            #then add j to list of indicies within LOS of i.
                    
                    norm_j = pt.Vec3(tri_norm[j,0], tri_norm[j,1], tri_norm[j,2]) #Normal vector of facet j
                    
                    #Get cosine from dot product of ray and normal vector
                    cos1 = r.direction.dot(norm_i)/(r.direction.length()*norm_i.length())
                    
                    r.direction = pt.Vec3(-r.direction.x, -r.direction.y, -r.direction.z) #flip ray
                    cos2 = r.direction.dot(norm_j)/(r.direction.length()*norm_j.length()) 
                    
                    #Calculate view factor
                    F = (tri_area[j]/(np.pi*idx_j[2,0]**2)) * cos1 * cos2 #idx_j[2,0] carries dist between facets
                    #F_i = np.append(F_i, F) #Add view factor to list of view factors within LOS of i
                    F_los[i,j] = F
    
        idx_los.append(idx_i)
        #F_los.append(F_i)
    
    #This whole thing might be faster if I used numpy vector operations instead of the pt.Vec3() class.
    
    return F_los, idx_los

def Shadowfax_3D(sa, cosz, tri_rays, tri_cent, tri_vert):
    #Determine which points are shadowed by rim of pit.
    #This function will have to be called at each time step.
    #Show the meaning of haste.
    """
    sa: solar azimuth angle
    cosz: cosine of the solar zenith angle
    """
    
    
    #calculate if ray from facet to sun intersects any other facets?
    #Or maybe calculate if the ray lies below or above the rim?
    #Or maybe calculate if the ray intersects the pit opening or not?
    #<-- Probs not the best option if I want to apply this to situations that aren't just holes in the ground.
    
    ########################################
    ########################################
    #OR MAYBE first calculate facet-to-facet ray that lies closest to facet-to-sun ray,
    #and then test if the facet-to-sun ray intersects the other facet.
    #If it doesn't intersect the closest facet, then there should be no shadowing?
    #<-- I can calculate the vectors from each facet to all other facets once at the
    #beginning of a model run so I don't need to loop through all the facets at each time step.
    #Can then calculate the minimum magnitude of the cross product between the facet-to-facet rays and the
    #facet-to-sun ray to find the smallest angle between them. Check if the facet associated with this smallest
    #angle intersects the solar ray.
    
    #Store facet-to-facet rays for each facet in one big old array. Use np.cross() for cross product and
    #np.amin() along one axis to calculate minimum for each facet. This avoids the use of for loops for now.
    #Then can I calculate when or not the solar ray intersects for each facet simulaneously without loops?
    #OR MAYBE I should use the maximum of the dot product to avoid degenracies with angles greater than 90.
    
    #Note: I think this will still work even if there are overhangs.
    t0 = time.time()
    #normalized ray pointing towards the sun (assume parallel solar rays)
    sinz = np.sin(np.arccos(cosz))
    solar_ray = np.array([np.cos(sa)*sinz, np.sin(sa)*sinz, cosz])
    t1 = time.time()
    print ('solar_ray time:', t1-t0)
    print ('solar_ray:', solar_ray)
    
    #Calculate the maximum dot product between the solar ray and the rays from each facet to all other facets.
    #Pretty sure I got the axes right here.
    closest_ray_idx = np.argmax( np.tensordot(tri_rays, solar_ray, axes=([2],[0])), axis=1)
    
    t2 = time.time()
    print ('closest_ray_idx time:', t2-t1)
    print ('closest_ray_idx:', closest_ray_idx)
    
    shadow_tri_vert = tri_vert[closest_ray_idx,:,:] #triangle vertices with potential to shadow
    
    t3 = time.time()
    print ('shadow_tri_vert time:', t3-t2)
    print ('shadow_tri_vert.shape:', shadow_tri_vert.shape)
    
    #Now check if solar ray intersects shadow triangles
    #intersect = ray_triangle_intersect(r_orig=tri_cent, r_dir=tri_rays[:,closest_ray_idx,:], \
    #                                   v0=shadow_tri_vert[:,0,:], v1=shadow_tri_vert[:,1,:], \
    #                                   v2=shadow_tri_vert[:,2,:])
    
    #intersect = ray_triangle_intersect(r_orig=tri_cent, r_dir=solar_ray, \
    #                                   v0=shadow_tri_vert[closest_ray_idx,0,:], \
    #                                   v1=shadow_tri_vert[closest_ray_idx,1,:], \
    #                                   v2=shadow_tri_vert[closest_ray_idx,2,:])
    
    intersect = ray_triangle_intersect(r_orig=tri_cent, r_dir=solar_ray, \
                                       v0=shadow_tri_vert[:,0,:], \
                                       v1=shadow_tri_vert[:,1,:], \
                                       v2=shadow_tri_vert[:,2,:])
    
    #If the solar ray intersects these triangles, then the facet lies in shadow.
    idx_shadow = np.where(intersect[:,0] == True)
    
    t4 = time.time()
    print ('intersect time:', t4-t3)
    ########################################
    ########################################
    
    #OR ALSO I could calculate a "skyline" for each facet by noting the elevation angle of the highest other 
    #facet edge visible to the facet of interest at each solar azimuth.
    
    #Also I could exclude a number of facets from any for loops I may have to use if I know those facets would
    #never be oriented towards direct sunlight given their latitude.
    
    return idx_shadow, closest_ray_idx, intersect

def Shadowfax_3D_2(sa, cosz, horizon_az, horizon_zen, intersect_sky):
    #This version is compatible with Find_Horizon_3()
    #Determine which points are shadowed by rim of pit.
    #This function will have to be called at each time step.
    #Shows the meaning of haste.
    """
    sa: solar azimuth angle
    cosz: cosine of the solar zenith angle
    """
    zen = np.arccos(cosz)
    idx_zen_closest = np.argmin( np.abs( horizon_zen-zen ) )
    idx_az_closest = np.argmin( np.abs( horizon_az-sa ) )
    
    idx_shadow = intersect_sky[:,idx_az_closest,idx_zen_closest]
    
    #Also I could exclude a number of facets from any for loops I may have to use if I know those facets would
    #never be oriented towards direct sunlight given their latitude.
    
    return idx_shadow

def Shadowfax_3D_3(sa, cosz, horizon_az, horizon_zen, intersect_sky):
    #This version is compatible with Find_Horizon_5()
    #Determines which points are shadowed by rim of pit.
    #Approximates what fraction of the solar disk is exposed if the sun is close to the horizon.
    #This function will have to be called at each time step.
    #Shows the meaning of haste.
    """
    sa: solar azimuth angle
    cosz: cosine of the solar zenith angle
    """
    solar_ang = 0.53*np.pi/180 #Angular diameter of the Sun viewed from the Moon
    
    zen = np.arccos(cosz)
    idx_zen_closest = np.argmin( np.abs( horizon_zen-zen ) )
    idx_az_closest = np.argmin( np.abs( horizon_az-sa ) )
    
    #Approximation for fraction of solar disk exposed
    idx_disk = np.where( np.abs(zen-horizon_zen)<solar_ang/2 )
    frac_shadow = np.squeeze(np.count_nonzero(~intersect_sky[:,idx_az_closest,idx_disk], axis=2)/idx_disk[0].size)
    
    return frac_shadow

def ray_triangle_intersect(r_orig, r_dir, v0, v1, v2):
    #Let's redo this function so it doesn't rely on this Vec3 class and instead uses numpy functions.
    #Also, let's make it so that I can do numerous intersection checks at once by inputing matricies.
    v0_dim = len(v0.shape)
    
    intersect = np.ones([v0.shape[0], 2])
    
    #v0v1 = v1.sub(v0) #ORIGINAL CODE
    v0v1 = v1 - v0
    #v0v2 = v2.sub(v0) #ORIGINAL CODE
    v0v2 = v2 - v0
    #pvec = r.direction.cross(v0v2) #ORIGINAL CODE
    pvec = np.cross(r_dir, v0v2)
    #print ('r_orig.shape:', r_orig.shape)
    #print ('r_orig:', r_orig)
    #print ('r_dir.shape:', r_dir.shape)
    #print ('r_dir:', r_dir)
    #print ('v0.shape:', v0.shape)
    #print ('v0:', v0)
    #print ('v1.shape:', v1.shape)
    #print ('v1:', v1)
    #print ('v2.shape:', v2.shape)
    #print ('v2:', v2, '\n')
    
    #print ('v0v1.shape:', v0v1.shape)
    #print ('v0v1:', v0v1)
    #print ('v0v2.shape:', v0v2.shape)
    #print ('v0v2:', v0v2, '\n')
    
    #print ('pvec.shape:', pvec.shape)
    #print ('pvec:', pvec)
    
    #det = v0v1.dot(pvec) #ORIGINAL CODE
    if (v0_dim==1):
        det = np.dot(v0v1, pvec)
        if det < 0.000001: 
            return (False, np.nan)
    else:
        det = np.sum(v0v1*pvec, axis=1) #np.tensordot(v0v1, pvec, axis=([1],[1])) #Is this right?
        intersect[det<0.000001, :] = False, np.nan

    #print ('det.shape:', det.shape)
    #print ('det:', det, '\n')
        
    #if det < 0.000001: #ORIGINAL CODE
    #    return False, float('-inf') #ORIGINAL CODE
    
    #intersect[det<0.000001, :] = False, float('-inf')

    invDet = 1.0 / det
    #tvec = r.orig.sub(v0) #ORIGINAL CODE
    tvec = r_orig - v0
    #u = tvec.dot(pvec) * invDet #ORIGINAL CODE
    if (v0_dim==1):
        u = np.dot(tvec, pvec) * invDet
        if u < 0 or u > 1:
            return (False, np.nan)
    else:
        u = np.sum(tvec*pvec, axis=1) * invDet
        intersect[(u<0) | (u>1), :] = False, np.nan

    #print ('tvec.shape:', tvec.shape)
    #print ('tvec:', tvec)
    #print ('u.shape:', u.shape)
    #print ('u:', u, '\n')
        
    #if u < 0 or u > 1: #ORIGINAL CODE
    #    return False, float('-inf') #ORIGINAL CODE
    #intersect[(u<0) or (u>1), :] = False, float('-inf')

    #qvec = tvec.cross(v0v1) #ORIGINAL CODE
    qvec = np.cross(tvec, v0v1)
    #v = r.direction.dot(qvec) * invDet #ORIGINAL CODE
    if (v0_dim==1):
        v = np.dot(r_dir, qvec) * invDet
        if v < 0 or u + v > 1:
            return (False, np.nan)
    else:
        v = np.sum(r_dir*qvec, axis=1) * invDet
        intersect[(v<0) | (u+v>1), :] = False, np.nan

    #print ('qvec.shape:', qvec.shape)
    #print ('qvec:', qvec)
    #print ('v.shape:', v.shape)
    #print ('v:', v, '\n')
        
    #if v < 0 or u + v > 1: #ORIGINAL CODE
    #    return False, float('-inf') #ORIGINAL CODE
    #intersect[(v<0) or (u+v>1), :] = False, float('-inf')
    
    if (v0_dim==1):
        return (True, np.dot(v0v2, qvec) * invDet)
    else:
        intersect[intersect[:,0]==True,1] = np.sum(v0v2*qvec, axis=1)[intersect[:,0]==True] * \
        invDet[intersect[:,0]==True]
    
    #print ('d.shape:', (np.sum(v0v2*qvec, axis=1) * invDet).shape)
    #print ('d:', np.sum(v0v2*qvec, axis=1) * invDet, '\n')
    
    return intersect#, invDet, intersect[:,1]/invDet #True, v0v2.dot(qvec) * invDet #ORIGINAL CODE

def Multi_Scattering(F_los, A0, emis, num_ref):
    #Need a function to calculate the coefficients necessary for the scattering computation that will occur
    #at each time step.
    """
    F_los: View factors of facets within LOS of each facet
    A0: Albedo at zero solar incidence angle
    emis: emissivity of surface facets
    num_ref: Maximum number of reflections accounted for
    """
    sigma = 5.67051196e-8 # Stefan-Boltzmann Constant
    
    #i = np.arccos(  ) # facet-to-facet ray incidence angle [rad]
    #
    #a = planet.albedoCoef[0]
    #b = planet.albedoCoef[1]
    #A = albedoVar(0.5, a, b, i)
    
    #Scattering coefficients for visible light (f_vis) and IR (f_IR) reflection
    x = np.zeros([F_los.shape[0], F_los.shape[0], num_ref])
    y = np.zeros([F_los.shape[0], F_los.shape[0], num_ref])
    
    x[:,:,0] = A0 * F_los
    y[:,:,0] = emis**2 * sigma * F_los #I'm pretty sure this is right
    for i in range(1, num_ref):
        #Calculate each set of higher order scattering coefficients from the previous coefficients
        x[:,:,i] = A0 * np.tensordot( x[:,:,i-1], F_los, axes=([1],[0]) )
        y[:,:,i] = (1 - emis) * np.tensordot( y[:,:,i-1], F_los, axes=([1],[0]) )
        
    f_vis = np.sum(x, axis=2)
    f_IR = np.sum(y, axis=2)
    
    return f_vis, f_IR

def viewFactors_3D_1(tri_vert, tri_cent, tri_area, tri_norm, tri_rays):
    #Calculate view factors of every facet seen by each other facet
    tri_idx = np.arange(0, tri_cent.shape[0])
    
    idx_los = [] #Empty Line-Of-Sight index. Will store indicies of facets within LOS of each facet
    F_los = np.zeros([tri_cent.shape[0], tri_cent.shape[0]]) #[] #View factors of facets within LOS of each facet
    
    #Calculate rays from centroid of facet of interest to all other facet centroids
    #Determine if any of these rays intersect other facets.
    #If any does, then the further facet is not seen by the facet of interest.
    
    for i in tri_idx:
        idx_i = np.array([]) #Will hold indicies of facets within LOS of i
        #F_i = np.array([]) #Will hold view factors of facets within LOS of i
        
        #norm_i = pt.Vec3(tri_norm[i,0], tri_norm[i,1], tri_norm[i,2]) #Normal vector of facet i
        #r = pt.Ray(orig=pt.Vec3(tri_cent[i,0], tri_cent[i,1], tri_cent[i,2])) #Initialize ij ray with origin at i
        r_orig = tri_cent[i,:]
        
        for j in np.where(tri_idx!=i)[0]:
            idx_j = np.zeros([3,1])
            idx_j[0,0] = j
            
            #r.direction = pt.Vec3(tri_cent[j,0], tri_cent[j,1], \
                                  #tri_cent[j,2]).sub(r.orig).normalize()
            r_dir = tri_rays[i,j,:]
            
            #v0 = pt.Vec3(tri_vert[j,0,0], tri_vert[j,0,1], tri_vert[j,0,2])
            #v1 = pt.Vec3(tri_vert[j,1,0], tri_vert[j,1,1], tri_vert[j,1,2])
            #v2 = pt.Vec3(tri_vert[j,2,0], tri_vert[j,2,1], tri_vert[j,2,2])
            
            #idx_j[1:,0] = pt.ray_triangle_intersect(r, v0, v1, v2) #Returns [intersect T/F, dist]
            idx_j[1:,0] = ray_triangle_intersect(r_orig, r_dir, tri_vert[j,0,:], tri_vert[j,1,:], tri_vert[j,2,:])
            
            #Check if there is any other facet blocking facet j from LOS of facet i
            if (idx_j[1,0] == True): #First check is facet j is oriented towards facet i
                for k in np.where((tri_idx!=i) & (tri_idx!=j))[0]:
                    #v0 = pt.Vec3(tri_vert[k,0,0], tri_vert[k,0,1], tri_vert[k,0,2])
                    #v1 = pt.Vec3(tri_vert[k,1,0], tri_vert[k,1,1], tri_vert[k,1,2])
                    #v2 = pt.Vec3(tri_vert[k,2,0], tri_vert[k,2,1], tri_vert[k,2,2])

                    #intersect = pt.ray_triangle_intersect(r, v0, v1, v2)
                    intersect = ray_triangle_intersect(r_orig, r_dir, tri_vert[k,0,:], \
                                                       tri_vert[k,1,:], tri_vert[k,2,:])
                    
                    if (intersect[0]==True): #If ray intersects a facet, add it to the list!
                        idx_j = np.append(idx_j, np.array([[k, intersect[0], intersect[1]]]).T, axis=1)
                
                #Have to check which facets are intersected AND which is the closest to the facet of interest.
                if (idx_j[0, np.argmin(idx_j[2,:])] == j): #If there is no other facet blocking LOS from i to j,
                    idx_i = np.append(idx_i, j)            #then add j to list of indicies within LOS of i.
                    
                    #norm_j = pt.Vec3(tri_norm[j,0], tri_norm[j,1], tri_norm[j,2]) #Normal vector of facet j
                    
                    #Get cosine from dot product of ray and normal vector
                    #cos1 = r.direction.dot(norm_i)/(r.direction.length()*norm_i.length())
                    cos1 = np.dot(r_dir, tri_norm[i,:])
                    
                    #r.direction = pt.Vec3(-r.direction.x, -r.direction.y, -r.direction.z) #flip ray
                    #cos2 = r.direction.dot(norm_j)/(r.direction.length()*norm_j.length()) 
                    cos2 = np.dot(-r_dir, tri_norm[j,:])
                    
                    #Calculate view factor
                    F = (tri_area[j]/(np.pi*idx_j[2,0]**2)) * cos1 * cos2 #idx_j[2,0] carries dist between facets
                    #F_i = np.append(F_i, F) #Add view factor to list of view factors within LOS of i
                    F_los[i,j] = F
    
        idx_los.append(idx_i)
        #F_los.append(F_i)
    
    #This whole thing might be faster if I used numpy vector operations instead of the pt.Vec3() class.
    
    return F_los, idx_los

def viewFactors_3D_2(tri_vert, tri_cent, tri_area, tri_norm, tri_rays):
    #Calculate view factors of every facet seen by each other facet
    tri_idx = np.arange(0, tri_cent.shape[0])
    
    idx_los = [] #Empty Line-Of-Sight index. Will store indicies of facets within LOS of each facet
    F_los = np.zeros([tri_cent.shape[0], tri_cent.shape[0]]) #[] #View factors of facets within LOS of each facet
    
    #Calculate rays from centroid of facet of interest to all other facet centroids
    #Determine if any of these rays intersect other facets.
    #If any does, then the further facet is not seen by the facet of interest.
    #t_i0 = time.time()
    for i in tri_idx:
        idx_i = np.array([]) #Will hold indicies of facets within LOS of i
        
        r_orig = tri_cent[i,:] #Initialize ij ray with origin at i
        
        idx_j = np.where(tri_idx!=i)[0]
        for j in idx_j:
            
            r_dir = tri_rays[i,j,:]
            x = (r_orig - tri_cent[j,:])
            x = x/np.sqrt(np.sum(x**2))
            
            intersect_j = ray_triangle_intersect(r_orig, r_dir, tri_vert[j,0,:], tri_vert[j,1,:], tri_vert[j,2,:])
            
            #Check if there is any other facet blocking facet j from LOS of facet i
            if (intersect_j[0] == True): #First check if facet j is oriented towards facet i
                
                intersect = ray_triangle_intersect(r_orig, r_dir, tri_vert[idx_j,0,:], tri_vert[idx_j,1,:], \
                                                    tri_vert[idx_j,2,:])
                
                
                idx_int = np.where( intersect[:,1]==np.nanmin( intersect[intersect[:,0]==True,1] ) )
                
                #Have to check which facets are intersected AND which is the closest to the facet of interest.
                if (np.any(idx_j[idx_int] == j)):       #If there is no other facet blocking LOS from i to j,
                    idx_i = np.append(idx_i, j) #then add j to list of indicies within LOS of i.
                    
                    #Get cosine from dot product of ray and normal vector
                    cos1 = np.dot(r_dir, tri_norm[i,:])
                    cos2 = np.dot(-r_dir, tri_norm[j,:])
                    
                    #Calculate view factor
                    F = (tri_area[j]/(np.pi*intersect[idx_int,1][0]**2)) * cos1 * cos2
                    F_los[i,j] = F
    
        idx_los.append(idx_i)
    
    return F_los, idx_los

def viewFactors_3D_3(tri_vert, tri_cent, tri_area, tri_norm, tri_rays):
    #Calculate view factors of every facet seen by each other facet
    #Now imma try to do this with only matricies since I updated the function ray_triangle_intersect_View_Factors_1()
    """
    (1) First, for any facet i, I want to check if the diagonals of this matrix are true: intersect[i,:,:,0]. If so, it means
    that when the verticies are for facet j and the direction is from i to j, then there is an intersection. In other words,
    facet j is facing i (unless the ray from i to j isn't on the same side as the normal of i?? uh-oh, I'll look into this).
    
    If the first condition is met, it's worth checking the second.
    (2) Second, for any facet i and for each facet j for which (1) is true, find all the other facets that ray i-j intersects.
    Then find the index of the facet with the minimum distance from i to the intersection.
    
    (3) Third, if j is this closest intersection, then we can calculate the view factor of j from i (otherwise the view factor is
    zero).
    """ 
    
    F_los = np.zeros([tri_cent.shape[0], tri_cent.shape[0]]) #[] #View factors of facets within LOS of each facet
    
    #intersect = ray_triangle_intersect_View_Factors_1(tri_cent[:,:], tri_rays[:,:,:], tri_vert[:,0,:], tri_vert[:,1,:],\
    #                                                  tri_vert[:,2,:])
    intersect0, intersect1 = ray_triangle_intersect_View_Factors_1(tri_cent[:,:], tri_rays[:,:,:], tri_vert[:,0,:], \
                                                                   tri_vert[:,1,:], tri_vert[:,2,:])
    print ('Step 0 Done')
    
    #Step (1)
    #idx_diag = np.where( (np.diagonal(intersect[:,:,:,0], axis1=1, axis2=2)==True) & \
    #                     (np.diagonal(intersect[:,:,:,1], axis1=1, axis2=2)>0) )
    idx_diag = np.where( (np.diagonal(intersect0[:,:,:], axis1=1, axis2=2)==True) & \
                         (np.diagonal(intersect1[:,:,:], axis1=1, axis2=2)>0) )
    print ('Step 1 Done')
    
    #Step (2)
    #intersect[:,:,:,1][intersect[:,:,:,1]<=0] = np.nan
    #idx_min = np.where( np.nanargmin(intersect[idx_diag[0],:,idx_diag[1],1], axis=1)==idx_diag[1] )
    intersect1[:,:,:][intersect1[:,:,:]<=0] = np.nan
    idx_min = np.where( np.nanargmin(intersect1[idx_diag[0],:,idx_diag[1]], axis=1)==idx_diag[1] )
    print ('Step 2 Done')
    
    #Step (3)
    i = idx_diag[0][idx_min]
    j = idx_diag[1][idx_min]
    
    #Get cosine from dot product of ray and normal vector
    cos1 = np.sum(tri_rays[i,j,:]*tri_norm[i,:], axis=1)
    cos2 = np.sum(-tri_rays[i,j,:]*tri_norm[j,:], axis=1)
    
    #Calculate view factor
    #F_los[i,j] = (tri_area[j]/(np.pi*intersect[i,j,j,1]**2)) * cos1 * cos2
    F_los[i,j] = (tri_area[j]/(np.pi*intersect1[i,j,j]**2)) * cos1 * cos2
    F_los[F_los<0] = 0 #This eliminates instances where facet i is facing away from facet j
    print ('Step 3 Done')
    
    #Ummmmmmmmmmm I think this is right! ¯\_(ツ)_/¯
    
    return F_los

def viewFactors_3D_4(tri_vert, tri_cent, tri_area, tri_norm, tri_rays):
    #Calculate view factors of every facet seen by each other facet
    tri_idx = np.arange(0, tri_cent.shape[0])
    F_los = np.zeros([tri_cent.shape[0], tri_cent.shape[0]]) #[] #View factors of facets within LOS of each facet
    
    #Calculate rays from centroid of facet of interest to all other facet centroids
    #Determine if any of these rays intersect other facets.
    #If any does, then the further facet is not seen by the facet of interest.
    #t_i0 = time.time()
    for i in tri_idx:
        r_orig = tri_cent[i,:] #Initialize ij ray with origin at i
        
        #idx_j = np.where(tri_idx!=i)[0]
        
        r_dir = tri_rays[i,:,:]
        intersect = ray_triangle_intersect_View_Factors(r_orig, r_dir, tri_vert[:,0,:], tri_vert[:,1,:], \
                                                            tri_vert[:,2,:])
        
        #Step (1)
        idx_diag = np.where( (intersect[tri_idx,tri_idx,0]==True) & \
                             (intersect[tri_idx,tri_idx,1]>0) )[0]
        
        #Step (2)
        intersect[:,:,1][intersect[:,:,1]<=0] = np.nan
        idx_min = np.where( (np.nanargmin(intersect[idx_diag,:,1], axis=1)==idx_diag))
        
        #Step (3)
        j = idx_diag[idx_min]

        #Get cosine from dot product of ray and normal vector
        cos1 = np.sum(tri_rays[i,j,:]*tri_norm[i,:], axis=1)
        cos2 = np.sum(-tri_rays[i,j,:]*tri_norm[j,:], axis=1)

        #Calculate view factor
        F_los[i,j] = (tri_area[j]/(np.pi*intersect[j,j,1]**2)) * cos1 * cos2
        
        F_los[i,F_los[i,:]<0] = 0 #This eliminates instances where facet i is facing away from facet j

        #Ummmmmmmmmmm I think this is right! ¯\_(ツ)_/¯
    
    return F_los

def viewFactors_3D_5(tri_vert, tri_cent, tri_area, tri_norm, tri_rays, pro_idx):
    #Calculate view factors of every facet seen by each other facet
    tri_idx = np.arange(0, tri_vert.shape[0])
    idx_i = np.arange(0, tri_cent.shape[0])
    F_los = np.zeros([tri_cent.shape[0], tri_vert.shape[0]], dtype=np.float32) #[] #View factors of facets within LOS of each facet
    
    #Calculate rays from centroid of facet of interest to all other facet centroids
    #Determine if any of these rays intersect other facets.
    #If any does, then the further facet is not seen by the facet of interest.
    #t_i0 = time.time()
    for i in idx_i:
        r_orig = tri_cent[i,:] #Initialize ij ray with origin at i
        
        #idx_j = np.where(tri_idx!=i)[0]
        
        r_dir = tri_rays[i,:,:]
        intersect = ray_triangle_intersect_View_Factors(r_orig, r_dir, tri_vert[:,0,:], tri_vert[:,1,:], \
                                                            tri_vert[:,2,:])
        
        #Step (1)
        idx_diag = np.where( (intersect[tri_idx,tri_idx,0]==True) & \
                             (intersect[tri_idx,tri_idx,1]>0) )[0]
        
        #Step (2)
        intersect[:,:,1][intersect[:,:,1]<=0] = np.nan
        idx_min = np.where( (np.nanargmin(intersect[idx_diag,:,1], axis=1)==idx_diag))
        
        #Step (3)
        j = idx_diag[idx_min]

        #Get cosine from dot product of ray and normal vector
        cos1 = np.sum(tri_rays[i,j,:]*tri_norm[pro_idx[i],:], axis=1)
        cos2 = np.sum(-tri_rays[i,j,:]*tri_norm[j,:], axis=1) #Need entire tri_norm array for the 'j's

        #Calculate view factor
        #aqui #F_los[i,j] = (tri_area[j]/(np.pi*intersect[j,j,1]**2)) * cos1 * cos2 #This is the main one I'm using that I know works
        #F_los[i,j] = (tri_area[i]/(np.pi*intersect[j,j,1]**2)) * cos1 * cos2 #I thought maybe tri_area[j] should be tri_area[i]
        
        ##### TESTING THIS VERSION OUT #####
        ###### Rezac and Zhao (2020) #######
        #Approximation 1#
        #F_los[i,j] = ( (4*np.sqrt(tri_area[i]*tri_area[j]))/(np.pi**2*tri_area[j]) ) * \
        #            np.arctan( (np.sqrt(np.pi*tri_area[j])*cos2)/(2*intersect[j,j,1]) ) * \
        #            np.arctan( (np.sqrt(np.pi*tri_area[i])*cos1)/(2*intersect[j,j,1]) )
        
        #Approximation 2#
        #F_los[i,j] = ( (4*np.sqrt(tri_area[i]*tri_area[j]))/(np.pi**2*tri_area[i]) ) * \
        #            np.arctan( (np.sqrt(np.pi*tri_area[i])*cos2)/(2*intersect[j,j,1]) ) * \
        #            np.arctan( (np.sqrt(np.pi*tri_area[j])*cos1)/(2*intersect[j,j,1]) )
        
        #Approximation 3#
        F_los[i,j] = ( (4*np.sqrt(tri_area[i]*tri_area[j]))/(np.pi**2*tri_area[i]) ) * \
                    np.arctan( (np.sqrt(np.pi*tri_area[j])*cos2)/(2*intersect[j,j,1]) ) * \
                    np.arctan( (np.sqrt(np.pi*tri_area[i])*cos1)/(2*intersect[j,j,1]) )
        ####################################
        
        F_los[i,F_los[i,:]<0] = 0 #This eliminates instances where facet i is facing away from facet j
        
        #Ummmmmmmmmmm I think this is right! ¯\_(ツ)_/¯
    #print ('np.append(F_los, pro_idx, axis=1):', np.append(F_los, pro_idx, axis=1))
    
    return np.append(F_los, pro_idx[:,None], axis=1)

def ray_triangle_intersect_View_Factors(r_orig, r_dir, v0, v1, v2):
    #Let's make it so that I can do numerous intersection checks at once by inputing matricies.
    intersect = np.ones([r_dir.shape[0], v0.shape[0], 2])

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pvec = np.cross(r_dir[:,None,:], v0v2)
    
    det = np.sum(v0v1*pvec, axis=2)
    idx = np.where(det<0.000001)
    intersect[idx[0],idx[1], :] = False, np.nan

    invDet = 1.0 / det
    tvec = r_orig - v0
    
    u = np.sum(tvec*pvec, axis=2) * invDet
    idx1 = np.where((u<0) | (u>1))
    intersect[idx1[0],idx1[1], :] = False, np.nan

    qvec = np.cross(tvec, v0v1)
    v = np.sum(r_dir[:,None,:]*qvec, axis=2) * invDet
    idx2 = np.where((v<0) | (u+v>1))
    intersect[idx2[0],idx2[1], :] = False, np.nan

    idx3 = np.where(intersect[:,:,0]==True)
    intersect[idx3[0],idx3[1],1] = \
        ((intersect[:,:,0]==True)*np.sum(v0v2*qvec, axis=1))[intersect[:,:,0]==True] * invDet[idx3[0],idx3[1]]
    
    return intersect

def ray_triangle_intersect_View_Factors_1(r_orig, r_dir, v0, v1, v2):
    """Let's make it so that I can do numerous intersection checks at once by inputing matricies.
    This function handles the case where the dimensions of the inputs are as follows,
    r_orig: (N,3)
    r_dir: (N,N,3)
    v0, v1, v2: (N,3)
    where N is the number of facets (triangle faces) in the surface.
    The resulting intersect matrix returned by this function has dimensions of: (N,N,N,2).
    An example input is: (tri_cent[:,:], tri_rays[:,:,:], tri_vert[:,0,:], tri_vert[:,1,:], tri_vert[:,2,:])
    """
    
    #intersect = np.ones([r_orig.shape[0], r_dir.shape[0], v0.shape[0], 2])
    intersect0 = np.ones([r_orig.shape[0], r_dir.shape[0], v0.shape[0]], dtype=bool)
    intersect1 = np.ones([r_orig.shape[0], r_dir.shape[0], v0.shape[0]], dtype=np.float16)

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pvec = np.cross(r_dir[:,None,:,:], v0v2[:,None,:])

    det = np.sum(v0v1[:,None,:]*pvec, axis=3)

    idx = np.where(det<0.000001)
    #intersect[idx[0], idx[1], idx[2], :] = False, np.nan
    intersect0[idx[0], idx[1], idx[2]], intersect1[idx[0], idx[1], idx[2]] = False, np.nan

    invDet = 1.0 / det
    tvec = r_orig - v0[:,None,:]

    u = np.sum(np.swapaxes(tvec,0,1)[:,:,None,:]*pvec, axis=3) * invDet

    idx1 = np.where((u<0) | (u>1))
    #intersect[idx1[0],idx1[1], idx1[2], :] = False, np.nan
    intersect0[idx1[0],idx1[1], idx1[2]], intersect1[idx1[0],idx1[1], idx1[2]] = False, np.nan

    qvec = np.cross(tvec, v0v1[:,None,:])
    v = np.sum(r_dir[:,None,:,:]*qvec, axis=3) * invDet

    idx2 = np.where((v<0) | (u+v>1))
    #intersect[idx2[0],idx2[1],idx2[2], :] = False, np.nan
    intersect0[idx2[0],idx2[1],idx2[2]], intersect1[idx2[0],idx2[1],idx2[2]] = False, np.nan

    idx3 = np.where(intersect[:,:,:,0]==True)
    #intersect[idx3[0],idx3[1],idx3[2],1] = \
    #    ((intersect[:,:,:,0]==True)*np.swapaxes(np.sum(v0v2[:,None,:]*qvec, axis=2),0,1)[:,:,None])\
    #        [intersect[:,:,:,0]==True] * invDet[idx3[0],idx3[1],idx3[2]]
    
    intersect1[idx3[0],idx3[1],idx3[2]] = \
        ((intersect1[:,:,:]==True)*np.swapaxes(np.sum(v0v2[:,None,:]*qvec, axis=2),0,1)[:,:,None])\
            [intersect1[:,:,:]==True] * invDet[idx3[0],idx3[1],idx3[2]]
    
    return intersect0, intersect1

def ray_triangle_intersect_Horizons(r_orig, r_dir, v0, v1, v2):
    """Let's make it so that I can do numerous intersection checks at once by inputing matricies.
    This function handles the case where the dimensions of the inputs are as follows,
    r_orig: (N,3)
    r_dir: (N,N,3)
    v0, v1, v2: (N,3)
    where N is the number of facets (triangle faces) in the surface.
    The resulting intersect matrix returned by this function has dimensions of: (N,N,N,2).
    """
    
    ##########NOTE: I should remake this function with intersect = np.ones([tri_cent.shape[0], tri_vert.shape[0]], dtype=bool)
    ########## since the horizons calculation doesn't need the distance to the intersection.
    ##########NOTE: I did the above and it seems to work :)
    
    #intersect = np.ones([r_orig.shape[0], v0.shape[0], 2])
    intersect = np.ones([r_orig.shape[0], v0.shape[0]], dtype=bool)

    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pvec = np.cross(r_dir, v0v2)
    
    det = np.sum(v0v1*pvec, axis=1)

    #intersect[:, det<0.000001, :] = False, np.nan
    intersect[:, det<0.000001] = False

    invDet = 1.0 / det
    tvec = r_orig - v0[:,None,:]

    u = np.sum(np.swapaxes(tvec,0,1)*pvec, axis=2) * invDet
    
    idx1 = np.where((u<0) | (u>1))
    #intersect[idx1[0],idx1[1], :] = False, np.nan
    intersect[idx1[0],idx1[1]] = False

    qvec = np.cross(tvec, v0v1[:,None,:])
    v = np.swapaxes(np.sum(r_dir*qvec, axis=2),0,1) * invDet

    idx2 = np.where((v<0) | (u+v>1))
    #intersect[idx2[0],idx2[1], :] = False, np.nan
    intersect[idx2[0],idx2[1]] = False

    #idx3 = np.where(intersect[:,:,0]==True)
    #intersect[idx3[0],idx3[1],1] = \
    #    ((intersect[:,:,0]==True)*np.swapaxes(np.sum(v0v2[:,None,:]*qvec, axis=2),0,1))[intersect[:,:,0]==True]*\
    #        invDet[idx3[1]]
    
    return intersect

def Find_Horizon_3(tri_cent, tri_vert, daz=1*np.pi/180, dzen=1*np.pi/180):
    """
    Find the horizon seen by each facet in terms of the azimuth angle and the zenith angles of the horizon at any particular
    azimuth angle.
    daz: azimuth step size [rad]
    dzen: zenith step size [rad]
    """
    
    horizon_az = np.arange(0, 2*np.pi, daz) #azimuth angles [rad]
    
    horizon_zen = np.arange(0, np.pi, dzen) #zenith angles [rad]
    
    intersect_sky = np.zeros([tri_cent.shape[0], horizon_az.size, horizon_zen.size], dtype=bool)
    for i in range(0,horizon_az.size):
        intersect_sky[:,i,0] = True
        for j in range(1,horizon_zen.size):

            sinz = np.sin(horizon_zen[j])
            cosz = np.cos(horizon_zen[j])
            r_dir = np.array([np.cos(horizon_az[i])*sinz, np.sin(horizon_az[i])*sinz, cosz])
            
            #intersect = ray_triangle_intersect_Horizons(tri_cent[:,:], r_dir, \
            #                                        tri_vert[:,0,:], tri_vert[:,1,:], tri_vert[:,2,:])[:,0]
            intersect = ray_triangle_intersect_Horizons(tri_cent[:,:], r_dir, \
                                                    tri_vert[:,0,:], tri_vert[:,1,:], tri_vert[:,2,:])

            
            intersect_sky[np.any(intersect==True, axis=1),i,j] = True
        
    return horizon_az, horizon_zen, intersect_sky

#@jit(nopython=True)
def Find_Horizon_4(tri_cent, tri_vert, tri_norm, horizon_az, horizon_zen, intersect, intersect_sky, daz=1*np.pi/180, \
                   dzen=1*np.pi/180):
    #I want this new version to include multiprocessing AND jit compiling
    #I can probably do this by splitting the first for loop (i) into different processors and then creating a funciton
    #for the second for loop (j) and jitting that function. I might have to initialize all the arrays before the j-loop
    #function in that case.
    #NOTE: I could also try just jitting this entire function since that would be way easier and probably still much faster.
    #Maybe I'll try the latter first and then do it the former way later and see which is faster.
    """
    Find the horizon seen by each facet in terms of the azimuth angle and the zenith angles of the horizon at any particular
    azimuth angle.
    daz: azimuth step size [rad]
    dzen: zenith step size [rad]
    """
    
    #horizon_az = np.arange(0, 2*np.pi, daz) #azimuth angles [rad]
    
    #horizon_zen = np.arange(0, np.pi, dzen) #zenith angles [rad]
    
    #intersect_null = np.ones([tri_cent.shape[0], tri_vert.shape[0]], dtype=bool)
    #intersect_sky = np.zeros([tri_cent.shape[0], horizon_az.size, horizon_zen.size], dtype=bool)
    intersect_dist = np.ones([tri_cent.shape[0], tri_vert.shape[0]])
    
    for i in range(0,horizon_az.size):
        #intersect_sky[:,i,0] = True
        for j in range(0,horizon_zen.size):

            sinz = np.sin(horizon_zen[j])
            cosz = np.cos(horizon_zen[j])
            r_dir = np.array([np.cos(horizon_az[i])*sinz, np.sin(horizon_az[i])*sinz, cosz])
            
            #intersect = ray_triangle_intersect_Horizons(tri_cent[:,:], r_dir, \
            #                                        tri_vert[:,0,:], tri_vert[:,1,:], tri_vert[:,2,:])[:,0]
            intersect[:,:] = True
            intersect = ray_triangle_intersect_Horizons_1(tri_cent[:,:], r_dir, \
                                        tri_vert[:,0,:], tri_vert[:,1,:], tri_vert[:,2,:], intersect, tri_norm, intersect_dist)
            #intersect = ray_triangle_intersect_Horizons_2(tri_cent[:,:], r_dir, \
            #                            tri_vert[:,0,:], tri_vert[:,1,:], tri_vert[:,2,:], intersect, tri_norm, intersect_dist)
            #intersect[:,:] = True
            #intersect = ray_triangle_intersect_Horizons_3(tri_cent[:,:], r_dir, \
            #                            tri_vert[:,0,:], tri_vert[:,1,:], tri_vert[:,2,:], intersect, tri_norm, intersect_dist)

            #um = np.abs(u - mu.data)
            #print ('um:', um)
            #um[um<1e-5] = 0
            #print ('um:', um)
            #print ('u:', u)
            #print ('mu:', mu)
            #print ('np.count_nonzero(um):', np.count_nonzero(um))
            #print ('um.shape:', um.shape)
            
            #tvecm = np.abs(tvec - mtvec.data)
            #print ('tvecm:', tvecm)
            #tvecm[tvecm<1e-5] = 0
            #print ('tvecm:', tvecm)
            #print ('tvecm:', tvec)
            #print ('mtvec:', mtvec)
            #print ('np.count_nonzero(tvecm):', np.count_nonzero(tvecm))
            #print ('tvecm.shape:', tvecm.shape)
            
            intersect_sky[np.any(intersect==True, axis=1),i,j] = True
            #intersect_sky[np_any_axis1(intersect),i,j] = True
        
    return horizon_az, horizon_zen, intersect_sky

@njit(cache=True)
def np_any_axis1(x):
    """Numba compatible version of np.any(x, axis=1)."""
    out = np.zeros(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_or(out, x[:, i])
    return out

#@jit(nopython=True)
def ray_triangle_intersect_Horizons_1(r_orig, r_dir, v0, v1, v2, intersect, tri_norm, intersect_dist):
    """Let's make it so that I can do numerous intersection checks at once by inputing matricies.
    This function handles the case where the dimensions of the inputs are as follows,
    r_orig: (N,3)
    r_dir: (N,N,3)
    v0, v1, v2: (N,3)
    where N is the number of facets (triangle faces) in the surface.
    The resulting intersect matrix returned by this function has dimensions of: (N,N,N,2).
    """
    
    ##########NOTE: I should remake this function with intersect = np.ones([tri_cent.shape[0], tri_vert.shape[0]], dtype=bool)
    ########## since the horizons calculation doesn't need the distance to the intersection.
    ##########NOTE: I did the above and it seems to work :)
    
    #intersect = np.ones([r_orig.shape[0], v0.shape[0], 2])
    #t0 = time.time()
    
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pvec = np.cross(r_dir, v0v2)
    #print ('pvec.shape:', pvec.shape)
    #print ('pvec:', pvec)
    
    det = np.sum(v0v1*pvec, axis=1)
    #print ('det.shape:', det.shape)
    #print ('det:', det)

    #intersect[:, det<0.000001, :] = False, np.nan
    intersect[:, det<0.000001] = False
    intersect_dist[:, det<0.000001] = np.nan
    #print ('det')
    #print ('intersect:', intersect)
    #print ('np.count_nonzero(intersect):', np.count_nonzero(intersect))

    invDet = 1.0 / det
    #tvec = r_orig - v0[:,None,:]
    #tvec = r_orig - v0.reshape((v0.shape[0],1,v0.shape[1]))
    #tvec = r_orig - np.reshape(v0,(v0.shape[0],1,v0.shape[1]))
    tvec = r_orig - np.ascontiguousarray(v0).reshape((v0.shape[0],1,v0.shape[1]))
    #print ('tvec.shape:', tvec.shape)
    #print ('tvec:', tvec)

    #u = np.sum(np.swapaxes(tvec,0,1)*pvec, axis=2) * invDet
    #u = np.sum(tvec.T*pvec, axis=2) * invDet
    #u = np.sum(np.transpose(tvec,(1,0))*pvec, axis=2) * invDet
    u = np.sum(tvec.transpose((1,0,2))*pvec, axis=2) * invDet
    #print ('u.shape:', u.shape)
    #print ('u:', u)
    
    idx1 = np.where((u<0) | (u>1))
    #print ('idx1:', idx1)
    #print ('idx1[0].shape:', idx1[0].shape)
    #print ('idx1[1].shape:', idx1[1].shape)
    #intersect[idx1[0],idx1[1], :] = False, np.nan
    intersect[idx1[0],idx1[1]] = False
    intersect_dist[idx1[0],idx1[1]] = np.nan
    #print ('idx1')
    #print ('intersect:', intersect)
    #print ('np.count_nonzero(intersect):', np.count_nonzero(intersect))
    
    ######
    #for i in range(0,idx1[0].size): #Doing this because numba only supports one advanced index per array
    #    #intersect[idx1[0][i],idx1[1][i],:] = False, np.nan
    #    intersect[idx1[0][i],idx1[1][i]] = False
    #    intersect_dist[idx1[0][i],idx1[1][i]] = np.nan
    ######
        
    #t1 = time.time()
    #print ('01:', t1-t0)
        
    #qvec = np.cross(tvec, v0v1[:,None,:])
    #qvec = np.cross(tvec, v0v1.reshape((v0v1.shape[0],1,v0v1.shape[1])))
    #qvec = np.cross(tvec, np.reshape(v0v1,(v0v1.shape[0],1,v0v1.shape[1])))
    qvec = np.cross(tvec, np.ascontiguousarray(v0v1).reshape((v0v1.shape[0],1,v0v1.shape[1])))
    #print ('qvec.shape:', qvec.shape)
    #print ('qvec:', qvec)
    #v = np.swapaxes(np.sum(r_dir*qvec, axis=2),0,1) * invDet
    #print ('r_dir.shape:', r_dir.shape)
    #print ('qvec.shape:', qvec.shape)
    #print ('np.sum(r_dir*qvec, axis=2).shape:', np.sum(r_dir*qvec, axis=2).shape)
    #v = np.swapaxes(np.sum(r_dir*qvec, axis=2),0,1) * invDet
    #v = np.sum(r_dir*qvec, axis=2).T * invDet
    #v = np.transpose(np.sum(r_dir*qvec, axis=2),(1,0)) * invDet
    v = np.sum(r_dir*qvec, axis=2).transpose((1,0)) * invDet
    #print ('v.shape:', v.shape)
    #print ('v:', v)

    idx2 = np.where((v<0) | (u+v>1))
    #intersect[idx2[0],idx2[1], :] = False, np.nan
    intersect[idx2[0],idx2[1]] = False
    intersect_dist[idx2[0],idx2[1]] = np.nan
    #print ('idx2')
    #print ('intersect:', intersect)
    #print ('np.count_nonzero(intersect):', np.count_nonzero(intersect))
    
    ######
    #for i in range(0,idx2[0].size): #Doing this because numba only supports one advanced index per array
    #    #intersect[idx2[0][i],idx2[1][i],:] = False, np.nan
    #    intersect[idx2[0][i],idx2[1][i]] = False
    #    intersect_dist[idx2[0][i],idx2[1][i]] = np.nan
    ######

    #t2 = time.time()
    #print ('12:', t2-t1)
        
    #idx3 = np.where(intersect[:,:,0]==True)
    #intersect[idx3[0],idx3[1],1] = \
    #    ((intersect[:,:,0]==True)*np.swapaxes(np.sum(v0v2[:,None,:]*qvec, axis=2),0,1))[intersect[:,:,0]==True]*\
    #        invDet[idx3[1]]
    
    idx3 = np.where(intersect[:,:]==True)
    intersect_dist[idx3[0],idx3[1]] = \
                (np.sum(np.ascontiguousarray(v0v2).reshape((v0v2.shape[0],1,v0v2.shape[1]))*qvec, axis=2).transpose((1,0)))\
                [idx3[0],idx3[1]]*invDet[idx3[1]]
    #print ('idx3')
    #print ('intersect:', intersect)
    #print ('np.count_nonzero(intersect):', np.count_nonzero(intersect))
    #loop for jit
    #for i in range(0,idx3[0].size): #Doing this because numba only supports one advanced index per array
    #    intersect_dist[idx3[0][i],idx3[1][i]] = \
    #            (np.sum(np.ascontiguousarray(v0v2).reshape((v0v2.shape[0],1,v0v2.shape[1]))*qvec, axis=2).transpose((1,0)))\
    #            [idx3[0][i],idx3[1][i]]*invDet[idx3[1][i]]
    
    
        #((intersect[:,:]==True)*\
        #       np.sum(np.ascontiguousarray(v0v2).reshape((v0v2.shape[0],1,v0v2.shape[1]))*qvec, axis=2).transpose((1,0))) \
        #      [intersect[:,:]==True]*invDet[idx3[1]]
        #(np.sum(v0v2[:,None,:]*qvec, axis=2).transpose((1,0)))[idx3[0],idx3[1]]*\
        #    invDet[idx3[1]]

    #t3 = time.time()
    #print ('23:', t3-t2)
    
    #If the intersecting facet is behind the facet of interest, don't count it as an intersection
    intersect[:,:][intersect_dist[:,:]<=0] = False
    intersect_dist[:,:][intersect_dist[:,:]<=0] = np.nan
    #print ('intersect_dist')
    #print ('intersect:', intersect)
    #print ('np.count_nonzero(intersect):', np.count_nonzero(intersect))
    #idx4 = np.where(intersect_dist<=0)
    #for i in range(0,idx4[0].size):
    #    intersect[idx4[0][i],idx4[1][i]] = False
    #    intersect_dist[idx4[0][i],idx4[1][i]] = np.nan
    
    #Make tri_rays nan where ray is on the opposite side of the face from the normal
    cos = np.sum(r_dir*tri_norm, axis=1)
    idx_norm = np.where(cos<0)
    #print ('r_dir.shape:', r_dir.shape)
    #print ('cos:', cos)
    #print ('cos[idx_norm].shape:', cos[idx_norm].shape)
    #print ('cos[idx_norm]:', cos[idx_norm])
    #print ('idx_norm:', idx_norm)
    #print ('idx_norm[0]:', idx_norm[0])
    intersect[idx_norm[0],:] = False
    #print ('intersect[idx_norm,:]:', intersect[idx_norm,:])
    
    #t4 = time.time()
    #print ('34:', t4-t3)
    
    return intersect

def ray_triangle_intersect_Horizons_2(r_orig, r_dir, v0, v1, v2, intersect, tri_norm, intersect_dist):
    """Let's make it so that I can do numerous intersection checks at once by inputing matricies.
    This function handles the case where the dimensions of the inputs are as follows,
    r_orig: (N,3)
    r_dir: (N,N,3)
    v0, v1, v2: (N,3)
    where N is the number of facets (triangle faces) in the surface.
    The resulting intersect matrix returned by this function has dimensions of: (N,N,N,2).
    """
    
    ##########NOTE: I should remake this function with intersect = np.ones([tri_cent.shape[0], tri_vert.shape[0]], dtype=bool)
    ########## since the horizons calculation doesn't need the distance to the intersection.
    ##########NOTE: I did the above and it seems to work :)
    
    #intersect = np.ones([r_orig.shape[0], v0.shape[0], 2])
    #t0 = time.time()
    
    print ('intersect.shape:', intersect.shape)
    print ('r_orig.shape:', r_orig.shape)
    print ('r_dir.shape:', r_dir.shape)
    
    v0v1 = v1 - v0
    v0v2 = v2 - v0
    pvec = np.cross(r_dir, v0v2)
    print ('v0v1.shape:', v0v1.shape)
    print ('v0v2.shape:', v0v2.shape)
    print ('pvec.shape:', pvec.shape)
    
    det = np.sum(v0v1*pvec, axis=1)
    print ('det.shape:', det.shape)

    #intersect[:, det<0.000001, :] = False, np.nan
    intersect[:, det<0.000001] = False
    intersect_dist[:, det<0.000001] = np.nan
    
    idx_skip = np.where(~(det<0.000001))
    print ('idx_skip[0].shape:', idx_skip[0].shape)
    print ('idx_skip:', idx_skip)
    print ('idx_skip[0]:', idx_skip[0])

    invDet = 1.0 / det[idx_skip]
    print ('invDet.shape:', invDet.shape)
    #tvec = r_orig - v0[:,None,:]
    #tvec = r_orig - v0.reshape((v0.shape[0],1,v0.shape[1]))
    #tvec = r_orig - np.reshape(v0,(v0.shape[0],1,v0.shape[1]))
    #tvec = r_orig - np.ascontiguousarray(v0).reshape((v0.shape[0],1,v0.shape[1]))
    tvec = r_orig - np.ascontiguousarray(v0[idx_skip,:]).reshape((idx_skip[0].shape[0],1,v0.shape[1]))
    print ('tvec.shape:', tvec.shape)

    print ('pvec[idx_skip,:].shape:', pvec[idx_skip,:].shape)
    #u = np.sum(np.swapaxes(tvec,0,1)*pvec, axis=2) * invDet
    #u = np.sum(tvec.T*pvec, axis=2) * invDet
    #u = np.sum(np.transpose(tvec,(1,0))*pvec, axis=2) * invDet
    #u = np.sum(tvec.transpose((1,0,2))*pvec, axis=2) * invDet
    u = np.sum(tvec.transpose((1,0,2))*pvec[idx_skip,:], axis=2) * invDet
    print ('u.shape:', u.shape)
    
    idx1 = np.where((u<0) | (u>1))
    #intersect[idx1[0],idx1[1], :] = False, np.nan
    #intersect[idx1[0],idx1[1]] = False
    #intersect_dist[idx1[0],idx1[1]] = np.nan
    print ('intersect[:,idx_skip[0]].shape:', intersect[:,idx_skip[0]].shape)
    print ('intersect[:,idx_skip[0]]:', intersect[:,idx_skip[0]])
    print ('intersect[:,idx_skip[0]][idx1[0],idx1[1]].shape:', intersect[:,idx_skip[0]][idx1[0],idx1[1]].shape)
    #intersect[:,idx_skip[0]][idx1[0],idx1[1]] = False
    #intersect_dist[:,idx_skip[0]][idx1[0],idx1[1]] = np.nan
    #print ('intersect[:,idx_skip[0]][idx1[0],idx1[1]]:', intersect[:,idx_skip[0]][idx1[0],idx1[1]])
    intersect[idx1[0],idx_skip[0][idx1[1]]] = False
    intersect_dist[idx1[0],idx_skip[0][idx1[1]]] = np.nan
    print ('idx1[1].shape:', idx1[1].shape)
    print ('idx_skip[0][idx1[1]]:', idx_skip[0][idx1[1]])
    print ('idx_skip[0][idx1[1]].shape:', idx_skip[0][idx1[1]].shape)
    print ('intersect[idx1[0],idx_skip[0][idx1[1]]]:', intersect[idx1[0],idx_skip[0][idx1[1]]])
    
    
    ######
    #for i in range(0,idx1[0].size): #Doing this because numba only supports one advanced index per array
    #    #intersect[idx1[0][i],idx1[1][i],:] = False, np.nan
    #    intersect[idx1[0][i],idx1[1][i]] = False
    #    intersect_dist[idx1[0][i],idx1[1][i]] = np.nan
    ######
        
    #t1 = time.time()
    #print ('01:', t1-t0)
        
    #qvec = np.cross(tvec, v0v1[:,None,:])
    #qvec = np.cross(tvec, v0v1.reshape((v0v1.shape[0],1,v0v1.shape[1])))
    #qvec = np.cross(tvec, np.reshape(v0v1,(v0v1.shape[0],1,v0v1.shape[1])))
    #qvec = np.cross(tvec, np.ascontiguousarray(v0v1).reshape((v0v1.shape[0],1,v0v1.shape[1])))
    qvec = np.cross(tvec, np.ascontiguousarray(v0v1[idx_skip,:]).reshape((idx_skip[0].shape[0],1,v0v1.shape[1])))
    
    #v = np.swapaxes(np.sum(r_dir*qvec, axis=2),0,1) * invDet
    #print ('r_dir.shape:', r_dir.shape)
    print ('qvec.shape:', qvec.shape)
    print ('qvec.transpose((1,0,2)).shape:', qvec.transpose((1,0,2)).shape)
    #print ('np.sum(r_dir*qvec, axis=2).shape:', np.sum(r_dir*qvec, axis=2).shape)
    #v = np.swapaxes(np.sum(r_dir*qvec, axis=2),0,1) * invDet
    #v = np.sum(r_dir*qvec, axis=2).T * invDet
    #v = np.transpose(np.sum(r_dir*qvec, axis=2),(1,0)) * invDet
    #print ('np.sum(r_dir*qvec, axis=2).transpose((1,0)).shape:', np.sum(r_dir*qvec, axis=2).transpose((1,0)).shape)
    #print ('np.sum(r_dir*qvec, axis=2).transpose((1,0)):', np.sum(r_dir*qvec, axis=2).transpose((1,0)))
    #print ('np.sum(r_dir*qvec.transpose((1,0,2)), axis=2).shape:', np.sum(r_dir*qvec.transpose((1,0,2)), axis=2).shape)
    #print ('np.sum(r_dir*qvec.transpose((1,0,2)), axis=2):', np.sum(r_dir*qvec.transpose((1,0,2)), axis=2))
    
    idx_skip1 = np.where(intersect[:,idx_skip[0]]==True)
    print ('idx_skip1[0].shape:', idx_skip1[0].shape)
    print ('idx_skip1[1].shape:', idx_skip1[1].shape)
    
    print ('qvec.transpose((1,0,2))[idx_skip1[0], idx_skip1[1], :].shape:', qvec.transpose((1,0,2))[idx_skip1[0], idx_skip1[1], :].shape)
    
    #v = np.sum(r_dir*qvec, axis=2).transpose((1,0)) * invDet
    v = np.sum(r_dir*qvec.transpose((1,0,2)), axis=2) * invDet
    v = np.sum(r_dir*qvec.transpose((1,0,2))[idx_skip1[0], idx_skip1[1], :], axis=1) * invDet
    print ('v.shape:', v.shape)

    idx2 = np.where((v<0) | (u+v>1))
    #intersect[idx2[0],idx2[1], :] = False, np.nan
    intersect[idx2[0],idx2[1]] = False
    intersect_dist[idx2[0],idx2[1]] = np.nan
    
    ######
    #for i in range(0,idx2[0].size): #Doing this because numba only supports one advanced index per array
    #    #intersect[idx2[0][i],idx2[1][i],:] = False, np.nan
    #    intersect[idx2[0][i],idx2[1][i]] = False
    #    intersect_dist[idx2[0][i],idx2[1][i]] = np.nan
    ######

    #t2 = time.time()
    #print ('12:', t2-t1)
        
    #idx3 = np.where(intersect[:,:,0]==True)
    #intersect[idx3[0],idx3[1],1] = \
    #    ((intersect[:,:,0]==True)*np.swapaxes(np.sum(v0v2[:,None,:]*qvec, axis=2),0,1))[intersect[:,:,0]==True]*\
    #        invDet[idx3[1]]
    
    idx3 = np.where(intersect[:,:]==True)
    intersect_dist[idx3[0],idx3[1]] = \
                (np.sum(np.ascontiguousarray(v0v2).reshape((v0v2.shape[0],1,v0v2.shape[1]))*qvec, axis=2).transpose((1,0)))\
                [idx3[0],idx3[1]]*invDet[idx3[1]]
    #loop for jit
    #for i in range(0,idx3[0].size): #Doing this because numba only supports one advanced index per array
    #    intersect_dist[idx3[0][i],idx3[1][i]] = \
    #            (np.sum(np.ascontiguousarray(v0v2).reshape((v0v2.shape[0],1,v0v2.shape[1]))*qvec, axis=2).transpose((1,0)))\
    #            [idx3[0][i],idx3[1][i]]*invDet[idx3[1][i]]
    
    
        #((intersect[:,:]==True)*\
        #       np.sum(np.ascontiguousarray(v0v2).reshape((v0v2.shape[0],1,v0v2.shape[1]))*qvec, axis=2).transpose((1,0))) \
        #      [intersect[:,:]==True]*invDet[idx3[1]]
        #(np.sum(v0v2[:,None,:]*qvec, axis=2).transpose((1,0)))[idx3[0],idx3[1]]*\
        #    invDet[idx3[1]]

    #t3 = time.time()
    #print ('23:', t3-t2)
    
    #If the intersecting facet is behind the facet of interest, don't count it as an intersection
    intersect[:,:][intersect_dist[:,:]<=0] = False
    intersect_dist[:,:][intersect_dist[:,:]<=0] = np.nan
    #idx4 = np.where(intersect_dist<=0)
    #for i in range(0,idx4[0].size):
    #    intersect[idx4[0][i],idx4[1][i]] = False
    #    intersect_dist[idx4[0][i],idx4[1][i]] = np.nan
    
    #Make tri_rays nan where ray is on the opposite side of the face from the normal
    cos = np.sum(r_dir*tri_norm, axis=1)
    idx_norm = np.where(cos<0)
    #print ('r_dir.shape:', r_dir.shape)
    #print ('cos:', cos)
    #print ('cos[idx_norm].shape:', cos[idx_norm].shape)
    #print ('cos[idx_norm]:', cos[idx_norm])
    #print ('idx_norm:', idx_norm)
    #print ('idx_norm[0]:', idx_norm[0])
    intersect[idx_norm[0],:] = False
    #print ('intersect[idx_norm,:]:', intersect[idx_norm,:])
    
    #t4 = time.time()
    #print ('34:', t4-t3)
    
    return intersect

def ray_triangle_intersect_Horizons_3(r_orig, r_dir, v0, v1, v2, intersect, tri_norm, intersect_dist):
    """Let's make it so that I can do numerous intersection checks at once by inputing matricies.
    This function handles the case where the dimensions of the inputs are as follows,
    r_orig: (N,3)
    r_dir: (N,N,3)
    v0, v1, v2: (N,3)
    where N is the number of facets (triangle faces) in the surface.
    The resulting intersect matrix returned by this function has dimensions of: (N,N,N,2).
    """
    
    ##########NOTE: I should remake this function with intersect = np.ones([tri_cent.shape[0], tri_vert.shape[0]], dtype=bool)
    ########## since the horizons calculation doesn't need the distance to the intersection.
    ##########NOTE: I did the above and it seems to work :)
    
    #intersect = np.ones([r_orig.shape[0], v0.shape[0], 2])
    #t0 = time.time()
    
    mr_orig, mr_dir, mv0, mv1, mv2, mintersect, mtri_norm, mintersect_dist = ma.array(r_orig), ma.array(r_dir), ma.array(v0),\
                                ma.array(v1), ma.array(v2), ma.array(intersect), ma.array(tri_norm), ma.array(intersect_dist)
    
    #print ('intersect.shape:', intersect.shape)
    #print ('r_orig.shape:', r_orig.shape)
    #print ('r_dir.shape:', r_dir.shape)
    #print ('\n Masked:')
    
    #v0v1 = v1 - v0
    #v0v2 = v2 - v0
    #pvec = np.cross(r_dir, v0v2)
    
    mv0v1 = mv1 - mv0
    mv0v2 = mv2 - mv0
    mpvec = ma.array(np.cross(mr_dir, mv0v2))
    #mpvec = np.cross(mr_dir, mv0v2)
    #print ('mpvec.shape:', mpvec.shape)
    #print ('mpvec:', mpvec)
    
    #det = np.sum(v0v1*pvec, axis=1)
    mdet = np.sum(mv0v1*mpvec, axis=1)
    #print ('mdet.shape:', mdet.shape)
    #print ('mdet:', mdet)

    #intersect[:, det<0.000001, :] = False, np.nan
    #intersect[:, det<0.000001] = False
    #intersect_dist[:, det<0.000001] = np.nan
    intersect[:, mdet<0.000001] = False
    intersect_dist[:, mdet<0.000001] = np.nan
    print ('mdet')
    print ('intersect:', intersect)
    print ('np.count_nonzero(intersect):', np.count_nonzero(intersect))
    
    #idx_skip = np.where(~(det<0.000001))
    mask = [~(mdet<0.000001)]
    #print ('idx_skip[0].shape:', idx_skip[0].shape)
    #print ('idx_skip:', idx_skip)
    #print ('idx_skip[0]:', idx_skip[0])

    mv0.mask = mask
    mv0v1.mask = mask
    mv0v2.mask = mask
    mpvec.mask = mask
    mdet.mask = mask
    #print ('np.count_nonzero(mdet.mask):', np.count_nonzero(mdet.mask))
    
    #invDet = 1.0 / det[idx_skip]
    minvDet = 1.0 / mdet
    #print ('invDet.shape:', invDet.shape)
    #tvec = r_orig - v0[:,None,:]
    #tvec = r_orig - v0.reshape((v0.shape[0],1,v0.shape[1]))
    #tvec = r_orig - np.reshape(v0,(v0.shape[0],1,v0.shape[1]))
    #tvec = r_orig - np.ascontiguousarray(v0).reshape((v0.shape[0],1,v0.shape[1]))
    #tvec = r_orig - np.ascontiguousarray(v0[idx_skip,:]).reshape((idx_skip[0].shape[0],1,v0.shape[1]))
    #print ('mv0.mask:', mv0.mask)
    #print ('np.count_nonzero(mv0.mask):', np.count_nonzero(mv0.mask))
    #mtvec = mr_orig - np.ascontiguousarray(mv0).reshape((mv0.shape[0],1,mv0.shape[1]))
    mtvec = mr_orig - mv0.reshape((mv0.shape[0],1,mv0.shape[1]))
    #print ('mtvec.shape:', mtvec.shape)
    #print ('mtvec:', mtvec)
    #print ('mtvec.mask:', mtvec.mask)
    #print ('np.count_nonzero(mtvec.mask):', np.count_nonzero(mtvec.mask))

    #print ('pvec[idx_skip,:].shape:', pvec[idx_skip,:].shape)
    #u = np.sum(np.swapaxes(tvec,0,1)*pvec, axis=2) * invDet
    #u = np.sum(tvec.T*pvec, axis=2) * invDet
    #u = np.sum(np.transpose(tvec,(1,0))*pvec, axis=2) * invDet
    #u = np.sum(tvec.transpose((1,0,2))*pvec, axis=2) * invDet
    #u = np.sum(tvec.transpose((1,0,2))*pvec[idx_skip,:], axis=2) * invDet
    mu = np.sum(mtvec.transpose((1,0,2))*mpvec, axis=2) * minvDet
    #print ('mu.shape:', mu.shape)
    #print ('mu:', mu)
    
    #idx1 = np.where((u<0) | (u>1))
    idx1 = np.where((mu<0) | (mu>1))
    print ('idx1:', idx1)
    print ('idx1[0].shape:', idx1[0].shape)
    print ('idx1[1].shape:', idx1[1].shape)
    #intersect[idx1[0],idx1[1], :] = False, np.nan
    #intersect[idx1[0],idx1[1]] = False
    #intersect_dist[idx1[0],idx1[1]] = np.nan
    #print ('intersect[:,idx_skip[0]].shape:', intersect[:,idx_skip[0]].shape)
    #print ('intersect[:,idx_skip[0]]:', intersect[:,idx_skip[0]])
    #print ('intersect[:,idx_skip[0]][idx1[0],idx1[1]].shape:', intersect[:,idx_skip[0]][idx1[0],idx1[1]].shape)
    ##intersect[:,idx_skip[0]][idx1[0],idx1[1]] = False
    ##intersect_dist[:,idx_skip[0]][idx1[0],idx1[1]] = np.nan
    ##print ('intersect[:,idx_skip[0]][idx1[0],idx1[1]]:', intersect[:,idx_skip[0]][idx1[0],idx1[1]])
    #intersect[idx1[0],idx_skip[0][idx1[1]]] = False
    #intersect_dist[idx1[0],idx_skip[0][idx1[1]]] = np.nan
    #print ('idx1[1].shape:', idx1[1].shape)
    #print ('idx_skip[0][idx1[1]]:', idx_skip[0][idx1[1]])
    #print ('idx_skip[0][idx1[1]].shape:', idx_skip[0][idx1[1]].shape)
    #print ('intersect[idx1[0],idx_skip[0][idx1[1]]]:', intersect[idx1[0],idx_skip[0][idx1[1]]])
    intersect[idx1[0],idx1[1]] = False
    intersect_dist[idx1[0],idx1[1]] = np.nan
    print ('idx1')
    print ('intersect:', intersect)
    print ('np.count_nonzero(intersect):', np.count_nonzero(intersect))
    
    
    ######
    #for i in range(0,idx1[0].size): #Doing this because numba only supports one advanced index per array
    #    #intersect[idx1[0][i],idx1[1][i],:] = False, np.nan
    #    intersect[idx1[0][i],idx1[1][i]] = False
    #    intersect_dist[idx1[0][i],idx1[1][i]] = np.nan
    ######
        
    #t1 = time.time()
    #print ('01:', t1-t0)
        
    #qvec = np.cross(tvec, v0v1[:,None,:])
    #qvec = np.cross(tvec, v0v1.reshape((v0v1.shape[0],1,v0v1.shape[1])))
    #qvec = np.cross(tvec, np.reshape(v0v1,(v0v1.shape[0],1,v0v1.shape[1])))
    #qvec = np.cross(tvec, np.ascontiguousarray(v0v1).reshape((v0v1.shape[0],1,v0v1.shape[1])))
    #qvec = np.cross(tvec, np.ascontiguousarray(v0v1[idx_skip,:]).reshape((idx_skip[0].shape[0],1,v0v1.shape[1])))
    #mqvec = ma.array(np.cross(mtvec, np.ascontiguousarray(mv0v1).reshape((mv0v1.shape[0],1,mv0v1.shape[1]))))
    mqvec = ma.array(np.cross(mtvec, mv0v1.reshape((mv0v1.shape[0],1,mv0v1.shape[1]))))
    
    mqvec.transpose((1,0,2)).mask = (mqvec.transpose((1,0,2)).mask) | (intersect==False)
    
    #print ('mqvec.shape:', mqvec.shape)
    #print ('mqvec:', mqvec)
    
    #v = np.swapaxes(np.sum(r_dir*qvec, axis=2),0,1) * invDet
    #print ('r_dir.shape:', r_dir.shape)
    #print ('qvec.shape:', qvec.shape)
    #print ('qvec.transpose((1,0,2)).shape:', qvec.transpose((1,0,2)).shape)
    #print ('np.sum(r_dir*qvec, axis=2).shape:', np.sum(r_dir*qvec, axis=2).shape)
    #v = np.swapaxes(np.sum(r_dir*qvec, axis=2),0,1) * invDet
    #v = np.sum(r_dir*qvec, axis=2).T * invDet
    #v = np.transpose(np.sum(r_dir*qvec, axis=2),(1,0)) * invDet
    #print ('np.sum(r_dir*qvec, axis=2).transpose((1,0)).shape:', np.sum(r_dir*qvec, axis=2).transpose((1,0)).shape)
    #print ('np.sum(r_dir*qvec, axis=2).transpose((1,0)):', np.sum(r_dir*qvec, axis=2).transpose((1,0)))
    #print ('np.sum(r_dir*qvec.transpose((1,0,2)), axis=2).shape:', np.sum(r_dir*qvec.transpose((1,0,2)), axis=2).shape)
    #print ('np.sum(r_dir*qvec.transpose((1,0,2)), axis=2):', np.sum(r_dir*qvec.transpose((1,0,2)), axis=2))
    
    #idx_skip1 = np.where(intersect[:,idx_skip[0]]==True)
    #print ('idx_skip1[0].shape:', idx_skip1[0].shape)
    #print ('idx_skip1[1].shape:', idx_skip1[1].shape)
    
    #print ('qvec.transpose((1,0,2))[idx_skip1[0], idx_skip1[1], :].shape:', qvec.transpose((1,0,2))[idx_skip1[0], idx_skip1[1], :].shape)
    
    #v = np.sum(r_dir*qvec, axis=2).transpose((1,0)) * invDet
    #v = np.sum(r_dir*qvec.transpose((1,0,2)), axis=2) * invDet
    #v = np.sum(r_dir*qvec.transpose((1,0,2))[idx_skip1[0], idx_skip1[1], :], axis=1) * invDet
    mv = np.sum(mr_dir*mqvec.transpose((1,0,2)), axis=2) * minvDet
    #print ('mv.shape:', mv.shape)
    #print ('mv:', mv)

    #idx2 = np.where((v<0) | (u+v>1))
    idx2 = np.where((mv<0) | (mu+mv>1))
    #intersect[idx2[0],idx2[1], :] = False, np.nan
    intersect[idx2[0],idx2[1]] = False
    intersect_dist[idx2[0],idx2[1]] = np.nan
    print ('idx2')
    print ('intersect:', intersect)
    print ('np.count_nonzero(intersect):', np.count_nonzero(intersect))
    
    ######
    #for i in range(0,idx2[0].size): #Doing this because numba only supports one advanced index per array
    #    #intersect[idx2[0][i],idx2[1][i],:] = False, np.nan
    #    intersect[idx2[0][i],idx2[1][i]] = False
    #    intersect_dist[idx2[0][i],idx2[1][i]] = np.nan
    ######

    #t2 = time.time()
    #print ('12:', t2-t1)
        
    #idx3 = np.where(intersect[:,:,0]==True)
    #intersect[idx3[0],idx3[1],1] = \
    #    ((intersect[:,:,0]==True)*np.swapaxes(np.sum(v0v2[:,None,:]*qvec, axis=2),0,1))[intersect[:,:,0]==True]*\
    #        invDet[idx3[1]]
    
    idx3 = np.where(intersect[:,:]==True)
    #intersect_dist[idx3[0],idx3[1]] = \
    #            (np.sum(np.ascontiguousarray(v0v2).reshape((v0v2.shape[0],1,v0v2.shape[1]))*qvec, axis=2).transpose((1,0)))\
    #            [idx3[0],idx3[1]]*invDet[idx3[1]]
    intersect_dist[idx3[0],idx3[1]] = \
                (np.sum(np.ascontiguousarray(mv0v2).reshape((mv0v2.shape[0],1,mv0v2.shape[1]))*mqvec, axis=2).transpose((1,0)))\
                [idx3[0],idx3[1]]*minvDet[idx3[1]]
    print ('idx3')
    print ('intersect:', intersect)
    print ('np.count_nonzero(intersect):', np.count_nonzero(intersect))
    #loop for jit
    #for i in range(0,idx3[0].size): #Doing this because numba only supports one advanced index per array
    #    intersect_dist[idx3[0][i],idx3[1][i]] = \
    #            (np.sum(np.ascontiguousarray(v0v2).reshape((v0v2.shape[0],1,v0v2.shape[1]))*qvec, axis=2).transpose((1,0)))\
    #            [idx3[0][i],idx3[1][i]]*invDet[idx3[1][i]]
    
    
        #((intersect[:,:]==True)*\
        #       np.sum(np.ascontiguousarray(v0v2).reshape((v0v2.shape[0],1,v0v2.shape[1]))*qvec, axis=2).transpose((1,0))) \
        #      [intersect[:,:]==True]*invDet[idx3[1]]
        #(np.sum(v0v2[:,None,:]*qvec, axis=2).transpose((1,0)))[idx3[0],idx3[1]]*\
        #    invDet[idx3[1]]

    #t3 = time.time()
    #print ('23:', t3-t2)
    
    #If the intersecting facet is behind the facet of interest, don't count it as an intersection
    #print ('intersect_dist')
    intersect[:,:][intersect_dist[:,:]<=0] = False
    intersect_dist[:,:][intersect_dist[:,:]<=0] = np.nan
    print ('intersect:', intersect)
    print ('np.count_nonzero(intersect):', np.count_nonzero(intersect))
    intersect[intersect_dist<=0] = False
    #intersect_dist[intersect_dist<=0] = np.nan
    #idx4 = np.where(intersect_dist<=0)
    #for i in range(0,idx4[0].size):
    #    intersect[idx4[0][i],idx4[1][i]] = False
    #    intersect_dist[idx4[0][i],idx4[1][i]] = np.nan
    
    #Make tri_rays nan where ray is on the opposite side of the face from the normal
    cos = np.sum(r_dir*tri_norm, axis=1)
    idx_norm = np.where(cos<0)
    #print ('r_dir.shape:', r_dir.shape)
    #print ('cos:', cos)
    #print ('cos[idx_norm].shape:', cos[idx_norm].shape)
    #print ('cos[idx_norm]:', cos[idx_norm])
    #print ('idx_norm:', idx_norm)
    #print ('idx_norm[0]:', idx_norm[0])
    intersect[idx_norm[0],:] = False
    #print ('intersect[idx_norm,:]:', intersect[idx_norm,:])
    
    #t4 = time.time()
    #print ('34:', t4-t3)
    
    return intersect, mu, mtvec

def Find_Horizon_5(tri_cent, tri_vert, tri_norm, horizon_az, horizon_zen, pro_idx, daz=1*np.pi/180, dzen=1*np.pi/180):
    #I want this new version to include multiprocessing AND jit compiling
    #I can probably do this by splitting the first for loop (i) into different processors and then creating a funciton
    #for the second for loop (j) and jitting that function. I might have to initialize all the arrays before the j-loop
    #function in that case.
    #NOTE: I could also try just jitting this entire function since that would be way easier and probably still much faster.
    #Maybe I'll try the latter first and then do it the former way later and see which is faster.
    """
    Find the horizon seen by each facet in terms of the azimuth angle and the zenith angles of the horizon at any particular
    azimuth angle.
    daz: azimuth step size [rad]
    dzen: zenith step size [rad]
    """
    
    #horizon_az = np.arange(0, 2*np.pi, daz) #azimuth angles [rad]
    
    #horizon_zen = np.arange(0, np.pi, dzen) #zenith angles [rad]
    
    intersect = np.ones([tri_cent.shape[0], tri_vert.shape[0]], dtype=bool)
    intersect_dist = np.ones([tri_cent.shape[0], tri_vert.shape[0]])
    #intersect = np.ones([tri_cent.shape[0], tri_vert.shape[0], 2])
    intersect_sky = np.zeros([tri_cent.shape[0], horizon_az.size, horizon_zen.size], dtype=bool)
    
    
    for i in range(0,horizon_az.size):
        #intersect_sky[:,i,0] = True
        for j in range(0,horizon_zen.size):

            sinz = np.sin(horizon_zen[j])
            cosz = np.cos(horizon_zen[j])
            r_dir = np.array([np.cos(horizon_az[i])*sinz, np.sin(horizon_az[i])*sinz, cosz])
            
            #intersect[:,:,0] = True
            #intersect[:,:,0] = ray_triangle_intersect_Horizons_1(tri_cent[:,:], r_dir, \
            #                                        tri_vert[:,0,:], tri_vert[:,1,:], tri_vert[:,2,:], intersect, tri_norm)[:,0]
            #intersect[:,:] = True
            #intersect = ray_triangle_intersect_Horizons_1(tri_cent[:,:], r_dir, \
            #                                        tri_vert[:,0,:], tri_vert[:,1,:], tri_vert[:,2,:], intersect, tri_norm)

            intersect[:,:] = True
            intersect = ray_triangle_intersect_Horizons_1(tri_cent[:,:], r_dir, \
                                        tri_vert[:,0,:], tri_vert[:,1,:], tri_vert[:,2,:], intersect, tri_norm, intersect_dist)
            
            
            #intersect_sky[np.any(intersect[:,:,0]==True, axis=1),i,j] = True
            intersect_sky[np.any(intersect==True, axis=1),i,j] = True
        
    return np.append(intersect_sky, \
                     np.resize(pro_idx[None,:],(intersect_sky.shape[0],intersect_sky.shape[1]))[:,:,None], axis=2)

def Solar_Rays(horizon_az, horizon_zen, tri_norm):
    r_dir = np.zeros([horizon_az.size, horizon_zen.size, 3])
    theta_sky = np.zeros([tri_norm.shape[0], horizon_az.size, horizon_zen.size])
    
    for i in range(0,horizon_az.size):
        for j in range(0,horizon_zen.size):
            sinz = np.sin(horizon_zen[j])
            cosz = np.cos(horizon_zen[j])
            r_dir[i,j,:] = np.array([np.cos(horizon_az[i])*sinz, np.sin(horizon_az[i])*sinz, cosz])
            theta_sky[:,i,j] = np.arccos( np.dot(tri_norm, r_dir[i,j,:])/\
            (np.linalg.norm(tri_norm)*np.linalg.norm(r_dir[i,j,:])) )
    
    return r_dir.reshape([horizon_az.size*horizon_zen.size,3]), \
            theta_sky.reshape([tri_norm.shape[0], horizon_az.size*horizon_zen.size])

def Find_Horizon_6(tri_cent, tri_vert, horizon_az, horizon_zen, r_dir, theta_sky, pro_idx):
    """
    Find the horizon seen by each facet in terms of the azimuth angle and the zenith angles of the horizon at any particular
    azimuth angle.
    
    Returns intersect_sky array, which contains a True/False for each facet and each solar azimuth and elevation angle. The
    array is True when there is an intersect. In other words, for each facet intersect_sky[i,:,:] will show the entire sky view
    of the facet, with Trues where the facet's sky view is blocked by other facets and Falses where the facet can see out to
    space.
    
    This version was adapted from Find_Horizon_5. Its purpose is to try and speed up the horizons calculation by eliminating
    the intersect check for any facet that is facing away from the sky (e.g, a facet on a cave roof). For these facets, if
    the angle between the surface normal and the solar ray is greater than 90 degrees, then the facet will not see the sun and
    the intersect check can be skipped.
    In order to do this, we must break up the multiprocessing by facet as opposed to by solar azimuth (as in Find_Horizon_5).
    
    I first wrote this on a plane to SF, so I can't really check it right now. Let's see if this works!
    """
    
    intersect = np.zeros([horizon_az.size*horizon_zen.size, tri_vert.shape[0]], dtype=bool)
    intersect_sky = np.zeros([tri_cent.shape[0], horizon_az.size*horizon_zen.size], dtype=bool)
    
    for i in range(0, tri_cent.shape[0]):
        
        intersect[theta_sky[i,:]>np.pi/2, :] = True
        
        #idx_dir = [theta_sky[i,:]<=np.pi/2]
        #intersect[idx_dir, i] = ray_triangle_intersect_View_Factors(tri_cent[i,:], r_dir[idx_dir,:], tri_vert[:,0,:],\
        #                                                               tri_vert[:,1,:], tri_vert[:,2,:])[0]
        intersect[:, i] = ray_triangle_intersect_View_Factors(tri_cent[i,:], r_dir[:,:], tri_vert[:,0,:],\
                                                                       tri_vert[:,1,:], tri_vert[:,2,:])[0]
        
        intersect_sky[i, np.any(intersect==True, axis=1)] = True
        
    intersect_sky = intersect_sky.reshape([tri_cent.shape[0], horizon_az.size, horizon_zen.size])
    
    return np.append(intersect_sky, \
                     np.resize(pro_idx[:,None],(intersect_sky.shape[0],intersect_sky.shape[1]))[:,:,None], axis=2)
    
def IsotropicVolatileModel(F_los, T, dt_T, dt, f_theta_i, daz, dzen, horizon_zen, intersect_sky, tri_area, num_save=int(1e4), \
                           planet=planets.Moon):
    #Volatile Hopping Model Inputs:
    #tri_area, F_los (view factors), T (surface temperatures), dt_T (thermal model timestep), 
    #dt (sublimation model timestep),
    #f_theta_i (initial surface density of H2O molecules on each facet as fraction of 1e19 for monolayer)
    
    #Outputs:
    #theta (surface density of H2O molecules on each facet at each timestep), time array
    #instantaneous (at each timestep) vapor pressure within pit/cave??
    Na = 6.02214076e23 #Avagadro Constant [mol-1]
    mw = 0.018 #molar mass of water [kg.mol-1]
    Rg = 8.314 #universal gas constant [J.K-1.mol-1]
    #theta_i: initial surface density of H2O molecules on each facet [m-2]
    theta_m = 1e19 #surface density of an h2o monolayer [molecules.m-2] (Schorghofer and Aharonson, 2014)
    
    N_steps = int(T.shape[0]*dt_T/dt)
    print ('N_steps:', N_steps)
    print ('T.shape[0]:', T.shape[0])
    num_facet = f_theta_i.size
    print ('num_facet:', num_facet)
    t = 0
    lt = np.zeros([num_save]) #local time
    f_theta = np.zeros([num_save, num_facet]) #number of H2O molecules on each facet at each timestep
    f_theta[0,:] = f_theta_i
    f_p = np.zeros_like(f_theta_i)
    #E_sub = np.zeros_like(f_theta_i)
    m_h2o = np.zeros([num_save, num_facet]) #[kg.m-2]
    m_h2o[0,:] = f_theta_i*theta_m*mw/Na
    
    ##############################################################
    ###### For calculating the total ice mass loss to space ######
    m_space = np.zeros([num_save, num_facet]) #Mass lost to space by each facet [kg]
    m_s = 0 #Total mass lost to space [kg]
    
    #Calculate solid angle of the sky seen by each facet
    #Solid angle corresponding to each elevation/azimuth pixel
    SA = daz*(np.cos(horizon_zen-(dzen/2))-np.cos(horizon_zen+(dzen/2)))
    SA[0] = daz*(np.cos(horizon_zen[0])-np.cos(horizon_zen[0]+(dzen/2)))
    SA[-1] = daz*(np.cos(horizon_zen[-1]-(dzen/2))-np.cos(horizon_zen[-1]))
    
    frac_sky = np.sum((~intersect_sky)*SA, axis=(1,2))/(2*np.pi)
    
    frac_const = tri_area*dt*frac_sky
    ##############################################################
    ##############################################################
    
    p_sat = satVaporPressureIce(T)
    rate_ice = p_sat*np.sqrt(mw/(2*np.pi*Rg*T)) #[kg.m-2.s-1]
    
    
    #Adjust F_los to account for known innaccuracies in view factor calculation
    F_adj = np.copy(F_los)
    idx = np.sum(F_los[:,:], axis=1)>1
    F_adj[idx,:] = F_adj[idx,:]/np.sum(F_los[idx,:], axis=1)[:,None]
    #F_adj[idx,:] = F_adj[idx,:]/np.sum(F_los[idx,:], axis=1)[None,:]
    
    save_count = 1
    m = f_theta_i*theta_m*mw/Na
    f = f_theta_i
    
    #for i in range(1, N_steps):
    for i in tqdm(range(1, N_steps)):
        #t0 = time.time()
        i_T = int(i*dt/dt_T)
        #print ('i:', i)
        #print ('i_T:', i_T)
        #print ('i:', i)
        #print ('i_T:', i_T, '\n')
        
        #f_p = AdsorptionFrac(f_theta[i-1,:])
        #E_sub = rate_ice[i_T,:]*f_p #[kg.m-2.s-1]
        #f_p = AdsorptionFrac(f, f_p)
        idx_ice = [f>0]
        #print ('idx_ice[0].shape:', idx_ice[0].shape)
        #print ('idx_ice[0]:', idx_ice[0], '\n')
        f_p[:] = 0
        f_p[idx_ice] = AdsorptionFrac(f[idx_ice], f_p[idx_ice])
        #t1 = time.time()
        #print ('AdsorptionFrac:', t1-t0)
        #print ('f_p:', f_p)
        #print ('rate_ice[i_T,:]:', rate_ice[i_T,:])
        E_sub = rate_ice[i_T,:]*f_p #[kg.m-2.s-1]
        #E_sub[:] = 0
        #E_sub[idx_ice] = rate_ice[i_T,idx_ice]*f_p[idx_ice] #[kg.m-2.s-1]
        #print ('E_sub:', E_sub)
        #if (i>80000):
        #    print ('np.mean(T[i_T,:]:', np.mean(T[i_T,:]))
        #    print ('np.count_nonzero(dt*E_sub>m):', np.count_nonzero(dt*E_sub>m))
        #    print ('dt*E_sub>m:', dt*E_sub>m)
        #    print ('dt*E_sub[dt*E_sub>m]:', dt*E_sub[dt*E_sub>m])
        #    print ('m[dt*E_sub>m]:', m[dt*E_sub>m])
        E_sub[dt*E_sub>m] = m[dt*E_sub>m]/dt #A facet can't lose more ice than it has
        #if (i>80000):
        #    print ('np.count_nonzero(E_sub<0):', np.count_nonzero(E_sub<0))
        #    print ('E_sub<0:', E_sub<0)
        #    print ('E_sub[E_sub<0]:', E_sub[E_sub<0])
        #    print ('m[dt*E_sub>m]:', m[dt*E_sub>m])
        #idx = dt*E_sub>f*theta_m*mw/Na
        #E_sub[idx] = f[idx]*theta_m*mw/Na/dt
        #t2 = time.time()
        #print ('E_sub:', t2-t1)
        
        #m_h2o[i,:] = m_h2o[i-1,:] + dt*( -E_sub + np.dot(E_sub, F_los.T) )
        #f_theta[i,:] = m_h2o[i,:]*Na/mw/theta_m
        
        m = m + dt*( -E_sub + np.dot(E_sub, F_los.T) )
        #m = m + dt*( -E_sub + np.dot(E_sub, F_los) ) #Trying out without the F_los transpose to see if it caused the mass loss
        #m = m + dt*( -E_sub + np.dot(E_sub, F_adj) ) #Trying with adjusted F_los to see if it caused the mass gain
        #m = m + dt*( -E_sub + np.dot(E_sub, F_adj.T) ) #This one seems to fix mass gain problem. Now to solve the mass loss one.
        
        #m = m + dt*( -E_sub + np.tensordot(E_sub, F_los.T,1) )
        #t3 = time.time()
        #print ('m:', t3-t2)
        #if (i>80000):
        #    print ('m<0', np.count_nonzero(m<0), '\n')
        m[m<0] = 0 #Can't have negative mass
        m[m<1e-12] = 0 #####Trying this out to see if it fixes mass loss
        #t4 = time.time()
        #print ('m=0:', t4-t3)
        f = m*Na/mw/theta_m
        #t5 = time.time()
        #print ('f:', t5-t4)
        #f = f + dt*( -E_sub + np.dot(E_sub, F_los.T) )*Na/mw/theta_m
        #f[f<0] = 0 #Can't have negative mass
        
        #Mass lost to space by each facet [kg]
        #mass that sublimated from each facet in this timestep times the fraction of the sky that each facet sees
        m_s += E_sub*frac_const
        #t6 = time.time()
        #print ('m_s:', t6-t5, '\n')
        
        
        if ( (i == save_count*int(N_steps/num_save)) & (save_count<num_save) ):
            lt[save_count] = t/planet.day*24.0 # local time [hr]
            m_h2o[save_count,:] = m
            f_theta[save_count,:] = f
            m_space[save_count,:] = m_s #Mass lost to space by each facet [kg]
            save_count += 1
        
        t += dt # Increment time
    
    #m_h2o = f_theta*theta_m*mw/Na
    
    return m_h2o, f_theta, lt, m_space

def IsotropicVolatileModel_VarTime(F_los, T, dt_T, dt, f_theta_i, daz, dzen, horizon_zen, intersect_sky, tri_area, \
                                   num_save=int(1e4), planet=planets.Moon, f_dt=1e1):
    #Volatile Hopping Model Inputs:
    #tri_area, F_los (view factors), T (surface temperatures), dt_T (thermal model timestep), 
    #dt (sublimation model timestep),
    #f_theta_i (initial surface density of H2O molecules on each facet as fraction of 1e19 for monolayer)
    
    #Outputs:
    #theta (surface density of H2O molecules on each facet at each timestep), time array
    #instantaneous (at each timestep) vapor pressure within pit/cave??
    Na = 6.02214076e23 #Avagadro Constant [mol-1]
    mw = 0.018 #molar mass of water [kg.mol-1]
    Rg = 8.314 #universal gas constant [J.K-1.mol-1]
    #theta_i: initial surface density of H2O molecules on each facet [m-2]
    theta_m = 1e19 #surface density of an h2o monolayer [molecules.m-2] (Schorghofer and Aharonson, 2014)
    
    t_end = T.shape[0]*dt_T
    print ('T.shape[0]:', T.shape[0])
    num_facet = f_theta_i.size
    print ('num_facet:', num_facet)
    t = 0
    lt = np.zeros([num_save]) #local time
    f_theta = np.zeros([num_save, num_facet]) #number of H2O monolayers on each facet at each timestep
    f_theta[0,:] = f_theta_i
    f_p = np.zeros_like(f_theta_i)
    #E_sub = np.zeros_like(f_theta_i)
    m_h2o = np.zeros([num_save, num_facet]) #[kg.m-2]
    m_h2o[0,:] = f_theta_i*theta_m*mw/Na
    
    ##############################################################
    ###### For calculating the total ice mass loss to space ######
    m_space = np.zeros([num_save, num_facet]) #Mass lost to space by each facet [kg]
    m_s = 0 #Total mass lost to space [kg]
    
    #Calculate solid angle of the sky seen by each facet
    #Solid angle corresponding to each elevation/azimuth pixel
    SA = daz*(np.cos(horizon_zen-(dzen/2))-np.cos(horizon_zen+(dzen/2)))
    SA[0] = daz*(np.cos(horizon_zen[0])-np.cos(horizon_zen[0]+(dzen/2)))
    SA[-1] = daz*(np.cos(horizon_zen[-1]-(dzen/2))-np.cos(horizon_zen[-1]))
    
    frac_sky = np.sum((~intersect_sky)*SA, axis=(1,2))/(2*np.pi)
    
    frac_const = tri_area*dt*frac_sky
    ##############################################################
    ##############################################################
    
    p_sat = satVaporPressureIce(T)
    rate_ice = p_sat*np.sqrt(mw/(2*np.pi*Rg*T)) #[kg.m-2.s-1]
    
    
    #Adjust F_los to account for known innaccuracies in view factor calculation
    F_adj = np.copy(F_los)
    idx = np.sum(F_los[:,:], axis=1)>1
    F_adj[idx,:] = F_adj[idx,:]/np.sum(F_los[idx,:], axis=1)[:,None]
    #F_adj[idx,:] = F_adj[idx,:]/np.sum(F_los[idx,:], axis=1)[None,:]
    
    save_count = 1
    m = f_theta_i*theta_m*mw/Na
    f = f_theta_i
    
    #Array of timesteps. Allow at least f_dt number of calculations before ablation of a monolayer
    dt_var = np.amax(f_theta_i)*theta_m*mw/Na/rate_ice/f_dt #[s] theta_m*mw/Na/rate_ice/f_dt #[s]
    t = np.array([0])
    i_T = 0
    #idx_ice = [f>0]
    #dt = np.amin(dt_var[i_T,idx_ice])
    #t = t.append(t[-1]+dt)
    
    pbar = tqdm(total=num_save) #Progress bar
    while(save_count<num_save):
        idx_ice = [f>0]
        #print (idx_ice)
        #print (idx_ice[0])
        dt = np.amin(dt_var[i_T,idx_ice[0]])
        print ('dt:', dt)
        if (dt>dt_T):
            dt = dt_T
        elif (dt<dt_T/1e3):
            dt = dt_T/1e3
        print ('dt:', dt)
        print ('i_T:', i_T)
        print ('np.amax(T[i_T,:]):', np.amax(T[i_T,:]))
        print ('np.count_nonzero(idx_ice[0]):', np.count_nonzero(idx_ice[0]), '\n')
        
        
        f_p[:] = 0
        f_p[idx_ice] = AdsorptionFrac(f[idx_ice], f_p[idx_ice])
        
        E_sub = rate_ice[i_T,:]*f_p #[kg.m-2.s-1]
        #TODO: check to make sure the facets without ice have E_sub=0, i.e. E_sub[i_T,f<=0]==0
        E_sub[dt*E_sub>m] = m[dt*E_sub>m]/dt #A facet can't lose more ice than it has
        
        m = m + dt*( -E_sub + np.dot(E_sub, F_adj.T) ) #This one seems to fix mass gain problem. Now to solve the mass loss one.
        m[m<0] = 0 #Can't have negative mass
        
        f = m*Na/mw/theta_m
        
        m_s += E_sub*frac_const
        
        t = np.append(t, t[-1]+dt)
        
        if ( (t[-1]>=save_count*t_end/num_save) ):
            lt[save_count] = t[-1]/planet.day*24.0 # local time [hr]
            m_h2o[save_count,:] = m
            f_theta[save_count,:] = f
            m_space[save_count,:] = m_s #Mass lost to space by each facet [kg]
            save_count += 1
            pbar.update(1)
        
        i_T = int(t[-1]/dt_T)
        
    pbar.close()
    
    return m_h2o, f_theta, lt, m_space

def IsotropicVolatileModel_AccSub(F_los, T, dt_T, dt, f_theta_i, daz, dzen, horizon_zen, intersect_sky, tri_area, \
                                  num_save=int(1e4), planet=planets.Moon, f_dm=1e2):
    #Volatile Hopping Model Inputs:
    #tri_area, F_los (view factors), T (surface temperatures), dt_T (thermal model timestep), 
    #dt (sublimation model timestep),
    #f_theta_i (initial surface density of H2O molecules on each facet as fraction of 1e19 for monolayer)
    
    #Outputs:
    #theta (surface density of H2O molecules on each facet at each timestep), time array
    #instantaneous (at each timestep) vapor pressure within pit/cave??
    Na = 6.02214076e23 #Avagadro Constant [mol-1]
    mw = 0.018 #molar mass of water [kg.mol-1]
    Rg = 8.314 #universal gas constant [J.K-1.mol-1]
    #theta_i: initial surface density of H2O molecules on each facet [m-2]
    theta_m = 1e19 #surface density of an h2o monolayer [molecules.m-2] (Schorghofer and Aharonson, 2014)
    
    N_steps = int(T.shape[0]*dt_T/dt)
    print ('N_steps:', N_steps)
    print ('T.shape[0]:', T.shape[0])
    num_facet = f_theta_i.size
    print ('num_facet:', num_facet)
    t = 0
    lt = np.zeros([num_save]) #local time
    f_theta = np.zeros([num_save, num_facet]) #number of H2O molecules on each facet at each timestep
    f_theta[0,:] = f_theta_i
    f_p = np.zeros_like(f_theta_i)
    #E_sub = np.zeros_like(f_theta_i)
    m_h2o = np.zeros([num_save, num_facet]) #[kg.m-2]
    m_h2o[0,:] = f_theta_i*theta_m*mw/Na
    
    ##############################################################
    ###### For calculating the total ice mass loss to space ######
    m_space = np.zeros([num_save, num_facet]) #Mass lost to space by each facet [kg]
    m_s = 0 #Total mass lost to space [kg]
    
    #Calculate solid angle of the sky seen by each facet
    #Solid angle corresponding to each elevation/azimuth pixel
    SA = daz*(np.cos(horizon_zen-(dzen/2))-np.cos(horizon_zen+(dzen/2)))
    SA[0] = daz*(np.cos(horizon_zen[0])-np.cos(horizon_zen[0]+(dzen/2)))
    SA[-1] = daz*(np.cos(horizon_zen[-1]-(dzen/2))-np.cos(horizon_zen[-1]))
    
    frac_sky = np.sum((~intersect_sky)*SA, axis=(1,2))/(2*np.pi)
    
    frac_const = tri_area*dt*frac_sky
    ##############################################################
    ##############################################################
    
    p_sat = satVaporPressureIce(T)
    rate_ice = p_sat*np.sqrt(mw/(2*np.pi*Rg*T)) #[kg.m-2.s-1]
    
    
    #Adjust F_los to account for known innaccuracies in view factor calculation
    F_adj = np.copy(F_los)
    idx = np.sum(F_los[:,:], axis=1)>1
    F_adj[idx,:] = F_adj[idx,:]/np.sum(F_los[idx,:], axis=1)[:,None]
    #F_adj[idx,:] = F_adj[idx,:]/np.sum(F_los[idx,:], axis=1)[None,:]
    
    save_count = 1
    m = f_theta_i*theta_m*mw/Na
    f = f_theta_i
    
    lim_dm = theta_m*mw/Na/f_dm
    print ('lim_dm:', lim_dm)
    
    for i in tqdm(range(1, N_steps)):
        i_T = int(i*dt/dt_T)
        idx_ice = [f>0]
        f_p[:] = 0
        f_p[idx_ice] = AdsorptionFrac(f[idx_ice], f_p[idx_ice])
        
        E_sub = rate_ice[i_T,:]*f_p #[kg.m-2.s-1]
        
        #Trying to make the sublimation calculation as accurate as possible
        #dm_old = dt*( -E_sub + np.dot(E_sub, F_adj.T) ) #This one seems to fix mass gain problem.
        dm_old = dt*( -E_sub + np.dot(E_sub, F_los.T) )
        m_old = m + dm_old
        E_sub[m_old<0] = E_sub[m_old<0] + m_old[m_old<0]/dt
        #dm_new = dt*( -E_sub + np.dot(E_sub, F_adj.T) ) #This one seems to fix mass gain problem.
        dm_new = dt*( -E_sub + np.dot(E_sub, F_los.T) )
        #print ('np.amax(np.abs(dm_old-dm_new)):', np.amax(np.abs(dm_old-dm_new)))
        
        while(np.amax(np.abs(dm_old-dm_new))>lim_dm):
            dm_old = dm_new
            m_old = m + dm_old
            E_sub[m_old<0] = E_sub[m_old<0] + m_old[m_old<0]/dt
            #dm_new = dt*( -E_sub + np.dot(E_sub, F_adj.T) ) #This one seems to fix mass gain problem.
            dm_new = dt*( -E_sub + np.dot(E_sub, F_los.T) )
            #print ('np.amax(np.abs(dm_old-dm_new)):', np.amax(np.abs(dm_old-dm_new)))
        #print ('\n')
        
        #Do I need one of these?:
        #E_sub[dt*E_sub>m] = m[dt*E_sub>m]/dt #A facet can't lose more ice than it has
        
        m = m + dm_new
        m[m<0] = 0 #Can't have negative mass. Might create some mass conservation issue here, so check this.
        
        f = m*Na/mw/theta_m
        
        #Mass lost to space by each facet [kg]
        #mass that sublimated from each facet in this timestep times the fraction of the sky that each facet sees
        m_s += E_sub*frac_const
        
        
        if ( (i == save_count*int(N_steps/num_save)) & (save_count<num_save) ):
            lt[save_count] = t/planet.day*24.0 # local time [hr]
            m_h2o[save_count,:] = m
            f_theta[save_count,:] = f
            m_space[save_count,:] = m_s #Mass lost to space by each facet [kg]
            save_count += 1
        
        t += dt # Increment time
    
    
    return m_h2o, f_theta, lt, m_space

def satVaporPressureIce(T):
    #(Buck 1981) originally from (Wexler 1977)
    x = np.exp( -5865.3696/T + 22.241033 + 0.013749042*T - \
                      0.34031775e-4*T**2 + 0.26967687e-7*T**3 + 0.6918651*np.log(T) )
    return x

def AdsorptionFrac(v, f_p):
    #Adjustment to sublimation rate (saturation vapor pressure) for micro-layers of H2O adsorbed onto regolith
    #From Schorghofer and Aharonson 2014 (originally from Cadenhead and Stetter 1974)
    theta_m = 1e19 #number of water molecules per area for H2O monolayer [m-2]
    
    a = 0.402
    v0 = 2.45
    b = -0.06 #b (and c) chosen so function is continuous at v0
    c = (1/(v0**2))*(1 - np.exp(-a*v0)) - b*v0
    
    #v = f_theta #theta/theta_m
    
    #if(v<=v0):
    #    f_p = b*v**3 + c*v**2
    #else:
    #    f_p = 1 - np.exp(-a*v)
        
    #f_p = np.zeros_like(v)
    f_p[v<=v0] = b*v[v<=v0]**3 + c*v[v<=v0]**2
    f_p[v>v0] = 1 - np.exp(-a*v[v>v0])
    
    f_p[v>50] = 1 #Make sure function doesn't blow up
        
    return f_p