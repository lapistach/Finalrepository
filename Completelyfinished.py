import numpy as np
from matplotlib import cm
from matplotlib.collections import EllipseCollection
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pickle

            
class CollisionEvent:
   
    def __init__(self, Type = 'wall or other', dt = np.inf, mono_1 = 0, mono_2 = 0, w_dir = 1):
       
        self.Type = Type
        self.dt = dt
        self.mono_1 = mono_1
        self.mono_2 = mono_2  # only importent for interparticle collisions
        self.w_dir = w_dir # only important for wall collisions
        
        
    def __str__(self):
        if self.Type == 'wall':
            return "Event type: {:s}, dt: {:.8f}, p1 = {:d}, dim = {:d}".format(self.Type, self.dt, self.mono_1, self.w_dir)
        else:
            return "Event type: {:s}, dt: {:.8f}, p1 = {:d}, p2 = {:d}".format(self.Type, self.dt, self.mono_1, self.mono_2)

class Monomers:
    
    def __init__(self, NumberOfMonomers = 4, L_xMin = 0, L_xMax = 1, L_yMin = 0, L_yMax = 1, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT = 1, FilePath = './Configuration.p'):
        try:
            self.__dict__ = pickle.load( open( FilePath, "rb" ) )
            print("IMPORTANT! System is initialized from file %s, i.e. other input parameters of __init__ are ignored!" % FilePath)
        except:
            assert ( NumberOfMonomers > 0 )
            assert ( (L_xMin < L_xMax) and (L_yMin < L_yMax) )
            self.NM = NumberOfMonomers
            self.DIM = 2 #dimension of system
            self.BoxLimMin = np.array([ L_xMin, L_yMin])
            self.BoxLimMax = np.array([ L_xMax, L_yMax])
            self.mass = -1*np.ones( self.NM ) # Masses, negative mass means not initialized
            self.rad = -1*np.ones( self.NM ) # Radiai, negative radiai means not initialized
            self.pos = np.empty( (self.NM, self.DIM) ) # Positions, not initalized but desired shape
            self.vel = np.empty( (self.NM, self.DIM) ) # Velocities, not initalized but desired shape
            self.mono_pairs = np.array( [ (k,l) for k in range(self.NM) for l in range( k+1,self.NM ) ] )
            self.next_wall_coll = CollisionEvent( 'wall', np.inf, 0, 0, 0)
            self.next_mono_coll = CollisionEvent( 'mono', np.inf, 0, 0, 0)
        
            self.assignRadiaiMassesVelocities(NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
            self.assignRandomMonoPos( )
            
    def save_configuration(self, FilePath = 'MonomerConfiguration.p'):
        '''Saves configuration. Callable at any time during simulation.'''
    
    def assignRadiaiMassesVelocities(self, NumberMono_per_kind = np.array([4]), Radiai_per_kind = 0.5*np.ones(1), Densities_per_kind = np.ones(1), k_BT = 1 ):
      
        assert( sum(NumberMono_per_kind) == self.NM )
        assert( isinstance(Radiai_per_kind,np.ndarray) and (Radiai_per_kind.ndim == 1) )
        assert( (Radiai_per_kind.shape == NumberMono_per_kind.shape) and (Radiai_per_kind.shape == Densities_per_kind.shape)) 

        total=0
        for i, number in enumerate(NumberMono_per_kind): #we are filling up radii and masses, the infinite density will give an infinite mass
           self.rad [total:total+number]=Radiai_per_kind[i]
           self.mass[total:total+number]=Densities_per_kind[i]*(np.pi*(Radiai_per_kind[i]))
           total+=number
        
        assert( k_BT > 0 )
        for i in range(self.NM):
            rand=np.random.random()*2*np.pi #we take random angles
            self.vel[i]=[np.cos(rand),np.sin(rand)] 
            if self.mass[i] >1000:
               self.vel[i] = [0.000001,0.000001]  #to avoid having a division by zero later on we give the seed particle a very very slow speed but not zero
            else:
                vitesse = np.sqrt( k_BT *2 / self.mass[i]) #formula of the speed 
                self.vel[i] *= vitesse        #random angles gives random speed directions to the particles
            
    
    def assignRandomMonoPos(self, start_index = 0 ):
        
        assert ( min(self.rad) > 0 ) #otherwise not initialized
        new_mono, infiniteLoopTest = start_index, 0
        if self.mass[new_mono]>1000: #if it is the seed particle then put it in the middle
           self.pos[new_mono,:]= [(L_xMax-L_xMin)/2 ,(L_yMax-L_yMin)/2 ] 
        else: #otherwise assign a random position
            self.pos[new_mono,:] = np.random.rand(1,2)*[ L_xMax - self.rad[new_mono], L_yMax - self.rad[new_mono]] + [self.rad[new_mono], self.rad[new_mono]]
        new_mono +=1 #initialize at 1 to compare with "old_mono"
        while new_mono < self.NM and infiniteLoopTest < 10**4:
            if self.mass[new_mono]>1000:
               self.pos[new_mono,:]= [(L_xMax-L_xMin)/2 ,(L_yMax-L_yMin)/2 ] 
            else:
                self.pos[new_mono,:] = np.random.rand(1,2)*[ L_xMax - self.rad[new_mono], L_yMax - self.rad[new_mono]] + [self.rad[new_mono], self.rad[new_mono]]
            NoOverlap = True
            old_mono = 0
            while old_mono < new_mono and NoOverlap:
                  dist = np.sqrt((self.pos[new_mono,0]-self.pos[old_mono,0])**2+(self.pos[new_mono,1]-self.pos[old_mono,1])**2)
                  doubleR = self.rad[new_mono]+self.rad[old_mono]
                  if dist < doubleR: #the sum of the particles radii must be superior to the distance that separes their center
                     NoOverlap = False
                  else:
                     old_mono += 1
            if NoOverlap:
                new_mono += 1
                infiniteLoopTest = 0
            else:
                infiniteLoopTest += 1
        
    
    def __str__(self, index = 'all'):
        if index == 'all':
            return "\nMonomers with:\nposition = " + str(self.pos) + "\nvelocity = " + str(self.vel) + "\nradius = " + str(self.rad) + "\nmass = " + str(self.mass)
        else:
            return "\nMonomer at index = " + str(index) + " with:\nposition = " + str(self.pos[index]) + "\nvelocity = " + str(self.vel[index]) + "\nradius = " + str(self.rad[index]) + "\nmass = " + str(self.mass[index])
        
    def Wall_time(self):
        
        coll_condition = np.where( self.vel > 0, self.BoxLimMax-self.rad[:,np.newaxis], self.BoxLimMin+self.rad[:,np.newaxis]) #the particle touches the wall (radius touching the wall)
        dt_List = ( coll_condition - self.pos) / self.vel #time before impact following the position and the wall
        MinTimeIndex = np.argmin(  dt_List ) #finding the smallest time to know which is the first particle to hit the wall
        collision_disk = MinTimeIndex // 2 # index of new position after collision
        wall_direction = MinTimeIndex % 2 #index of new direction after collision
        
        self.next_wall_coll.dt = dt_List[collision_disk][wall_direction] # time before impact
        self.next_wall_coll.mono_1 = collision_disk 
        self.next_wall_coll.w_dir = wall_direction 
        
        print(self.next_wall_coll)
        
        
    def Mono_pair_time(self): 
        
        mono_i = self.mono_pairs[:,0] 
        mono_j = self.mono_pairs[:,1]
        
        d_vx=self.vel[mono_i,0]-self.vel[mono_j,0]
        d_vy=self.vel[mono_i,1]-self.vel[mono_j,1]
        d_x0=self.pos[mono_i,0]-self.pos[mono_j,0]
        d_y0=self.pos[mono_i,1]-self.pos[mono_j,1]

        a=d_vx**2+d_vy**2
        b=2*(d_vx*d_x0+d_vy*d_y0)
        c=d_x0**2+d_y0**2-(self.rad[mono_i]+self.rad[mono_j])**2
       
        delta = b**2-4*a*c                                            #all of this calculus has be done during class, it can be looked up in the collaboratory
        
        condition = np.empty((1,len(b)))
        for i in range(len(delta)):   #our condition for the np.where 
            condition[0,i] = delta[i] >=0 and b[i] < 0 
        
        deltat = np.where(condition ,(-b-np.sqrt(delta))/(2*a),np.array([np.inf]*len(condition))) #a simple physical analysis shows that we need b<0

        index_min = np.argmin(deltat[0])
        
        self.next_mono_coll.dt = deltat[0][index_min]
        self.next_mono_coll.mono_1 = self.mono_pairs[index_min,0]
        self.next_mono_coll.mono_2 = self.mono_pairs[index_min,1]

        print(self.next_mono_coll)
        
        
    def compute_next_event(self):
        
        self.Mono_pair_time()
        self.Wall_time()
        if self.next_wall_coll.dt < self.next_mono_coll.dt :
            return self.next_wall_coll
        else :
            return self.next_mono_coll
            
        
            
    def compute_new_velocities(self, next_event):

        if next_event.Type == 'wall' :
                self.vel[next_event.mono_1,next_event.w_dir]= self.vel[next_event.mono_1,next_event.w_dir]*-1
                
        else:

            
            mono_1 = next_event.mono_1
            mono_2 = next_event.mono_2
           
            if self.NM>2 : #as long as we didn't reach the last particle
                if (self.mass[mono_1]>1000) or (self.mass[mono_2]>1000) : #every particle that hits the fractal must stay stuck to it
                    self.vel[mono_1]=[0.000001,0.000001] 
                    self.vel[mono_2]=[0.000001,0.000001]
                    self.mass[mono_1]=10000
                    self.mass[mono_2]=10000
                    self.NM -= 1
             
                else : #if they just hit each other outside the seed particle then it's an unelastic collison
                     diff = self.pos[mono_2] - self.pos[mono_1]
                     diff=diff/np.linalg.norm(diff)
                     m1=self.mass[mono_1]
                     m2=self.mass[mono_2]
                     Dv=self.vel[mono_1]-self.vel[mono_2]
                     self.vel[mono_1]=self.vel[mono_1] - (2*m2)/(m1+m2)*np.inner(diff,Dv)*diff
                     self.vel[mono_2]=self.vel[mono_2] + (2*m1)/(m1+m2)*np.inner(diff,Dv)*diff

            else : #once we arrive to the last particle, it all happens like a simple collision, without sticking to the fractal
                 diff = self.pos[mono_2] - self.pos[mono_1]
                 diff=diff/np.linalg.norm(diff)
                 m1=self.mass[mono_1]
                 m2=self.mass[mono_2]
                 Dv=self.vel[mono_1]-self.vel[mono_2]
                 self.vel[mono_1]=self.vel[mono_1] - (2*m2)/(m1+m2)*np.inner(diff,Dv)*diff
                 self.vel[mono_2]=self.vel[mono_2] + (2*m1)/(m1+m2)*np.inner(diff,Dv)*diff


    def snapshot(self, FileName = './snapshot.png', Title = '$t = $?'):
        
        fig, ax = plt.subplots( dpi=300 )
        L_xMin, L_xMax = self.BoxLimMin[0], self.BoxLimMax[0]
        L_yMin, L_yMax = self.BoxLimMin[1], self.BoxLimMax[1]
        BorderGap = 0.1*(L_xMax - L_xMin)
        ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
        ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)

        #--->plot hard walls (rectangle)
        rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
        ax.add_patch(rect)
        ax.set_aspect('equal')
        ax.set_xlabel('$x$ position')
        ax.set_ylabel('$y$ position')
        
        #--->plot monomer positions as circles
        MonomerColors = np.linspace( 0.2, 0.95, self.NM)
        Width, Hight, Angle = 2*self.rad, 2*self.rad, np.zeros( self.NM )
        collection = EllipseCollection( Width, Hight, Angle, units='x', offsets=self.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
        collection.set_array(MonomerColors)
        collection.set_clim(0, 1) # <--- we set the limit for the color code
        ax.add_collection(collection)

        #--->plot velocities as arrows
        ax.quiver( self.pos[:,0], self.pos[:,1], self.vel[:,0], self.vel[:,1] , units = 'dots', scale_units = 'dots')
        
        plt.title(Title)
        plt.savefig(FileName)
        plt.close()


import os
from matplotlib.animation import FuncAnimation

np.random.seed(999)
Snapshot_output_dir = './SnapshotsMonomers'
if not os.path.exists(Snapshot_output_dir): os.makedirs(Snapshot_output_dir)

Conf_output_dir = './ConfsMonomers'
Path_ToConfiguration = Conf_output_dir+'/FinalMonomerConf.p'
if False: #os.path.isfile( Path_ToConfiguration ):
    '''Initialize system from existing file'''
    mols = Monomers( FilePath = Path_ToConfiguration )
else:
    '''Initialize system with following parameters'''
    # create directory if it does not exist
    if not os.path.exists(Conf_output_dir): os.makedirs(Conf_output_dir)
    #define parameters
    NumberOfMonomers = 10
    L_xMin, L_xMax = 0, 100
    L_yMin, L_yMax = 0, 50
    NumberMono_per_kind = np.array([NumberOfMonomers-1,1]) #here we put the seed particle in the last position
    Radiai_per_kind = np.array([ 1.5, 1.5 ]) #we give the same size to the particles and the seed particle, but we could change it!
    Densities_per_kind = np.append(np.ones(len(NumberMono_per_kind)-1),np.inf) #we give the seed particle an infinite density for it to by infinitely heavy
    k_BT = 100000
    # call constructor, which should initialize the configuration
    mols = Monomers(NumberOfMonomers, L_xMin, L_xMax, L_yMin, L_yMax, NumberMono_per_kind, Radiai_per_kind, Densities_per_kind, k_BT )
    
mols.snapshot( FileName = Snapshot_output_dir+'/InitialConf.png', Title = '$t = 0$')

t = 0.0
dt = 0.02
NumberOfFrames = 1500

next_event = mols.compute_next_event()
def MolecularDynamicsLoop( frame ):
    
    global t, mols, next_event
    
    next_time_vel_change = t + next_event.dt
    future_time_next_frame = t + dt
    while next_time_vel_change < future_time_next_frame :
        mols.pos += mols.vel * next_event.dt
        t += next_event.dt
        mols.compute_new_velocities(next_event)
        next_event=mols.compute_next_event()
        next_time_vel_change=t+next_event.dt
            
    timeremaining = future_time_next_frame - t
    mols.pos += timeremaining*mols.vel
    t += timeremaining
    next_event.dt -= timeremaining
    
    if mols.NM > 2:
        plt.title( '$t = %.4f$, remaining frames = %d, Planetesimal is forming! ' % (t, NumberOfFrames-(frame+1)) )
    else:
        plt.title( '$t = %.4f$, remaining frames = %d, Saffronov number higher than 1 ' % (t, NumberOfFrames-(frame+1)) )
    collection.set_offsets( mols.pos )
    return collection


fig, ax = plt.subplots()
L_xMin, L_yMin = mols.BoxLimMin #not defined if initalized by file
L_xMax, L_yMax = mols.BoxLimMax #not defined if initalized by file
BorderGap = 0.1*(L_xMax - L_xMin)
ax.set_xlim(L_xMin-BorderGap, L_xMax+BorderGap)
ax.set_ylim(L_yMin-BorderGap, L_yMax+BorderGap)
ax.set_aspect('equal')

# confining hard walls plotted as dashed lines
rect = mpatches.Rectangle((L_xMin,L_yMin), L_xMax-L_xMin, L_yMax-L_yMin, linestyle='dashed', ec='gray', fc='None')
ax.add_patch(rect)


# plotting all monomers as solid circles of individual color
MonomerColors = np.linspace(0.2,0.95,mols.NM)
Width, Hight, Angle = 2*mols.rad, 2*mols.rad, np.zeros(mols.NM)
collection = EllipseCollection(Width, Hight, Angle, units='x', offsets=mols.pos,
                       transOffset=ax.transData, cmap='nipy_spectral', edgecolor = 'k')
collection.set_array(MonomerColors)
collection.set_clim(0, 1) # <--- we set the limit for the color code
ax.add_collection(collection)

'''Create the animation, i.e. looping NumberOfFrames over the update function'''
Delay_in_ms = 33.3 #delay between images/frames for plt.show()
ani = FuncAnimation(fig, MolecularDynamicsLoop, frames=NumberOfFrames, interval=Delay_in_ms, blit=False, repeat=False)
plt.show()

'''Save the final configuration and make a snapshot.'''
#write the function to save the final configuration
mols.save_configuration(Path_ToConfiguration)
mols.snapshot( FileName = Snapshot_output_dir + '/FinalConf.png', Title = '$t = %.4f$' % t)
