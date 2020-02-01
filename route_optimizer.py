# TODO
# - 1: Node figs, Area figs, Charts
# - 2: Validation
# - 3: getLiftProb -method
# - 5: Data saving

# IMPORT
import numpy
import math
import cProfile
#import multiprocessing

import matplotlib.pyplot as plt
#from math import asin, acos, sqrt, sin, cos, pi

# DEFINE GLOBAL CONSTANTS
kts_to_kmh = numpy.dot(numpy.dot(6000,12),2.54) / 100 / 1000
kts_to_ms = numpy.dot(kts_to_kmh,1000) / 3600
nm_to_km = 1.852
ft_to_m = 0.3048

# DEFINE CLASSES
class LiftMap:
    def __init__(self, matrix, lift_strenght):
        self.liftStrength = lift_strenght
        self.probMatrix = matrix
        (self.ySize, self.xSize)=matrix.shape
        self.latMin = 6669870 #south limit
        self.latMax = 6957974 #north limit
        self.lonMin = 221705 # west limit
        self.lonMax = 494046 # east limit
        self.xGridSize=(self.lonMax-self.lonMin)/self.xSize
        self.yGridSize=(self.latMax-self.latMin)/self.ySize

    #TODO
    def getLiftProb(self, start_point=None,end_point=None):
        return numpy.array([0.25, 0.75]),numpy.array([3.0, 0.0])

class Chart:
    #TODO
    def __init__(self,lift_map ,height_band,circle_intervals,n_circles, polar, target):
        
        self.polar = polar
        self.liftMap = lift_map
        
        self.heightBand = height_band
                
        self.target = target
        self.nCircles=n_circles
        self.circleIntervals = circle_intervals / nm_to_km

        self.hGridSize = 500
        self.hStep = height_band/ft_to_m/self.hGridSize
        self.hGrid=(numpy.arange(0,height_band/ft_to_m,self.hStep)).T.reshape([-1,1])
        
        self.liftGridSize = 2

        #Computatation parameters
        # Porpoise factor -- cruse strength/climb
        self.porp = 0.0
        # Maximum points for landout 
        self.alpha = 0.65
        # Min altitude for thermaling in ft
        self.minclimb = 500
        # How many miles back for first step?
        self.firstStep = 1
        # Sigma altidude in feet per mile = smoothing
        self.mix = 50
        # Course length in NM
        self.xmax = 150 #needed for landouts
        self.routeDist = 150

        # used to scale up or down wining speed assumption
        self.hCap=0.96
    
        vdotwin = numpy.sqrt(2/self.polar.a*(self.liftMap.liftStrength/kts_to_ms)+self.polar.minSinkSpeed**2)
        sdotwin = self.polar.getSinkRate(vdotwin)
        vwin=(self.liftMap.liftStrength/kts_to_ms-self.polar.minSink)/(self.liftMap.liftStrength/kts_to_ms-self.polar.minSink+sdotwin)*vdotwin                            

        # Winner's handicapped speed in kts 
        self.vWin=vwin / self.hCap
        self.tWin = self.xmax / self.vWin                        #  Winner's time -- used to evaluate landouts
 
        self.dataStructure = DataStructure(self)
        self.calculateChartTimer()

    def calculateChart(self):
        for circle in self.dataStructure.circles:
            pool = multiprocessing.Pool(multiprocessing.cpu_count())
            output=pool.map(circle.calculateNode, range(circle.nNodes))
            for i in range(circle.nNodes):
                circle.nodes[i]=output[i]

            pool.close()
            pool.join()
            
            print(circle.index + 1, "circle(s) completed!")

    def calculateChartTimer(self):
        for circle in self.dataStructure.circles:
            for i in range(circle.nNodes):
                circle.nodes[i].calculateNode()         
            print(circle.index + 1, "circle(s) completed!")
    

class DataStructure:
    def __init__(self,chart):
        self.circles = [Circle(i,chart) for i in range(n_circles)] #list

    def plotCurve(self, circle_index, node_index):
        node = self.circles[circle_index].nodes[node_index]
        hgrid = node.chart.hGrid
        
        plt.subplot(2,3,1)
        data = node.mcCready
        plt.plot(data,hgrid)

        plt.subplot(2,3,2)
        data = node.optimalDirection
        plt.plot(data,hgrid)

        plt.subplot(2,3,3)
        data = node.expectedPoints
        plt.plot(data,hgrid)

        plt.subplot(2,3,4)
        data = node.expectedTimeToGo
        plt.plot(data,hgrid)

        plt.subplot(2,3,5)
        data = node.expectedOutlandPoints
        plt.plot(data,hgrid)

        plt.subplot(2,3,6)
        data = node.expectedOutlandPoints
        plt.plot(data,hgrid)

        plt.show()
        #fig, ax = plt.subplot(3,2,1)
        #ax.plot(lamfin_discus[hgrid[hgrid >= minclimb],numpy.array([8,16,54,150])] / 2,hgrid[hgrid >= minclimb] / 3)
        #ax.plot(array ,hgrid)
        #ax.legend(['10 nm','20 nm','100 nm','150 nm'])
    # newmc.m:695
        #ax.plot(mcvals / 2,((g2(numpy.array([0]),mcvals)*numpy.array([10]))*6000) / 3)
        #ax.grid(True)
        #set(gca,'Ytick',numpy.concatenate([0,500,1000,1500,2000]))
        #ax.set_xlabel('MacCready (kts)')
        #ax.set_ylabel('Altitude (ft)')
        #ax.set_ylabel(numpy.concatenate(['H',char(246),'he (m)']),'Interpreter','none')
        #ax.set_xlim(0,5)
        #ax.set_ylim(0,6000)
    
    def plotArea(self, altitude):
        # Define area
        n_radials = 20
        n_circles = len(self.circles)

        #Interpolate data
        data = [[0 for i in range(n_radials)] for j in range(n_circles)]
        for i in range(n_circles):
            for j in range(n_radials):
                radialFromTarget =  1.0 * j / self.circles[i].nNodes * 2 * math.pi
                data[i,j] =  InterpolationNode(self.circles[i],self.cirles[0].nodes[0].chart, radialFromTarget)
                data[i,j].calculateNode

        #Draw figure
        fig, ax = plt.subplot(2,3,1)

        data = numpy.array([n_circles, n_radials])
        InterpolationNode(circle,chart, radialFromTarget)
        plt.subplot(2,3,1)

        plt.matshow(data)
        # 

    def plotChart(self):
        data = [] #x, y, dir x, dir y
        for alt in altitudes:
            for i in circles:
                for j in nodes:
                    data.append()

        plt.barbs


class Circle:
    def __init__(self,index, chart):
        self.index = index
        self.distanceFromTarget = chart.firstStep * nm_to_km + (index) * (chart.circleIntervals)
     
        if index==0: 
            self.nNodes = 1
            self.nodes = [FinishNode(self,chart)]
        else:
            self.nNodes = int(2 * math.pi * self.distanceFromTarget / chart.circleIntervals)
            self.nodes = [EnRouteNode(i,self,chart, i / self.nNodes * 2 * math.pi) for i in range(self.nNodes)] 

    def calculateNode(self,i):
            self.nodes[i].calculateNode()
            return self.nodes[i]

class Node:
    def __init__(self,circle,chart):
        self.parentCircle = circle
        self.chart = chart     

        self.distanceFromTarget = circle.distanceFromTarget
        self.expectedPoints = numpy.zeros([chart.hGridSize]) #vector
        self.expectedTimeToGo = numpy.zeros([chart.hGridSize]) #vector
        self.expectedFinishProb = numpy.zeros([chart.hGridSize])
        self.expectedOutlandPoints = numpy.zeros([chart.hGridSize])
        self.wtv = - numpy.ones([chart.hGridSize]) / chart.tWin # wt (h,x) if no landout
        self.whv = numpy.zeros([chart.hGridSize]) # wh (h,x)
        self.mcCready = numpy.zeros([chart.hGridSize]) #vector

        self.optimalDirection = numpy.zeros([chart.hGridSize]) #vector

    #old name: mixer
    def mixer(self,y=None,x=None,mix=None):
        # smooth functions across altitude according to normal distribution -- 
        # equivalent to adding random altitude per mile */
        if mix == 0:
            yn=numpy.copy(y)
        else:
            yn=numpy.zeros([y.shape[0]])
            delx=x[2] - x[1]
            maxind=3*mix
            indxs=(numpy.arange(-maxind,maxind+1,1)).T
            wts=numpy.exp(- 1 / 2*((indxs*delx) ** 2) / (mix ** 2))
            i=0
            while i < x.shape[0]:
                test_cond=numpy.stack(((i + indxs >= 0) , (i + indxs < x.shape[0])), axis=0)
                test=numpy.all(test_cond,axis=0)
                #wl = selif(wts,test);
                wl=wts[test]
                wl=wl / numpy.sum(wl)
                il=indxs[test]
                yn[i]=numpy.sum(wl*y[i + il.astype(int)].T)
                i=i + 1
        return yn

    #old name: interp
    def interp(self,x=None,xgrid=None,ygrid=None,indxl=-1):
        if numpy.amin(xgrid[numpy.arange(1,xgrid.shape[0]-1)] - xgrid[numpy.arange(0,xgrid.shape[0] - 2)]) < 0:
            print('interp: error. xgrid must be organized from low to high')
        T=xgrid.shape[0]-1
        if x < xgrid[0]:
            y=ygrid[0] - ((x - xgrid[0])*(ygrid[1] - ygrid[0])) / (xgrid[1] - xgrid[0])
        else:
            if x >= xgrid[-1]:
                y=ygrid[T]+ ((x - xgrid[T])*(ygrid[T] - ygrid[T - 1])) / (xgrid[T] - xgrid[T - 1])
            else:
                indxv=(numpy.arange(xgrid.shape[0]))
                dif=x - xgrid
                low=dif >= 0
                indxl=numpy.argmax(indxv*low.T)
                indxh=indxl+1 
                xl=xgrid[indxl]
                yl=ygrid[indxl]
                xh=xgrid[indxh]
                yh=ygrid[indxh]
                y=((yh - yl) / (xh - xl))*(x - xl) + yl
        return y

    def interp2(self,x=None,xgrid=None,ygrid=None,indxl=-1):
        #if numpy.amin(xgrid[numpy.arange(1,xgrid.shape[0]-1)] - xgrid[numpy.arange(0,xgrid.shape[0] - 2)]) < 0:
        #    print('interp: error. xgrid must be organized from low to high')
        T = xgrid.shape[0]-1
        if x < xgrid[0]:
            y = ygrid[0] - ((x - xgrid[0])*(ygrid[1] - ygrid[0])) / (xgrid[1] - xgrid[0])
        else:
            if x >= xgrid[-1]:
                y = ygrid[T]+ ((x - xgrid[T])*(ygrid[T] - ygrid[T - 1])) / (xgrid[T] - xgrid[T - 1])
            else:
                #indxv = (numpy.arange(xgrid.shape[0]))
                if self.indxl == None:
                    low = (x >= xgrid)
                    self.indxl = numpy.argmin(low) - 1
                
                indxh = self.indxl+1 
                xl = xgrid[self.indxl]
                yl = ygrid[self.indxl]
                xh = xgrid[indxh]
                yh = ygrid[indxh]
                y=((yh - yl) / (xh - xl))*(x - xl) + yl
        return y
class FinishNode(Node):
    def __init__(self,circle,chart):
        super().__init__(circle, chart)
    
    def calculateNode(self):
    # -------------------------------- */
        # Find values one mile out         */
        # simply fly home, no lift or sink */
        # -------------------------------- */
        # Initialize grids */
        w=numpy.zeros([self.chart.hGridSize,1])           # Speed at finish
        wh=numpy.zeros([self.chart.hGridSize,1])          # wh (h,x)
        wt=- numpy.ones([self.chart.hGridSize,1]) / self.chart.tWin  # wt (h,x) if no landout
        wt[0]=0                                 # at zero, you've landed out
        w[0]=self.chart.alpha*(self.chart.xmax - self.chart.firstStep) / self.chart.xmax      # landout rules, assuming winning speedtonow
        wh[0]=numpy.NaN                         # Code for not known yet                       # Code for not known yet
        hi = 1
        while hi < self.chart.hGridSize:
            h = self.chart.hGrid[hi,0]
            v = self.chart.polar.getSpeed(0, 6000*self.chart.firstStep / h)          # speed to just exhaust altitude in this lift
        
            if v >= self.chart.polar.minSinkSpeed:                        # Can make it home
        
                w[hi] = self.chart.xmax / (self.chart.tWin*(self.chart.xmax - self.chart.firstStep) / self.chart.xmax + self.chart.firstStep / v) / self.chart.vWin    # Speed at finish, assuming you've flown at winning speed so far
                self.whv[hi]= 1 / self.chart.tWin / (self.chart.polar.getMcCready(v))
                self.expectedTimeToGo[hi] = self.chart.firstStep / v * 3600
                self.expectedFinishProb[hi] = 1
            else:
                self.expectedOutlandPoints[hi]=(self.chart.routeDist - self.chart.firstStep + h/6000*self.chart.polar.ldmax) / self.chart.routeDist * self.chart.alpha
                self.expectedFinishProb[hi]=0

                if v == - 1:                # Can't make it home
                    self.wtv[hi]=0
                    w[hi] = numpy.nan
                    self.whv[hi] = numpy.nan
        
                else:
                    print('error: speed {:.0f} reported to get home'.format(v))
            hi = hi + 1
        
        
        # now fill in wh for landout h values by linearly interpolating w */
        i = 1
        while numpy.isnan(w[i]):
            i = i + 1
        # Now i is the smallest index of a valid height
        
        while i <= self.chart.hGridSize:
            testwh = 6000*(w[i] - w[0]) / (self.chart.hGrid[i,0] - self.chart.hGrid[0,0])   # Guess at wh in landout region
            if testwh > wh[i]:                                  # n't want a huge peak in wh
                break                                           # (infinite value of an inch)
            else:                                               # hence use linear w until we reach a 
                i = i + 1                                         # reasonable wh from the fly home rule
        
        self.whv[0:i] = testwh*numpy.ones([i])
        w[0:(i )] = w[0] + ((self.chart.hGrid[0:(i ),:] - self.chart.hGrid[0,:]) / (self.chart.hGrid[i,:] - self.chart.hGrid[0,:]))*(w[i] - w[0])
        
        # take average over lift values */
        #whv = numpy.copy(wh) TODO: remove
        #wtv = numpy.copy(wt) TODO: remove
        self.mcCready = numpy.nan_to_num( - self.wtv / self.whv )
        
        # mix up h 
        self.whv=self.mixer(self.whv,self.chart.hGrid,self.chart.mix)						# adds subtracts random amount to h -- smooths wh, wt over h
        self.wtv=self.mixer(self.wtv,self.chart.hGrid,self.chart.mix)
        #lamfin=lamv*numpy.ones([1,int(fstp)]) TODO: remove

        # we're done! we have wh and wt one mile out. */

class InterpolationNode(Node):
    def __init__(self,circle,chart, radialFromTarget):
        super().__init__(circle, chart)
        self.radialFromTarget = radialFromTarget

    def calculateNode(self):
        step = 2 * math.pi / self.parentCircle.nNodes
        position = self.radialFromTarget / step
        up_index = int(math.ceil(position))
        if up_index == self.parentCircle.nNodes:
            up_index = 0     
        down_index = int(math.floor(position))
        weight = (position - math.floor(position)) / step
         
        #TODO handle node creation when radial outside of 0,2pi
        node_index = [down_index, up_index]
        weights = [1 - weight, weight]
       
        for i in range(2):
            self.wtv += self.parentCircle.nodes[node_index[i]].wtv * weights[i]
            self.whv += self.parentCircle.nodes[node_index[i]].whv * weights[i]
            self.mcCready += self.parentCircle.nodes[node_index[i]].mcCready * weights[i]
            self.expectedPoints += self.parentCircle.nodes[node_index[i]].expectedPoints * weights[i]
            self.expectedTimeToGo += self.parentCircle.nodes[node_index[i]].expectedTimeToGo * weights[i]
            self.expectedFinishProb += self.parentCircle.nodes[node_index[i]].expectedFinishProb * weights[i]
            self.expectedOutlandPoints += self.parentCircle.nodes[node_index[i]].expectedOutlandPoints * weights[i]

class AuxNode(Node):
    def __init__(self,circle,chart, radialFromTarget, auxNodeDirection):
        super().__init__(circle, chart)
        self.radialFromTarget = radialFromTarget
        


    def calculateNode(self, auxNodeDirection):
        
        nextNode_circle_index = self.parentCircle.index - 1
        target_circle = self.chart.dataStructure.circles[nextNode_circle_index]
 
        if auxNodeDirection == 0.0:
            beta = 0 
            target_radial = self.radialFromTarget
            distanceToNextNode = self.distanceFromTarget - target_circle.distanceFromTarget
        else:
            beta = math.pi - auxNodeDirection - math.asin(self.parentCircle.distanceFromTarget/target_circle.distanceFromTarget*math.sin(auxNodeDirection))
            target_radial = (self.radialFromTarget - beta) % math.pi 
            distanceToNextNode = target_circle.distanceFromTarget * math.sin(beta) / math.sin(auxNodeDirection)


        nextNode = InterpolationNode(target_circle, self.chart, target_radial)
        nextNode.calculateNode()

        

        # now iterate back  */
        # ----------------- */
        whl = numpy.zeros([self.chart.hGridSize,self.chart.liftGridSize])			# wh(h,x,l)
        wtl = numpy.zeros([self.chart.hGridSize,self.chart.liftGridSize])
        whl[0,:] = (self.chart.alpha*self.chart.polar.ldmax / self.chart.xmax)*numpy.ones([1,self.chart.liftGridSize])	# Landout value */
        
        holdg = numpy.copy(self.chart.hGrid)
        hold_p_finish = numpy.zeros([self.chart.hGridSize])
        hold_t_to_go = numpy.zeros([self.chart.hGridSize])
        hold_outlanding_points = numpy.zeros([self.chart.hGridSize])

        results_outlanding_points = numpy.zeros([self.chart.hGridSize,self.chart.liftGridSize])
        results_p_finish = numpy.zeros([self.chart.hGridSize,self.chart.liftGridSize])
        results_t_to_go = numpy.zeros([self.chart.hGridSize,self.chart.liftGridSize])

        # place to store height at x -dx */
        xi = self.parentCircle.index  #fstp + 1											# index of distance
        
        self.whv=numpy.copy(nextNode.whv)
        self.wtv=numpy.copy(nextNode.wtv)
        li=0

        [lprb, lift_strengts]= self.chart.liftMap.getLiftProb()
                                                    # look over all lift values
        while li < self.chart.liftGridSize:
            l =  lift_strengts[li]					
            hi = 0										# look over all altitudes; hi = h index
            while hi < self.chart.hGridSize:
                # for each altitude and lift, we have the mcready, so we can work
                # back and find the altitude at which you leave to get here */
                glide_angle = self.chart.polar.getInverseGlideRatio(numpy.array([l*self.chart.porp]),nextNode.mcCready[hi])
                holdg[hi] = self.chart.hGrid[hi] + glide_angle * distanceToNextNode*6000
                
                # save also p finish, t to go and expected outlanding points
                hold_p_finish[hi] = nextNode.expectedFinishProb[hi]
                hold_t_to_go[hi] = nextNode.expectedTimeToGo[hi] + 3600 * distanceToNextNode / self.chart.polar.getSpeed(l,1/glide_angle-0.01)
                hold_outlanding_points[hi] = nextNode.expectedOutlandPoints[hi]
                
                hi = hi + 1

            # at this point, whl and wtl are wh and wt, but at the new h.
            # next task is to interpolate this to give wh and wt on the old h grid*/
            hi = 1	# one always means land out
            while hi < self.chart.hGridSize:
                h = self.chart.hGrid[hi]
                if h <= numpy.amin(holdg):	# you're out
                    whl[hi,li] = self.chart.alpha * self.chart.polar.ldmax / self.chart.routeDist  # Validate!!
                    wtl[hi,li] = 0
                    
                    results_outlanding_points[hi,li] = (self.chart.routeDist-self.distanceFromTarget+h/6000*self.chart.polar.ldmax)/self.chart.routeDist*self.chart.alpha
                    results_p_finish[hi,li] = 0
                    results_t_to_go[hi,li] = 0 
                else:
                    self.indxl = None
                    whl[hi,li] = self.interp2(h,holdg,self.whv)	#interpolates
                    wtl[hi,li] = self.interp2(h,holdg,self.wtv)

                    #results_outlanding_points_current=(route_dist-distance_to_go+h/6000*ldmax)/route_dist*alpha
                    #if results_outlanding_points_current>alpha:
                    #    results_outlanding_points[hi,int(xi-1),li+1]=results_outlanding_points_current
                    #else:
                    #    results_outlanding_points[hi,int(xi-1),li+1]=results_outlanding_points_current
                    results_p_finish[hi,li] = self.interp2(h,holdg,hold_p_finish)
                    results_t_to_go[hi,li] = self.interp2(h,holdg,hold_t_to_go)
                    results_outlanding_points[hi,li] = self.interp2(h,holdg,hold_outlanding_points)
                hi = hi + 1
        
            # 	At this point, we have wht and wlt assuming that you will
            #   glide. However, you have the option to climb. If l-smin > mc
            #   then you climb up to the point that l-smin = mc (mc rises with
            #   altitude) or the top of the lift, whichever comes first. */
            if numpy.amax(whl[:,li] == 0):
                print('error: whl = 0 whl = ')
                print(whl[:,li].T)
            lam = - wtl[:,li] / whl[:,li]					# MacCready if you criuse
            flg = 0											# flag for, have you foud point to leave yet
            if numpy.amin(lam) < l - self.chart.polar.minSink:	# if there are any altitudes where you climb
                                                            # find altitude where you stop thermaling */
                indx = self.chart.hGridSize - 1					# start at top
                while self.chart.hGrid[indx] > self.chart.minclimb:	# can't thermal below minclimb
                    if lam[indx] < l - self.chart.polar.minSink:	# if lift too weak, leave here
                        if flg == 0:						# if this is where you leave
                            wtall=wtl[indx,li]
                            if indx == (self.chart.hGrid.size-1):		# if leave at top, wh jumps
                                whall=- wtall / (l - self.chart.polar.minSink)
                            else:
                                whall=- wtall / (l - self.chart.polar.minSink)	# let wh jump to soak approx error
                            flg = 1
                        if flg == 1:						#  if you will leave higher, Mc work  wn
                            wtl[indx,li] = wtall
                            whl[indx,li] = whall

                            results_p_finish[indx,li] = results_p_finish[min([indx+1,self.chart.hGridSize-1]),li]
                            results_t_to_go[indx,li] = results_t_to_go[min([indx+1,self.chart.hGridSize-1]),li] + self.chart.hStep/(l - self.chart.polar.minSink) ##TODO: Do not add if max alt
                            results_outlanding_points[indx,li] =  results_outlanding_points[min([indx+1,self.chart.hGridSize-1]),li]
                    indx = indx - 1
            li = li + 1
        
        self.whv = numpy.sum(numpy.multiply(whl,lprb.reshape([1,-1])),axis=1)
        self.wtv = numpy.sum(numpy.multiply(wtl,lprb.reshape([1,-1])),axis=1)
        self.mcCready = - self.wtv / self.whv

        self.expectedFinishProb = numpy.sum(numpy.multiply(results_p_finish,lprb.reshape([1,-1])),axis=1) 
        self.expectedTimeToGo = numpy.nan_to_num(numpy.divide(numpy.sum(numpy.multiply(numpy.multiply(results_t_to_go,lprb.reshape([1,-1])),results_p_finish),axis=1),numpy.sum(numpy.multiply(lprb.reshape([1,-1]),results_p_finish)))) 
        self.expectedOutlandPoints = numpy.sum(numpy.multiply(results_outlanding_points,lprb.reshape([1,-1])),axis=1)
        self.expectedPoints = numpy.matmul(results_outlanding_points,lprb)+numpy.matmul(results_p_finish,lprb)*(1-self.chart.alpha) * numpy.minimum(self.chart.routeDist/numpy.matmul(results_t_to_go,lprb)/3600/self.chart.vWin,numpy.ones([self.chart.hGridSize]))

        #lamfin=numpy.concatenate((lamfin,lamv.reshape([-1,1])),axis=1)

        # mix up h 
        self.whv = self.mixer(self.whv,self.chart.hGrid,self.chart.mix)						# adds subtracts random amount to h -- smooths wh, wt over h
        self.wtv = self.mixer(self.wtv,self.chart.hGrid,self.chart.mix)

        del nextNode

class EnRouteNode(Node):
    def __init__(self,index,circle,chart, radialFromTarget):
        super().__init__(circle, chart)
        self.radialFromTarget = index / circle.nNodes * 2 * math.pi

    def calculateNode(self):
        # Create nodes
        aux_node_interval = 10.0 / 360.0 * 2.0 * math.pi
        max_offset = math.asin((self.distanceFromTarget - self.chart.circleIntervals) / self.distanceFromTarget)
        n_auxnodes_per_side = int(max_offset/aux_node_interval) 
        n_auxnode = 2 * n_auxnodes_per_side + 1
        max_auxnode_offset = n_auxnodes_per_side * aux_node_interval
        
        # Calculate nodes and choose optimal node
        for i in range(n_auxnode):
            # Find out aux node's radial 
            auxNodeDirection = max_auxnode_offset - i * aux_node_interval

            # Create and Calculate parentNode
            auxNode = AuxNode(self.parentCircle, self.chart, self.radialFromTarget, auxNodeDirection) 
            auxNode.calculateNode(auxNodeDirection)

            # Compare to previous nodes
            isBetter = numpy.argwhere(auxNode.expectedPoints > self.expectedPoints)

            self.expectedPoints[isBetter] = auxNode.expectedPoints[isBetter]
            self.expectedTimeToGo[isBetter] = auxNode.expectedTimeToGo[isBetter]
            self.expectedFinishProb[isBetter] = auxNode.expectedFinishProb[isBetter]
            self.expectedOutlandPoints[isBetter] = auxNode.expectedOutlandPoints[isBetter]

            self.wtv[isBetter] = auxNode.wtv[isBetter]
            self.whv[isBetter] = auxNode.whv[isBetter]
            self.mcCready[isBetter] = auxNode.mcCready[isBetter]

            self.optimalDirection[isBetter] = auxNodeDirection
        
            # Remove auxNode
            del auxNode

class Point:
    def __init__(self, x_target, y_target):
        self.xTarget = x_target
        self.yTarget = y_target

    def setPositionXY(self, x, y):
        self.x = x
        self.y = y 
        x_diff = self.x - self.xTarget
        y_diff = self.y - self.yTarget
        self.distanceFromTarget = math.sqrt(x_diff**2 + y_diff**2)
        acos_y = math.acos(y_diff / self.distanceFromTarget)  
        if x_diff >= 0:
            self.radialFromTarget = acos_y 
        else: 
            self.radialFromTarget = 2 * math.pi - acos_y

    def setPositionRadDist(self, rad, dist):
        self.radialFromTarget = rad
        self.distanceFromTarget = dist 
        self.x = self.xTarget + math.sin(self.radialFromTarget) * self.distanceFromTarget
        self.y = self.yTarget + math.cos(self.radialFromTarget) * self.distanceFromTarget

class GlidePolar:
    def __init__(self):
        #Dry LS7 33.9 km/m2
        self.minSink=1.22
        self.minSinkSpeed=46
        self.highSpeed=92
        self.sinkForHighSpeed=3.88
        self.a=numpy.dot(2,(self.highSpeed - self.minSink)) / (self.highSpeed - self.minSinkSpeed) ** 2
        self.ldmax=self.getGlideRatio(numpy.array([0]),numpy.array([0]))

    #old name: vofg
    def getSpeed(self,m=numpy.nan,gl=numpy.nan):
        if gl < 0:
            print('vofg: being asked for negative glide angle')
        if ((m < self.minSink) and (gl > self.getGlideRatio(numpy.array([m]),numpy.array([0])))):
            vl=(- 1)
        else:
            al= self.a / 2
            bl=- (1 / gl + self.a*self.minSinkSpeed)
            cl=self.minSink + self.a / 2* (self.minSinkSpeed * self.minSinkSpeed) - m
            disc=bl * bl - 4*al*cl
            if disc < 0:
                print('vofg error: negative discriminant.  g={} l={} gmax={}'.format(gl,m,self.getGlideRatio(numpy.array([m]),0).item()))
                vl=(- 1)
            else:
                vl=(- bl + (disc) ** 0.5) / (2*al)
        return vl

    #old name: s
    def getSinkRate(self,v=None):
        if numpy.amin(numpy.amin(v).T) < 0:
            print('s(v) error: asking for speed {:.3f} less than zero v'.format(v))
        sink=(numpy.multiply((v >= self.minSinkSpeed),(self.minSink + (self.a / 2 *(v - self.minSinkSpeed) ** 2))) + numpy.multiply((v < self.minSinkSpeed),self.minSink))
        return sink

    #old name: rev, replace by numpy.flip(x,0)

    #old name: mcinv
    def getMcCreadyInv(self, lam=None):
        mc_value=(numpy.multiply((lam > - self.minSink),(abs(2/self.a*(lam + self.minSink) + self.minSinkSpeed ** 2) ** 0.5)) + (lam <= self.minSink)*0)
        return mc_value 

    #old name: mc
    def getMcCready(self,v=None):
        mc_value=self.a/2*(v**2-self.minSinkSpeed**2)-self.minSink
        return mc_value   

    #old name: g
    def getGlideRatio(self,m=None,lam=None):
        vl = self.getMcCreadyInv(lam-m.T)
        gl = vl / (self.getSinkRate(vl) - m.T)
        return gl

    # old name: g2
    # g2 returns inverse glide angle (height lost per mile) */
    # this can handle negatives (porpoising) */
    # speed is restricted to be at least self.minSinkSpeed so you can't get infinite up */
    def getInverseGlideRatio(self,m=None,lam=None):
        if ((m.size > 1) and (lam.size > 1)):
            print('G2 error: one of m, lam may be a vector but not both')
        i=0
        while i < m.size:
            ml=m[i]
            j=0
            while j < lam.size:
                if lam.size==1:
                    laml=lam
                else:
                    laml=lam[j]
                vl=self.getMcCreadyInv(laml - ml)
                if vl == 0:
                    vl=numpy.copy(self.minSinkSpeed)
                if ((i == 0) and (j == 0)):
                    gl=(self.getSinkRate(vl) - ml) / vl
                else:
                    gl=numpy.append(gl,((self.getSinkRate(vl) - ml) / vl))
                j=j + 1
            i=i + 1
        return gl

# MAIN FUNCTION
def main(lift_map_file,turn_point_file,lift_strenght,height_band,circle_intervals,n_circles,turnpoints):
    #Initialize objects
    polar = GlidePolar()
    lift_map = LiftMap(lift_map_file,lift_strenght)
    charts=[]

    for i in turnpoints:
        index=numpy.argwhere(turn_point_file[0,]==i)
        target=Point(turn_point_file[1,index] , turn_point_file[2,index])
        charts.append(Chart(lift_map,height_band,circle_intervals,n_circles,polar, target))
    
    #Create maps
    charts[0].dataStructure.plotCurve(0,0)

# PARSE ARGUMENTS
if __name__ == '__main__':
    #Thermal map
    lift_map_file=numpy.load("Prediction_raster.npy")
    #Turnpoint file
    turn_point_file=numpy.load("Turnpoints.npy")

    #Lift strenght, m/s
    lift_strenght = 3.0
    #Height band, cloudbase, m
    height_band = 1500
    #todo: wind

    #Turnpoints
    turnpoints=[1] # EFRY
    #Chart parameters, km
    circle_intervals = 1.852
    n_circles = 1
    #ADD node_interval

    cProfile.run('main(lift_map_file,turn_point_file,lift_strenght,height_band,circle_intervals,n_circles,turnpoints)', 'run_stats')