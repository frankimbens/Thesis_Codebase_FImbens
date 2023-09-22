#prerequisites: 
#Visual studio 2019.
#CUDA 11.5 (note, earlier versions might work, 11.6 definitely does not).
#Both NumPy and CuPy libraries, CUPY requires the two above.
#Pillow for exporting an image

#A quick note that variables starting with 'the' are local in the function/loop, whereas glob/global are global variables

import sys
import math
import numpy as np
import cupy as cp
import time
from PIL import Image

#The three constants that define our black hole
globalGM = 0.5
globalA = 0.49
globalQ = 0.0 #Has no function for the Kerr version
#render resolution and screen width
screenResolution = (300,300)
screenWidth = (2.0,2.0)
screenDist = 20.0*globalGM
focalLength = 0.5
obsTheta = -np.pi/2*0.1

def cartCameraCast(locScreenResolution,locScreenWidth,locScreenDist,locFocalLength):
    locX = np.ones(locScreenResolution,dtype=np.float32)*locScreenDist
    locYPREMESH = np.linspace(-locScreenWidth[0],locScreenWidth[0],locScreenResolution[0],dtype=np.float32)
    locZPREMESH = np.linspace(-locScreenWidth[1],locScreenWidth[1],locScreenResolution[1],dtype=np.float32)
    locZ, locY = np.meshgrid(locZPREMESH,locYPREMESH)

    locVX = np.ones(locScreenResolution,dtype=np.float32)*-1.0
    locVY = locY/locScreenWidth[0]*locFocalLength
    locVZ = locZ/locScreenWidth[0]*locFocalLength
    
    return((locX,locY,locZ,locVX,locVY,locVZ))

def rotateObservation(locX,locY,locZ,locVX,locVY,locVZ,obsTheta):
    obsRotMatrix = [[np.cos(obsTheta),0.0,-np.sin(obsTheta)],[0.0,1.0,0.0],[np.sin(obsTheta),0.0,np.cos(obsTheta)]]

    locX2 = cp.asarray(locX*obsRotMatrix[0][0] + locY*obsRotMatrix[0][1] + locZ*obsRotMatrix[0][2])
    locY2 = cp.asarray(locX*obsRotMatrix[1][0] + locY*obsRotMatrix[1][1] + locZ*obsRotMatrix[1][2])
    locZ2 = cp.asarray(locX*obsRotMatrix[2][0] + locY*obsRotMatrix[2][1] + locZ*obsRotMatrix[2][2])

    locVX2 = cp.asarray(locVX*obsRotMatrix[0][0] + locVY*obsRotMatrix[0][1] + locVZ*obsRotMatrix[0][2])
    locVY2 = cp.asarray(locVX*obsRotMatrix[1][0] + locVY*obsRotMatrix[1][1] + locVZ*obsRotMatrix[1][2])
    locVZ2 = cp.asarray(locVX*obsRotMatrix[2][0] + locVY*obsRotMatrix[2][1] + locVZ*obsRotMatrix[2][2])
    
    return((locX2,locY2,locZ2,locVX2,locVY2,locVZ2))
    

coordKernel = cp.ElementwiseKernel(
   'float32 theX, float32 theY, float32 theZ, float32 theVX, float32 theVY, float32 theVZ, float32 theGM, float32 theA, float32 theVarQ',
   'float32 theT, float32 theR, float32 theTheta, float32 thePhi, float32 thePT, float32 thePR, float32 thePTheta, float32 thePPhi',
   '''

//float thePos[3] = {theX, theY, theZ};
float theVel[3] = {theVX, theVY, theVZ};

float theNewPos[3] = {0,0,0};
float theNewVel[3] = {0,0,0};

BLfromCart(theNewPos, theX, theY, theZ, theA);

theT = 0.0;
theR = theNewPos[0];
theTheta = theNewPos[1];
thePhi = theNewPos[2];

float theInvCoordDerivative[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
float theCoordDerivative[3][3] = {{0,0,0},{0,0,0},{0,0,0}};

theInvCoordDerivative[0][0] = theR/sqrtf(theR*theR+theA*theA)*sinf(theTheta)*cosf(thePhi);
theInvCoordDerivative[0][1] = sqrtf(theR*theR+theA*theA)*cosf(theTheta)*cosf(thePhi);
theInvCoordDerivative[0][2] = -sqrtf(theR*theR+theA*theA)*sinf(theTheta)*sinf(thePhi);
                     
theInvCoordDerivative[1][0] = theR/sqrtf(theR*theR+theA*theA)*sinf(theTheta)*sinf(thePhi);
theInvCoordDerivative[1][1] = sqrtf(theR*theR+theA*theA)*cosf(theTheta)*sinf(thePhi);
theInvCoordDerivative[1][2] = sqrtf(theR*theR+theA*theA)*sinf(theTheta)*cosf(thePhi);
                     
theInvCoordDerivative[2][0] = cosf(theTheta);
theInvCoordDerivative[2][1] = -theR*sinf(theTheta);
theInvCoordDerivative[2][2] = 0;

matrixInverse(theInvCoordDerivative, theCoordDerivative);

for(int mu = 0 ; mu < 3 ; mu++) {
for(int nu = 0 ; nu < 3 ; nu++) {
    theNewVel[mu] += theCoordDerivative[mu][nu]*theVel[nu];
};
};

float theVT = 0.0;
float theVR = theNewVel[0];
float theVTheta = theNewVel[1];
float theVPhi = theNewVel[2];

float Metric[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
metricAtPoint(Metric,theR,theTheta,theGM,theA,theVarQ);

theVT = lightlikeT(theVR,theVTheta,theVPhi,Metric);

thePT     = Metric[0][0]*theVT + Metric[0][1]*theVR + Metric[0][2]*theVTheta + Metric[0][3]*theVPhi;
thePR     = Metric[1][0]*theVT + Metric[1][1]*theVR + Metric[1][2]*theVTheta + Metric[1][3]*theVPhi;
thePTheta = Metric[2][0]*theVT + Metric[2][1]*theVR + Metric[2][2]*theVTheta + Metric[2][3]*theVPhi;
thePPhi   = Metric[3][0]*theVT + Metric[3][1]*theVR + Metric[3][2]*theVTheta + Metric[3][3]*theVPhi;

theT += theVT    *(Metric[0][0]*theVT + Metric[0][1]*theVR + Metric[0][2]*theVTheta + Metric[0][3]*theVPhi);
theT += theVR    *(Metric[1][0]*theVT + Metric[1][1]*theVR + Metric[1][2]*theVTheta + Metric[1][3]*theVPhi);
theT += theVTheta*(Metric[2][0]*theVT + Metric[2][1]*theVR + Metric[2][2]*theVTheta + Metric[2][3]*theVPhi);
theT += theVPhi  *(Metric[3][0]*theVT + Metric[3][1]*theVR + Metric[3][2]*theVTheta + Metric[3][3]*theVPhi);

   ''',
   'geodesic',
   preamble='''
__device__ void BLfromCart(float oCoord[3], float locX, float locY, float locZ, float locA)
{
    float locRad = sqrtf(locX*locX + locY*locY + locZ*locZ);
    float locR = sqrtf( 0.5*( (locRad*locRad-locA*locA) + sqrtf( (locA*locA-locRad*locRad)*(locA*locA-locRad*locRad) + 4.0*locZ*locZ*locA*locA) ) );

    float locTheta = acosf(locZ/locR);

    float locPhi = 1.5707963;
    if(locX > 0) {
        locPhi = atanf(locY/locX);
    } else if(locX < 0) {
        locPhi = atanf(locY/locX) + 3.14159265;
    }
    
    oCoord[0] = locR;
    oCoord[1] = locTheta;
    oCoord[2] = locPhi;
};

__device__ void metricAtPoint(float ioMetric[4][4], float locR, float locTheta, float locGM,float locSpinA,float locQ) //NOTE THIS IS NOT THE INVERTED ONE
{
    float locA2 = locSpinA*locSpinA;
    float locR2 = locR*locR;
    float locSinTheta = sin(locTheta);
    float locCosTheta = cos(locTheta);
    float locSin2Theta = 2.0*locSinTheta*locCosTheta;
    float locRho2 = locR2 + (locSpinA*locCosTheta)*(locSpinA*locCosTheta);
    float locDelta = locR2 - 2*locGM*locR + locA2;
    
    ioMetric[0][0] = -1.0+2.0*locGM*locR/locRho2;
    ioMetric[1][1] = locRho2/locDelta;
    ioMetric[2][2] = locRho2;
    ioMetric[3][3] = locSinTheta*locSinTheta/locRho2*((locR*locR+locSpinA*locSpinA)*(locR*locR+locSpinA*locSpinA) - locSpinA*locSpinA*locDelta*locSinTheta*locSinTheta);
    ioMetric[0][3] = -2.0*locGM*locSpinA*locR*locSinTheta*locSinTheta/locRho2;
    ioMetric[3][0] = ioMetric[0][3];
};

__device__ float lightlikeT(float locVX,float locVY,float locVZ,float locMetric[4][4])
{
    float locQuadA = locMetric[0][0];
    float locVel[3] = {locVX,locVY,locVZ};
    float locQuadB = 0;
    for (int i = 0 ; i < 3 ; i++) { 
        locQuadB += 2.0*locVel[i]*locMetric[0][i+1];
    };
    float locQuadC = 0;
    for (int i = 0 ; i < 3 ; i++) { 
        for (int j = 0 ; j < 3 ; j++) { 
            locQuadC += locVel[i]*locMetric[i+1][j+1]*locVel[j];
        };
    };
    float locVT = (-locQuadB - sqrtf(locQuadB*locQuadB - 4.0*locQuadA*locQuadC))/(2.0*locQuadA);
    return(locVT);
};

__device__ void matrixInverse(float iMetric[3][3], float oMetric[3][3])
{
    oMetric[0][0] = iMetric[1][1]*iMetric[2][2] - iMetric[1][2]*iMetric[2][1];
    oMetric[1][0] = - iMetric[1][0]*iMetric[2][2] + iMetric[1][2]*iMetric[2][0];
    oMetric[2][0] = iMetric[1][0]*iMetric[2][1] - iMetric[1][1]*iMetric[2][0];

    oMetric[0][1] = - iMetric[0][1]*iMetric[2][2] + iMetric[0][2]*iMetric[2][1];
    oMetric[1][1] = iMetric[0][0]*iMetric[2][2] - iMetric[0][2]*iMetric[2][0];
    oMetric[2][1] = - iMetric[0][0]*iMetric[2][1] + iMetric[0][1]*iMetric[2][0];

    oMetric[0][2] = iMetric[0][1]*iMetric[1][2] - iMetric[0][2]*iMetric[1][1];
    oMetric[1][2] = - iMetric[0][0]*iMetric[1][2] + iMetric[0][2]*iMetric[1][0];
    oMetric[2][2] = iMetric[0][0]*iMetric[1][1] - iMetric[1][1]*iMetric[0][0];

    float Determinant = iMetric[0][0]*oMetric[0][0] + iMetric[0][1]*oMetric[1][0] + iMetric[0][2]*oMetric[2][0];
    if(Determinant != 0) {
        for(int mu = 0 ; mu < 3 ; mu++) {
        for(int nu = 0 ; nu < 3 ; nu++) {
            oMetric[mu][nu] = oMetric[mu][nu]/Determinant;
        };
        };
    } else {
        for(int mu = 0 ; mu < 3 ; mu++) {
        for(int nu = 0 ; nu < 3 ; nu++) {
            oMetric[mu][nu] = 0;
        };
        };
    };
};
   ''')
   
geodesicKernel = cp.ElementwiseKernel(
   'float32 theT, float32 theR, float32 theTheta, float32 thePhi, float32 thePT, float32 thePR, float32 thePTheta, float32 thePPhi, float32 theM, float32 theA, float32 theVarQ, int32 theN',
   'float32 theEndT, float32 theEndR,float32 theEndTheta,float32 theEndPhi,float32 theEndPT, float32 theEndPR,float32 theEndPTheta,float32 theEndPPhi',
   '''

double theQ[4] = {theT, theR, theTheta, thePhi};
double theP[4] = {thePT, thePR, thePTheta, thePPhi};
double theR0 = 30*theM;

double derPrecision = 0.00000001;
double stepPrecision = 0.0015;

double Metric[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
double dRMetric[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
double dThetaMetric[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
double DerMetric[4][4][4] = {{{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},{{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},{{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}},{{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}}};

double theDeltaP1[4] = {0,0,0,0};
double theDeltaQ1[4] = {0,0,0,0};
double theDeltaP2[4] = {0,0,0,0};
double theDeltaQ2[4] = {0,0,0,0};
double theDeltaP3[4] = {0,0,0,0};
double theDeltaQ3[4] = {0,0,0,0};
double theDeltaP4[4] = {0,0,0,0};
double theDeltaQ4[4] = {0,0,0,0};

for (int i = 0 ; i < theN ; i++) { 
    
//---------------------- K1
    invMetricAtPoint(Metric,theQ[1],theQ[2],theM,theA,theVarQ);
    invMetricAtPoint(dRMetric,theQ[1] + derPrecision,theQ[2],theM,theA,theVarQ);
    invMetricAtPoint(dThetaMetric,theQ[1],theQ[2] + derPrecision,theM,theA,theVarQ);
    
    for(int mu = 0 ; mu < 4 ; mu++) {
    for(int nu = 0 ; nu < 4 ; nu++) {
        DerMetric[1][mu][nu] = (dRMetric[mu][nu] - Metric[mu][nu])/derPrecision;
        DerMetric[2][mu][nu] = (dThetaMetric[mu][nu] - Metric[mu][nu])/derPrecision;
    };
    };
    
    for(int mu = 0 ; mu < 4 ; mu++) {
        theDeltaP1[mu] = 0;
    for(int nu = 0 ; nu < 4 ; nu++) {
    for(int sigma = 0 ; sigma < 4 ; sigma++) {
        theDeltaP1[mu]+= -0.5*DerMetric[mu][nu][sigma]*theP[nu]*theP[sigma];
    };
    };
    };
    
    for(int mu = 0 ; mu < 4 ; mu++) {
        theDeltaQ1[mu] = 0;
    for(int nu = 0 ; nu < 4 ; nu++) {
        theDeltaQ1[mu]+= Metric[mu][nu]*theP[nu];
    };
    };
    
    double timeScale = abs(stepPrecision/theDeltaQ1[0]);

//---------------------- K2
    invMetricAtPoint(Metric,theQ[1]+0.5*timeScale*theDeltaQ1[1],theQ[2]+0.5*timeScale*theDeltaQ1[2],theM,theA,theVarQ);
    invMetricAtPoint(dRMetric,theQ[1]+0.5*timeScale*theDeltaQ1[1] + derPrecision,theQ[2]+0.5*timeScale*theDeltaQ1[2],theM,theA,theVarQ);
    invMetricAtPoint(dThetaMetric,theQ[1]+0.5*timeScale*theDeltaQ1[1],theQ[2]+0.5*timeScale*theDeltaQ1[2] + derPrecision,theM,theA,theVarQ);
    
    for(int mu = 0 ; mu < 4 ; mu++) {
    for(int nu = 0 ; nu < 4 ; nu++) {
        DerMetric[1][mu][nu] = (dRMetric[mu][nu] - Metric[mu][nu])/derPrecision;
        DerMetric[2][mu][nu] = (dThetaMetric[mu][nu] - Metric[mu][nu])/derPrecision;
    };
    };
    
    for(int mu = 0 ; mu < 4 ; mu++) {
        theDeltaP2[mu] = 0;
    for(int nu = 0 ; nu < 4 ; nu++) {
    for(int sigma = 0 ; sigma < 4 ; sigma++) {
        theDeltaP2[mu]+= -0.5*DerMetric[mu][nu][sigma]*(theP[nu]+0.5*timeScale*theDeltaP1[nu])*(theP[sigma]+0.5*timeScale*theDeltaP1[sigma]);
    };
    };
    };
    
    for(int mu = 0 ; mu < 4 ; mu++) {
        theDeltaQ2[mu] = 0;
    for(int nu = 0 ; nu < 4 ; nu++) {
        theDeltaQ2[mu]+= Metric[mu][nu]*(theP[nu]+0.5*timeScale*theDeltaP1[nu]);
    };
    };

//---------------------- K3
    invMetricAtPoint(Metric,theQ[1]+0.5*timeScale*theDeltaQ2[1],theQ[2]+0.5*timeScale*theDeltaQ2[2],theM,theA,theVarQ);
    invMetricAtPoint(dRMetric,theQ[1]+0.5*timeScale*theDeltaQ2[1] + derPrecision,theQ[2]+0.5*timeScale*theDeltaQ2[2],theM,theA,theVarQ);
    invMetricAtPoint(dThetaMetric,theQ[1]+0.5*timeScale*theDeltaQ2[1],theQ[2]+0.5*timeScale*theDeltaQ2[2] + derPrecision,theM,theA,theVarQ);
    
    for(int mu = 0 ; mu < 4 ; mu++) {
    for(int nu = 0 ; nu < 4 ; nu++) {
        DerMetric[1][mu][nu] = (dRMetric[mu][nu] - Metric[mu][nu])/derPrecision;
        DerMetric[2][mu][nu] = (dThetaMetric[mu][nu] - Metric[mu][nu])/derPrecision;
    };
    };
    
    for(int mu = 0 ; mu < 4 ; mu++) {
        theDeltaP3[mu] = 0;
    for(int nu = 0 ; nu < 4 ; nu++) {
    for(int sigma = 0 ; sigma < 4 ; sigma++) {
        theDeltaP3[mu]+= -0.5*DerMetric[mu][nu][sigma]*(theP[nu]+0.5*timeScale*theDeltaP2[nu])*(theP[sigma]+0.5*timeScale*theDeltaP2[sigma]);
    };
    };
    };
    
    for(int mu = 0 ; mu < 4 ; mu++) {
        theDeltaQ3[mu] = 0;
    for(int nu = 0 ; nu < 4 ; nu++) {
        theDeltaQ3[mu]+= Metric[mu][nu]*(theP[nu]+0.5*timeScale*theDeltaP2[nu]);
    };
    };

//---------------------- K4
    invMetricAtPoint(Metric,theQ[1]+timeScale*theDeltaQ3[1],theQ[2]+timeScale*theDeltaQ3[2],theM,theA,theVarQ);
    invMetricAtPoint(dRMetric,theQ[1]+timeScale*theDeltaQ3[1] + derPrecision,theQ[2]+timeScale*theDeltaQ3[2],theM,theA,theVarQ);
    invMetricAtPoint(dThetaMetric,theQ[1]+timeScale*theDeltaQ3[1],theQ[2]+timeScale*theDeltaQ3[2] + derPrecision,theM,theA,theVarQ);
    
    for(int mu = 0 ; mu < 4 ; mu++) {
    for(int nu = 0 ; nu < 4 ; nu++) {
        DerMetric[1][mu][nu] = (dRMetric[mu][nu] - Metric[mu][nu])/derPrecision;
        DerMetric[2][mu][nu] = (dThetaMetric[mu][nu] - Metric[mu][nu])/derPrecision;
    };
    };
    
    for(int mu = 0 ; mu < 4 ; mu++) {
        theDeltaP4[mu] = 0;
    for(int nu = 0 ; nu < 4 ; nu++) {
    for(int sigma = 0 ; sigma < 4 ; sigma++) {
        theDeltaP4[mu]+= -0.5*DerMetric[mu][nu][sigma]*(theP[nu]+timeScale*theDeltaP3[nu])*(theP[sigma]+timeScale*theDeltaP3[sigma]);
    };
    };
    };
    
    for(int mu = 0 ; mu < 4 ; mu++) {
        theDeltaQ4[mu] = 0;
    for(int nu = 0 ; nu < 4 ; nu++) {
        theDeltaQ4[mu]+= Metric[mu][nu]*(theP[nu]+timeScale*theDeltaP3[nu]);
    };
    };
    
    
    
    for(int mu = 0 ; mu < 4 ; mu++) {
        theP[mu] += timeScale*0.1667*(theDeltaP1[mu] + 2*theDeltaP2[mu] + 2*theDeltaP3[mu] + theDeltaP4[mu]);
        theQ[mu] += timeScale*0.1667*(theDeltaQ1[mu] + 2*theDeltaQ2[mu] + 2*theDeltaQ3[mu] + theDeltaQ4[mu]);
    };
    
    if(theQ[1] > theR0) break;
    if(timeScale < stepPrecision*0.00003) break;
    if( 150*theQ[1]*(theQ[2]-1.5707963)*(theQ[2]-1.5707963) + (theQ[1]-9*theM)*(theQ[1]-9*theM) < 6.5*theM) break;
};

theEndT = theQ[0];
theEndR = theQ[1];
theEndTheta = theQ[2];
theEndPhi = theQ[3];

theEndPT = theP[0];
theEndPR = theP[1];
theEndPTheta = theP[2];
theEndPPhi = theP[3];
   ''',
   'geodesic',
   preamble='''
__device__ void invMetricAtPoint(double ioMetric[4][4], double locR, double locTheta, double locM,double locSpinA, double locQ)
{
    double locA2 = locSpinA*locSpinA;
    double locR2 = locR*locR;
    double locSinTheta = sin(locTheta);
    double locCosTheta = cos(locTheta);
    double locSin2Theta = 2.0*locSinTheta*locCosTheta;
    double locRho2 = locR2 + (locSpinA*locCosTheta)*(locSpinA*locCosTheta);
    double locDelta = locR2 - 2*locM*locR + locA2;
    
    ioMetric[0][0] = -1.0-2.0*locM*locR*(locA2 + locR2)/(locRho2*locDelta);
    ioMetric[1][1] = locDelta/locRho2;
    ioMetric[2][2] = 1.0/locRho2;
    ioMetric[3][3] = (locRho2-2.0*locM*locR)/(locSinTheta*locSinTheta*locRho2*locDelta);
    ioMetric[0][3] = -2.0*locM*locSpinA*locR/(locRho2*locDelta);
    ioMetric[3][0] = ioMetric[0][3];
};
   ''')


def makeData(THEscreenResolution,THEscreenWidth,THEscreenDist,THEfocalLength,THEobsTheta,THEGM,THEA,THEQ,saveLoc):
    
    #Making cartesian coordinates in 3dim space
    start_time = time.time()
    print("creating coordinates")

    globXpreRot,globYpreRot,globZpreRot,globVXpreRot,globVYpreRot,globVZpreRot = cartCameraCast(THEscreenResolution,THEscreenWidth,THEscreenDist,THEfocalLength)
    globX,globY,globZ,globVX,globVY,globVZ = rotateObservation(globXpreRot,globYpreRot,globZpreRot,globVXpreRot,globVYpreRot,globVZpreRot,THEobsTheta)

    print(time.time()-start_time)


    #Converting constants to correct data type
    inputGM = cp.float32(THEGM)
    inputA = cp.float32(THEA)
    inputQ = cp.float32(THEQ)


    #Turning Cartesian coordinates and velocity to (p,q) in BL coordinates
    start_time = time.time()
    print("converting coordinates")

    globT0,globR0,globTheta0,globPhi0,globPT0,globPR0,globPTheta0,globPPhi0 = coordKernel(globX,globY,globZ,globVX,globVY,globVZ,  inputGM,inputA,inputQ)
    cp.cuda.Stream.null.synchronize()

    # print('x')
    # print(globX)
    # print( cp.sqrt(globR0*globR0 + globalA*globalA)*cp.sin(globTheta0)*cp.cos(globPhi0) )
    # print('y')
    # print(globY)
    # print( cp.sqrt(globR0*globR0 + globalA*globalA)*cp.sin(globTheta0)*cp.sin(globPhi0) )
    # print('z')
    # print(globZ)
    # print( globR0*cp.cos(globTheta0) )

    print(time.time()-start_time)

    #Geodesic calculations
    start_time = time.time()
    print("propagating geodesics")

    Tboard,Rboard,ThetaBoard,PhiBoard,PTboard,PRboard,PThetaBoard,PPhiBoard = geodesicKernel(globT0,globR0,globTheta0,globPhi0,globPT0,globPR0,globPTheta0,globPPhi0,inputGM,inputA,inputQ,100000)
    cp.cuda.Stream.null.synchronize()

    print(time.time()-start_time)

    # print('t')
    # print(globT0)
    # print('r')
    # print(globR0)
    # print('th')
    # print(globTheta0)
    # print('ph')
    # print(globPhi0)

    #Saving Output
    np.save(saveLoc + "/outputTBoard.npy",Tboard)
    np.save(saveLoc + "/outputRBoard.npy",Rboard)
    np.save(saveLoc + "/outputThetaBoard.npy",ThetaBoard)
    np.save(saveLoc + "/outputPhiBoard.npy",PhiBoard)
    
    np.save(saveLoc + "/outputPTBoard.npy",PTboard)
    np.save(saveLoc + "/outputPRBoard.npy",PRboard)
    np.save(saveLoc + "/outputPThetaBoard.npy",PThetaBoard)
    np.save(saveLoc + "/outputPPhiBoard.npy",PPhiBoard)

makeData(screenResolution,screenWidth,screenDist,focalLength,obsTheta,globalGM,globalA,globalQ,"data2")