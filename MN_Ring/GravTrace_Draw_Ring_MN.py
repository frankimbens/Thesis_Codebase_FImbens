import numpy as np
import cupy as cp
from PIL import Image
import time


print("input Folder")
inputLoadstring = input()
start_time = time.time()
print("starting image stuff")

globalGM = 0.5
globalA = 0.49
globalQ = 2.0


ringColourKernel = cp.ElementwiseKernel(
    'float32 theT, float32 theR, float32 theTheta, float32 thePhi, float32 thePT, float32 thePR, float32 thePTheta, float32 thePPhi, float32 theM, float32 theA, float32 theVarQ',
    'int32 colors',
    '''
    double theQ[4] = {theT, theR, theTheta, thePhi};
    double theP[4] = {thePT,thePR,thePTheta,thePPhi};
    
    
    
    if (theQ[1] > 10.0) {
        colors = 0xff3c0000;
    } else if (theQ[1] > 1.5) {
        
        double theBandP[4] = {0,0,0,0};
        double theBandPPhi = 0;
        double theBandPT = 0;
        double theBandPLength = 0;
        
        double theRedShift = 0.0;
        
        double Metric[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        double derPrecision = 0.00001;
        double DRMetric[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        double DerMetric[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        
        invMetricAtPoint(Metric,theQ[1],theQ[2],theM,theA,theVarQ);
        invMetricAtPoint(DRMetric,theQ[1]+derPrecision,theQ[2],theM,theA,theVarQ);
        for(int mu = 0 ; mu < 4 ; mu++) {
        for(int nu = 0 ; nu < 4 ; nu++) {
            DerMetric[mu][nu] = (DRMetric[mu][nu] - Metric[mu][nu])/derPrecision;
        };
        };
        
        theBandPPhi = -1;
        theBandPT = ( -2*theBandPPhi*DerMetric[0][3] - sqrtf(4*theBandPPhi*DerMetric[0][3]*theBandPPhi*DerMetric[0][3] - 4*DerMetric[0][0]*theBandPPhi*theBandPPhi*DerMetric[3][3]) )/( 2 * DerMetric[0][0]);
        theBandPLength = ( theBandPT*theBandPT*Metric[0][0] + 2*theBandPPhi*theBandPT*Metric[0][3] + theBandPPhi*theBandPPhi*Metric[3][3] );
        //if(theBandPLength < 0.0) {
            theBandPLength = sqrtf( -( theBandPT*theBandPT*Metric[0][0] + 2*theBandPPhi*theBandPT*Metric[0][3] + theBandPPhi*theBandPPhi*Metric[3][3] ) );
        //} else if(theBandPLength> 0.0) {
        //    theBandPLength = sqrtf( ( theBandPT*theBandPT*Metric[0][0] + 2*theBandPPhi*theBandPT*Metric[0][3] + theBandPPhi*theBandPPhi*Metric[3][3] ) );
        //};
        theBandP[0] = -theBandPT/theBandPLength;
        theBandP[3] = theBandPPhi/theBandPLength;
        
        //-------Schwarzschild-Analytical components
        //theBandPPhi = sqrtf(2*theM/(( 2/(theR) - 2*theM/(theR*theR) ) ));
        //theBandPT = sqrtf( ( 1+ theBandPPhi*theBandPPhi/(theR*theR) )*( 1 - 2*theM/theR ) );
        //theBandP[0] = theBandPT;
        //theBandP[3] = theBandPPhi;
                
        for(int mu = 0 ; mu < 4 ; mu++) {
        for(int nu = 0 ; nu < 4 ; nu++) {
            theRedShift+= Metric[mu][nu]*theP[nu]*theBandP[mu];
        };
        };
        
        int scale = 0;
        int colorB =   0x010100;
        int colorR =   0x000001;
        
        colors = 0xff000000;
        if(theRedShift > 1.0) {
            scale = 255 * (1.0 - 1.0*(1.0 - 1.0/theRedShift));
            colors += colorB*scale+colorR*255;
        } else if(theRedShift < 1.0 and theRedShift > 0/0) {
            scale = 255 * (1.0 - 1.0*(1.0 - theRedShift));
            colors += colorR*scale+colorB*255;
        } else if(theRedShift < 0.0) {
            colors += 0x00ffff;
        } else {
            colors += 0x00ff00;
        }
        
        
    } else {
        colors = 0xff000000;
    }
    ''',
   'colours',
    preamble='''
    __device__ void invMetricAtPoint(double ioMetric[4][4], double locR, double locTheta, double locM,double locSpinA,double locQ)
    {
        double locS = locSpinA*locM;
        double locAlpha = (  -locM + sqrt(locM*locM - (locS/locM)*(locS/locM))  )/(locS/locM);
        double locK = locM*(1-locAlpha*locAlpha)/(1+locAlpha*locAlpha);
        double locBeta = locQ*(locM*locM*locM)/(locK*locK*locK);
        
        double locX = (locR-locM)/locK;
        double locY = cos(locTheta);
        
        double locRR = sqrt(locX*locX + locY*locY - 1);
        double locP1 = locX*locY/locRR;
        double locP2 = 0.5*(3*locP1*locP1 - 1);
        double locP3 = 0.5*(5*locP1*locP1*locP1 - 3*locP1);
        
        double locaa = -locAlpha*exp(  -2*locBeta*( -1 + (locX-locY)/locRR + (locX-locY)/(locRR*locRR)*locP1 + (locX-locY)/(locRR*locRR*locRR)*locP2 )  );
        double locbb = locAlpha*exp(  2*locBeta*( 1 - (locX+locY)/locRR + (locX+locY)/(locRR*locRR)*locP1 - (locX+locY)/(locRR*locRR*locRR)*locP2 )  );
        
        double locGammaPrime = 0.5*log( (locX*locX-1)/(locX*locX-locY*locY) ) + 1.5*locBeta*locBeta/(locRR*locRR*locRR*locRR*locRR*locRR)*(locP3*locP3 - locP2*locP2) + locBeta*(  -2 + (locX-locY + (locX+locY))/locRR + (locX-locY - (locX+locY))/(locRR*locRR)*locP1 + (locX-locY + (locX+locY))/(locRR*locRR*locRR)*locP2  );
        double locPsi = locBeta*locP2/(locRR*locRR*locRR);
        
        double locA = (locX*locX - 1)*(1+locaa*locbb)*(1+locaa*locbb) - (1 - locY*locY)*(locbb-locaa)*(locbb-locaa);
        double locB = ( (locX + 1) + (locX - 1)*locaa*locbb )*( (locX + 1) + (locX - 1)*locaa*locbb ) + ( (1+locY)*locaa + (1-locY)*locbb )*( (1+locY)*locaa + (1-locY)*locbb );
        double locC = (locX*locX - 1)*(1+locaa*locbb)*(locbb-locaa - locY*(locaa+locbb))  +  (1 - locY*locY)*(locbb-locaa)*(1+locaa*locbb + locX*(1-locaa*locbb));
        
        double locExpGamma = exp(locGammaPrime)* locA/( (locX*locX-1)*(1-locAlpha*locAlpha)*(1-locAlpha*locAlpha) );
        double locOmega = 2*locK*exp(-2*locPsi)*locC/locA-4*locK*locAlpha/(1-locAlpha*locAlpha);
        double locF = exp(2*locPsi)*locA/locB;
        
        double locRho2 = (locR - locM)*(locR - locM) - locK*locK*locY*locY;
        double locDelta = (locR - locM)*(locR - locM) - locK*locK;
        double locSinTheta = sin(locTheta);
     
        ioMetric[0][0] = -1/locF + locF*locOmega*locOmega/(locDelta*locSinTheta*locSinTheta);
        ioMetric[1][1] = locF*locDelta/(locExpGamma*locRho2);
        ioMetric[2][2] = locF/(locExpGamma*locRho2);
        ioMetric[3][3] = locF/(locDelta*locSinTheta*locSinTheta);
        ioMetric[0][3] = locF*locOmega/(locDelta*locSinTheta*locSinTheta);
        ioMetric[3][0] = ioMetric[0][3];
    };
   '''
   )

    
def drawRing(theLoadString,theTBoard,theRBoard,theThetaBoard,thePhiBoard,thePTBoard,thePRBoard,thePThetaBoard,thePPhiBoard):
    CUPYTBoard     = cp.asarray(theTBoard)
    CUPYRBoard     = cp.asarray(theRBoard)
    CUPYThetaBoard = cp.asarray(theThetaBoard)
    CUPYPhiBoard   = cp.asarray(thePhiBoard)

    CUPYPTBoard     = cp.asarray(thePTBoard)
    CUPYPRBoard     = cp.asarray(thePRBoard)
    CUPYPThetaBoard = cp.asarray(thePThetaBoard)
    CUPYPPhiBoard   = cp.asarray(thePPhiBoard)
    
    #Turning the output positions into a bmp
    inputGM = cp.float32(globalGM)
    inputA = cp.float32(globalA)
    inputQ = cp.float32(globalQ)
    spaceColourBoard = np.transpose(ringColourKernel(CUPYTBoard,CUPYRBoard,CUPYThetaBoard,CUPYPhiBoard,CUPYPTBoard,CUPYPRBoard,CUPYPThetaBoard,CUPYPPhiBoard,inputGM,inputA,inputQ))
    cp.cuda.Stream.null.synchronize()

    #Turning out array back to numpy to be able to draw it
    spaceColourBoardNP = cp.asnumpy(spaceColourBoard)

    #Saves the array as an image
    rgb_img = im = Image.fromarray(spaceColourBoardNP, 'RGBA')
    rgb_img.save(theLoadString + 'RingRender.png')


def drawAll(loadString):
    TBoard     = np.load(loadString + "/outputTBoard.npy")
    RBoard     = np.load(loadString + "/outputRBoard.npy")
    ThetaBoard = np.load(loadString + "/outputThetaBoard.npy")
    PhiBoard   = np.load(loadString + "/outputPhiBoard.npy")

    PTBoard     = np.load(loadString + "/outputPTBoard.npy")
    PRBoard     = np.load(loadString + "/outputPRBoard.npy")
    PThetaBoard = np.load(loadString + "/outputPThetaBoard.npy")
    PPhiBoard   = np.load(loadString + "/outputPPhiBoard.npy")

    drawRing(loadString,TBoard,RBoard,ThetaBoard,PhiBoard,PTBoard,PRBoard,PThetaBoard,PPhiBoard)

drawAll(inputLoadstring)

print(time.time()-start_time)



