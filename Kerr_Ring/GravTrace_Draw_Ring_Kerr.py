import numpy as np
import cupy as cp
from PIL import Image
import time


print("directory:")
inputLoadstring = input()
globalGM = 0.5
globalA = 0.49
globalQ = 0.0
start_time = time.time()
print("starting image stuff")

ringColourKernel = cp.ElementwiseKernel(
    'float32 theT, float32 theR, float32 theTheta, float32 thePhi, float32 thePT, float32 thePR, float32 thePTheta, float32 thePPhi, float32 theM, float32 theA',
    'int32 colors',
    '''
    float theQ[4] = {theT, theR, theTheta, thePhi};
    float theP[4] = {thePT,thePR,thePTheta,thePPhi};
    
    
    
    if (theQ[1] > theM*20) {
        colors = 0xff3c0000;
    } else if (theQ[1] > theM*3) {
        
        float theBandP[4] = {0,0,0,0};
        float theBandPPhi = 0;
        float theBandPT = 0;
        float theBandPLength = 0;
        
        float theRedShift = 0.0;
        
        float Metric[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        float derPrecision = 0.0001;
        float DRMetric[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        float DerMetric[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
        
        invMetricAtPoint(Metric,theQ[1],theQ[2],theM,theA);
        invMetricAtPoint(DRMetric,theQ[1]+derPrecision,theQ[2],theM,theA);
        for(int mu = 0 ; mu < 4 ; mu++) {
        for(int nu = 0 ; nu < 4 ; nu++) {
            DerMetric[mu][nu] = (DRMetric[mu][nu] - Metric[mu][nu])/derPrecision;
        };
        };
        
        theBandPPhi = -1;
        theBandPT = ( -2*theBandPPhi*DerMetric[0][3] - sqrtf(4*theBandPPhi*DerMetric[0][3]*theBandPPhi*DerMetric[0][3] - 4*DerMetric[0][0]*theBandPPhi*theBandPPhi*DerMetric[3][3]) )/( 2 * DerMetric[0][0]);
        theBandPLength = sqrtf( -( theBandPT*theBandPT*Metric[0][0] + 2*theBandPPhi*theBandPT*Metric[0][3] + theBandPPhi*theBandPPhi*Metric[3][3] ) );
        theBandP[0] = -theBandPT/theBandPLength;
        theBandP[3] = theBandPPhi/theBandPLength;
                
        for(int mu = 0 ; mu < 4 ; mu++) {
        for(int nu = 0 ; nu < 4 ; nu++) {
            theRedShift+= Metric[mu][nu]*theP[nu]*theBandP[mu];
        };
        };
        
        //--Debugging code
        //if(theRedShift < 1.0) {
        //colors = 0xffff0000;
        //} else {
        //colors = 0xffffffff;
        //}

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
    __device__ void invMetricAtPoint(float ioMetric[4][4], float locR, float locTheta, float locGM,float locA)
    {
        //float locS = locA*locM;
        float locA2 = locA*locA;
        float locR2 = locR*locR;
        float locSinTheta = sin(locTheta);
        float locCosTheta = cos(locTheta);
        float locSin2Theta = 2.0*locSinTheta*locCosTheta;
        float locRho2 = locR2 + (locA*locCosTheta)*(locA*locCosTheta);
        float locDelta = locR2 - 2*locGM*locR + locA2;
        
        ioMetric[0][0] = -1.0-2.0*locGM*locR*(locA2 + locR2)/(locRho2*locDelta);
        ioMetric[1][1] = locDelta/locRho2;
        ioMetric[2][2] = 1.0/locRho2;
        ioMetric[3][3] = (locRho2-2.0*locGM*locR)/(locSinTheta*locSinTheta*locRho2*locDelta);
        ioMetric[0][3] = -2.0*locGM*locA*locR/(locRho2*locDelta);
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
    
    spaceColourBoard = np.transpose(ringColourKernel(CUPYTBoard,CUPYRBoard,CUPYThetaBoard,CUPYPhiBoard,CUPYPTBoard,CUPYPRBoard,CUPYPThetaBoard,CUPYPPhiBoard,inputGM,inputA))
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



