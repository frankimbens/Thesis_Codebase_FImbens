import numpy as np
import cupy as cp
from PIL import Image
import time


print("directory:")
inputLoadstring = input()
start_time = time.time()

spaceColourKernel = cp.ElementwiseKernel(
   'float32 theT, float32 theR, float32 theTheta, float32 thePhi',
   'int32 colors',
   '''
   float theQ[4] = {theT,theR,theTheta,thePhi};
if (theQ[1] > 8.0) {
    if (__sinf(theQ[3])*__sinf(theQ[3]) < 0.01) {
        colors = 0xff0000ff;
    } else if (__cosf(theQ[3])*__cosf(theQ[3]) < 0.01) {
        colors = 0xff00ff00;
    } else if (__sinf(2.0*theQ[2])*__sinf(2.0*theQ[2]) < 0.01) {
        colors = 0xffff0000;
    } else {
        colors = 0xff3c0000;
    }
//} else if (theQ[1] > 5.0*0.5) {
//        colors = 0xff00ff00;
} else {
    colors = 0xff000000;
}   ''',
   'colours')

timeColourKernel = cp.ElementwiseKernel(
    'float32 theT, float32 theR, float32 theTheta, float32 thePhi',
    'int32 colours',
    '''
    float theR0 = 12.0;
    float scale = powf(abs(theT/(0.00015*500000)),10);
    float scale2 = powf(abs(theT/(0.00015*500000)),1);
    if(theR < theR0 - 0.1) {
        colours = 0xffffff00;
    } else if(scale < 1.01) {
        //int colour = 0.5*(scale*0x000000ff + scale2*0x000000ff);
        int colourG = scale*0x000000ff;
        colourG *=256;
        
        int colourB = scale2*0x00000ff*7/8;
        colourB *=256*256;
        
        int colourB2 = scale2*0x000ff00;
        colourB2 = ((colourB2%0xff)/8);
        colourB2 *=256*256;
        
        colours = 0xff000000 + colourG + colourB + colourB2;
    } else {
        colours = 0xff00ff00;
    };
//    float scale = powf(1.0-theR/12,10);
//    int colourR = 0x000000ff*scale;
//    colours = 0xff000000 + colourR;
   ''',
   'colours')     

bandColourKernel = cp.ElementwiseKernel(
   'float32 theT, float32 theR, float32 theTheta, float32 thePhi',
   'int32 colors',
   '''
   float theQ[4] = {theT,theR,theTheta,thePhi};
if (theQ[1] > 8.0) {
    if ((theQ[2]-1.5708)*(theQ[2]-1.5708) < 0.0005) {
        colors = 0xffffffff;
    } else if ((theQ[2]-1.5708)*(theQ[2]-1.5708) < 0.003) {
        colors = 0xffaaaaaa;
    } else if ((theQ[2]-1.5708)*(theQ[2]-1.5708) < 0.01) {
        colors = 0xff666666;
    } else {
        colors = 0xff3c0000;
    }
//} else if (theQ[1] > 5.0*0.5) {
//        colors = 0xff00ff00;
} else {
    colors = 0xff000000;
}   ''',
   'colours')

newtonColourKernel = cp.ElementwiseKernel(
   'float32 theT, float32 theR, float32 theTheta, float32 thePhi',
   'int32 colors',
   '''
   float theQ[4] = {theT,theR,theTheta,thePhi};
if (theQ[1] > 8.0) {
    float dist = 40*(  (cosf(theQ[2]))*(cosf(theQ[2])) + (sinf(theQ[3]))*(sinf(theQ[3]))  );
    if (dist < 1.0) {
        colors = 0xff3c0000;
        colors += 0xff*(1-dist);
    } else {
        colors = 0xff3c0000;
    }
} else {
    colors = 0xff000000;
}   ''',
   'colours')

def doubleArray(theTBoard, theRBoard, theThetaBoard, thePhiBoard):
    theTBoardDoubled = np.concatenate((theTBoard, np.flip(theTBoard, axis=1)), axis=1)
    theRBoardDoubled = np.concatenate((theRBoard, np.flip(theRBoard, axis=1)), axis=1)
    theThetaBoardDoubled = np.concatenate((theThetaBoard, np.flip((np.pi-theThetaBoard), axis=1)), axis=1)
    thePhiBoardDoubled = np.concatenate((thePhiBoard, np.flip(thePhiBoard, axis=1)), axis=1)
    return(theTBoardDoubled,theRBoardDoubled,theThetaBoardDoubled,thePhiBoardDoubled)

def intForBilinearRender(theThetaBoardDoubled,thePhiBoardDoubled,theTextureShape):
    theThetaArray = (theThetaBoardDoubled) / np.pi * theTextureShape[0]
    thePhiArray = np.remainder(thePhiBoardDoubled, 2*np.pi) / (2*np.pi) * theTextureShape[1]

    theThetaInt = theThetaArray.astype(np.int32)
    thePhiInt = thePhiArray.astype(np.int32)

    theThetaBilinearFac = theThetaArray - theThetaInt
    thePhiBilinearFac = thePhiArray - thePhiInt
    
    return(theThetaInt,thePhiInt,theThetaBilinearFac,thePhiBilinearFac)
    
def drawBilinear(theLoadString,theTBoard,theRBoard,theThetaBoard,thePhiBoard):

    theTextureImage = Image.open('background_3.png')
    #print(theTextureImage.format)
    #print(theTextureImage.size)
    #print(theTextureImage.mode)

    theTexture = np.asarray(theTextureImage)
    #print(theTexture.dtype)
    TextureShape = theTexture.shape
    #print(TextureShape)


    imageShape = theThetaBoard.shape
    outputImage = np.zeros(imageShape[::-1],dtype=np.uint32) #the tuple is reversed here
    print(imageShape)

    ThetaInt,PhiInt,ThetaBilinearFac,PhiBilinearFac = intForBilinearRender(theThetaBoard,thePhiBoard,TextureShape)

    colourRGB = 0

    for i in range(imageShape[0]):
        for j in range(imageShape[1]):
            if theRBoard[i,j] > 8:
                if PhiInt[i,j] >=0 and ThetaInt[i,j] >=0:
                    if PhiInt[i,j] < TextureShape[1]-1 and ThetaInt[i,j] < TextureShape[0]-1:
                        colourRGB11 = theTexture[ThetaInt[i][j],PhiInt[i][j]]
                        colourRGB12 = theTexture[ThetaInt[i][j]+1,PhiInt[i][j]]
                        colourRGB21 = theTexture[ThetaInt[i][j],PhiInt[i][j]+1]
                        colourRGB22 = theTexture[ThetaInt[i][j]+1,PhiInt[i][j]+1]
                        
                        colourR = int(colourRGB11[0]*(1-ThetaBilinearFac[i,j])*(1-PhiBilinearFac[i,j]) + colourRGB12[0]*ThetaBilinearFac[i,j]*(1-PhiBilinearFac[i,j]) + colourRGB21[0]*(1-ThetaBilinearFac[i,j])*PhiBilinearFac[i,j] + colourRGB22[0]*ThetaBilinearFac[i,j]*PhiBilinearFac[i,j])
                        colourG = int(colourRGB11[1]*(1-ThetaBilinearFac[i,j])*(1-PhiBilinearFac[i,j]) + colourRGB12[1]*ThetaBilinearFac[i,j]*(1-PhiBilinearFac[i,j]) + colourRGB21[1]*(1-ThetaBilinearFac[i,j])*PhiBilinearFac[i,j] + colourRGB22[1]*ThetaBilinearFac[i,j]*PhiBilinearFac[i,j])
                        colourB = int(colourRGB11[2]*(1-ThetaBilinearFac[i,j])*(1-PhiBilinearFac[i,j]) + colourRGB12[2]*ThetaBilinearFac[i,j]*(1-PhiBilinearFac[i,j]) + colourRGB21[2]*(1-ThetaBilinearFac[i,j])*PhiBilinearFac[i,j] + colourRGB22[2]*ThetaBilinearFac[i,j]*PhiBilinearFac[i,j])
                        colourInt = 0xff000000 + 256*256*int(colourB) + 256*int(colourG) + int(colourR)
                        outputImage[j,i] = colourInt
            else:
                colourInt = 0xff000000
                outputImage[j,i] = colourInt

    #Saves the array as an image
    rgb_img = im = Image.fromarray(outputImage, 'RGBA')
    rgb_img.save(theLoadString + 'TextureRender.png')
    
def drawSpace(theLoadString,theTBoard,theRBoard,theThetaBoard,thePhiBoard):
    CUPYTBoard     = cp.asarray(theTBoard)
    CUPYRBoard     = cp.asarray(theRBoard)
    CUPYThetaBoard = cp.asarray(theThetaBoard)
    CUPYPhiBoard   = cp.asarray(thePhiBoard)
    
    #Turning the output positions into a bmp
    spaceColourBoard = np.transpose(spaceColourKernel(CUPYTBoard,CUPYRBoard,CUPYThetaBoard,CUPYPhiBoard))
    cp.cuda.Stream.null.synchronize()

    #Turning out array back to numpy to be able to draw it
    spaceColourBoardNP = cp.asnumpy(spaceColourBoard)

    #Saves the array as an image
    rgb_img = im = Image.fromarray(spaceColourBoardNP, 'RGBA')
    rgb_img.save(theLoadString + 'SpaceRender.png')

def drawTime(theLoadString,theTBoard,theRBoard,theThetaBoard,thePhiBoard):
    CUPYTBoard     = cp.asarray(theTBoard)
    CUPYRBoard     = cp.asarray(theRBoard)
    CUPYThetaBoard = cp.asarray(theThetaBoard)
    CUPYPhiBoard   = cp.asarray(thePhiBoard)
    
    #Turning the output positions into a bmp
    timeColourBoard = np.transpose(timeColourKernel(CUPYTBoard,CUPYRBoard,CUPYThetaBoard,CUPYPhiBoard))
    cp.cuda.Stream.null.synchronize()

    #Turning out array back to numpy to be able to draw it
    timeColourBoardNP = cp.asnumpy(timeColourBoard)

    #Saves the array as an image
    rgb_img = im = Image.fromarray(timeColourBoardNP, 'RGBA')
    rgb_img.save(theLoadString + 'TimeRender.png')

def drawBand(theLoadString,theTBoard,theRBoard,theThetaBoard,thePhiBoard):
    CUPYTBoard     = cp.asarray(theTBoard)
    CUPYRBoard     = cp.asarray(theRBoard)
    CUPYThetaBoard = cp.asarray(theThetaBoard)
    CUPYPhiBoard   = cp.asarray(thePhiBoard)
    
    #Turning the output positions into a bmp
    bandColourBoard = np.transpose(bandColourKernel(CUPYTBoard,CUPYRBoard,CUPYThetaBoard,CUPYPhiBoard))
    cp.cuda.Stream.null.synchronize()

    #Turning out array back to numpy to be able to draw it
    bandColourBoardNP = cp.asnumpy(bandColourBoard)

    #Saves the array as an image
    rgb_img = im = Image.fromarray(bandColourBoardNP, 'RGBA')
    rgb_img.save(theLoadString + 'BandRender.png')

def drawNewtonRing(theLoadString,theTBoard,theRBoard,theThetaBoard,thePhiBoard):
    CUPYTBoard     = cp.asarray(theTBoard)
    CUPYRBoard     = cp.asarray(theRBoard)
    CUPYThetaBoard = cp.asarray(theThetaBoard)
    CUPYPhiBoard   = cp.asarray(thePhiBoard)
    
    #Turning the output positions into a bmp
    newtonColourBoard = np.transpose(newtonColourKernel(CUPYTBoard,CUPYRBoard,CUPYThetaBoard,CUPYPhiBoard))
    cp.cuda.Stream.null.synchronize()

    #Turning out array back to numpy to be able to draw it
    newtonColourBoardNP = cp.asnumpy(newtonColourBoard)

    #Saves the array as an image
    rgb_img = im = Image.fromarray(newtonColourBoardNP, 'RGBA')
    rgb_img.save(theLoadString + 'NewtonRingRender.png')

def drawAll(loadString):
    TBoard     = np.load(loadString + "/outputTBoard.npy")
    RBoard     = np.load(loadString + "/outputRBoard.npy")
    ThetaBoard = np.load(loadString + "/outputThetaBoard.npy")
    PhiBoard   = np.load(loadString + "/outputPhiBoard.npy")

    #TBoardDoubled,RBoardDoubled,ThetaBoardDoubled,PhiBoardDoubled = doubleArray(TBoard,RBoard,ThetaBoard,PhiBoard)
    
    #drawBilinear(loadString,TBoard,RBoard,ThetaBoard,PhiBoard)
    drawSpace(loadString,TBoard,RBoard,ThetaBoard,PhiBoard)
    drawTime(loadString,TBoard,RBoard,ThetaBoard,PhiBoard)
    drawBand(loadString,TBoard,RBoard,ThetaBoard,PhiBoard)
    #drawNewtonRing(loadString,TBoard,RBoard,ThetaBoard,PhiBoard)

drawAll(inputLoadstring)

print(time.time()-start_time)



