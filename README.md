# Thesis_Codebase_FImbens
Dependencies: NumPy, CuPy, Pillow
Tested with CUDA 12.2, CUPY-CUDA12X on a GTX1080, RTX2080, RTX4080
Runtimes: 10s for Kerr, 60s for MN spacetime on RTX4080 at 300x300 resolution.

# Short Documentation
The code consists of two files. The first script generates and propagates the geodesics, and saves them as a numpy array. The second script then draws images based on these saved arrays.
## Generation
Rays are generated in a square grid which forms the camera, in carthesian space coordinates. This is then translated to Boyer-Lindquist coordinates on the GPU, a $t$ coordinate is added to make the input vectors lightlike. They are normalised in this same GPU kernel and then converted to covectors. After this a second kernel will apply an RK4 integrator to the geodesics, and propagate them until they either leave a certain radius (theR0 in the kernel) or "freeze" (the timestep becomes smaller than stepPrecision*0.00003). In the case of objects in the space, such as a disk, they are also terminated if they enter the bounding box of this object. After this process the location (theQ) and momenta (theP) are saved.
## Drawing
The drawing script has a lot of different drawing functions. Most use as input only the exit location of the geodesics. The rays that do not terminate are often drawn in black, the rest is sorted by where they hit the background and given a colour based on that position. The time drawing script is special, as it gives colour based on the exit time. A little bit of banding is added to this function to allow for more accurate reading of the low exit time regions. The background can also be given a texture, here bilinear filtering is used to draw the texture smoothly.
For the drawing scripts that include a disk only a single function exists. This simply has a black pixel for rays that do not terminate (that fall into the black hole), or a blue one for the background. All the pixels that terminate on the disk are then given a colour based on the redshift they undergo compared to the point at which they are sent out on the ring.

# Aim
The aim of the code is to be simple but powerfull, to allow easy changes to input metrics and easy analysis of the output data. In general global variables are declared as such, while local variables start with the, or at an even lower level with loc. This is to make the code easier to read, and thus easier to edit and apply to other projects. Please not that the dependent libraries, especially CuPy may need some external programs to be able to run, specifically Cuda. This is a proprietary language that runs only on recent (around 10 years) nVidia graphical processors, the code will not run on other devices. This language was used due to it's simplicity.
