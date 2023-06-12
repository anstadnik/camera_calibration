1. **It’s strange to have \alpha coefficient dimension as pixels/mm with default
   = 1 (3.2.3 Camera to image projection, formula 3.3) Why 1 pixel == 1 mm as
default?**  
Corrected. The assumption is just that the pixels are squared, not that the
scale is equal to 1.
2. **A division model was introduced by A.W. Fitzgibbon, so it’s strange to have
   citation to Pritts et al., 2021 in this sentence: “A division model is very
powerful and can express a wide range of distortions (Pritts et al., 2021)“**  
Added a reference to the origin. Pritts is cited because I re-used and expanded
their approach to the inversion.
3. **“,” and “.” are often in the beginning of the line (after formulas)**  
Corrected.
4. **H from 3x3 matrix becomes 6-elements vector in “Now, by rewriting the
   equation in the matrix form M · H = 0, where  H=(r_11, r_12, r_12, r_22, t_1,
t_2)”**  
Changed the variable name to avoid confusion, and to correctly align with the
notation.
5. **“To find t1 and t2, note that r1 and r2 are orthonormal“ - they are
   orthonormal as 3d vectors, but one component is omitted. Do we suppose
this?**  
That's a typo. I [used symbolic math toolkit to verify my derivation](https://github.com/anstadnik/camera_calibration/blob/5dd416adbd8fb483ac8c3e23ea60d4d2d0eb374e/notebooks/Solver%20Scaramuzza.ipynb), but used
rows instead of columns to initialize the vectors. Actually, t_1 and t_2 are
already found on the previous step, and now we're finding r_31 and r_32. Fixed.
6. **(4.12) and (4.32) seems to be absent formulas (or redundant label)**  
Corrected, thanks.
7. **Only the change to reprojection error was given as a camera calibration
   result. No comparison was made to classical camera calibration results and no
discussion why results differ.**  
The main contribution of the paper is the recovery of the previously undetected
features using an intermediate model (i.e. feature detection step). The obtained
features can be used as an input to camera calibration toolchains, or the user
can use the camera calibrations, provided by the proposed algorithm. However, we don't
claim that the proposed camera calibration pipeline is better than the current SOTA.  
These newly found features will further constrain the camera calibration (also,
typically it's the points near the edge of the image are not initially
detected due to high distortion. These points contain
more information about the camera parameters compared to the barely distorted
points near the center of distortion).  
I added a clarification about that to the Metrics section, extended the
description of the effect of additional points on the resulting camera
calibration, and extended the description of the used metrics.
8. **In the following sentence I don’t understand why the reference was made to
   section 5.7.3: “However, because of the occlusions, or similar to the board
patterns in the background, there were false positives (section 5.7.3).“**  
Turns out, if the figure lacks a caption (as it was the case, only subfigures
had ones), the reference to it breaks. Fixed.
