#  MATLAB code for coupled plasticity-diffusion problem
Numerical example of degradation of plate with a circulate hole for
> Mohammad. S. Joshaghani and Kalyana. B. Nakshatrala,
> ``A maximum-principle-satisfying modeling framework for coupling plasticity with species diffusion", available in 
> [arXiv](https://arxiv.org/abs/2011.06652).

In this repoository, we have provided MATLAB computer codes for coupled elastoplastic-diffusion problem. This code entails different coupling strategies and degradation models for both Continuous Galerkin (CG) and Non-negative (NN) formulations. 
These two formulations are discussed in the paper.

We have implemented the proposed computational framework by combining the capabilities of COMSOL Multiphysics and MATLAB , and by using LiveLink for MATLAB and COMSOL Java API interfaces. Java API provides a user’s interface to access finite element data structures and libraries in COMSOL, while LiveLink provides a bidirectional interface between COMSOL and MATLAB. The deformation subproblem is solved using the elastoplasticity module in COMSOL, and the diffusion subproblem is solved using a MATLAB computer code. The optimization solvers, needed in the non-negative formulation for the diffusion subproblem, are also from MATLAB optimization toolbox.

## Requirements
Running the example `plate_with_hole.m` requires installation of:
* COMSOL Multiphysics (version > 5.2)
* COMSOL Java API (version > 4.3)
* MATLAB. version 7.10.0 or higher
* MATLAB Optimization Toolbox

## Runscript
In your bash, you first need to start the COMSOL server and afterwards MATLAB 
```bash
comsol server -silent -port 12345 -tmpdir $TMPDIR &
sleep 10
matlab -nodisplay -singleCompThread -r plate_with_hole(<formulation>,<hardening>,<anisotropy>,<sizeMesh>,<solid>,<coupling>,<coupling param)> ;
```
The COMSOL server is started in the background. With the *-port* you can specify the port that is used by the COMSOL server to communicate with MATLAB. After starting up the COMSOL server, we recommend to run a sleep command, as the server needs some time before it is ready to communicate with MATLAB. In the last step, MATLAB is started. 

## Matlab script

* MATLAB script needs to know where to find COMSOL and through which port the communication should take place. Make sure to include
```
addpath('/path-to-apps/comsol/5.3/x86_64/mli');
mphstart(12345);
```

* Declare variables in plate_with_hole function as:

| variable        |   value        |
| ------------- |-------------| 
| formulation       | `CG` or `NN`| 
| hardening     | `Model I` or `model II` | 
| anisotropy      | `None` or `Low` or `High` |
| sizeMesh | values `1` to `6`, one is the coarsest |
| solid | `el` for elastic; `ppl` for perfectly plastic, or `elpl` for elastoplastic |
| coupling | `None`for uncoupled, `One−way`, or `Two−way`  |
| coupling param | set `$c_ref$ value` if model *model I* used and `$zeta$ value` if *model II* used|

## Representative results

![](./Figures/Video1.gif)
<p><center>
<em>Continuous Galerkin (CG) vs. non-negative (NN) formulations:</em>
This video illustrates the diffusional and plastic response of a plate with a hole undergoes one cycle of uniaxial loading-unloading. 
The results are generated with degradation model II under two-way coupling, which is discussed in the paper.
The concentration field should be between 0 and 1. 
The CG formulation violated the lower bound (the gray regions), and/or the upper bound (the black regions) at every loading step. 
It is also evident that CG formulation propagates the violation of the non-negative constraint to the deformation subproblem. Hence, the predictions for the stress profile, the effective plastic strain contours, and the evolution of plastic zones contain numerical errors and are not accurate. However, the proposed NN formulation successfully 
eliminated all these numerical artifacts (see the right column).
</center>
</p>

## Related links 
My [poster presentation](./Figures/Poster_for_EMI2019.pdf) at Engineering Mechanics Institute (EMI) Conference, Caltech, CA, 2019.
