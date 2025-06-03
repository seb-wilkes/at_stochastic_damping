
#include "atelem.c"
#include "atrandom.c"

struct elem 
{
    double kappa_x_mean;
    double kappa_y_mean;
    double kappa_s_mean;
    double kappa_x_std;
    double kappa_y_std;
    double kappa_s_std;
    double dispx;
    double dispxp;
    double dispy;
    double dispyp;
    double EnergyLossFactor;
};

void StochasticRadiationPass(double *r_in,
                           double kappa_x_mean, double kappa_y_mean, double kappa_s_mean,
                           double kappa_x_std, double kappa_y_std, double kappa_s_std,
                           double dispx, double dispxp, double dispy, double dispyp,
                           double EnergyLossFactor,
                           pcg32_random_t *rng, int num_particles)
{
    int c;
    double *r6;
    double x, xp, y, yp, dpp;
    double kappa_x, kappa_y, kappa_s;
    double shrink_x, shrink_y, shrink_s;

    #pragma omp parallel for if (num_particles > OMP_PARTICLE_THRESHOLD*10) \
    default(shared) shared(r_in, num_particles, rng) \
    private(c, r6, x, xp, y, yp, dpp, kappa_x, kappa_y, kappa_s, shrink_x, shrink_y, shrink_s)
    for (c = 0; c < num_particles; c++) {
        r6 = r_in + c * 6;
        
        if (!atIsNaN(r6[0])) {
            // Sample stochastic kappa values (Gaussian approximation)
            kappa_x = kappa_x_mean + kappa_x_std * atrandn_r(rng, 0.0, 1.0);
            kappa_y = kappa_y_mean + kappa_y_std * atrandn_r(rng, 0.0, 1.0);
            kappa_s = kappa_s_mean + kappa_s_std * atrandn_r(rng, 0.0, 1.0);
            
            // Calculate shrinkage factors using your direct formula
            shrink_x = sqrt(1.0 - kappa_x);
            shrink_y = sqrt(1.0 - kappa_y);
            shrink_s = sqrt(1.0 - kappa_s);
            
            // Extract coordinates with dispersion correction
            dpp = r6[4];
            x = r6[0] - dispx * dpp;
            xp = r6[1] - dispxp * dpp;
            y = r6[2] - dispy * dpp;
            yp = r6[3] - dispyp * dpp;
            
            // Apply stochastic damping
            x *= shrink_x;
            xp *= shrink_x;
            y *= shrink_y;
            yp *= shrink_y;
            dpp *= shrink_s;
            r6[5] *= shrink_s;  // longitudinal position
            
            // Restore coordinates with updated momentum
            r6[0] = x + dispx * dpp;
            r6[1] = xp + dispxp * dpp;
            r6[2] = y + dispy * dpp;
            r6[3] = yp + dispyp * dpp;
            r6[4] = dpp;
            
            // Apply energy loss
            r6[4] -= EnergyLossFactor;
        }
    }
}

#if defined(MATLAB_MEX_FILE) || defined(PYAT)
ExportMode struct elem *trackFunction(const atElem *ElemData,struct elem *Elem,
                double *r_in, int num_particles, struct parameters *Param)
{
    if (!Elem) {
        double U0, EnergyLossFactor;
        double dispx, dispxp, dispy, dispyp;
        double kappa_x_mean, kappa_y_mean, kappa_s_mean;
        double kappa_x_std, kappa_y_std, kappa_s_std;
        
        // Get the new stochastic parameters
        kappa_x_mean = atGetDouble(ElemData, "kappa_x_mean"); check_error();
        kappa_y_mean = atGetDouble(ElemData, "kappa_y_mean"); check_error();
        kappa_s_mean = atGetDouble(ElemData, "kappa_s_mean"); check_error();
        kappa_x_std = atGetDouble(ElemData, "kappa_x_std"); check_error();
        kappa_y_std = atGetDouble(ElemData, "kappa_y_std"); check_error();
        kappa_s_std = atGetDouble(ElemData, "kappa_s_std"); check_error();
        
        U0 = atGetDouble(ElemData, "U0"); check_error();
        dispx = atGetOptionalDouble(ElemData, "dispx", 0.0); check_error();
        dispxp = atGetOptionalDouble(ElemData, "dispxp", 0.0); check_error();
        dispy = atGetOptionalDouble(ElemData, "dispy", 0.0); check_error();
        dispyp = atGetOptionalDouble(ElemData, "dispyp", 0.0); check_error();
        
        Elem = (struct elem*)atMalloc(sizeof(struct elem));
        Elem->kappa_x_mean = kappa_x_mean;
        Elem->kappa_y_mean = kappa_y_mean;
        Elem->kappa_s_mean = kappa_s_mean;
        Elem->kappa_x_std = kappa_x_std;
        Elem->kappa_y_std = kappa_y_std;
        Elem->kappa_s_std = kappa_s_std;
        Elem->dispx = dispx;
        Elem->dispxp = dispxp;
        Elem->dispy = dispy;
        Elem->dispyp = dispyp;
        Elem->EnergyLossFactor = U0/Param->energy;
    }
    
    // Call YOUR function, not SimpleRadiationRadPass
    StochasticRadiationPass(r_in, 
                          Elem->kappa_x_mean, Elem->kappa_y_mean, Elem->kappa_s_mean,
                          Elem->kappa_x_std, Elem->kappa_y_std, Elem->kappa_s_std,
                          Elem->dispx, Elem->dispxp, Elem->dispy, Elem->dispyp,
                          Elem->EnergyLossFactor,
                          Param->thread_rng, num_particles);
        return Elem;
}

MODULE_DEF(StochasticRadiationPass)        /* Change module name too */

#endif /*defined(MATLAB_MEX_FILE) || defined(PYAT)*/

#if defined(MATLAB_MEX_FILE)
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  if (nrhs >= 2) {
    double *r_in;
    const mxArray *ElemData = prhs[0];
    int num_particles = mxGetN(prhs[1]);
    double U0, EnergyLossFactor;
    double dispx, dispxp, dispy, dispyp;
    double kappa_x_mean, kappa_y_mean, kappa_s_mean;
    double kappa_x_std, kappa_y_std, kappa_s_std;

    // Get stochastic parameters instead of damp_mat_diag
    kappa_x_mean = atGetDouble(ElemData, "kappa_x_mean"); check_error();
    kappa_y_mean = atGetDouble(ElemData, "kappa_y_mean"); check_error();
    kappa_s_mean = atGetDouble(ElemData, "kappa_s_mean"); check_error();
    kappa_x_std = atGetDouble(ElemData, "kappa_x_std"); check_error();
    kappa_y_std = atGetDouble(ElemData, "kappa_y_std"); check_error();
    kappa_s_std = atGetDouble(ElemData, "kappa_s_std"); check_error();
    
    dispx = atGetOptionalDouble(ElemData, "dispx", 0.0); check_error();
    dispy = atGetOptionalDouble(ElemData, "dispy", 0.0); check_error();
    dispxp = atGetOptionalDouble(ElemData, "dispxp", 0.0); check_error();
    dispyp = atGetOptionalDouble(ElemData, "dispyp", 0.0); check_error();
    U0 = atGetDouble(ElemData, "U0"); check_error();
    EnergyLossFactor = U0/6e9;  // Or whatever energy reference you want
    
    if (mxGetM(prhs[1]) != 6) mexErrMsgIdAndTxt("AT:WrongArg","Second argument must be a 6 x N matrix");
    
    /* ALLOCATE memory for the output array of the same size as the input  */
    plhs[0] = mxDuplicateArray(prhs[1]);
    r_in = mxGetDoubles(plhs[0]);
    
    // Need to create a dummy RNG for MATLAB - this is a limitation
    pcg32_random_t rng;
    pcg32_srandom_r(&rng, time(NULL), (intptr_t)&rng);
    
    // Call YOUR function
    StochasticRadiationPass(r_in, 
                          kappa_x_mean, kappa_y_mean, kappa_s_mean,
                          kappa_x_std, kappa_y_std, kappa_s_std,
                          dispx, dispxp, dispy, dispyp,
                          EnergyLossFactor,
                          &pcg32_global, num_particles);  // ✅ Use &pcg32_global like QuantDiff
  }      
  else if (nrhs == 0) {
      /* list of required fields */
      plhs[0] = mxCreateCellMatrix(7,1);  // Changed from 2 to 7
      mxSetCell(plhs[0],0,mxCreateString("kappa_x_mean"));
      mxSetCell(plhs[0],1,mxCreateString("kappa_y_mean"));
      mxSetCell(plhs[0],2,mxCreateString("kappa_s_mean"));
      mxSetCell(plhs[0],3,mxCreateString("kappa_x_std"));
      mxSetCell(plhs[0],4,mxCreateString("kappa_y_std"));
      mxSetCell(plhs[0],5,mxCreateString("kappa_s_std"));
      mxSetCell(plhs[0],6,mxCreateString("U0"));
      
      if (nlhs>1) {
          /* list of optional fields */
          plhs[1] = mxCreateCellMatrix(4,1);
          mxSetCell(plhs[1],0,mxCreateString("dispx"));
          mxSetCell(plhs[1],1,mxCreateString("dispxp"));
          mxSetCell(plhs[1],2,mxCreateString("dispy"));
          mxSetCell(plhs[1],3,mxCreateString("dispyp"));
      }
  }
  else {
      mexErrMsgIdAndTxt("AT:WrongArg","Needs 0 or 2 arguments");
  }
}
#endif /*defined(MATLAB_MEX_FILE)*/
