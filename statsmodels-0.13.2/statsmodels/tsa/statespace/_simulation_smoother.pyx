#cython: profile=False
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=False
"""
State Space Models

Author: Chad Fulton  
License: Simplified-BSD
"""

# ## Constants

# ### Simulation smoothers
# TODO actually just copy the values from SMOOTHING_STATE, SMOOTHING_DISTURBANCE
# because we always want them to be identical
cdef int SIMULATE_STATE = 0x01           # Durbin and Koopman (2012), Chapter 4.9.1
cdef int SIMULATE_DISTURBANCE = 0x04     # Durbin and Koopman (2012), Chapter 4.9.2
cdef int SIMULATE_ALL = (
    SIMULATE_STATE | SIMULATE_DISTURBANCE
)

from statsmodels.tsa.statespace._kalman_filter cimport (
    FILTER_CONVENTIONAL, FILTER_COLLAPSED, FILTER_UNIVARIATE,
    INVERT_UNIVARIATE, SOLVE_CHOLESKY, TIMING_INIT_PREDICTED,
    STABILITY_FORCE_SYMMETRY, MEMORY_STORE_ALL
)
from statsmodels.tsa.statespace._kalman_smoother cimport (
    SMOOTHER_ALL
)

# Typical imports
import numpy as np
import warnings
cimport numpy as np
cimport cython

np.import_array()

cimport scipy.linalg.cython_blas as blas
cimport scipy.linalg.cython_lapack as lapack
cimport statsmodels.tsa.statespace._tools as tools

cdef int FORTRAN = 1

cdef class sSimulationSmoother(object):
    # ### Statespace model
    # cdef readonly sStatespace model
    # ### Kalman filter
    # cdef readonly sKalmanFilter kfilter
    # ### Kalman smoother
    # cdef readonly sKalmanSmoother smoother

    # ### Simulated Statespace model
    # cdef readonly sStatespace simulated_model
    # ### Simulated Kalman filter
    # cdef readonly sKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    # cdef readonly sKalmanSmoother simulated_smoother

    # ### Secondary Simulated Statespace model
    # Note: currently only used in the case of missing data
    # cdef readonly sStatespace secondary_simulated_model
    # ### Simulated Kalman filter
    # cdef readonly sKalmanFilter secondary_simulated_kfilter
    # ### Simulated Kalman smoother
    # cdef readonly sKalmanSmoother secondary_simulated_smoother

    # ### Simulation parameters
    # cdef public int simulation_output
    # cdef readonly int has_missing

    # ### Random variates
    # cdef int n_disturbance_variates
    # cdef readonly np.float32_t [:] disturbance_variates
    # cdef int n_initial_state_variates
    # cdef readonly np.float32_t [:] initial_state_variates

    # ### Simulated Data
    # cdef readonly np.float32_t [::1,:] simulated_measurement_disturbance
    # cdef readonly np.float32_t [::1,:] simulated_state_disturbance
    # cdef readonly np.float32_t [::1,:] simulated_state

    # ### Generated Data
    # cdef readonly np.float32_t [::1,:] generated_obs
    # cdef readonly np.float32_t [::1,:] generated_state

    # ### Temporary arrays
    # cdef readonly np.float32_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    # cdef np.float32_t * _tmp0
    # cdef np.float32_t * _tmp1
    # cdef np.float32_t * _tmp2

    def __init__(self,
                 sStatespace model,
                 int filter_method=FILTER_CONVENTIONAL,
                 int inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY,
                 int stability_method=STABILITY_FORCE_SYMMETRY,
                 int conserve_memory=MEMORY_STORE_ALL,
                 int filter_timing=TIMING_INIT_PREDICTED,
                 np.float64_t tolerance=1e-19,
                 int loglikelihood_burn=0,
                 int smoother_output=SMOOTHER_ALL,
                 int simulation_output=SIMULATE_ALL,
                 int nobs=-1):
        cdef int inc = 1
        cdef:
            np.npy_intp dim1[1]
            np.npy_intp dim2[2]
        cdef np.float32_t [::1,:] obs
        cdef np.float32_t [::1,:] secondary_obs
        cdef int nobs_endog

        # Use model nobs by default
        if nobs == -1:
            nobs = model.nobs
        # Only allow more nobs if a time-invariant model
        elif nobs > model.nobs and model.time_invariant == 0:
            raise ValueError('In a time-varying model, cannot create more'
                             ' simulations than there are observations.')
        elif nobs <= 0:
            raise ValueError('Invalid number of simulations; must be'
                             ' positive.')

        self.nobs = nobs
        nobs_endog = self.nobs * model.k_endog

        self.pretransformed_disturbance_variates = False
        self.pretransformed_initial_state_variates = False
        self.fixed_initial_state = False

        # Model objects
        self.model = model
        # self.kfilter = sKalmanFilter(
        #     self.model, filter_method, inversion_method,
        #     stability_method, conserve_memory,
        #     tolerance, loglikelihood_burn
        # )
        # self.smoother = sKalmanSmoother(
        #     self.model, self.kfilter, smoother_output
        # )

        # Simulated model objects
        dim2[0] = model.k_endog; dim2[1] = self.nobs;
        obs = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN)
        self.simulated_model = sStatespace(
            obs, model.design, model.obs_intercept, model.obs_cov,
            model.transition, model.state_intercept, model.selection,
            model.state_cov
        )
        self.simulated_kfilter = sKalmanFilter(
            self.simulated_model, filter_method, inversion_method,
            stability_method, conserve_memory, filter_timing,
            tolerance, loglikelihood_burn
        )
        self.simulated_smoother = sKalmanSmoother(
            self.simulated_model, self.simulated_kfilter, smoother_output
        )

        # Secondary simulated model objects
        # Currently only used if there is missing data (since then the
        # approach in which the Kalman filter only has to be run over the
        # series y_t^* = y_t - y_t^+ is infeasible), although it could also
        # allow drawing multiple samples at the same time, see Durbin and
        # Koopman (2002).
        self.has_missing = model.has_missing
        if self.has_missing:
            dim2[0] = model.k_endog; dim2[1] = self.nobs;
            secondary_obs = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN)
            blas.scopy(&nobs_endog, &model.obs[0,0], &inc, &secondary_obs[0,0], &inc)
            self.secondary_simulated_model = sStatespace(
                secondary_obs, model.design, model.obs_intercept, model.obs_cov,
                model.transition, model.state_intercept, model.selection,
                model.state_cov
            )
            self.secondary_simulated_kfilter = sKalmanFilter(
                self.secondary_simulated_model, filter_method, inversion_method,
                stability_method, conserve_memory, filter_timing,
                tolerance, loglikelihood_burn
            )
            self.secondary_simulated_smoother = sKalmanSmoother(
                self.secondary_simulated_model, self.secondary_simulated_kfilter, smoother_output
            )
        # In the case of non-missing data, the Kalman filter will actually
        # be run over y_t^* = y_t - y_t^+, which means the observation equation
        # intercept should be zero; make sure that it is
        else:
            dim2[0] = self.model.k_endog; dim2[1] = self.model.obs_intercept.shape[1];
            self.simulated_model.obs_intercept = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN)
            dim2[0] = self.model.k_states; dim2[1] = self.model.state_intercept.shape[1];
            self.simulated_model.state_intercept = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN)


        # Initialize the simulated model memoryviews
        # Note: the actual initialization is replaced in the simulate()
        # function below, but will complain if the memoryviews have not been
        # first initialized, which this call does.
        self.simulated_model.initialize_approximate_diffuse()
        if self.has_missing:
            self.secondary_simulated_model.initialize_approximate_diffuse()

        # Parameters
        self.simulation_output = simulation_output
        self.n_disturbance_variates = self.nobs * (self.model.k_endog + self.model.k_posdef)
        self.n_initial_state_variates = self.model.k_states

        # Random variates
        dim1[0] = self.n_disturbance_variates;
        self.disturbance_variates = np.PyArray_ZEROS(1, dim1, np.NPY_FLOAT32, FORTRAN)
        dim1[0] = self.n_initial_state_variates;
        self.initial_state_variates = np.PyArray_ZEROS(1, dim1, np.NPY_FLOAT32, FORTRAN)

        # Simulated data (\tilde eta_t, \tilde eps_t, \tilde alpha_t)
        # Note that these are (k_endog x nobs), (k_posdef x nobs), (k_states x nobs)
        if filter_method & FILTER_COLLAPSED:
            dim2[0] = self.model.k_states; dim2[1] = self.nobs;
        else:
            dim2[0] = self.model.k_endog; dim2[1] = self.nobs;
        self.simulated_measurement_disturbance = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN)
        dim2[0] = self.model.k_posdef; dim2[1] = self.nobs;
        self.simulated_state_disturbance = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN)
        dim2[0] = self.model.k_states; dim2[1] = self.nobs;
        self.simulated_state = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN)

        # Generated data (y_t^+, alpha_t^+)
        dim2[0] = self.model.k_endog; dim2[1] = self.nobs;
        self.generated_obs = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN)
        dim2[0] = self.model.k_states; dim2[1] = self.nobs+1;
        self.generated_state = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN)

        # Temporary arrays
        dim2[0] = self.model.k_states; dim2[1] = self.model.k_states;
        self.tmp0 = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN) # chol(P_1)
        dim2[0] = self.model.k_posdef; dim2[1] = self.model.k_posdef;
        self.tmp2 = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN) # chol(Q_t)
        dim2[0] = self.model.k_endog; dim2[1] = self.model.k_endog;
        self.tmp1 = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN) # chol(H_t)

        # Pointers
        self._tmp0 = &self.tmp0[0,0]
        self._tmp1 = &self.tmp1[0,0]
        self._tmp2 = &self.tmp2[0,0]

    def __reduce__(self):
        args = (self.model, self.model.filter_method, self.model.inversion_method,
                self.model.stability_method, self.model.conserve_memory, self.model.filter_timing,
                self.model.tolerance, self.model.loglikelihood_burn, self.simulated_smoother.smoother_output,
                self.simulation_output, self.nobs, self.pretransformed_variates)
        state = {
            'disturbance_variates': np.array(self.disturbance_variates, copy=True, order='F'),
            'initial_state_variates': np.array(self.initial_state_variates, copy=True, order='F'),
            'simulated_measurement_disturbance': np.array(self.simulated_measurement_disturbance, copy=True, order='F'),
            'simulated_state_disturbance': np.array(self.simulated_state_disturbance, copy=True, order='F'),
            'simulated_state': np.array(self.simulated_state, copy=True, order='F'),
            'generated_obs': np.array(self.generated_obs, copy=True, order='F'),
            'generated_state': np.array(self.generated_state, copy=True, order='F'),
            'tmp0': np.array(self.tmp0, copy=True, order='F'),
            'tmp2': np.array(self.tmp2, copy=True, order='F'),
            'tmp1': np.array(self.tmp1, copy=True, order='F')
        }
        return (self.__class__, args, state)

    def __setstate__(self, state):
        self.disturbance_variates = state['disturbance_variates']
        self.initial_state_variates = state['initial_state_variates']
        self.simulated_measurement_disturbance  = state['simulated_measurement_disturbance']
        self.simulated_state_disturbance = state['simulated_state_disturbance']
        self.simulated_state = state['simulated_state']
        self.generated_obs = state['generated_obs']
        self.generated_state = state['generated_state']
        self.tmp0 = state['tmp0']
        self.tmp2 = state['tmp2']
        self.tmp1 = state['tmp1']

    cdef void _reinitialize_temp_pointers(self) except *:
        self._tmp0 = &self.tmp0[0,0]
        self._tmp1 = &self.tmp1[0,0]
        self._tmp2 = &self.tmp2[0,0]


    cpdef draw_disturbance_variates(self):
        self.disturbance_variates = np.random.normal(size=self.n_disturbance_variates)
        self.pretransformed_disturbance_variates = False

    cpdef draw_initial_state_variates(self):
        self.initial_state_variates = np.random.normal(size=self.n_initial_state_variates)
        self.pretransformed_initial_state_variates = False
        self.fixed_initial_state = False

    cpdef set_disturbance_variates(self, np.float32_t [:] variates, int pretransformed=0):
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('disturbance variates', &variates.shape[0],
                                    self.n_disturbance_variates)
        self.disturbance_variates = variates
        self.pretransformed_disturbance_variates = pretransformed

    cpdef set_initial_state_variates(self, np.float32_t [:] variates, int pretransformed=0):
        # Note that the initial state is set to be:
        # initial_state = mod.initial_state + initial_state_variate * cholesky(mod.initial_state_cov)
        # so is can be difficult to set the initial state itself via this method;
        # see instead set_initial_state
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('initial state variates',
                                    &variates.shape[0],
                                    self.n_initial_state_variates)
        self.initial_state_variates = variates
        self.pretransformed_initial_state_variates = pretransformed
        self.fixed_initial_state = False

    cpdef set_initial_state(self, np.float32_t [:] initial_state):
        # Using this method sets a flag that indicates the self.initial_state_variates
        # variable should be interpreted as the actual initial_state.
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('initial state',
                                    &initial_state.shape[0],
                                    self.n_initial_state_variates)
        self.initial_state_variates = initial_state
        self.pretransformed_initial_state_variates = True
        self.fixed_initial_state = True

    cpdef simulate(self, int simulation_output=-1):
        """
        Draw a simulation
        """
        cdef:
            int inc = 1
            int info
            int measurement_idx, state_idx, t
            int k_endog = self.model.k_endog
            int k_states = self.model.k_states
            int k_states2 = self.model.k_states**2
            int k_posdef = self.model.k_posdef
            int k_posdef2 = self.model.k_posdef**2
            int nobs_endog = self.nobs * self.model.k_endog
            int nobs_kstates = self.nobs * self.model.k_states
            int nobs1_kstates = (self.nobs+1) * self.model.k_states
            int nobs_posdef = self.nobs * self.model.k_posdef
        cdef:
            np.float32_t alpha = 1.0
            np.float32_t gamma = -1.0
        cdef sStatespace collapsing_model
        cdef np.float32_t [:,:] collapsing_obs
        cdef np.npy_intp dim2[2]


        if simulation_output == -1:
            simulation_output = self.simulation_output
        
        # Forwards recursion
        # 0. Statespace initialization
        if not self.model.initialized:
            raise RuntimeError("Statespace model not initialized.")
        blas.scopy(&k_states, &self.model.initial_state[0], &inc, &self.simulated_model.initial_state[0], &inc)
        blas.scopy(&k_states2, &self.model.initial_state_cov[0,0], &inc, &self.simulated_model.initial_state_cov[0,0], &inc)
        blas.scopy(&k_states2, &self.model.initial_diffuse_state_cov[0,0], &inc, &self.simulated_model.initial_diffuse_state_cov[0,0], &inc)

        if self.has_missing:
            blas.scopy(&k_states, &self.model.initial_state[0], &inc, &self.secondary_simulated_model.initial_state[0], &inc)
            blas.scopy(&k_states2, &self.model.initial_state_cov[0,0], &inc, &self.secondary_simulated_model.initial_state_cov[0,0], &inc)
            blas.scopy(&k_states2, &self.model.initial_diffuse_state_cov[0,0], &inc, &self.secondary_simulated_model.initial_diffuse_state_cov[0,0], &inc)

        # 0. Kalman filter initialization: get alpha_1^+ ~ N(a_1, P_1)
        # Usually, this means transforming the N(0,1) random variate
        # into a N(initial_state, initial_state_cov) random variate.
        # alpha_1^+ = initial_state + variate * chol(initial_state_cov)
        # If pretransformed_variates is True, then the variates should already
        # be N(0, initial_state_cov), and then we just need:
        # alpha_1^+ = initial_state + variate
        # However, if fixed_initial_state is True, then we just set:
        # alpha_1^+ = variate
        blas.scopy(&k_states, &self.initial_state_variates[0], &inc, &self.generated_state[0,0], &inc)
        if not self.fixed_initial_state:
            self.cholesky(&self.model.initial_state_cov[0,0], self._tmp0, k_states)
            if not self.pretransformed_initial_state_variates:
                self.transform_variates(&self.generated_state[0,0], self._tmp0, k_states)
            # In the case of no missing data, we want to keep the initial state at zero
            if self.has_missing:
                blas.saxpy(&k_states, &alpha, &self.model.initial_state[0], &inc, &self.generated_state[0,0], &inc)


        self.simulated_kfilter.seek(0) # reset the filter
        if self.has_missing:
            self.secondary_simulated_kfilter.seek(0) # reset the filter
        measurement_idx = 0
        state_idx = nobs_endog
        # Note: it is only when we are simulating a series that we might have
        # self.model.nobs != self.simulated_model.nobs, and in that case we do
        # not need to copy over the observations.
        # TODO: Need more explicit support for the pure simulation case, both
        # to avoid errors (like the copy statement below) and because we can
        # probably speed up the process by not running the filter parts.
        if not self.has_missing and (self.model.nobs == self.simulated_model.nobs):
            # reset the obs data in the primary simulated model
            # (but only if there is not missing data - in that case we will
            # combine the actual data with the generated data in the primary
            # model, so copy the actual data here and subtract data below)
            blas.scopy(&nobs_endog, &self.model.obs[0,0], &inc, &self.simulated_model.obs[0,0], &inc)
        for t in range(self.nobs):
            # 1. Transform independent draws to w_t^+: eps_t^+ = ind_eps * chol(H_t)
            #                                          eta_t^+ = ind_eta * chol(Q_t)

            # 2. Construct y_t^+ = d_t + Z_t alpha_t^+ + eps_t^+
            #      alpha_{t+1}^+ = c_t + T_t alpha_t^+ + eta_t^+

            #    Measurement disturbance (eps)
            # self._tmp1 = chol(H_t)
            if t == 0 or self.model.obs_cov.shape[2] > 1:
                self.cholesky(&self.model.obs_cov[0,0,t], self._tmp1, k_endog)

            # eps_t^+ = ind_eps * chol(H_t)
            if not self.pretransformed_disturbance_variates:
                self.transform_variates(&self.disturbance_variates[measurement_idx], self._tmp1, k_endog)
            # y_t^+
            self.generate_obs(t, &self.generated_obs[0,t], &self.generated_state[0,t], &self.disturbance_variates[measurement_idx])

            measurement_idx += k_endog

            #    State disturbance (eta)
            # self._tmp1 = chol(Q_t)
            if t == 0 or self.model.state_cov.shape[2] > 1:
                self.cholesky(&self.model.state_cov[0,0,t], self._tmp2, k_posdef)

            # eta_t^+ = ind_eta * chol(Q_t)
            if not self.pretransformed_disturbance_variates:
                self.transform_variates(&self.disturbance_variates[state_idx], self._tmp2, k_posdef)
            # alpha_t+1^+
            self.generate_state(t, &self.generated_state[0,t+1], &self.generated_state[0,t], &self.disturbance_variates[state_idx])

            state_idx += k_posdef

            # If we are just generating new series (i.e. all we want is
            # generated_obs, generated_state), go to the next iteration
            if self.simulation_output == 0:
                continue

            # Typically, rather than running the Kalman filter separately for
            # y_t^+ and y_t, we can instead run it over y_t^* = y_t - y_t^+
            if not self.has_missing:
                #    Construct y_t^* = - y_t^+ + y_t
                blas.saxpy(&k_endog, &gamma, &self.generated_obs[0,t], &inc, &self.simulated_model.obs[0, t], &inc)

                # 3. Iterate Kalman filter, based on y_t^*
                #    (this will give us alpha_t+1^*)
                next(self.simulated_kfilter)
            # In the case of missing data, we have to run them separately
            else:
                # 3-1. Iterate the Kalman filter on the y_t^+ data
                #      to get alpha_t+1^+
                blas.scopy(&k_endog, &self.generated_obs[0,t], &inc, &self.simulated_model.obs[0, t], &inc)
                next(self.simulated_kfilter)

                # 3-2. Iterate the Kalman filter on the y_t data
                #      to get alpha_t+1
                next(self.secondary_simulated_kfilter)

        # If we are just generating new series (i.e. all we want is
        # generated_obs, generated_state), return now
        if self.simulation_output == 0:
            return

        # Backwards recursion
        # This gives us \hat w_t^* = \hat w_t - \hat w_t^+                    (simulation_output & SIMULATE_DISTURBANCE)
        #               \hat alpha_t+1^* = \hat alpha_t+1 - \hat alpha_t+1^+  (simulation_output & SIMULATE_STATE)
        # or if there is missing data:
        # this gives us \hat w_t^+
        #               \hat alpha_t+1
        # and we construct starred versions below
        self.simulated_smoother.smoother_output = simulation_output
        self.simulated_smoother()

        # In the FILTER_COLLAPSED case, we now need to replace nobs_endog with
        # nobs_kstates
        if self.simulated_kfilter.filter_method & FILTER_COLLAPSED:
            nobs_endog = nobs_kstates

        if self.has_missing:
            # This gives us \hat w_t
            #               \hat alpha_t+1
            self.secondary_simulated_smoother.smoother_output = simulation_output
            self.secondary_simulated_smoother()

            # Construct \hat w_t^* = \hat w_t - \hat w_t^+
            #           \hat alpha_t+1^* = \hat alpha_t+1 - \hat alpha_t+1^+
            # Note: this overwrites the values in self.simulated_smoother,
            # so that the steps below will be the same regardless of whether or
            # not there was missing data
            if self.simulation_output & SIMULATE_DISTURBANCE:
                # If there are partially missing entries, we need to re-order
                # the smoothed measurment disturbances.
                tools.sreorder_missing_vector(self.secondary_simulated_smoother.smoothed_measurement_disturbance, self.model.missing)
                blas.sswap(&nobs_endog, &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                 &self.secondary_simulated_smoother.smoothed_measurement_disturbance[0,0], &inc)
                blas.saxpy(&nobs_endog, &gamma, &self.secondary_simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                         &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc)
                blas.sswap(&nobs_posdef, &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                  &self.secondary_simulated_smoother.smoothed_state_disturbance[0,0], &inc)
                blas.saxpy(&nobs_posdef, &gamma, &self.secondary_simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                          &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc)

            if self.simulation_output & SIMULATE_STATE:
                blas.sswap(&nobs_kstates, &self.simulated_smoother.smoothed_state[0,0], &inc,
                                                   &self.secondary_simulated_smoother.smoothed_state[0,0], &inc)
                blas.saxpy(&nobs_kstates, &gamma, &self.secondary_simulated_smoother.smoothed_state[0,0], &inc,
                                                           &self.simulated_smoother.smoothed_state[0,0], &inc)

        # Construct the final simulated variables
        # This gives us \tilde w_t = \hat w_t^* + w_t^+                (simulation_output & SIMULATE_DISTURBANCE)
        #               \tilde alpha_t+1 = \hat alpha_t^* + alpha_t^+  (simulation_output & SIMULATE_STATE)
        if self.simulation_output & SIMULATE_DISTURBANCE:
            # \tilde eps_t = \hat eps_t^* + eps_t^+
            # In the FILTER_COLLAPSED case, we have to use a collapsed version of the disturbance variates,
            # so we construct a new representation that can perform the collapse on the variates
            if self.simulated_kfilter.filter_method & FILTER_COLLAPSED:
                dim2[0] = k_endog; dim2[1] = self.nobs;
                collapsing_obs = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT32, FORTRAN)
                nobs_endog = k_endog * self.nobs
                blas.scopy(&nobs_endog, &self.disturbance_variates[0], &inc, &collapsing_obs[0, 0], &inc)
                nobs_endog = nobs_kstates

                collapsing_model = sStatespace(
                    collapsing_obs,
                    self.model.design, self.model.obs_intercept, self.model.obs_cov,
                    self.model.transition, self.model.state_intercept, self.model.selection,
                    self.model.state_cov
                )
                collapsing_model.initialized = True
                for t in range(self.nobs):
                    collapsing_model.seek(t, self.simulated_kfilter.univariate_filter[t],
                                             self.simulated_kfilter.filter_method & FILTER_COLLAPSED)
                    blas.scopy(&k_states, &collapsing_model.collapse_obs[0], &inc, &self.simulated_measurement_disturbance[0,t], &inc)
            else:
                blas.scopy(&nobs_endog, &self.disturbance_variates[0], &inc, &self.simulated_measurement_disturbance[0,0], &inc)

            blas.saxpy(&nobs_endog, &alpha, &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                     &self.simulated_measurement_disturbance[0,0], &inc)

            # \tilde eta_t = \hat eta_t^* + eta_t^+
            blas.scopy(&nobs_posdef, &self.disturbance_variates[k_endog * self.nobs], &inc, &self.simulated_state_disturbance[0,0], &inc)
            blas.saxpy(&nobs_posdef, &alpha, &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                     &self.simulated_state_disturbance[0,0], &inc)

        if self.simulation_output & SIMULATE_STATE:
            # \tilde alpha_t = \hat alpha_t^* + alpha_t^+
            blas.scopy(&nobs_kstates, &self.generated_state[0,0], &inc, &self.simulated_state[0,0], &inc)
            blas.saxpy(&nobs_kstates, &alpha, &self.simulated_smoother.smoothed_state[0,0], &inc,
                                                       &self.simulated_state[0,0], &inc)

    cdef np.float32_t generate_obs(self, int t, np.float32_t * obs, np.float32_t * state, np.float32_t * variates):
        cdef:
            int inc = 1
            int k_endog = self.model.k_endog
            int k_states = self.model.k_states
            int design_t = 0
            int obs_intercept_t = 0
        cdef:
            np.float32_t alpha = 1.0

        # Get indices for possibly time-varying arrays
        # Note that this would be problematic if simulating a
        # time-varying system if the number of simulated observations was
        # greater than the number of observations in the model (because then
        # e.g. self.model.design[0,0,design_t] would be an invalid memory
        # location).
        # However, this problem is ruled out in the constructor since for
        # time-varying systems, the number of simulated observations must be
        # less than the number of observations in the model.
        if not self.model.time_invariant:
            if self.model.design.shape[2] > 1:             design_t = t
            if self.model.obs_intercept.shape[1] > 1:      obs_intercept_t = t

        # \\# = d_t + \varepsilon_t
        blas.scopy(&k_endog, variates, &inc, obs, &inc)
        blas.saxpy(&k_endog, &alpha, &self.model.obs_intercept[0,obs_intercept_t], &inc, obs, &inc)

        # y_t = \\# + Z_t alpha_t
        blas.sgemv("N", &k_endog, &k_states,
                            &alpha, &self.model.design[0,0,design_t], &k_endog,
                                    state, &inc,
                            &alpha, obs, &inc)

    cdef np.float32_t generate_state(self, int t, np.float32_t * state, np.float32_t * input_state, np.float32_t * variates):
        cdef:
            int inc = 1
            int k_states = self.model.k_states
            int k_posdef = self.model.k_posdef
            int state_intercept_t = 0
            int transition_t = 0
            int selection_t = 0
        cdef:
            np.float32_t alpha = 1.0

        # Get indices for possibly time-varying arrays
        if not self.model.time_invariant:
            if self.model.state_intercept.shape[1] > 1:      state_intercept_t = t
            if self.model.transition.shape[2] > 1:           transition_t = t
            if self.model.selection.shape[2] > 1:            selection_t = t

        # \\# = R_t eta_t + c_t
        blas.scopy(&k_states, &self.model.state_intercept[0,state_intercept_t], &inc, state, &inc)
        blas.sgemv("N", &k_states, &k_posdef,
                            &alpha, &self.model.selection[0,0,selection_t], &k_states,
                                    variates, &inc,
                            &alpha, state, &inc)

        # alpha_{t+1} = T_t alpha_t + \\#
        blas.sgemv("N", &k_states, &k_states,
                            &alpha, &self.model.transition[0,0,transition_t], &k_states,
                                    input_state, &inc,
                            &alpha, state, &inc)

    cdef void cholesky(self, np.float32_t * source, np.float32_t * destination, int n):
        cdef:
            int inc = 1
            int n2 = n**2
            int info
        if n == 1:
            destination[0] = source[0]**0.5
        else:
            blas.scopy(&n2, source, &inc, destination, &inc)
            lapack.spotrf("L", &n, destination, &n, &info)

    cdef void transform_variates(self, np.float32_t * variates, np.float32_t * cholesky_factor, int n):
        cdef:
            int inc = 1

        # Overwrites variate
        if n == 1:
            variates[0] = cholesky_factor[0] * variates[0]
        else:
            blas.strmv("L", "N", "N", &n, cholesky_factor, &n,
                                                   variates, &inc)

cdef class dSimulationSmoother(object):
    # ### Statespace model
    # cdef readonly dStatespace model
    # ### Kalman filter
    # cdef readonly dKalmanFilter kfilter
    # ### Kalman smoother
    # cdef readonly dKalmanSmoother smoother

    # ### Simulated Statespace model
    # cdef readonly dStatespace simulated_model
    # ### Simulated Kalman filter
    # cdef readonly dKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    # cdef readonly dKalmanSmoother simulated_smoother

    # ### Secondary Simulated Statespace model
    # Note: currently only used in the case of missing data
    # cdef readonly dStatespace secondary_simulated_model
    # ### Simulated Kalman filter
    # cdef readonly dKalmanFilter secondary_simulated_kfilter
    # ### Simulated Kalman smoother
    # cdef readonly dKalmanSmoother secondary_simulated_smoother

    # ### Simulation parameters
    # cdef public int simulation_output
    # cdef readonly int has_missing

    # ### Random variates
    # cdef int n_disturbance_variates
    # cdef readonly np.float64_t [:] disturbance_variates
    # cdef int n_initial_state_variates
    # cdef readonly np.float64_t [:] initial_state_variates

    # ### Simulated Data
    # cdef readonly np.float64_t [::1,:] simulated_measurement_disturbance
    # cdef readonly np.float64_t [::1,:] simulated_state_disturbance
    # cdef readonly np.float64_t [::1,:] simulated_state

    # ### Generated Data
    # cdef readonly np.float64_t [::1,:] generated_obs
    # cdef readonly np.float64_t [::1,:] generated_state

    # ### Temporary arrays
    # cdef readonly np.float64_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    # cdef np.float64_t * _tmp0
    # cdef np.float64_t * _tmp1
    # cdef np.float64_t * _tmp2

    def __init__(self,
                 dStatespace model,
                 int filter_method=FILTER_CONVENTIONAL,
                 int inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY,
                 int stability_method=STABILITY_FORCE_SYMMETRY,
                 int conserve_memory=MEMORY_STORE_ALL,
                 int filter_timing=TIMING_INIT_PREDICTED,
                 np.float64_t tolerance=1e-19,
                 int loglikelihood_burn=0,
                 int smoother_output=SMOOTHER_ALL,
                 int simulation_output=SIMULATE_ALL,
                 int nobs=-1):
        cdef int inc = 1
        cdef:
            np.npy_intp dim1[1]
            np.npy_intp dim2[2]
        cdef np.float64_t [::1,:] obs
        cdef np.float64_t [::1,:] secondary_obs
        cdef int nobs_endog

        # Use model nobs by default
        if nobs == -1:
            nobs = model.nobs
        # Only allow more nobs if a time-invariant model
        elif nobs > model.nobs and model.time_invariant == 0:
            raise ValueError('In a time-varying model, cannot create more'
                             ' simulations than there are observations.')
        elif nobs <= 0:
            raise ValueError('Invalid number of simulations; must be'
                             ' positive.')

        self.nobs = nobs
        nobs_endog = self.nobs * model.k_endog

        self.pretransformed_disturbance_variates = False
        self.pretransformed_initial_state_variates = False
        self.fixed_initial_state = False

        # Model objects
        self.model = model
        # self.kfilter = dKalmanFilter(
        #     self.model, filter_method, inversion_method,
        #     stability_method, conserve_memory,
        #     tolerance, loglikelihood_burn
        # )
        # self.smoother = dKalmanSmoother(
        #     self.model, self.kfilter, smoother_output
        # )

        # Simulated model objects
        dim2[0] = model.k_endog; dim2[1] = self.nobs;
        obs = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN)
        self.simulated_model = dStatespace(
            obs, model.design, model.obs_intercept, model.obs_cov,
            model.transition, model.state_intercept, model.selection,
            model.state_cov
        )
        self.simulated_kfilter = dKalmanFilter(
            self.simulated_model, filter_method, inversion_method,
            stability_method, conserve_memory, filter_timing,
            tolerance, loglikelihood_burn
        )
        self.simulated_smoother = dKalmanSmoother(
            self.simulated_model, self.simulated_kfilter, smoother_output
        )

        # Secondary simulated model objects
        # Currently only used if there is missing data (since then the
        # approach in which the Kalman filter only has to be run over the
        # series y_t^* = y_t - y_t^+ is infeasible), although it could also
        # allow drawing multiple samples at the same time, see Durbin and
        # Koopman (2002).
        self.has_missing = model.has_missing
        if self.has_missing:
            dim2[0] = model.k_endog; dim2[1] = self.nobs;
            secondary_obs = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN)
            blas.dcopy(&nobs_endog, &model.obs[0,0], &inc, &secondary_obs[0,0], &inc)
            self.secondary_simulated_model = dStatespace(
                secondary_obs, model.design, model.obs_intercept, model.obs_cov,
                model.transition, model.state_intercept, model.selection,
                model.state_cov
            )
            self.secondary_simulated_kfilter = dKalmanFilter(
                self.secondary_simulated_model, filter_method, inversion_method,
                stability_method, conserve_memory, filter_timing,
                tolerance, loglikelihood_burn
            )
            self.secondary_simulated_smoother = dKalmanSmoother(
                self.secondary_simulated_model, self.secondary_simulated_kfilter, smoother_output
            )
        # In the case of non-missing data, the Kalman filter will actually
        # be run over y_t^* = y_t - y_t^+, which means the observation equation
        # intercept should be zero; make sure that it is
        else:
            dim2[0] = self.model.k_endog; dim2[1] = self.model.obs_intercept.shape[1];
            self.simulated_model.obs_intercept = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN)
            dim2[0] = self.model.k_states; dim2[1] = self.model.state_intercept.shape[1];
            self.simulated_model.state_intercept = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN)


        # Initialize the simulated model memoryviews
        # Note: the actual initialization is replaced in the simulate()
        # function below, but will complain if the memoryviews have not been
        # first initialized, which this call does.
        self.simulated_model.initialize_approximate_diffuse()
        if self.has_missing:
            self.secondary_simulated_model.initialize_approximate_diffuse()

        # Parameters
        self.simulation_output = simulation_output
        self.n_disturbance_variates = self.nobs * (self.model.k_endog + self.model.k_posdef)
        self.n_initial_state_variates = self.model.k_states

        # Random variates
        dim1[0] = self.n_disturbance_variates;
        self.disturbance_variates = np.PyArray_ZEROS(1, dim1, np.NPY_FLOAT64, FORTRAN)
        dim1[0] = self.n_initial_state_variates;
        self.initial_state_variates = np.PyArray_ZEROS(1, dim1, np.NPY_FLOAT64, FORTRAN)

        # Simulated data (\tilde eta_t, \tilde eps_t, \tilde alpha_t)
        # Note that these are (k_endog x nobs), (k_posdef x nobs), (k_states x nobs)
        if filter_method & FILTER_COLLAPSED:
            dim2[0] = self.model.k_states; dim2[1] = self.nobs;
        else:
            dim2[0] = self.model.k_endog; dim2[1] = self.nobs;
        self.simulated_measurement_disturbance = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN)
        dim2[0] = self.model.k_posdef; dim2[1] = self.nobs;
        self.simulated_state_disturbance = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN)
        dim2[0] = self.model.k_states; dim2[1] = self.nobs;
        self.simulated_state = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN)

        # Generated data (y_t^+, alpha_t^+)
        dim2[0] = self.model.k_endog; dim2[1] = self.nobs;
        self.generated_obs = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN)
        dim2[0] = self.model.k_states; dim2[1] = self.nobs+1;
        self.generated_state = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN)

        # Temporary arrays
        dim2[0] = self.model.k_states; dim2[1] = self.model.k_states;
        self.tmp0 = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN) # chol(P_1)
        dim2[0] = self.model.k_posdef; dim2[1] = self.model.k_posdef;
        self.tmp2 = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN) # chol(Q_t)
        dim2[0] = self.model.k_endog; dim2[1] = self.model.k_endog;
        self.tmp1 = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN) # chol(H_t)

        # Pointers
        self._tmp0 = &self.tmp0[0,0]
        self._tmp1 = &self.tmp1[0,0]
        self._tmp2 = &self.tmp2[0,0]

    def __reduce__(self):
        args = (self.model, self.model.filter_method, self.model.inversion_method,
                self.model.stability_method, self.model.conserve_memory, self.model.filter_timing,
                self.model.tolerance, self.model.loglikelihood_burn, self.simulated_smoother.smoother_output,
                self.simulation_output, self.nobs, self.pretransformed_variates)
        state = {
            'disturbance_variates': np.array(self.disturbance_variates, copy=True, order='F'),
            'initial_state_variates': np.array(self.initial_state_variates, copy=True, order='F'),
            'simulated_measurement_disturbance': np.array(self.simulated_measurement_disturbance, copy=True, order='F'),
            'simulated_state_disturbance': np.array(self.simulated_state_disturbance, copy=True, order='F'),
            'simulated_state': np.array(self.simulated_state, copy=True, order='F'),
            'generated_obs': np.array(self.generated_obs, copy=True, order='F'),
            'generated_state': np.array(self.generated_state, copy=True, order='F'),
            'tmp0': np.array(self.tmp0, copy=True, order='F'),
            'tmp2': np.array(self.tmp2, copy=True, order='F'),
            'tmp1': np.array(self.tmp1, copy=True, order='F')
        }
        return (self.__class__, args, state)

    def __setstate__(self, state):
        self.disturbance_variates = state['disturbance_variates']
        self.initial_state_variates = state['initial_state_variates']
        self.simulated_measurement_disturbance  = state['simulated_measurement_disturbance']
        self.simulated_state_disturbance = state['simulated_state_disturbance']
        self.simulated_state = state['simulated_state']
        self.generated_obs = state['generated_obs']
        self.generated_state = state['generated_state']
        self.tmp0 = state['tmp0']
        self.tmp2 = state['tmp2']
        self.tmp1 = state['tmp1']

    cdef void _reinitialize_temp_pointers(self) except *:
        self._tmp0 = &self.tmp0[0,0]
        self._tmp1 = &self.tmp1[0,0]
        self._tmp2 = &self.tmp2[0,0]


    cpdef draw_disturbance_variates(self):
        self.disturbance_variates = np.random.normal(size=self.n_disturbance_variates)
        self.pretransformed_disturbance_variates = False

    cpdef draw_initial_state_variates(self):
        self.initial_state_variates = np.random.normal(size=self.n_initial_state_variates)
        self.pretransformed_initial_state_variates = False
        self.fixed_initial_state = False

    cpdef set_disturbance_variates(self, np.float64_t [:] variates, int pretransformed=0):
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('disturbance variates', &variates.shape[0],
                                    self.n_disturbance_variates)
        self.disturbance_variates = variates
        self.pretransformed_disturbance_variates = pretransformed

    cpdef set_initial_state_variates(self, np.float64_t [:] variates, int pretransformed=0):
        # Note that the initial state is set to be:
        # initial_state = mod.initial_state + initial_state_variate * cholesky(mod.initial_state_cov)
        # so is can be difficult to set the initial state itself via this method;
        # see instead set_initial_state
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('initial state variates',
                                    &variates.shape[0],
                                    self.n_initial_state_variates)
        self.initial_state_variates = variates
        self.pretransformed_initial_state_variates = pretransformed
        self.fixed_initial_state = False

    cpdef set_initial_state(self, np.float64_t [:] initial_state):
        # Using this method sets a flag that indicates the self.initial_state_variates
        # variable should be interpreted as the actual initial_state.
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('initial state',
                                    &initial_state.shape[0],
                                    self.n_initial_state_variates)
        self.initial_state_variates = initial_state
        self.pretransformed_initial_state_variates = True
        self.fixed_initial_state = True

    cpdef simulate(self, int simulation_output=-1):
        """
        Draw a simulation
        """
        cdef:
            int inc = 1
            int info
            int measurement_idx, state_idx, t
            int k_endog = self.model.k_endog
            int k_states = self.model.k_states
            int k_states2 = self.model.k_states**2
            int k_posdef = self.model.k_posdef
            int k_posdef2 = self.model.k_posdef**2
            int nobs_endog = self.nobs * self.model.k_endog
            int nobs_kstates = self.nobs * self.model.k_states
            int nobs1_kstates = (self.nobs+1) * self.model.k_states
            int nobs_posdef = self.nobs * self.model.k_posdef
        cdef:
            np.float64_t alpha = 1.0
            np.float64_t gamma = -1.0
        cdef dStatespace collapsing_model
        cdef np.float64_t [:,:] collapsing_obs
        cdef np.npy_intp dim2[2]


        if simulation_output == -1:
            simulation_output = self.simulation_output
        
        # Forwards recursion
        # 0. Statespace initialization
        if not self.model.initialized:
            raise RuntimeError("Statespace model not initialized.")
        blas.dcopy(&k_states, &self.model.initial_state[0], &inc, &self.simulated_model.initial_state[0], &inc)
        blas.dcopy(&k_states2, &self.model.initial_state_cov[0,0], &inc, &self.simulated_model.initial_state_cov[0,0], &inc)
        blas.dcopy(&k_states2, &self.model.initial_diffuse_state_cov[0,0], &inc, &self.simulated_model.initial_diffuse_state_cov[0,0], &inc)

        if self.has_missing:
            blas.dcopy(&k_states, &self.model.initial_state[0], &inc, &self.secondary_simulated_model.initial_state[0], &inc)
            blas.dcopy(&k_states2, &self.model.initial_state_cov[0,0], &inc, &self.secondary_simulated_model.initial_state_cov[0,0], &inc)
            blas.dcopy(&k_states2, &self.model.initial_diffuse_state_cov[0,0], &inc, &self.secondary_simulated_model.initial_diffuse_state_cov[0,0], &inc)

        # 0. Kalman filter initialization: get alpha_1^+ ~ N(a_1, P_1)
        # Usually, this means transforming the N(0,1) random variate
        # into a N(initial_state, initial_state_cov) random variate.
        # alpha_1^+ = initial_state + variate * chol(initial_state_cov)
        # If pretransformed_variates is True, then the variates should already
        # be N(0, initial_state_cov), and then we just need:
        # alpha_1^+ = initial_state + variate
        # However, if fixed_initial_state is True, then we just set:
        # alpha_1^+ = variate
        blas.dcopy(&k_states, &self.initial_state_variates[0], &inc, &self.generated_state[0,0], &inc)
        if not self.fixed_initial_state:
            self.cholesky(&self.model.initial_state_cov[0,0], self._tmp0, k_states)
            if not self.pretransformed_initial_state_variates:
                self.transform_variates(&self.generated_state[0,0], self._tmp0, k_states)
            # In the case of no missing data, we want to keep the initial state at zero
            if self.has_missing:
                blas.daxpy(&k_states, &alpha, &self.model.initial_state[0], &inc, &self.generated_state[0,0], &inc)


        self.simulated_kfilter.seek(0) # reset the filter
        if self.has_missing:
            self.secondary_simulated_kfilter.seek(0) # reset the filter
        measurement_idx = 0
        state_idx = nobs_endog
        # Note: it is only when we are simulating a series that we might have
        # self.model.nobs != self.simulated_model.nobs, and in that case we do
        # not need to copy over the observations.
        # TODO: Need more explicit support for the pure simulation case, both
        # to avoid errors (like the copy statement below) and because we can
        # probably speed up the process by not running the filter parts.
        if not self.has_missing and (self.model.nobs == self.simulated_model.nobs):
            # reset the obs data in the primary simulated model
            # (but only if there is not missing data - in that case we will
            # combine the actual data with the generated data in the primary
            # model, so copy the actual data here and subtract data below)
            blas.dcopy(&nobs_endog, &self.model.obs[0,0], &inc, &self.simulated_model.obs[0,0], &inc)
        for t in range(self.nobs):
            # 1. Transform independent draws to w_t^+: eps_t^+ = ind_eps * chol(H_t)
            #                                          eta_t^+ = ind_eta * chol(Q_t)

            # 2. Construct y_t^+ = d_t + Z_t alpha_t^+ + eps_t^+
            #      alpha_{t+1}^+ = c_t + T_t alpha_t^+ + eta_t^+

            #    Measurement disturbance (eps)
            # self._tmp1 = chol(H_t)
            if t == 0 or self.model.obs_cov.shape[2] > 1:
                self.cholesky(&self.model.obs_cov[0,0,t], self._tmp1, k_endog)

            # eps_t^+ = ind_eps * chol(H_t)
            if not self.pretransformed_disturbance_variates:
                self.transform_variates(&self.disturbance_variates[measurement_idx], self._tmp1, k_endog)
            # y_t^+
            self.generate_obs(t, &self.generated_obs[0,t], &self.generated_state[0,t], &self.disturbance_variates[measurement_idx])

            measurement_idx += k_endog

            #    State disturbance (eta)
            # self._tmp1 = chol(Q_t)
            if t == 0 or self.model.state_cov.shape[2] > 1:
                self.cholesky(&self.model.state_cov[0,0,t], self._tmp2, k_posdef)

            # eta_t^+ = ind_eta * chol(Q_t)
            if not self.pretransformed_disturbance_variates:
                self.transform_variates(&self.disturbance_variates[state_idx], self._tmp2, k_posdef)
            # alpha_t+1^+
            self.generate_state(t, &self.generated_state[0,t+1], &self.generated_state[0,t], &self.disturbance_variates[state_idx])

            state_idx += k_posdef

            # If we are just generating new series (i.e. all we want is
            # generated_obs, generated_state), go to the next iteration
            if self.simulation_output == 0:
                continue

            # Typically, rather than running the Kalman filter separately for
            # y_t^+ and y_t, we can instead run it over y_t^* = y_t - y_t^+
            if not self.has_missing:
                #    Construct y_t^* = - y_t^+ + y_t
                blas.daxpy(&k_endog, &gamma, &self.generated_obs[0,t], &inc, &self.simulated_model.obs[0, t], &inc)

                # 3. Iterate Kalman filter, based on y_t^*
                #    (this will give us alpha_t+1^*)
                next(self.simulated_kfilter)
            # In the case of missing data, we have to run them separately
            else:
                # 3-1. Iterate the Kalman filter on the y_t^+ data
                #      to get alpha_t+1^+
                blas.dcopy(&k_endog, &self.generated_obs[0,t], &inc, &self.simulated_model.obs[0, t], &inc)
                next(self.simulated_kfilter)

                # 3-2. Iterate the Kalman filter on the y_t data
                #      to get alpha_t+1
                next(self.secondary_simulated_kfilter)

        # If we are just generating new series (i.e. all we want is
        # generated_obs, generated_state), return now
        if self.simulation_output == 0:
            return

        # Backwards recursion
        # This gives us \hat w_t^* = \hat w_t - \hat w_t^+                    (simulation_output & SIMULATE_DISTURBANCE)
        #               \hat alpha_t+1^* = \hat alpha_t+1 - \hat alpha_t+1^+  (simulation_output & SIMULATE_STATE)
        # or if there is missing data:
        # this gives us \hat w_t^+
        #               \hat alpha_t+1
        # and we construct starred versions below
        self.simulated_smoother.smoother_output = simulation_output
        self.simulated_smoother()

        # In the FILTER_COLLAPSED case, we now need to replace nobs_endog with
        # nobs_kstates
        if self.simulated_kfilter.filter_method & FILTER_COLLAPSED:
            nobs_endog = nobs_kstates

        if self.has_missing:
            # This gives us \hat w_t
            #               \hat alpha_t+1
            self.secondary_simulated_smoother.smoother_output = simulation_output
            self.secondary_simulated_smoother()

            # Construct \hat w_t^* = \hat w_t - \hat w_t^+
            #           \hat alpha_t+1^* = \hat alpha_t+1 - \hat alpha_t+1^+
            # Note: this overwrites the values in self.simulated_smoother,
            # so that the steps below will be the same regardless of whether or
            # not there was missing data
            if self.simulation_output & SIMULATE_DISTURBANCE:
                # If there are partially missing entries, we need to re-order
                # the smoothed measurment disturbances.
                tools.dreorder_missing_vector(self.secondary_simulated_smoother.smoothed_measurement_disturbance, self.model.missing)
                blas.dswap(&nobs_endog, &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                 &self.secondary_simulated_smoother.smoothed_measurement_disturbance[0,0], &inc)
                blas.daxpy(&nobs_endog, &gamma, &self.secondary_simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                         &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc)
                blas.dswap(&nobs_posdef, &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                  &self.secondary_simulated_smoother.smoothed_state_disturbance[0,0], &inc)
                blas.daxpy(&nobs_posdef, &gamma, &self.secondary_simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                          &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc)

            if self.simulation_output & SIMULATE_STATE:
                blas.dswap(&nobs_kstates, &self.simulated_smoother.smoothed_state[0,0], &inc,
                                                   &self.secondary_simulated_smoother.smoothed_state[0,0], &inc)
                blas.daxpy(&nobs_kstates, &gamma, &self.secondary_simulated_smoother.smoothed_state[0,0], &inc,
                                                           &self.simulated_smoother.smoothed_state[0,0], &inc)

        # Construct the final simulated variables
        # This gives us \tilde w_t = \hat w_t^* + w_t^+                (simulation_output & SIMULATE_DISTURBANCE)
        #               \tilde alpha_t+1 = \hat alpha_t^* + alpha_t^+  (simulation_output & SIMULATE_STATE)
        if self.simulation_output & SIMULATE_DISTURBANCE:
            # \tilde eps_t = \hat eps_t^* + eps_t^+
            # In the FILTER_COLLAPSED case, we have to use a collapsed version of the disturbance variates,
            # so we construct a new representation that can perform the collapse on the variates
            if self.simulated_kfilter.filter_method & FILTER_COLLAPSED:
                dim2[0] = k_endog; dim2[1] = self.nobs;
                collapsing_obs = np.PyArray_ZEROS(2, dim2, np.NPY_FLOAT64, FORTRAN)
                nobs_endog = k_endog * self.nobs
                blas.dcopy(&nobs_endog, &self.disturbance_variates[0], &inc, &collapsing_obs[0, 0], &inc)
                nobs_endog = nobs_kstates

                collapsing_model = dStatespace(
                    collapsing_obs,
                    self.model.design, self.model.obs_intercept, self.model.obs_cov,
                    self.model.transition, self.model.state_intercept, self.model.selection,
                    self.model.state_cov
                )
                collapsing_model.initialized = True
                for t in range(self.nobs):
                    collapsing_model.seek(t, self.simulated_kfilter.univariate_filter[t],
                                             self.simulated_kfilter.filter_method & FILTER_COLLAPSED)
                    blas.dcopy(&k_states, &collapsing_model.collapse_obs[0], &inc, &self.simulated_measurement_disturbance[0,t], &inc)
            else:
                blas.dcopy(&nobs_endog, &self.disturbance_variates[0], &inc, &self.simulated_measurement_disturbance[0,0], &inc)

            blas.daxpy(&nobs_endog, &alpha, &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                     &self.simulated_measurement_disturbance[0,0], &inc)

            # \tilde eta_t = \hat eta_t^* + eta_t^+
            blas.dcopy(&nobs_posdef, &self.disturbance_variates[k_endog * self.nobs], &inc, &self.simulated_state_disturbance[0,0], &inc)
            blas.daxpy(&nobs_posdef, &alpha, &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                     &self.simulated_state_disturbance[0,0], &inc)

        if self.simulation_output & SIMULATE_STATE:
            # \tilde alpha_t = \hat alpha_t^* + alpha_t^+
            blas.dcopy(&nobs_kstates, &self.generated_state[0,0], &inc, &self.simulated_state[0,0], &inc)
            blas.daxpy(&nobs_kstates, &alpha, &self.simulated_smoother.smoothed_state[0,0], &inc,
                                                       &self.simulated_state[0,0], &inc)

    cdef np.float64_t generate_obs(self, int t, np.float64_t * obs, np.float64_t * state, np.float64_t * variates):
        cdef:
            int inc = 1
            int k_endog = self.model.k_endog
            int k_states = self.model.k_states
            int design_t = 0
            int obs_intercept_t = 0
        cdef:
            np.float64_t alpha = 1.0

        # Get indices for possibly time-varying arrays
        # Note that this would be problematic if simulating a
        # time-varying system if the number of simulated observations was
        # greater than the number of observations in the model (because then
        # e.g. self.model.design[0,0,design_t] would be an invalid memory
        # location).
        # However, this problem is ruled out in the constructor since for
        # time-varying systems, the number of simulated observations must be
        # less than the number of observations in the model.
        if not self.model.time_invariant:
            if self.model.design.shape[2] > 1:             design_t = t
            if self.model.obs_intercept.shape[1] > 1:      obs_intercept_t = t

        # \\# = d_t + \varepsilon_t
        blas.dcopy(&k_endog, variates, &inc, obs, &inc)
        blas.daxpy(&k_endog, &alpha, &self.model.obs_intercept[0,obs_intercept_t], &inc, obs, &inc)

        # y_t = \\# + Z_t alpha_t
        blas.dgemv("N", &k_endog, &k_states,
                            &alpha, &self.model.design[0,0,design_t], &k_endog,
                                    state, &inc,
                            &alpha, obs, &inc)

    cdef np.float64_t generate_state(self, int t, np.float64_t * state, np.float64_t * input_state, np.float64_t * variates):
        cdef:
            int inc = 1
            int k_states = self.model.k_states
            int k_posdef = self.model.k_posdef
            int state_intercept_t = 0
            int transition_t = 0
            int selection_t = 0
        cdef:
            np.float64_t alpha = 1.0

        # Get indices for possibly time-varying arrays
        if not self.model.time_invariant:
            if self.model.state_intercept.shape[1] > 1:      state_intercept_t = t
            if self.model.transition.shape[2] > 1:           transition_t = t
            if self.model.selection.shape[2] > 1:            selection_t = t

        # \\# = R_t eta_t + c_t
        blas.dcopy(&k_states, &self.model.state_intercept[0,state_intercept_t], &inc, state, &inc)
        blas.dgemv("N", &k_states, &k_posdef,
                            &alpha, &self.model.selection[0,0,selection_t], &k_states,
                                    variates, &inc,
                            &alpha, state, &inc)

        # alpha_{t+1} = T_t alpha_t + \\#
        blas.dgemv("N", &k_states, &k_states,
                            &alpha, &self.model.transition[0,0,transition_t], &k_states,
                                    input_state, &inc,
                            &alpha, state, &inc)

    cdef void cholesky(self, np.float64_t * source, np.float64_t * destination, int n):
        cdef:
            int inc = 1
            int n2 = n**2
            int info
        if n == 1:
            destination[0] = source[0]**0.5
        else:
            blas.dcopy(&n2, source, &inc, destination, &inc)
            lapack.dpotrf("L", &n, destination, &n, &info)

    cdef void transform_variates(self, np.float64_t * variates, np.float64_t * cholesky_factor, int n):
        cdef:
            int inc = 1

        # Overwrites variate
        if n == 1:
            variates[0] = cholesky_factor[0] * variates[0]
        else:
            blas.dtrmv("L", "N", "N", &n, cholesky_factor, &n,
                                                   variates, &inc)

cdef class cSimulationSmoother(object):
    # ### Statespace model
    # cdef readonly cStatespace model
    # ### Kalman filter
    # cdef readonly cKalmanFilter kfilter
    # ### Kalman smoother
    # cdef readonly cKalmanSmoother smoother

    # ### Simulated Statespace model
    # cdef readonly cStatespace simulated_model
    # ### Simulated Kalman filter
    # cdef readonly cKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    # cdef readonly cKalmanSmoother simulated_smoother

    # ### Secondary Simulated Statespace model
    # Note: currently only used in the case of missing data
    # cdef readonly cStatespace secondary_simulated_model
    # ### Simulated Kalman filter
    # cdef readonly cKalmanFilter secondary_simulated_kfilter
    # ### Simulated Kalman smoother
    # cdef readonly cKalmanSmoother secondary_simulated_smoother

    # ### Simulation parameters
    # cdef public int simulation_output
    # cdef readonly int has_missing

    # ### Random variates
    # cdef int n_disturbance_variates
    # cdef readonly np.complex64_t [:] disturbance_variates
    # cdef int n_initial_state_variates
    # cdef readonly np.complex64_t [:] initial_state_variates

    # ### Simulated Data
    # cdef readonly np.complex64_t [::1,:] simulated_measurement_disturbance
    # cdef readonly np.complex64_t [::1,:] simulated_state_disturbance
    # cdef readonly np.complex64_t [::1,:] simulated_state

    # ### Generated Data
    # cdef readonly np.complex64_t [::1,:] generated_obs
    # cdef readonly np.complex64_t [::1,:] generated_state

    # ### Temporary arrays
    # cdef readonly np.complex64_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    # cdef np.complex64_t * _tmp0
    # cdef np.complex64_t * _tmp1
    # cdef np.complex64_t * _tmp2

    def __init__(self,
                 cStatespace model,
                 int filter_method=FILTER_CONVENTIONAL,
                 int inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY,
                 int stability_method=STABILITY_FORCE_SYMMETRY,
                 int conserve_memory=MEMORY_STORE_ALL,
                 int filter_timing=TIMING_INIT_PREDICTED,
                 np.float64_t tolerance=1e-19,
                 int loglikelihood_burn=0,
                 int smoother_output=SMOOTHER_ALL,
                 int simulation_output=SIMULATE_ALL,
                 int nobs=-1):
        cdef int inc = 1
        cdef:
            np.npy_intp dim1[1]
            np.npy_intp dim2[2]
        cdef np.complex64_t [::1,:] obs
        cdef np.complex64_t [::1,:] secondary_obs
        cdef int nobs_endog

        # Use model nobs by default
        if nobs == -1:
            nobs = model.nobs
        # Only allow more nobs if a time-invariant model
        elif nobs > model.nobs and model.time_invariant == 0:
            raise ValueError('In a time-varying model, cannot create more'
                             ' simulations than there are observations.')
        elif nobs <= 0:
            raise ValueError('Invalid number of simulations; must be'
                             ' positive.')

        self.nobs = nobs
        nobs_endog = self.nobs * model.k_endog

        self.pretransformed_disturbance_variates = False
        self.pretransformed_initial_state_variates = False
        self.fixed_initial_state = False

        # Model objects
        self.model = model
        # self.kfilter = cKalmanFilter(
        #     self.model, filter_method, inversion_method,
        #     stability_method, conserve_memory,
        #     tolerance, loglikelihood_burn
        # )
        # self.smoother = cKalmanSmoother(
        #     self.model, self.kfilter, smoother_output
        # )

        # Simulated model objects
        dim2[0] = model.k_endog; dim2[1] = self.nobs;
        obs = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN)
        self.simulated_model = cStatespace(
            obs, model.design, model.obs_intercept, model.obs_cov,
            model.transition, model.state_intercept, model.selection,
            model.state_cov
        )
        self.simulated_kfilter = cKalmanFilter(
            self.simulated_model, filter_method, inversion_method,
            stability_method, conserve_memory, filter_timing,
            tolerance, loglikelihood_burn
        )
        self.simulated_smoother = cKalmanSmoother(
            self.simulated_model, self.simulated_kfilter, smoother_output
        )

        # Secondary simulated model objects
        # Currently only used if there is missing data (since then the
        # approach in which the Kalman filter only has to be run over the
        # series y_t^* = y_t - y_t^+ is infeasible), although it could also
        # allow drawing multiple samples at the same time, see Durbin and
        # Koopman (2002).
        self.has_missing = model.has_missing
        if self.has_missing:
            dim2[0] = model.k_endog; dim2[1] = self.nobs;
            secondary_obs = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN)
            blas.ccopy(&nobs_endog, &model.obs[0,0], &inc, &secondary_obs[0,0], &inc)
            self.secondary_simulated_model = cStatespace(
                secondary_obs, model.design, model.obs_intercept, model.obs_cov,
                model.transition, model.state_intercept, model.selection,
                model.state_cov
            )
            self.secondary_simulated_kfilter = cKalmanFilter(
                self.secondary_simulated_model, filter_method, inversion_method,
                stability_method, conserve_memory, filter_timing,
                tolerance, loglikelihood_burn
            )
            self.secondary_simulated_smoother = cKalmanSmoother(
                self.secondary_simulated_model, self.secondary_simulated_kfilter, smoother_output
            )
        # In the case of non-missing data, the Kalman filter will actually
        # be run over y_t^* = y_t - y_t^+, which means the observation equation
        # intercept should be zero; make sure that it is
        else:
            dim2[0] = self.model.k_endog; dim2[1] = self.model.obs_intercept.shape[1];
            self.simulated_model.obs_intercept = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN)
            dim2[0] = self.model.k_states; dim2[1] = self.model.state_intercept.shape[1];
            self.simulated_model.state_intercept = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN)


        # Initialize the simulated model memoryviews
        # Note: the actual initialization is replaced in the simulate()
        # function below, but will complain if the memoryviews have not been
        # first initialized, which this call does.
        self.simulated_model.initialize_approximate_diffuse()
        if self.has_missing:
            self.secondary_simulated_model.initialize_approximate_diffuse()

        # Parameters
        self.simulation_output = simulation_output
        self.n_disturbance_variates = self.nobs * (self.model.k_endog + self.model.k_posdef)
        self.n_initial_state_variates = self.model.k_states

        # Random variates
        dim1[0] = self.n_disturbance_variates;
        self.disturbance_variates = np.PyArray_ZEROS(1, dim1, np.NPY_COMPLEX64, FORTRAN)
        dim1[0] = self.n_initial_state_variates;
        self.initial_state_variates = np.PyArray_ZEROS(1, dim1, np.NPY_COMPLEX64, FORTRAN)

        # Simulated data (\tilde eta_t, \tilde eps_t, \tilde alpha_t)
        # Note that these are (k_endog x nobs), (k_posdef x nobs), (k_states x nobs)
        if filter_method & FILTER_COLLAPSED:
            dim2[0] = self.model.k_states; dim2[1] = self.nobs;
        else:
            dim2[0] = self.model.k_endog; dim2[1] = self.nobs;
        self.simulated_measurement_disturbance = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN)
        dim2[0] = self.model.k_posdef; dim2[1] = self.nobs;
        self.simulated_state_disturbance = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN)
        dim2[0] = self.model.k_states; dim2[1] = self.nobs;
        self.simulated_state = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN)

        # Generated data (y_t^+, alpha_t^+)
        dim2[0] = self.model.k_endog; dim2[1] = self.nobs;
        self.generated_obs = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN)
        dim2[0] = self.model.k_states; dim2[1] = self.nobs+1;
        self.generated_state = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN)

        # Temporary arrays
        dim2[0] = self.model.k_states; dim2[1] = self.model.k_states;
        self.tmp0 = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN) # chol(P_1)
        dim2[0] = self.model.k_posdef; dim2[1] = self.model.k_posdef;
        self.tmp2 = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN) # chol(Q_t)
        dim2[0] = self.model.k_endog; dim2[1] = self.model.k_endog;
        self.tmp1 = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN) # chol(H_t)

        # Pointers
        self._tmp0 = &self.tmp0[0,0]
        self._tmp1 = &self.tmp1[0,0]
        self._tmp2 = &self.tmp2[0,0]

    def __reduce__(self):
        args = (self.model, self.model.filter_method, self.model.inversion_method,
                self.model.stability_method, self.model.conserve_memory, self.model.filter_timing,
                self.model.tolerance, self.model.loglikelihood_burn, self.simulated_smoother.smoother_output,
                self.simulation_output, self.nobs, self.pretransformed_variates)
        state = {
            'disturbance_variates': np.array(self.disturbance_variates, copy=True, order='F'),
            'initial_state_variates': np.array(self.initial_state_variates, copy=True, order='F'),
            'simulated_measurement_disturbance': np.array(self.simulated_measurement_disturbance, copy=True, order='F'),
            'simulated_state_disturbance': np.array(self.simulated_state_disturbance, copy=True, order='F'),
            'simulated_state': np.array(self.simulated_state, copy=True, order='F'),
            'generated_obs': np.array(self.generated_obs, copy=True, order='F'),
            'generated_state': np.array(self.generated_state, copy=True, order='F'),
            'tmp0': np.array(self.tmp0, copy=True, order='F'),
            'tmp2': np.array(self.tmp2, copy=True, order='F'),
            'tmp1': np.array(self.tmp1, copy=True, order='F')
        }
        return (self.__class__, args, state)

    def __setstate__(self, state):
        self.disturbance_variates = state['disturbance_variates']
        self.initial_state_variates = state['initial_state_variates']
        self.simulated_measurement_disturbance  = state['simulated_measurement_disturbance']
        self.simulated_state_disturbance = state['simulated_state_disturbance']
        self.simulated_state = state['simulated_state']
        self.generated_obs = state['generated_obs']
        self.generated_state = state['generated_state']
        self.tmp0 = state['tmp0']
        self.tmp2 = state['tmp2']
        self.tmp1 = state['tmp1']

    cdef void _reinitialize_temp_pointers(self) except *:
        self._tmp0 = &self.tmp0[0,0]
        self._tmp1 = &self.tmp1[0,0]
        self._tmp2 = &self.tmp2[0,0]


    cpdef draw_disturbance_variates(self):
        self.disturbance_variates = np.random.normal(size=self.n_disturbance_variates)
        self.pretransformed_disturbance_variates = False

    cpdef draw_initial_state_variates(self):
        self.initial_state_variates = np.random.normal(size=self.n_initial_state_variates)
        self.pretransformed_initial_state_variates = False
        self.fixed_initial_state = False

    cpdef set_disturbance_variates(self, np.complex64_t [:] variates, int pretransformed=0):
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('disturbance variates', &variates.shape[0],
                                    self.n_disturbance_variates)
        self.disturbance_variates = variates
        self.pretransformed_disturbance_variates = pretransformed

    cpdef set_initial_state_variates(self, np.complex64_t [:] variates, int pretransformed=0):
        # Note that the initial state is set to be:
        # initial_state = mod.initial_state + initial_state_variate * cholesky(mod.initial_state_cov)
        # so is can be difficult to set the initial state itself via this method;
        # see instead set_initial_state
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('initial state variates',
                                    &variates.shape[0],
                                    self.n_initial_state_variates)
        self.initial_state_variates = variates
        self.pretransformed_initial_state_variates = pretransformed
        self.fixed_initial_state = False

    cpdef set_initial_state(self, np.complex64_t [:] initial_state):
        # Using this method sets a flag that indicates the self.initial_state_variates
        # variable should be interpreted as the actual initial_state.
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('initial state',
                                    &initial_state.shape[0],
                                    self.n_initial_state_variates)
        self.initial_state_variates = initial_state
        self.pretransformed_initial_state_variates = True
        self.fixed_initial_state = True

    cpdef simulate(self, int simulation_output=-1):
        """
        Draw a simulation
        """
        cdef:
            int inc = 1
            int info
            int measurement_idx, state_idx, t
            int k_endog = self.model.k_endog
            int k_states = self.model.k_states
            int k_states2 = self.model.k_states**2
            int k_posdef = self.model.k_posdef
            int k_posdef2 = self.model.k_posdef**2
            int nobs_endog = self.nobs * self.model.k_endog
            int nobs_kstates = self.nobs * self.model.k_states
            int nobs1_kstates = (self.nobs+1) * self.model.k_states
            int nobs_posdef = self.nobs * self.model.k_posdef
        cdef:
            np.complex64_t alpha = 1.0
            np.complex64_t gamma = -1.0
        cdef cStatespace collapsing_model
        cdef np.complex64_t [:,:] collapsing_obs
        cdef np.npy_intp dim2[2]


        if simulation_output == -1:
            simulation_output = self.simulation_output
        
        # Forwards recursion
        # 0. Statespace initialization
        if not self.model.initialized:
            raise RuntimeError("Statespace model not initialized.")
        blas.ccopy(&k_states, &self.model.initial_state[0], &inc, &self.simulated_model.initial_state[0], &inc)
        blas.ccopy(&k_states2, &self.model.initial_state_cov[0,0], &inc, &self.simulated_model.initial_state_cov[0,0], &inc)
        blas.ccopy(&k_states2, &self.model.initial_diffuse_state_cov[0,0], &inc, &self.simulated_model.initial_diffuse_state_cov[0,0], &inc)

        if self.has_missing:
            blas.ccopy(&k_states, &self.model.initial_state[0], &inc, &self.secondary_simulated_model.initial_state[0], &inc)
            blas.ccopy(&k_states2, &self.model.initial_state_cov[0,0], &inc, &self.secondary_simulated_model.initial_state_cov[0,0], &inc)
            blas.ccopy(&k_states2, &self.model.initial_diffuse_state_cov[0,0], &inc, &self.secondary_simulated_model.initial_diffuse_state_cov[0,0], &inc)

        # 0. Kalman filter initialization: get alpha_1^+ ~ N(a_1, P_1)
        # Usually, this means transforming the N(0,1) random variate
        # into a N(initial_state, initial_state_cov) random variate.
        # alpha_1^+ = initial_state + variate * chol(initial_state_cov)
        # If pretransformed_variates is True, then the variates should already
        # be N(0, initial_state_cov), and then we just need:
        # alpha_1^+ = initial_state + variate
        # However, if fixed_initial_state is True, then we just set:
        # alpha_1^+ = variate
        blas.ccopy(&k_states, &self.initial_state_variates[0], &inc, &self.generated_state[0,0], &inc)
        if not self.fixed_initial_state:
            self.cholesky(&self.model.initial_state_cov[0,0], self._tmp0, k_states)
            if not self.pretransformed_initial_state_variates:
                self.transform_variates(&self.generated_state[0,0], self._tmp0, k_states)
            # In the case of no missing data, we want to keep the initial state at zero
            if self.has_missing:
                blas.caxpy(&k_states, &alpha, &self.model.initial_state[0], &inc, &self.generated_state[0,0], &inc)


        self.simulated_kfilter.seek(0) # reset the filter
        if self.has_missing:
            self.secondary_simulated_kfilter.seek(0) # reset the filter
        measurement_idx = 0
        state_idx = nobs_endog
        # Note: it is only when we are simulating a series that we might have
        # self.model.nobs != self.simulated_model.nobs, and in that case we do
        # not need to copy over the observations.
        # TODO: Need more explicit support for the pure simulation case, both
        # to avoid errors (like the copy statement below) and because we can
        # probably speed up the process by not running the filter parts.
        if not self.has_missing and (self.model.nobs == self.simulated_model.nobs):
            # reset the obs data in the primary simulated model
            # (but only if there is not missing data - in that case we will
            # combine the actual data with the generated data in the primary
            # model, so copy the actual data here and subtract data below)
            blas.ccopy(&nobs_endog, &self.model.obs[0,0], &inc, &self.simulated_model.obs[0,0], &inc)
        for t in range(self.nobs):
            # 1. Transform independent draws to w_t^+: eps_t^+ = ind_eps * chol(H_t)
            #                                          eta_t^+ = ind_eta * chol(Q_t)

            # 2. Construct y_t^+ = d_t + Z_t alpha_t^+ + eps_t^+
            #      alpha_{t+1}^+ = c_t + T_t alpha_t^+ + eta_t^+

            #    Measurement disturbance (eps)
            # self._tmp1 = chol(H_t)
            if t == 0 or self.model.obs_cov.shape[2] > 1:
                self.cholesky(&self.model.obs_cov[0,0,t], self._tmp1, k_endog)

            # eps_t^+ = ind_eps * chol(H_t)
            if not self.pretransformed_disturbance_variates:
                self.transform_variates(&self.disturbance_variates[measurement_idx], self._tmp1, k_endog)
            # y_t^+
            self.generate_obs(t, &self.generated_obs[0,t], &self.generated_state[0,t], &self.disturbance_variates[measurement_idx])

            measurement_idx += k_endog

            #    State disturbance (eta)
            # self._tmp1 = chol(Q_t)
            if t == 0 or self.model.state_cov.shape[2] > 1:
                self.cholesky(&self.model.state_cov[0,0,t], self._tmp2, k_posdef)

            # eta_t^+ = ind_eta * chol(Q_t)
            if not self.pretransformed_disturbance_variates:
                self.transform_variates(&self.disturbance_variates[state_idx], self._tmp2, k_posdef)
            # alpha_t+1^+
            self.generate_state(t, &self.generated_state[0,t+1], &self.generated_state[0,t], &self.disturbance_variates[state_idx])

            state_idx += k_posdef

            # If we are just generating new series (i.e. all we want is
            # generated_obs, generated_state), go to the next iteration
            if self.simulation_output == 0:
                continue

            # Typically, rather than running the Kalman filter separately for
            # y_t^+ and y_t, we can instead run it over y_t^* = y_t - y_t^+
            if not self.has_missing:
                #    Construct y_t^* = - y_t^+ + y_t
                blas.caxpy(&k_endog, &gamma, &self.generated_obs[0,t], &inc, &self.simulated_model.obs[0, t], &inc)

                # 3. Iterate Kalman filter, based on y_t^*
                #    (this will give us alpha_t+1^*)
                next(self.simulated_kfilter)
            # In the case of missing data, we have to run them separately
            else:
                # 3-1. Iterate the Kalman filter on the y_t^+ data
                #      to get alpha_t+1^+
                blas.ccopy(&k_endog, &self.generated_obs[0,t], &inc, &self.simulated_model.obs[0, t], &inc)
                next(self.simulated_kfilter)

                # 3-2. Iterate the Kalman filter on the y_t data
                #      to get alpha_t+1
                next(self.secondary_simulated_kfilter)

        # If we are just generating new series (i.e. all we want is
        # generated_obs, generated_state), return now
        if self.simulation_output == 0:
            return

        # Backwards recursion
        # This gives us \hat w_t^* = \hat w_t - \hat w_t^+                    (simulation_output & SIMULATE_DISTURBANCE)
        #               \hat alpha_t+1^* = \hat alpha_t+1 - \hat alpha_t+1^+  (simulation_output & SIMULATE_STATE)
        # or if there is missing data:
        # this gives us \hat w_t^+
        #               \hat alpha_t+1
        # and we construct starred versions below
        self.simulated_smoother.smoother_output = simulation_output
        self.simulated_smoother()

        # In the FILTER_COLLAPSED case, we now need to replace nobs_endog with
        # nobs_kstates
        if self.simulated_kfilter.filter_method & FILTER_COLLAPSED:
            nobs_endog = nobs_kstates

        if self.has_missing:
            # This gives us \hat w_t
            #               \hat alpha_t+1
            self.secondary_simulated_smoother.smoother_output = simulation_output
            self.secondary_simulated_smoother()

            # Construct \hat w_t^* = \hat w_t - \hat w_t^+
            #           \hat alpha_t+1^* = \hat alpha_t+1 - \hat alpha_t+1^+
            # Note: this overwrites the values in self.simulated_smoother,
            # so that the steps below will be the same regardless of whether or
            # not there was missing data
            if self.simulation_output & SIMULATE_DISTURBANCE:
                # If there are partially missing entries, we need to re-order
                # the smoothed measurment disturbances.
                tools.creorder_missing_vector(self.secondary_simulated_smoother.smoothed_measurement_disturbance, self.model.missing)
                blas.cswap(&nobs_endog, &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                 &self.secondary_simulated_smoother.smoothed_measurement_disturbance[0,0], &inc)
                blas.caxpy(&nobs_endog, &gamma, &self.secondary_simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                         &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc)
                blas.cswap(&nobs_posdef, &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                  &self.secondary_simulated_smoother.smoothed_state_disturbance[0,0], &inc)
                blas.caxpy(&nobs_posdef, &gamma, &self.secondary_simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                          &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc)

            if self.simulation_output & SIMULATE_STATE:
                blas.cswap(&nobs_kstates, &self.simulated_smoother.smoothed_state[0,0], &inc,
                                                   &self.secondary_simulated_smoother.smoothed_state[0,0], &inc)
                blas.caxpy(&nobs_kstates, &gamma, &self.secondary_simulated_smoother.smoothed_state[0,0], &inc,
                                                           &self.simulated_smoother.smoothed_state[0,0], &inc)

        # Construct the final simulated variables
        # This gives us \tilde w_t = \hat w_t^* + w_t^+                (simulation_output & SIMULATE_DISTURBANCE)
        #               \tilde alpha_t+1 = \hat alpha_t^* + alpha_t^+  (simulation_output & SIMULATE_STATE)
        if self.simulation_output & SIMULATE_DISTURBANCE:
            # \tilde eps_t = \hat eps_t^* + eps_t^+
            # In the FILTER_COLLAPSED case, we have to use a collapsed version of the disturbance variates,
            # so we construct a new representation that can perform the collapse on the variates
            if self.simulated_kfilter.filter_method & FILTER_COLLAPSED:
                dim2[0] = k_endog; dim2[1] = self.nobs;
                collapsing_obs = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX64, FORTRAN)
                nobs_endog = k_endog * self.nobs
                blas.ccopy(&nobs_endog, &self.disturbance_variates[0], &inc, &collapsing_obs[0, 0], &inc)
                nobs_endog = nobs_kstates

                collapsing_model = cStatespace(
                    collapsing_obs,
                    self.model.design, self.model.obs_intercept, self.model.obs_cov,
                    self.model.transition, self.model.state_intercept, self.model.selection,
                    self.model.state_cov
                )
                collapsing_model.initialized = True
                for t in range(self.nobs):
                    collapsing_model.seek(t, self.simulated_kfilter.univariate_filter[t],
                                             self.simulated_kfilter.filter_method & FILTER_COLLAPSED)
                    blas.ccopy(&k_states, &collapsing_model.collapse_obs[0], &inc, &self.simulated_measurement_disturbance[0,t], &inc)
            else:
                blas.ccopy(&nobs_endog, &self.disturbance_variates[0], &inc, &self.simulated_measurement_disturbance[0,0], &inc)

            blas.caxpy(&nobs_endog, &alpha, &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                     &self.simulated_measurement_disturbance[0,0], &inc)

            # \tilde eta_t = \hat eta_t^* + eta_t^+
            blas.ccopy(&nobs_posdef, &self.disturbance_variates[k_endog * self.nobs], &inc, &self.simulated_state_disturbance[0,0], &inc)
            blas.caxpy(&nobs_posdef, &alpha, &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                     &self.simulated_state_disturbance[0,0], &inc)

        if self.simulation_output & SIMULATE_STATE:
            # \tilde alpha_t = \hat alpha_t^* + alpha_t^+
            blas.ccopy(&nobs_kstates, &self.generated_state[0,0], &inc, &self.simulated_state[0,0], &inc)
            blas.caxpy(&nobs_kstates, &alpha, &self.simulated_smoother.smoothed_state[0,0], &inc,
                                                       &self.simulated_state[0,0], &inc)

    cdef np.complex64_t generate_obs(self, int t, np.complex64_t * obs, np.complex64_t * state, np.complex64_t * variates):
        cdef:
            int inc = 1
            int k_endog = self.model.k_endog
            int k_states = self.model.k_states
            int design_t = 0
            int obs_intercept_t = 0
        cdef:
            np.complex64_t alpha = 1.0

        # Get indices for possibly time-varying arrays
        # Note that this would be problematic if simulating a
        # time-varying system if the number of simulated observations was
        # greater than the number of observations in the model (because then
        # e.g. self.model.design[0,0,design_t] would be an invalid memory
        # location).
        # However, this problem is ruled out in the constructor since for
        # time-varying systems, the number of simulated observations must be
        # less than the number of observations in the model.
        if not self.model.time_invariant:
            if self.model.design.shape[2] > 1:             design_t = t
            if self.model.obs_intercept.shape[1] > 1:      obs_intercept_t = t

        # \\# = d_t + \varepsilon_t
        blas.ccopy(&k_endog, variates, &inc, obs, &inc)
        blas.caxpy(&k_endog, &alpha, &self.model.obs_intercept[0,obs_intercept_t], &inc, obs, &inc)

        # y_t = \\# + Z_t alpha_t
        blas.cgemv("N", &k_endog, &k_states,
                            &alpha, &self.model.design[0,0,design_t], &k_endog,
                                    state, &inc,
                            &alpha, obs, &inc)

    cdef np.complex64_t generate_state(self, int t, np.complex64_t * state, np.complex64_t * input_state, np.complex64_t * variates):
        cdef:
            int inc = 1
            int k_states = self.model.k_states
            int k_posdef = self.model.k_posdef
            int state_intercept_t = 0
            int transition_t = 0
            int selection_t = 0
        cdef:
            np.complex64_t alpha = 1.0

        # Get indices for possibly time-varying arrays
        if not self.model.time_invariant:
            if self.model.state_intercept.shape[1] > 1:      state_intercept_t = t
            if self.model.transition.shape[2] > 1:           transition_t = t
            if self.model.selection.shape[2] > 1:            selection_t = t

        # \\# = R_t eta_t + c_t
        blas.ccopy(&k_states, &self.model.state_intercept[0,state_intercept_t], &inc, state, &inc)
        blas.cgemv("N", &k_states, &k_posdef,
                            &alpha, &self.model.selection[0,0,selection_t], &k_states,
                                    variates, &inc,
                            &alpha, state, &inc)

        # alpha_{t+1} = T_t alpha_t + \\#
        blas.cgemv("N", &k_states, &k_states,
                            &alpha, &self.model.transition[0,0,transition_t], &k_states,
                                    input_state, &inc,
                            &alpha, state, &inc)

    cdef void cholesky(self, np.complex64_t * source, np.complex64_t * destination, int n):
        cdef:
            int inc = 1
            int n2 = n**2
            int info
        if n == 1:
            destination[0] = source[0]**0.5
        else:
            blas.ccopy(&n2, source, &inc, destination, &inc)
            lapack.cpotrf("L", &n, destination, &n, &info)

    cdef void transform_variates(self, np.complex64_t * variates, np.complex64_t * cholesky_factor, int n):
        cdef:
            int inc = 1

        # Overwrites variate
        if n == 1:
            variates[0] = cholesky_factor[0] * variates[0]
        else:
            blas.ctrmv("L", "N", "N", &n, cholesky_factor, &n,
                                                   variates, &inc)

cdef class zSimulationSmoother(object):
    # ### Statespace model
    # cdef readonly zStatespace model
    # ### Kalman filter
    # cdef readonly zKalmanFilter kfilter
    # ### Kalman smoother
    # cdef readonly zKalmanSmoother smoother

    # ### Simulated Statespace model
    # cdef readonly zStatespace simulated_model
    # ### Simulated Kalman filter
    # cdef readonly zKalmanFilter simulated_kfilter
    # ### Simulated Kalman smoother
    # cdef readonly zKalmanSmoother simulated_smoother

    # ### Secondary Simulated Statespace model
    # Note: currently only used in the case of missing data
    # cdef readonly zStatespace secondary_simulated_model
    # ### Simulated Kalman filter
    # cdef readonly zKalmanFilter secondary_simulated_kfilter
    # ### Simulated Kalman smoother
    # cdef readonly zKalmanSmoother secondary_simulated_smoother

    # ### Simulation parameters
    # cdef public int simulation_output
    # cdef readonly int has_missing

    # ### Random variates
    # cdef int n_disturbance_variates
    # cdef readonly np.complex128_t [:] disturbance_variates
    # cdef int n_initial_state_variates
    # cdef readonly np.complex128_t [:] initial_state_variates

    # ### Simulated Data
    # cdef readonly np.complex128_t [::1,:] simulated_measurement_disturbance
    # cdef readonly np.complex128_t [::1,:] simulated_state_disturbance
    # cdef readonly np.complex128_t [::1,:] simulated_state

    # ### Generated Data
    # cdef readonly np.complex128_t [::1,:] generated_obs
    # cdef readonly np.complex128_t [::1,:] generated_state

    # ### Temporary arrays
    # cdef readonly np.complex128_t [::1,:] tmp0, tmp1, tmp2

    # ### Pointers
    # cdef np.complex128_t * _tmp0
    # cdef np.complex128_t * _tmp1
    # cdef np.complex128_t * _tmp2

    def __init__(self,
                 zStatespace model,
                 int filter_method=FILTER_CONVENTIONAL,
                 int inversion_method=INVERT_UNIVARIATE | SOLVE_CHOLESKY,
                 int stability_method=STABILITY_FORCE_SYMMETRY,
                 int conserve_memory=MEMORY_STORE_ALL,
                 int filter_timing=TIMING_INIT_PREDICTED,
                 np.float64_t tolerance=1e-19,
                 int loglikelihood_burn=0,
                 int smoother_output=SMOOTHER_ALL,
                 int simulation_output=SIMULATE_ALL,
                 int nobs=-1):
        cdef int inc = 1
        cdef:
            np.npy_intp dim1[1]
            np.npy_intp dim2[2]
        cdef np.complex128_t [::1,:] obs
        cdef np.complex128_t [::1,:] secondary_obs
        cdef int nobs_endog

        # Use model nobs by default
        if nobs == -1:
            nobs = model.nobs
        # Only allow more nobs if a time-invariant model
        elif nobs > model.nobs and model.time_invariant == 0:
            raise ValueError('In a time-varying model, cannot create more'
                             ' simulations than there are observations.')
        elif nobs <= 0:
            raise ValueError('Invalid number of simulations; must be'
                             ' positive.')

        self.nobs = nobs
        nobs_endog = self.nobs * model.k_endog

        self.pretransformed_disturbance_variates = False
        self.pretransformed_initial_state_variates = False
        self.fixed_initial_state = False

        # Model objects
        self.model = model
        # self.kfilter = zKalmanFilter(
        #     self.model, filter_method, inversion_method,
        #     stability_method, conserve_memory,
        #     tolerance, loglikelihood_burn
        # )
        # self.smoother = zKalmanSmoother(
        #     self.model, self.kfilter, smoother_output
        # )

        # Simulated model objects
        dim2[0] = model.k_endog; dim2[1] = self.nobs;
        obs = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN)
        self.simulated_model = zStatespace(
            obs, model.design, model.obs_intercept, model.obs_cov,
            model.transition, model.state_intercept, model.selection,
            model.state_cov
        )
        self.simulated_kfilter = zKalmanFilter(
            self.simulated_model, filter_method, inversion_method,
            stability_method, conserve_memory, filter_timing,
            tolerance, loglikelihood_burn
        )
        self.simulated_smoother = zKalmanSmoother(
            self.simulated_model, self.simulated_kfilter, smoother_output
        )

        # Secondary simulated model objects
        # Currently only used if there is missing data (since then the
        # approach in which the Kalman filter only has to be run over the
        # series y_t^* = y_t - y_t^+ is infeasible), although it could also
        # allow drawing multiple samples at the same time, see Durbin and
        # Koopman (2002).
        self.has_missing = model.has_missing
        if self.has_missing:
            dim2[0] = model.k_endog; dim2[1] = self.nobs;
            secondary_obs = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN)
            blas.zcopy(&nobs_endog, &model.obs[0,0], &inc, &secondary_obs[0,0], &inc)
            self.secondary_simulated_model = zStatespace(
                secondary_obs, model.design, model.obs_intercept, model.obs_cov,
                model.transition, model.state_intercept, model.selection,
                model.state_cov
            )
            self.secondary_simulated_kfilter = zKalmanFilter(
                self.secondary_simulated_model, filter_method, inversion_method,
                stability_method, conserve_memory, filter_timing,
                tolerance, loglikelihood_burn
            )
            self.secondary_simulated_smoother = zKalmanSmoother(
                self.secondary_simulated_model, self.secondary_simulated_kfilter, smoother_output
            )
        # In the case of non-missing data, the Kalman filter will actually
        # be run over y_t^* = y_t - y_t^+, which means the observation equation
        # intercept should be zero; make sure that it is
        else:
            dim2[0] = self.model.k_endog; dim2[1] = self.model.obs_intercept.shape[1];
            self.simulated_model.obs_intercept = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN)
            dim2[0] = self.model.k_states; dim2[1] = self.model.state_intercept.shape[1];
            self.simulated_model.state_intercept = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN)


        # Initialize the simulated model memoryviews
        # Note: the actual initialization is replaced in the simulate()
        # function below, but will complain if the memoryviews have not been
        # first initialized, which this call does.
        self.simulated_model.initialize_approximate_diffuse()
        if self.has_missing:
            self.secondary_simulated_model.initialize_approximate_diffuse()

        # Parameters
        self.simulation_output = simulation_output
        self.n_disturbance_variates = self.nobs * (self.model.k_endog + self.model.k_posdef)
        self.n_initial_state_variates = self.model.k_states

        # Random variates
        dim1[0] = self.n_disturbance_variates;
        self.disturbance_variates = np.PyArray_ZEROS(1, dim1, np.NPY_COMPLEX128, FORTRAN)
        dim1[0] = self.n_initial_state_variates;
        self.initial_state_variates = np.PyArray_ZEROS(1, dim1, np.NPY_COMPLEX128, FORTRAN)

        # Simulated data (\tilde eta_t, \tilde eps_t, \tilde alpha_t)
        # Note that these are (k_endog x nobs), (k_posdef x nobs), (k_states x nobs)
        if filter_method & FILTER_COLLAPSED:
            dim2[0] = self.model.k_states; dim2[1] = self.nobs;
        else:
            dim2[0] = self.model.k_endog; dim2[1] = self.nobs;
        self.simulated_measurement_disturbance = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN)
        dim2[0] = self.model.k_posdef; dim2[1] = self.nobs;
        self.simulated_state_disturbance = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN)
        dim2[0] = self.model.k_states; dim2[1] = self.nobs;
        self.simulated_state = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN)

        # Generated data (y_t^+, alpha_t^+)
        dim2[0] = self.model.k_endog; dim2[1] = self.nobs;
        self.generated_obs = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN)
        dim2[0] = self.model.k_states; dim2[1] = self.nobs+1;
        self.generated_state = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN)

        # Temporary arrays
        dim2[0] = self.model.k_states; dim2[1] = self.model.k_states;
        self.tmp0 = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN) # chol(P_1)
        dim2[0] = self.model.k_posdef; dim2[1] = self.model.k_posdef;
        self.tmp2 = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN) # chol(Q_t)
        dim2[0] = self.model.k_endog; dim2[1] = self.model.k_endog;
        self.tmp1 = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN) # chol(H_t)

        # Pointers
        self._tmp0 = &self.tmp0[0,0]
        self._tmp1 = &self.tmp1[0,0]
        self._tmp2 = &self.tmp2[0,0]

    def __reduce__(self):
        args = (self.model, self.model.filter_method, self.model.inversion_method,
                self.model.stability_method, self.model.conserve_memory, self.model.filter_timing,
                self.model.tolerance, self.model.loglikelihood_burn, self.simulated_smoother.smoother_output,
                self.simulation_output, self.nobs, self.pretransformed_variates)
        state = {
            'disturbance_variates': np.array(self.disturbance_variates, copy=True, order='F'),
            'initial_state_variates': np.array(self.initial_state_variates, copy=True, order='F'),
            'simulated_measurement_disturbance': np.array(self.simulated_measurement_disturbance, copy=True, order='F'),
            'simulated_state_disturbance': np.array(self.simulated_state_disturbance, copy=True, order='F'),
            'simulated_state': np.array(self.simulated_state, copy=True, order='F'),
            'generated_obs': np.array(self.generated_obs, copy=True, order='F'),
            'generated_state': np.array(self.generated_state, copy=True, order='F'),
            'tmp0': np.array(self.tmp0, copy=True, order='F'),
            'tmp2': np.array(self.tmp2, copy=True, order='F'),
            'tmp1': np.array(self.tmp1, copy=True, order='F')
        }
        return (self.__class__, args, state)

    def __setstate__(self, state):
        self.disturbance_variates = state['disturbance_variates']
        self.initial_state_variates = state['initial_state_variates']
        self.simulated_measurement_disturbance  = state['simulated_measurement_disturbance']
        self.simulated_state_disturbance = state['simulated_state_disturbance']
        self.simulated_state = state['simulated_state']
        self.generated_obs = state['generated_obs']
        self.generated_state = state['generated_state']
        self.tmp0 = state['tmp0']
        self.tmp2 = state['tmp2']
        self.tmp1 = state['tmp1']

    cdef void _reinitialize_temp_pointers(self) except *:
        self._tmp0 = &self.tmp0[0,0]
        self._tmp1 = &self.tmp1[0,0]
        self._tmp2 = &self.tmp2[0,0]


    cpdef draw_disturbance_variates(self):
        self.disturbance_variates = np.random.normal(size=self.n_disturbance_variates)
        self.pretransformed_disturbance_variates = False

    cpdef draw_initial_state_variates(self):
        self.initial_state_variates = np.random.normal(size=self.n_initial_state_variates)
        self.pretransformed_initial_state_variates = False
        self.fixed_initial_state = False

    cpdef set_disturbance_variates(self, np.complex128_t [:] variates, int pretransformed=0):
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('disturbance variates', &variates.shape[0],
                                    self.n_disturbance_variates)
        self.disturbance_variates = variates
        self.pretransformed_disturbance_variates = pretransformed

    cpdef set_initial_state_variates(self, np.complex128_t [:] variates, int pretransformed=0):
        # Note that the initial state is set to be:
        # initial_state = mod.initial_state + initial_state_variate * cholesky(mod.initial_state_cov)
        # so is can be difficult to set the initial state itself via this method;
        # see instead set_initial_state
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('initial state variates',
                                    &variates.shape[0],
                                    self.n_initial_state_variates)
        self.initial_state_variates = variates
        self.pretransformed_initial_state_variates = pretransformed
        self.fixed_initial_state = False

    cpdef set_initial_state(self, np.complex128_t [:] initial_state):
        # Using this method sets a flag that indicates the self.initial_state_variates
        # variable should be interpreted as the actual initial_state.
        # TODO allow variates to be an iterator or callback
        tools.validate_vector_shape('initial state',
                                    &initial_state.shape[0],
                                    self.n_initial_state_variates)
        self.initial_state_variates = initial_state
        self.pretransformed_initial_state_variates = True
        self.fixed_initial_state = True

    cpdef simulate(self, int simulation_output=-1):
        """
        Draw a simulation
        """
        cdef:
            int inc = 1
            int info
            int measurement_idx, state_idx, t
            int k_endog = self.model.k_endog
            int k_states = self.model.k_states
            int k_states2 = self.model.k_states**2
            int k_posdef = self.model.k_posdef
            int k_posdef2 = self.model.k_posdef**2
            int nobs_endog = self.nobs * self.model.k_endog
            int nobs_kstates = self.nobs * self.model.k_states
            int nobs1_kstates = (self.nobs+1) * self.model.k_states
            int nobs_posdef = self.nobs * self.model.k_posdef
        cdef:
            np.complex128_t alpha = 1.0
            np.complex128_t gamma = -1.0
        cdef zStatespace collapsing_model
        cdef np.complex128_t [:,:] collapsing_obs
        cdef np.npy_intp dim2[2]


        if simulation_output == -1:
            simulation_output = self.simulation_output
        
        # Forwards recursion
        # 0. Statespace initialization
        if not self.model.initialized:
            raise RuntimeError("Statespace model not initialized.")
        blas.zcopy(&k_states, &self.model.initial_state[0], &inc, &self.simulated_model.initial_state[0], &inc)
        blas.zcopy(&k_states2, &self.model.initial_state_cov[0,0], &inc, &self.simulated_model.initial_state_cov[0,0], &inc)
        blas.zcopy(&k_states2, &self.model.initial_diffuse_state_cov[0,0], &inc, &self.simulated_model.initial_diffuse_state_cov[0,0], &inc)

        if self.has_missing:
            blas.zcopy(&k_states, &self.model.initial_state[0], &inc, &self.secondary_simulated_model.initial_state[0], &inc)
            blas.zcopy(&k_states2, &self.model.initial_state_cov[0,0], &inc, &self.secondary_simulated_model.initial_state_cov[0,0], &inc)
            blas.zcopy(&k_states2, &self.model.initial_diffuse_state_cov[0,0], &inc, &self.secondary_simulated_model.initial_diffuse_state_cov[0,0], &inc)

        # 0. Kalman filter initialization: get alpha_1^+ ~ N(a_1, P_1)
        # Usually, this means transforming the N(0,1) random variate
        # into a N(initial_state, initial_state_cov) random variate.
        # alpha_1^+ = initial_state + variate * chol(initial_state_cov)
        # If pretransformed_variates is True, then the variates should already
        # be N(0, initial_state_cov), and then we just need:
        # alpha_1^+ = initial_state + variate
        # However, if fixed_initial_state is True, then we just set:
        # alpha_1^+ = variate
        blas.zcopy(&k_states, &self.initial_state_variates[0], &inc, &self.generated_state[0,0], &inc)
        if not self.fixed_initial_state:
            self.cholesky(&self.model.initial_state_cov[0,0], self._tmp0, k_states)
            if not self.pretransformed_initial_state_variates:
                self.transform_variates(&self.generated_state[0,0], self._tmp0, k_states)
            # In the case of no missing data, we want to keep the initial state at zero
            if self.has_missing:
                blas.zaxpy(&k_states, &alpha, &self.model.initial_state[0], &inc, &self.generated_state[0,0], &inc)


        self.simulated_kfilter.seek(0) # reset the filter
        if self.has_missing:
            self.secondary_simulated_kfilter.seek(0) # reset the filter
        measurement_idx = 0
        state_idx = nobs_endog
        # Note: it is only when we are simulating a series that we might have
        # self.model.nobs != self.simulated_model.nobs, and in that case we do
        # not need to copy over the observations.
        # TODO: Need more explicit support for the pure simulation case, both
        # to avoid errors (like the copy statement below) and because we can
        # probably speed up the process by not running the filter parts.
        if not self.has_missing and (self.model.nobs == self.simulated_model.nobs):
            # reset the obs data in the primary simulated model
            # (but only if there is not missing data - in that case we will
            # combine the actual data with the generated data in the primary
            # model, so copy the actual data here and subtract data below)
            blas.zcopy(&nobs_endog, &self.model.obs[0,0], &inc, &self.simulated_model.obs[0,0], &inc)
        for t in range(self.nobs):
            # 1. Transform independent draws to w_t^+: eps_t^+ = ind_eps * chol(H_t)
            #                                          eta_t^+ = ind_eta * chol(Q_t)

            # 2. Construct y_t^+ = d_t + Z_t alpha_t^+ + eps_t^+
            #      alpha_{t+1}^+ = c_t + T_t alpha_t^+ + eta_t^+

            #    Measurement disturbance (eps)
            # self._tmp1 = chol(H_t)
            if t == 0 or self.model.obs_cov.shape[2] > 1:
                self.cholesky(&self.model.obs_cov[0,0,t], self._tmp1, k_endog)

            # eps_t^+ = ind_eps * chol(H_t)
            if not self.pretransformed_disturbance_variates:
                self.transform_variates(&self.disturbance_variates[measurement_idx], self._tmp1, k_endog)
            # y_t^+
            self.generate_obs(t, &self.generated_obs[0,t], &self.generated_state[0,t], &self.disturbance_variates[measurement_idx])

            measurement_idx += k_endog

            #    State disturbance (eta)
            # self._tmp1 = chol(Q_t)
            if t == 0 or self.model.state_cov.shape[2] > 1:
                self.cholesky(&self.model.state_cov[0,0,t], self._tmp2, k_posdef)

            # eta_t^+ = ind_eta * chol(Q_t)
            if not self.pretransformed_disturbance_variates:
                self.transform_variates(&self.disturbance_variates[state_idx], self._tmp2, k_posdef)
            # alpha_t+1^+
            self.generate_state(t, &self.generated_state[0,t+1], &self.generated_state[0,t], &self.disturbance_variates[state_idx])

            state_idx += k_posdef

            # If we are just generating new series (i.e. all we want is
            # generated_obs, generated_state), go to the next iteration
            if self.simulation_output == 0:
                continue

            # Typically, rather than running the Kalman filter separately for
            # y_t^+ and y_t, we can instead run it over y_t^* = y_t - y_t^+
            if not self.has_missing:
                #    Construct y_t^* = - y_t^+ + y_t
                blas.zaxpy(&k_endog, &gamma, &self.generated_obs[0,t], &inc, &self.simulated_model.obs[0, t], &inc)

                # 3. Iterate Kalman filter, based on y_t^*
                #    (this will give us alpha_t+1^*)
                next(self.simulated_kfilter)
            # In the case of missing data, we have to run them separately
            else:
                # 3-1. Iterate the Kalman filter on the y_t^+ data
                #      to get alpha_t+1^+
                blas.zcopy(&k_endog, &self.generated_obs[0,t], &inc, &self.simulated_model.obs[0, t], &inc)
                next(self.simulated_kfilter)

                # 3-2. Iterate the Kalman filter on the y_t data
                #      to get alpha_t+1
                next(self.secondary_simulated_kfilter)

        # If we are just generating new series (i.e. all we want is
        # generated_obs, generated_state), return now
        if self.simulation_output == 0:
            return

        # Backwards recursion
        # This gives us \hat w_t^* = \hat w_t - \hat w_t^+                    (simulation_output & SIMULATE_DISTURBANCE)
        #               \hat alpha_t+1^* = \hat alpha_t+1 - \hat alpha_t+1^+  (simulation_output & SIMULATE_STATE)
        # or if there is missing data:
        # this gives us \hat w_t^+
        #               \hat alpha_t+1
        # and we construct starred versions below
        self.simulated_smoother.smoother_output = simulation_output
        self.simulated_smoother()

        # In the FILTER_COLLAPSED case, we now need to replace nobs_endog with
        # nobs_kstates
        if self.simulated_kfilter.filter_method & FILTER_COLLAPSED:
            nobs_endog = nobs_kstates

        if self.has_missing:
            # This gives us \hat w_t
            #               \hat alpha_t+1
            self.secondary_simulated_smoother.smoother_output = simulation_output
            self.secondary_simulated_smoother()

            # Construct \hat w_t^* = \hat w_t - \hat w_t^+
            #           \hat alpha_t+1^* = \hat alpha_t+1 - \hat alpha_t+1^+
            # Note: this overwrites the values in self.simulated_smoother,
            # so that the steps below will be the same regardless of whether or
            # not there was missing data
            if self.simulation_output & SIMULATE_DISTURBANCE:
                # If there are partially missing entries, we need to re-order
                # the smoothed measurment disturbances.
                tools.zreorder_missing_vector(self.secondary_simulated_smoother.smoothed_measurement_disturbance, self.model.missing)
                blas.zswap(&nobs_endog, &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                 &self.secondary_simulated_smoother.smoothed_measurement_disturbance[0,0], &inc)
                blas.zaxpy(&nobs_endog, &gamma, &self.secondary_simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                         &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc)
                blas.zswap(&nobs_posdef, &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                  &self.secondary_simulated_smoother.smoothed_state_disturbance[0,0], &inc)
                blas.zaxpy(&nobs_posdef, &gamma, &self.secondary_simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                          &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc)

            if self.simulation_output & SIMULATE_STATE:
                blas.zswap(&nobs_kstates, &self.simulated_smoother.smoothed_state[0,0], &inc,
                                                   &self.secondary_simulated_smoother.smoothed_state[0,0], &inc)
                blas.zaxpy(&nobs_kstates, &gamma, &self.secondary_simulated_smoother.smoothed_state[0,0], &inc,
                                                           &self.simulated_smoother.smoothed_state[0,0], &inc)

        # Construct the final simulated variables
        # This gives us \tilde w_t = \hat w_t^* + w_t^+                (simulation_output & SIMULATE_DISTURBANCE)
        #               \tilde alpha_t+1 = \hat alpha_t^* + alpha_t^+  (simulation_output & SIMULATE_STATE)
        if self.simulation_output & SIMULATE_DISTURBANCE:
            # \tilde eps_t = \hat eps_t^* + eps_t^+
            # In the FILTER_COLLAPSED case, we have to use a collapsed version of the disturbance variates,
            # so we construct a new representation that can perform the collapse on the variates
            if self.simulated_kfilter.filter_method & FILTER_COLLAPSED:
                dim2[0] = k_endog; dim2[1] = self.nobs;
                collapsing_obs = np.PyArray_ZEROS(2, dim2, np.NPY_COMPLEX128, FORTRAN)
                nobs_endog = k_endog * self.nobs
                blas.zcopy(&nobs_endog, &self.disturbance_variates[0], &inc, &collapsing_obs[0, 0], &inc)
                nobs_endog = nobs_kstates

                collapsing_model = zStatespace(
                    collapsing_obs,
                    self.model.design, self.model.obs_intercept, self.model.obs_cov,
                    self.model.transition, self.model.state_intercept, self.model.selection,
                    self.model.state_cov
                )
                collapsing_model.initialized = True
                for t in range(self.nobs):
                    collapsing_model.seek(t, self.simulated_kfilter.univariate_filter[t],
                                             self.simulated_kfilter.filter_method & FILTER_COLLAPSED)
                    blas.zcopy(&k_states, &collapsing_model.collapse_obs[0], &inc, &self.simulated_measurement_disturbance[0,t], &inc)
            else:
                blas.zcopy(&nobs_endog, &self.disturbance_variates[0], &inc, &self.simulated_measurement_disturbance[0,0], &inc)

            blas.zaxpy(&nobs_endog, &alpha, &self.simulated_smoother.smoothed_measurement_disturbance[0,0], &inc,
                                                     &self.simulated_measurement_disturbance[0,0], &inc)

            # \tilde eta_t = \hat eta_t^* + eta_t^+
            blas.zcopy(&nobs_posdef, &self.disturbance_variates[k_endog * self.nobs], &inc, &self.simulated_state_disturbance[0,0], &inc)
            blas.zaxpy(&nobs_posdef, &alpha, &self.simulated_smoother.smoothed_state_disturbance[0,0], &inc,
                                                     &self.simulated_state_disturbance[0,0], &inc)

        if self.simulation_output & SIMULATE_STATE:
            # \tilde alpha_t = \hat alpha_t^* + alpha_t^+
            blas.zcopy(&nobs_kstates, &self.generated_state[0,0], &inc, &self.simulated_state[0,0], &inc)
            blas.zaxpy(&nobs_kstates, &alpha, &self.simulated_smoother.smoothed_state[0,0], &inc,
                                                       &self.simulated_state[0,0], &inc)

    cdef np.complex128_t generate_obs(self, int t, np.complex128_t * obs, np.complex128_t * state, np.complex128_t * variates):
        cdef:
            int inc = 1
            int k_endog = self.model.k_endog
            int k_states = self.model.k_states
            int design_t = 0
            int obs_intercept_t = 0
        cdef:
            np.complex128_t alpha = 1.0

        # Get indices for possibly time-varying arrays
        # Note that this would be problematic if simulating a
        # time-varying system if the number of simulated observations was
        # greater than the number of observations in the model (because then
        # e.g. self.model.design[0,0,design_t] would be an invalid memory
        # location).
        # However, this problem is ruled out in the constructor since for
        # time-varying systems, the number of simulated observations must be
        # less than the number of observations in the model.
        if not self.model.time_invariant:
            if self.model.design.shape[2] > 1:             design_t = t
            if self.model.obs_intercept.shape[1] > 1:      obs_intercept_t = t

        # \\# = d_t + \varepsilon_t
        blas.zcopy(&k_endog, variates, &inc, obs, &inc)
        blas.zaxpy(&k_endog, &alpha, &self.model.obs_intercept[0,obs_intercept_t], &inc, obs, &inc)

        # y_t = \\# + Z_t alpha_t
        blas.zgemv("N", &k_endog, &k_states,
                            &alpha, &self.model.design[0,0,design_t], &k_endog,
                                    state, &inc,
                            &alpha, obs, &inc)

    cdef np.complex128_t generate_state(self, int t, np.complex128_t * state, np.complex128_t * input_state, np.complex128_t * variates):
        cdef:
            int inc = 1
            int k_states = self.model.k_states
            int k_posdef = self.model.k_posdef
            int state_intercept_t = 0
            int transition_t = 0
            int selection_t = 0
        cdef:
            np.complex128_t alpha = 1.0

        # Get indices for possibly time-varying arrays
        if not self.model.time_invariant:
            if self.model.state_intercept.shape[1] > 1:      state_intercept_t = t
            if self.model.transition.shape[2] > 1:           transition_t = t
            if self.model.selection.shape[2] > 1:            selection_t = t

        # \\# = R_t eta_t + c_t
        blas.zcopy(&k_states, &self.model.state_intercept[0,state_intercept_t], &inc, state, &inc)
        blas.zgemv("N", &k_states, &k_posdef,
                            &alpha, &self.model.selection[0,0,selection_t], &k_states,
                                    variates, &inc,
                            &alpha, state, &inc)

        # alpha_{t+1} = T_t alpha_t + \\#
        blas.zgemv("N", &k_states, &k_states,
                            &alpha, &self.model.transition[0,0,transition_t], &k_states,
                                    input_state, &inc,
                            &alpha, state, &inc)

    cdef void cholesky(self, np.complex128_t * source, np.complex128_t * destination, int n):
        cdef:
            int inc = 1
            int n2 = n**2
            int info
        if n == 1:
            destination[0] = source[0]**0.5
        else:
            blas.zcopy(&n2, source, &inc, destination, &inc)
            lapack.zpotrf("L", &n, destination, &n, &info)

    cdef void transform_variates(self, np.complex128_t * variates, np.complex128_t * cholesky_factor, int n):
        cdef:
            int inc = 1

        # Overwrites variate
        if n == 1:
            variates[0] = cholesky_factor[0] * variates[0]
        else:
            blas.ztrmv("L", "N", "N", &n, cholesky_factor, &n,
                                                   variates, &inc)
