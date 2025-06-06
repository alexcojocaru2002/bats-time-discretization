#define INFINITY __int_as_float(0x7f800000)

extern "C" {
    __device__ void get_sample_params(const float **discrete_spike_times,
                                      const float **spike_times,
                                      const float **exp_tau_s,
                                      const float **exp_tau,
                                      const float **spike_weights,
                                      int n_neurons, int sample_idx, int neuron_idx, int max_n_pre_spike) {
        int sample_start_idx = sample_idx * max_n_pre_spike;

        *discrete_spike_times += sample_start_idx;
        *spike_times += sample_start_idx;
        *exp_tau_s += sample_start_idx;
        *exp_tau += sample_start_idx;
        *spike_weights += (sample_idx * n_neurons + neuron_idx) * max_n_pre_spike;
    }

    __device__ void get_neuron_results(int **n_spikes,
                                       float **a,
                                       float **x,
                                       float **spike_times,
                                       float **discrete_spike_times,
                                       float **post_exp_tau,
                                       int n_neurons, int sample_idx, int neuron_idx, int max_n_post_spike) {
        int sample_neuron_idx = (sample_idx * n_neurons + neuron_idx);
        int res_start_idx = sample_neuron_idx * max_n_post_spike;

        *n_spikes += sample_neuron_idx;
        *a += res_start_idx; // Assuming the spike numbers is the same
        *x += res_start_idx; // Might have a problem as we miss more discrete spikes
        *spike_times += res_start_idx;
        *discrete_spike_times += res_start_idx;
        *post_exp_tau += res_start_idx;
    }

    __device__ bool compute_spikes(const float c,
                                   int *n_spikes,
                                   float *a,
                                   float *x,
                                   float *spike_times,
                                   float *discrete_spike_times,
                                   float *post_exp_tau,
                                   float cumul_a,
                                   float *cumul_b,
                                   float last_spike,
                                   float discrete_last_spike,
                                   float next_spike,
                                   float delta_theta_tau,
                                   float tau,
                                   float time_delta,
                                   float max_simulation,
                                   int neuron_idx,
                                   int max_n_post_spike,
                                   int sample_idx) {
        float x_tmp, inside_log, tmp, tmp_disc;

        // Compute until there is no spike anymore
        while (true) {
            tmp = (*cumul_b) * (*cumul_b) - 4.0 * cumul_a * c;

            if (tmp < 0) // Negative discriminant, no spike --> stop
                return false;
            x_tmp = sqrtf(tmp);
            tmp = x_tmp + (*cumul_b); // X_k + b_k

            if (tmp == 0.0) // Division per zero, no spike --> stop
                return false;
            inside_log = 2 * cumul_a / tmp;
            if (inside_log < 0) // Negative log, no spike --> stop
                return false;

            tmp = tau * __logf(inside_log); // t_k
            
            float potential = tau * ((*cumul_b) * __expf(-tmp / tau) - cumul_a * __expf(-2 * tmp / tau));
            
            // 0.0001 can be adjusted depending on desired margin for error
            if (potential + 0.0001 < (c * tau)) {
            	return false;
            }

            // Add timeshift
//             tmp = tmp + ((3 * time_delta) / 4) ;
            tmp_disc = tmp;

            // Discretize to next time step
            if (time_delta != 0.0f)
                tmp_disc = tmp_disc + (time_delta - fmodf(tmp, time_delta));
            else
                tmp_disc = tmp;

            // Spike time is before the last pre-spike or after the next spike --> stop
            if (tmp_disc <= discrete_last_spike || tmp_disc > max_simulation || tmp_disc > next_spike)
                return false;

            discrete_spike_times[*n_spikes] = tmp_disc;
            // Spike time is valid
            //printf("%f %f\n", tmp, discrete_spike_times[*n_spikes]);
            a[*n_spikes] = cumul_a;
            x[*n_spikes] = x_tmp;
            spike_times[*n_spikes] = tmp;
            discrete_last_spike = discrete_spike_times[*n_spikes]; // we use the discrete spikes
            last_spike = tmp;
            post_exp_tau[*n_spikes] = inside_log;
            *cumul_b -= delta_theta_tau * inside_log; // Apply reset to b
            (*n_spikes)++;
            if (*n_spikes >= max_n_post_spike) {
                return true;
            }
        }
    }

    __global__ void compute_spike_times_kernel(// Parameters
                                               const float *discrete_spike_times,
                                               const float *spike_times,
                                               const float *exp_tau_s,
                                               const float *exp_tau,
                                               const float *spike_weights,
                                               const float c,
                                               float delta_theta_tau,
                                               float tau,
                                               float time_delta,
                                               float max_simulation,
                                               int max_n_pre_spike,
                                               int max_n_post_spike,
                                               // Outputs	
                                               int *n_spikes,
                                               float *a,
                                               float *x,
                                               float *out_spike_times,
                                               float *out_discrete_spike_times,
                                               float *post_exp_tau) {
        int n_neurons = gridDim.x;
        int sample_idx = threadIdx.x;
        int neuron_idx = blockIdx.x;

        //if(spike_times[0] != 0 )
            //printf("%f %f\n", spike_times[0], discrete_spike_times[0]);

        get_sample_params(&discrete_spike_times, &spike_times, &exp_tau_s, &exp_tau, &spike_weights,
                          n_neurons, sample_idx, neuron_idx, max_n_pre_spike);
        get_neuron_results(&n_spikes, &a, &x, &out_spike_times, &out_discrete_spike_times, &post_exp_tau,
                           n_neurons, sample_idx, neuron_idx, max_n_post_spike);

        float cumul_a = 0.0;
        float cumul_b = 0.0;
        float weight;
        int next_i;
        float next_spike;

        for (int i = 0; i < max_n_pre_spike; i++) {
            if (discrete_spike_times[i] == INFINITY) // No spike anymore --> stop
                break;
            weight = spike_weights[i];

            cumul_a += weight * exp_tau_s[i];
            cumul_b += weight * exp_tau[i];


            next_i = i + 1;

            //printf("%f %f\n", spike_times[next_i], discrete_spike_times[next_i]);

            if (next_i < max_n_pre_spike)
                next_spike = discrete_spike_times[next_i]; // we use discrete spike times
            else                                  // replacing this with discrete breaks everything
                next_spike = INFINITY;
            if (compute_spikes(c, n_spikes, a, x, out_spike_times, out_discrete_spike_times, post_exp_tau,
                               cumul_a, &cumul_b, spike_times[i], discrete_spike_times[i], next_spike, delta_theta_tau, tau, time_delta,
                               max_simulation, neuron_idx, max_n_post_spike, sample_idx))
                break; // Buffer full
            //if(out_spike_times[n_spikes[i]] != INFINITY)
            //printf("%d\n", n_spikes[i]);
            //  printf("%f %f\n", out_spike_times[n_spikes[i]], out_discrete_spike_times[n_spikes[i]]);
            //n_spikes[i] is the spike
        }
    }
}
