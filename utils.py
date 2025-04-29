import numpy as np
from scipy import signal
from scipy.signal import hilbert

def compute_speed(x, y, t):
    """
    Compute speed from x, y, and t data.
    """
    speed = np.zeros(x.shape)
    speed[1:] = np.sqrt(np.diff(x)**2 + np.diff(y)**2) / np.diff(t)
    speed[0] = speed[1]
    return speed

# def extract_high_speed_periods(speed, time, speed_thresh=5.0, min_duration=0.5):
#     """
#     Extracts contiguous periods where speed exceeds a threshold.

#     Parameters:
#     -----------
#     speed : np.ndarray
#         1D array of speed values.
#     time : np.ndarray
#         1D array of time values corresponding to each speed sample.
#     speed_thresh : float
#         Minimum speed to define a "high-speed" period (in same units as `speed`).
#     min_duration : float
#         Minimum duration (in seconds) for a high-speed period to be considered valid.

#     Returns:
#     --------
#     List of tuples:
#         Each tuple is (start_time, end_time) of a high-speed period.
#     """
#     above_thresh = speed > speed_thresh
#     if not np.any(above_thresh):
#         return []

#     # Find changes in the high-speed mask
#     diff = np.diff(above_thresh.astype(int))
#     start_indices = np.where(diff == 1)[0] + 1
#     end_indices = np.where(diff == -1)[0] + 1

#     # Handle if high-speed period starts from the first frame
#     if above_thresh[0]:
#         start_indices = np.insert(start_indices, 0, 0)
#     # Handle if high-speed period continues till the last frame
#     if above_thresh[-1]:
#         end_indices = np.append(end_indices, len(speed))

#     # Filter by minimum duration
#     periods = []
#     for start, end in zip(start_indices, end_indices):
#         duration = time[end - 1] - time[start]
#         if duration >= min_duration:
#             periods.append((time[start], time[end - 1]))

#     return periods

def extract_high_speed_periods(speed, time, position, speed_thresh=5.0, min_duration=0.5, max_angle_std=np.pi/6):
    """
    Extracts contiguous high-speed periods where trajectory is also roughly straight.

    Parameters:
    -----------
    speed : np.ndarray
        1D array of speed values.
    time : np.ndarray
        1D array of time values (same length as speed).
    position : np.ndarray
        2D array of shape (2, N) giving x and y positions.
    speed_thresh : float
        Speed threshold to define "high-speed".
    min_duration : float
        Minimum duration (seconds) to consider a valid period.
    max_angle_std : float
        Maximum circular standard deviation (radians) of heading angles for a "straight" segment.

    Returns:
    --------
    List of tuples:
        Each tuple is (start_time, end_time) for valid high-speed straight-line segments.
    """
    above_thresh = speed > speed_thresh
    if not np.any(above_thresh):
        return []

    diff = np.diff(above_thresh.astype(int))
    start_indices = np.where(diff == 1)[0] + 1
    end_indices = np.where(diff == -1)[0] + 1

    if above_thresh[0]:
        start_indices = np.insert(start_indices, 0, 0)
    if above_thresh[-1]:
        end_indices = np.append(end_indices, len(speed))

    periods = []
    for start, end in zip(start_indices, end_indices):
        duration = time[end - 1] - time[start]
        if duration < min_duration:
            continue

        # Compute heading angles from position differences
        dx = np.diff(position[start:end, 0])
        dy = np.diff(position[start:end, 1])
        angles = np.arctan2(dy, dx)

        # Circular standard deviation
        mean_angle = np.angle(np.mean(np.exp(1j * angles)))
        angle_diffs = np.angle(np.exp(1j * (angles - mean_angle)))
        angle_std = np.sqrt(np.mean(angle_diffs**2))

        if angle_std <= max_angle_std:
            periods.append((time[start], time[end - 1]))

    return periods

def compute_spike_array(spike_times, timestamps):
    """
    Compute the spike array from spike times and timestamps.
    """
    
    #for loop is too slow:
    # spike_array = np.zeros(timestamps.shape)
    # for spike_time in spike_times:
    #     # Find the index of the closest time in t to this spike_time
    #     idx = np.argmin(np.abs(timestamps - spike_time))
    #     spike_array[idx] += 1
    
    # Assume t is sorted. If not, you must sort it first.
    indices = np.searchsorted(timestamps, spike_times, side='right')

    # Ensure no out-of-bounds indices
    indices = np.clip(indices, 0, len(timestamps) - 1)

    # Adjust indices to always refer to the nearest timestamp
    # If the index points to the start of the array, no need to adjust
    # If not, check if the previous index is closer
    prev_close = (indices > 0) & (np.abs(timestamps[indices - 1] - spike_times) < np.abs(timestamps[indices] - spike_times))
    indices[prev_close] -= 1

    spike_array = np.zeros_like(timestamps)
    np.add.at(spike_array, indices, 1) #very fast!
    
    return spike_array

def bandpassfilter(data, lowcut=6, highcut=12, fs=100):
    """
    band pass filter of the signal
    """
    lowcut = 5
    highcut = 11
    order = 5
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype="band")
    filtereddata = signal.filtfilt(b, a, data)

    return filtereddata


def get_phase(filtered_lfp):
    """
    get the instantaneous phase of the filtered lfp signal
    """
    
    analytic_signal = hilbert(filtered_lfp)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    # wrap the instantaneous_phase to -pi and pi
    instantaneous_phase = np.mod(instantaneous_phase + np.pi, 2 * np.pi)
    
    return instantaneous_phase

def maximum_a_posteriori_estimate(posterior_density):
    '''

    Parameters
    ----------
    posterior_density : xarray.DataArray, shape (n_time, n_x_bins, n_y_bins)

    Returns
    -------
    map_estimate : xarray.DataArray, shape (n_time,)

    '''
    stacked_posterior = np.log(posterior_density.stack(
        z=['x_position', 'y_position']))
    map_estimate = stacked_posterior.z[stacked_posterior.argmax('z')]
    return map_estimate